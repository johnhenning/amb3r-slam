"""Streaming API and bounded asynchronous capture.

Frontend state belongs to the caller thread. Backend state belongs to one
worker. Corrections cross that boundary only through completed immutable-by-
convention BackendUpdate objects. Separate model instances avoid unsafe reuse.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import numpy as np

from .backend import HierarchicalBackend, SlamConfig
from .frontend import Frontend
from .store import FrameStore
from .types import Frame


@dataclass
class TrackingResult:
    frame_id: int
    timestamp: float
    pose: np.ndarray
    confidence: float
    latency_ms: float
    backend_pending: bool


class SlamSystem:
    """``push(frame)`` returns immediately after tracking in asynchronous mode.

    Backpressure waits only when a new mapping window must be submitted. This
    preserves the n/3 mapping stride and span-2 overlap instead of silently
    skipping mandatory submaps. Live capture drops stale frames upstream.
    """

    def __init__(self, frontend_model, backend_model, directory, config=None, asynchronous=True):
        self.config = config or SlamConfig()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any((self.directory / "frames").glob("*.npz")):
            raise ValueError("Run directory already contains frames; choose a new directory")
        self.store = FrameStore(self.directory / "frames", self.config.window * 2)
        self.frontend = Frontend(frontend_model)
        self.backend = HierarchicalBackend(backend_model, self.directory, self.config)
        self.executor = ThreadPoolExecutor(max_workers=1) if asynchronous else None
        self.pending = None
        self.count = 0
        self.last_timestamp = -np.inf
        self.last_mapped = -1
        self.closed = False
        self.latencies = []
        self.timestamps = []
        self.online_poses = []

    def _collect(self, wait=False):
        if self.pending is not None and (wait or self.pending.done()):
            update = self.pending.result()  # Worker exceptions must reach caller.
            self.pending = None
            self.frontend.apply_update(update, self.store.get(update.anchor_id))

    def _map(self, start, end):
        self._collect(wait=True)
        frames = [self.store.get(i) for i in range(start, end)]
        scores = [self.frontend.scores[f.index] for f in frames]
        if self.executor:
            self.pending = self.executor.submit(self.backend.process, frames, scores)
        else:
            update = self.backend.process(frames, scores)
            self.frontend.apply_update(update, self.store.get(update.anchor_id))
        self.last_mapped = end - 1

    def push(self, frame: Frame) -> TrackingResult:
        if self.closed:
            raise RuntimeError("Cannot push after close")
        if (
            frame.index != self.count
            or not np.isfinite(frame.timestamp)
            or frame.timestamp <= self.last_timestamp
        ):
            raise ValueError("Frames require contiguous indices and increasing finite timestamps")
        start = time.perf_counter()
        self._collect()
        self.store.put(frame)
        pose, confidence = self.frontend.track(frame)
        self.count += 1
        self.last_timestamp = frame.timestamp
        self.timestamps.append(frame.timestamp)
        self.online_poses.append(pose.copy())
        if (
            self.count >= self.config.window
            and (self.count - self.config.window) % self.config.step == 0
        ):
            self._map(self.count - self.config.window, self.count)
        latency = (time.perf_counter() - start) * 1000
        self.latencies.append(latency)
        return TrackingResult(
            frame.index,
            frame.timestamp,
            pose,
            confidence,
            latency,
            self.pending is not None,
        )

    def close(self):
        if self.closed:
            return
        try:
            self._collect(wait=True)
            if self.count > 1 and self.last_mapped < self.count - 1:
                self._map(max(0, self.count - self.config.window), self.count)
                self._collect(wait=True)
        finally:
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True

    def trajectory(self):
        return (
            np.stack([self.frontend.poses[i] for i in range(self.count)])
            if self.count
            else np.empty((0, 4, 4))
        )

    def statistics(self):
        if not self.latencies:
            return {"frames": 0}
        return {
            "frames": self.count,
            "push_latency_ms_p50": float(np.median(self.latencies)),
            "push_latency_ms_p95": float(np.percentile(self.latencies, 95)),
            "push_throughput_fps": 1000 / float(np.mean(self.latencies)),
            "submaps": self.backend.count,
            "edges": len(self.backend.graph.edges),
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            if self.executor:
                self.executor.shutdown(wait=True, cancel_futures=True)
            self.closed = True


class LatestFrameCapture:
    """A capacity-one queue trades frame completeness for bounded capture age."""

    def __init__(self, source):
        self.source = source
        self.queue = Queue(maxsize=1)
        self.stop = Event()
        self.finished = Event()
        self.error = None
        self.dropped = 0
        self.thread = Thread(target=self._run, daemon=True)

    def _run(self):
        import cv2

        capture = cv2.VideoCapture(self.source)
        try:
            if not capture.isOpened():
                raise OSError(f"Cannot open capture source {self.source!r}")
            while not self.stop.is_set():
                ok, image = capture.read()
                if not ok:
                    break
                item = (time.monotonic(), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                try:
                    self.queue.put_nowait(item)
                except Full:
                    try:
                        self.queue.get_nowait()
                        self.dropped += 1
                    except Empty:
                        pass
                    self.queue.put_nowait(item)
        except Exception as error:
            self.error = error
        finally:
            capture.release()
            self.finished.set()

    def __iter__(self):
        self.thread.start()
        try:
            while not self.finished.is_set() or not self.queue.empty():
                try:
                    yield self.queue.get(timeout=0.1)
                except Empty:
                    continue
            if self.error:
                raise self.error
        finally:
            self.stop.set()
            self.thread.join(timeout=2.0)
