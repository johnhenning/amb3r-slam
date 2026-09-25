"""Streaming API and bounded asynchronous capture.

The caller owns tracking, frame persistence, and correction application. Local
mapping owns prepared submaps until handoff. The graph worker owns backend state.
The two backend stages serialize model calls but otherwise run independently.
Completed BackendUpdate messages are transferred to the caller in FIFO order.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread
from types import TracebackType

import numpy as np

from .backend import BackendUpdate, HierarchicalBackend, SlamConfig, Submap
from .contracts import Array, PathLike, RunStatistics
from .frontend import Frontend
from .store import FrameStore
from .types import Frame, GeometryModel


@dataclass
class TrackingResult:
    frame_id: int
    timestamp: float
    pose: Array
    confidence: float
    latency_ms: float
    backend_pending: bool


class SlamSystem:
    """Tracking, local reconstruction, and graph processing form a bounded pipeline.

    Backpressure waits only when a new mapping window must be submitted. This
    preserves the n/3 mapping stride and span-2 overlap instead of silently
    skipping mandatory submaps. Live capture drops stale frames upstream.
    """

    def __init__(
        self,
        frontend_model: GeometryModel,
        backend_model: GeometryModel,
        directory: PathLike,
        config: SlamConfig | None = None,
        asynchronous: bool = True,
        max_pending_windows: int = 2,
    ) -> None:
        if max_pending_windows < 1:
            raise ValueError("max_pending_windows must be positive")
        if asynchronous and frontend_model is backend_model:
            raise ValueError("Concurrent execution requires separate frontend/backend models")
        self.config = config or SlamConfig()
        self.max_pending_windows = max_pending_windows
        self.asynchronous = asynchronous
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any((self.directory / "frames").glob("*.npz")):
            raise ValueError("Run directory already contains frames; choose a new directory")
        self.store = FrameStore(self.directory / "frames", self.config.window * 2)
        self.frontend = Frontend(frontend_model)
        self.backend = HierarchicalBackend(backend_model, self.directory, self.config)
        self.executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="amber-graph")
            if asynchronous
            else None
        )
        self.mapper = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="amber-mapping")
            if asynchronous
            else None
        )
        self._pending: deque[tuple[Future[Submap], Future[BackendUpdate]]] = deque()
        self._submitted = 0
        self._submaps = 0
        self._edges = 0
        self._pending_peak = 0
        self._backpressure_ms = 0.0
        self._backpressure_waits = 0
        self.count = 0
        self.last_timestamp = -np.inf
        self.last_mapped = -1
        self.closed = False
        self.latencies = []
        self.timestamps = []
        self.online_poses = []

    @property
    def pending(self) -> Future[BackendUpdate] | None:
        """Oldest outstanding correction, retained for API compatibility."""
        return self._pending[0][1] if self._pending else None

    def _apply(self, update: BackendUpdate) -> None:
        self.frontend.apply_update(update, self.store.get(update.anchor_id))
        self._submaps = update.graph_report.get("submaps", self._submaps)
        self._edges = update.graph_report.get("edges", self._edges)

    def _collect(self, wait: bool = False) -> None:
        # FIFO corrections avoid stale anchors overwriting newer graph results.
        while self._pending:
            _, future = self._pending[0]
            if not wait and not future.done():
                break
            update = future.result()  # Worker exceptions must reach caller.
            self._pending.popleft()
            self._apply(update)

    def _integrate(self, prepared: Future[Submap], frames: list[Frame]) -> BackendUpdate:
        return self.backend.integrate(prepared.result(), frames)

    def _map(self, start: int, end: int) -> None:
        self._collect()
        if len(self._pending) >= self.max_pending_windows:
            started = time.perf_counter()
            self._backpressure_waits += 1
            # Make room for ONE window, rather than draining the entire pipeline.
            assert self.pending is not None
            update = self.pending.result()
            self._pending.popleft()
            self._apply(update)
            self._backpressure_ms += (time.perf_counter() - started) * 1000
        frames = [self.store.get(i) for i in range(start, end)]
        scores = [self.frontend.scores[f.index] for f in frames]
        if self.executor is not None and self.mapper is not None:
            prepared = self.mapper.submit(self.backend.prepare, frames, scores, self._submitted)
            integrated = self.executor.submit(self._integrate, prepared, frames)
            self._pending.append((prepared, integrated))
            self._pending_peak = max(self._pending_peak, len(self._pending))
        else:
            self._apply(self.backend.process(frames, scores))
        self._submitted += 1
        self.last_mapped = end - 1

    def _shutdown(self) -> None:
        # Cancel queued jobs before joining. Active native calls cannot be killed;
        # shutdown waits for them so no worker can mutate files after return.
        for prepared, integrated in self._pending:
            integrated.cancel()
            prepared.cancel()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
        if self.mapper is not None:
            self.mapper.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self.closed = True

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

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._collect(wait=True)
            if self.count > 1 and self.last_mapped < self.count - 1:
                self._map(max(0, self.count - self.config.window), self.count)
                self._collect(wait=True)
        finally:
            self._shutdown()

    def trajectory(self) -> Array:
        return (
            np.stack([self.frontend.poses[i] for i in range(self.count)])
            if self.count
            else np.empty((0, 4, 4))
        )

    def statistics(self) -> RunStatistics:
        if not self.latencies:
            return {"frames": 0}
        return {
            "frames": self.count,
            "push_latency_ms_p50": float(np.median(self.latencies)),
            "push_latency_ms_p95": float(np.percentile(self.latencies, 95)),
            "push_throughput_fps": 1000 / float(np.mean(self.latencies)),
            # Only completed messages are read; graph state belongs to its worker.
            "submaps": self._submaps,
            "edges": self._edges,
            "execution_mode": "concurrent" if self.asynchronous else "sequential",
            "max_pending_windows": self.max_pending_windows,
            "pending_windows_peak": self._pending_peak,
            "backpressure_waits": self._backpressure_waits,
            "backpressure_ms": self._backpressure_ms,
        }

    def __enter__(self) -> SlamSystem:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            self._shutdown()


class LatestFrameCapture:
    """A capacity-one queue trades frame completeness for bounded capture age."""

    def __init__(self, source: str | int) -> None:
        self.source = source
        self.queue: Queue[tuple[float, Array]] = Queue(maxsize=1)
        self.stop = Event()
        self.finished = Event()
        self.error: Exception | None = None
        self.dropped = 0
        self.thread = Thread(target=self._run, daemon=True)

    def _run(self) -> None:
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

    def __iter__(self) -> Iterator[tuple[float, Array]]:
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
