"""Linux replay-scoped process metrics and overlapping stage traces."""

from __future__ import annotations

import csv
import functools
import json
import os
import resource
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import TracebackType
from typing import ParamSpec, TypeVar

import numpy as np
import psutil

from .contracts import JsonObject

P = ParamSpec("P")
T = TypeVar("T")


def _cpu_stat() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/cpu.stat")
    if not path.is_file():
        return {}
    return {
        key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())
    }


class ReplayProfiler:
    """100% CPU means one core; memory includes loaded model weights.

    RSS/USS peaks are sampled, not allocator high-water marks. Measures this
    process and all its threads, not child processes. Stage wall times overlap;
    stage thread-CPU times exclude native worker threads. Sampling overhead is
    included. The context writes artifacts only after the timed replay ends.
    """

    def __init__(self, directory: Path, interval_s: float = 0.2) -> None:
        if interval_s <= 0:
            raise ValueError("Sample interval must be positive")
        self.directory = directory
        self.interval_s = interval_s
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="amber-profiler", daemon=True)
        self.samples: list[JsonObject] = []
        self.events: list[JsonObject] = []
        self.lock = threading.Lock()
        self.error: Exception | None = None
        self.started = 0.0
        self.summary: JsonObject = {}

    def wrap(self, name: str, function: Callable[P, T]) -> Callable[P, T]:
        """Keep the callable contract while recording each invocation's span."""

        @functools.wraps(function)
        def measured(*args: P.args, **kwargs: P.kwargs) -> T:
            start, cpu = time.perf_counter(), time.thread_time()
            try:
                return function(*args, **kwargs)
            finally:
                event: JsonObject = {
                    "name": name,
                    "start_s": start - self.started,
                    "duration_s": time.perf_counter() - start,
                    "caller_thread_cpu_s": time.thread_time() - cpu,
                    "tid": threading.get_native_id(),
                }
                with self.lock:
                    self.events.append(event)

        return measured

    def _sample(self) -> None:
        memory = self.process.memory_full_info()
        cpu = self.process.cpu_times()
        self.samples.append(
            {
                "elapsed_s": time.perf_counter() - self.started,
                "cpu_time_s": float(cpu.user + cpu.system),
                "rss_bytes": int(memory.rss),
                "uss_bytes": int(memory.uss),
                "threads": self.process.num_threads(),
            }
        )

    def _run(self) -> None:
        try:
            while not self.stop.wait(self.interval_s):
                self._sample()
        except Exception as error:
            self.error = error

    def __enter__(self) -> ReplayProfiler:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.usage_start = resource.getrusage(resource.RUSAGE_SELF)
        self.io_start = self.process.io_counters()
        self.cgroup_start = _cpu_stat()
        self.started = time.perf_counter()
        self._sample()
        self.thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            elapsed = time.perf_counter() - self.started
            usage = resource.getrusage(resource.RUSAGE_SELF)
            io = self.process.io_counters()
            cgroup = _cpu_stat()
        finally:
            self.stop.set()
            self.thread.join()
        if exc_type is not None:
            return
        if self.error is not None:
            raise RuntimeError("Resource sampler failed") from self.error
        self._sample()
        cpu = (
            usage.ru_utime + usage.ru_stime - self.usage_start.ru_utime - self.usage_start.ru_stime
        )
        dt = np.diff([s["elapsed_s"] for s in self.samples])
        utilization = 100 * np.diff([s["cpu_time_s"] for s in self.samples]) / dt
        stages = {}
        for name in sorted({e["name"] for e in self.events}):
            events = [e for e in self.events if e["name"] == name]
            stages[name] = {
                "calls": len(events),
                "wall_sum_s": sum(e["duration_s"] for e in events),
                "wall_p50_s": float(np.median([e["duration_s"] for e in events])),
                "wall_p95_s": float(np.percentile([e["duration_s"] for e in events], 95)),
                "caller_thread_cpu_sum_s": sum(e["caller_thread_cpu_s"] for e in events),
            }
        self.summary = {
            "scope": "Replay including tracking, mapping, graph, I/O and final drain; excludes model loading and report generation",
            "elapsed_s": elapsed,
            "cpu_total_s": cpu,
            "cpu_user_s": usage.ru_utime - self.usage_start.ru_utime,
            "cpu_system_s": usage.ru_stime - self.usage_start.ru_stime,
            "cpu_percent_mean": 100 * cpu / elapsed,
            "cpu_percent_p95_sampled": float(np.percentile(utilization, 95)),
            "cpu_percent_peak_sampled": float(utilization.max()),
            "rss_peak_bytes_sampled": max(s["rss_bytes"] for s in self.samples),
            "uss_peak_bytes_sampled": max(s["uss_bytes"] for s in self.samples),
            "threads_peak_sampled": max(s["threads"] for s in self.samples),
            "minor_page_faults": usage.ru_minflt - self.usage_start.ru_minflt,
            "major_page_faults": usage.ru_majflt - self.usage_start.ru_majflt,
            "voluntary_context_switches": usage.ru_nvcsw - self.usage_start.ru_nvcsw,
            "involuntary_context_switches": usage.ru_nivcsw - self.usage_start.ru_nivcsw,
            "storage_read_bytes": io.read_bytes - self.io_start.read_bytes,
            "storage_write_bytes": io.write_bytes - self.io_start.write_bytes,
            "read_syscalls": io.read_count - self.io_start.read_count,
            "write_syscalls": io.write_count - self.io_start.write_count,
            "cgroup_cpu_stat_delta": {
                k: v - self.cgroup_start.get(k, v) for k, v in cgroup.items()
            },
            "cgroup_scope": "container-wide, may include other processes; not host-wide scheduling data",
            "cpu_affinity_count": len(self.process.cpu_affinity()),
            "wait_environment": {
                k: os.environ.get(k) for k in ["OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME"]
            },
            "sample_interval_s": self.interval_s,
            "samples": len(self.samples),
            "maximum_sample_gap_s": float(dt.max()),
            "stages": stages,
            "stage_semantics": "Stage times overlap; caller thread CPU excludes native worker CPU",
        }
        (self.directory / "summary.json").write_text(json.dumps(self.summary, indent=2) + "\n")
        (self.directory / "stages.json").write_text(json.dumps(self.events, indent=2) + "\n")
        with (self.directory / "samples.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.samples[0]))
            writer.writeheader()
            writer.writerows(self.samples)
