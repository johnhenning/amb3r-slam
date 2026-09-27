"""Method-neutral RGB replay boundary; ground truth stays in the evaluator."""

from __future__ import annotations

import copy
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from .benchmark import sha256_file, tum_rgb_samples
from .contracts import Array, JsonObject
from .evaluation import evaluate, read_tum, write_tum
from .profiling import ReplayProfiler
from .types import Frame


@dataclass
class Trajectory:
    """Camera-to-world rigid poses at original input timestamps, in input order."""

    timestamps: Array
    poses: Array

    def validate(self, input_times: Array) -> None:
        t, p = np.asarray(self.timestamps), np.asarray(self.poses)
        if t.ndim != 1 or len(t) < 3 or p.shape != (len(t), 4, 4):
            raise ValueError("Trajectory requires >=3 timestamped 4x4 poses")
        if not np.isfinite(t).all() or not np.isfinite(p).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Trajectory must be finite and strictly ordered")
        if not np.isin(t, input_times).all():
            raise ValueError("Trajectory timestamps must be original sampled timestamps")
        r = p[:, :3, :3]
        if not (
            np.allclose(p[:, 3], [0, 0, 0, 1], atol=1e-5)
            and np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-4)
            and np.allclose(np.linalg.det(r), 1, atol=1e-4)
        ):
            raise ValueError("Trajectory poses must be rigid camera-to-world transforms")


class SlamApproach(Protocol):
    """In-process RGB-only approach. finish drains workers; close always cleans up.

    Construction/model downloads precede replay. push/finish must not read ground
    truth. Missing estimates may be omitted, never replaced by reference poses.
    """

    def push(self, frame: Frame) -> None: ...

    def finish(self) -> Trajectory: ...

    def close(self) -> None: ...


def run_experiment(
    directory: Path,
    output: Path,
    factory: Callable[[Path], SlamApproach],
    *,
    approach: str,
    revision: str,
    config: JsonObject,
    stride: int = 10,
    max_frames: int | None = None,
    environment: JsonObject | None = None,
) -> JsonObject:
    """Run one isolated experiment directory using the existing TUM evaluator.

    Scope: RGB decode + push + final drain. Construction, evaluation, plotting,
    and upload excluded. CPU/RAM profiling covers this process, not subprocesses.
    Sim(3) evaluation is explicitly restricted to this monocular RGB protocol.
    """
    if not approach.strip() or not revision.strip():
        raise ValueError("Provide an approach name and immutable source revision")
    # Freeze provenance before user factory code can mutate its configuration.
    config = copy.deepcopy(config)
    json.dumps(config, allow_nan=False)
    samples, source_count = tum_rgb_samples(directory, stride, max_frames)
    input_times = np.array([t for t, _ in samples])
    output.mkdir(parents=True, exist_ok=False)
    (output / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                **(environment or {}),
            },
            indent=2,
            allow_nan=False,
        )
    )
    manifest = [(t, p.relative_to(directory).as_posix(), sha256_file(p)) for t, p in samples]
    (output / "inputs.json").write_text(json.dumps(manifest, indent=2))
    method = factory(output / "runtime")
    latencies = []
    try:
        with ReplayProfiler(output / "profile"):
            started = time.perf_counter()
            for index, (timestamp, path) in enumerate(samples):
                with Image.open(path) as image:
                    frame = Frame(index, timestamp, np.array(image.convert("RGB")))
                tick = time.perf_counter()
                method.push(frame)
                latencies.append(1000 * (time.perf_counter() - tick))
            trajectory = method.finish()
            elapsed = time.perf_counter() - started
    except BaseException as error:
        try:
            method.close()
        except BaseException as cleanup_error:
            error.add_note(f"Adapter cleanup also failed: {cleanup_error!r}")
        raise
    else:
        method.close()
    trajectory.validate(input_times)
    # Reference poses are not loaded until the method has finished and closed.
    gt_times, gt_poses = read_tum(directory / "groundtruth.txt")
    write_tum(output / "reference.tum", gt_times, gt_poses)
    write_tum(output / "trajectory.tum", trajectory.timestamps, trajectory.poses)
    np.savetxt(
        output / "latency.csv",
        np.column_stack([np.arange(len(samples)), input_times, latencies]),
        delimiter=",",
        header="frame,timestamp,push_latency_ms",
        comments="",
    )
    metrics = evaluate(output / "reference.tum", output / "trajectory.tum")
    result: JsonObject = {
        "schema_version": 1,
        "approach": approach,
        "approach_revision": revision,
        "protocol": "tum-monocular-rgb-sim3-v1",
        "sequence": directory.name,
        "source_rgb_frames": source_count,
        "sampled_frames": len(samples),
        "sampling_stride": stride,
        "max_frames": max_frames,
        "sampled_duration_s": float(input_times[-1] - input_times[0]),
        "sampled_gt_match_fraction": metrics["matched_poses"] / len(samples),
        "estimate_fraction": len(trajectory.timestamps) / len(samples),
        "input_manifest_sha256": sha256_file(output / "inputs.json"),
        "groundtruth_sha256": sha256_file(directory / "groundtruth.txt"),
        "rgb_list_sha256": sha256_file(directory / "rgb.txt"),
        "config": config,
        "metrics": {"final": metrics},
        "statistics": {
            "execution_mode": config.get("execution_mode", "unspecified"),
            "replay_elapsed_s": elapsed,
            "end_to_end_fps": len(samples) / elapsed,
        },
        "measurement_scope": "in-process RGB decode, push and finish; setup excluded",
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result
