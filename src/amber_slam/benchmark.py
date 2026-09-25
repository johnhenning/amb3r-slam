"""Repeatable RGB-only trajectory benchmarks on TUM RGB-D sequences.

Ground truth is held by the evaluator: only RGB and timestamps enter SLAM.
Sampling is applied to the original RGB list, before ground-truth association.
This avoids selecting an easier trajectory using ground-truth availability.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from .backend import SlamConfig
from .contracts import Array, BenchmarkEnvironment, BenchmarkResult, PathLike
from .datasets import _list_file
from .evaluation import (
    align_trajectory,
    associate,
    evaluate,
    read_tum,
    write_tum,
)
from .runtime import SlamSystem
from .types import Frame, GeometryModel

TUM_SEQUENCES = {
    "freiburg1_xyz": {
        "sha256": "a0236d97b8c30cd93b653656d2b6c293ff7c982a4130ef2a1a8beecdb124ef98",
        "url": "https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
    },
    "freiburg1_desk": {
        "sha256": "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c",
        "url": "https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
    },
}


def sha256_file(path: PathLike) -> str:
    """Hash large checkpoints without loading their bytes into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tum_rgb_samples(
    directory: PathLike, stride: int = 10, max_frames: int | None = None
) -> tuple[list[tuple[float, Path]], int]:
    """Return timestamp/path pairs with an explicit, deterministic sampling rule."""
    if stride < 1 or (max_frames is not None and max_frames < 3):
        raise ValueError("stride must be positive; max_frames must be >=3 or None")
    directory = Path(directory)
    rows = _list_file(directory / "rgb.txt")
    times = np.array([t for t, _ in rows])
    if len(rows) < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("RGB timestamps must be finite, increasing, and contain >=3 frames")
    selected = rows[::stride][:max_frames]
    if len(selected) < 3:
        raise ValueError("Sampling leaves fewer than three frames")
    return [(t, directory / name) for t, name in selected], len(rows)


def trajectory_diagnostics(
    reference_path: PathLike, estimate_path: PathLike, tolerance: float = 0.02
) -> tuple[Array, Array, Array, Array]:
    """Return independently aligned matched trajectories and per-pose errors."""
    tr, reference = read_tum(reference_path)
    te, estimate = read_tum(estimate_path)
    ir, ie = associate(tr, te, tolerance)
    aligned, _ = align_trajectory(estimate[ie], reference[ir], "sim3")
    errors = np.linalg.norm(aligned[:, :3, 3] - reference[ir, :3, 3], axis=1)
    return te[ie], reference[ir], aligned, errors


def plot_sequence(directory: PathLike, title: str) -> None:
    """Save a static, exportable figure; no plotting dependency in SLAM runtime."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    directory = Path(directory)
    fig = plt.figure(figsize=(16, 5))
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.16, top=0.84, wspace=0.42)
    trajectory_ax = fig.add_subplot(131, projection="3d")
    error_ax = fig.add_subplot(132)
    latency_ax = fig.add_subplot(133)
    for label, filename, color in [
        ("Online", "trajectory_online.tum", "#d97706"),
        ("Corrected", "trajectory.tum", "#2563eb"),
    ]:
        timestamps, reference, aligned, errors = trajectory_diagnostics(
            directory / "reference.tum", directory / filename
        )
        xyz = aligned[:, :3, 3]
        trajectory_ax.plot(*xyz.T, color=color, label=label, linewidth=1.5)
        error_ax.plot(timestamps - timestamps[0], errors, label=label, color=color)
    truth = reference[:, :3, 3]
    trajectory_ax.plot(*truth.T, color="#111827", label="Ground truth", linewidth=1.5)
    trajectory_ax.set(
        xlabel="x (m)", ylabel="y (m)", zlabel="z (m)", title="Sim(3)-aligned trajectory"
    )
    trajectory_ax.set_box_aspect(np.maximum(np.ptp(truth, axis=0), 0.01))
    for axis in (trajectory_ax.xaxis, trajectory_ax.yaxis, trajectory_ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
    trajectory_ax.legend(fontsize=8)
    error_ax.set(
        xlabel="Elapsed sequence time (s)",
        ylabel="Position error (m)",
        title="Absolute trajectory error",
    )
    error_ax.grid(alpha=0.2)
    latencies = np.loadtxt(directory / "latency.csv", delimiter=",", skiprows=1, ndmin=2)
    latency_ax.plot(latencies[:, 0], latencies[:, 2], color="#7c3aed")
    latency_ax.set(
        xlabel="Sampled frame index",
        ylabel="Synchronous push latency (ms)",
        title="Tracking + scheduled mapping",
    )
    latency_ax.grid(alpha=0.2)
    fig.suptitle(title, fontsize=14)
    fig.savefig(directory / "diagnostics.png", dpi=160)
    plt.close(fig)


def run_tum_sequence(
    directory: PathLike,
    output: PathLike,
    frontend: GeometryModel,
    backend: GeometryModel,
    config: SlamConfig,
    stride: int = 10,
    max_frames: int | None = None,
    asynchronous: bool = True,
    max_pending_windows: int = 2,
) -> BenchmarkResult:
    """Run one sequence; preserve raw trajectories, timing, and evaluation metadata.

    Model construction/download is excluded from measured replay time. RGB decode,
    frame persistence, mapping, optimization, and final backend flush are included.
    Select asynchronous=False for repeatable sequential correction scheduling.
    """
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark: {output}")
    samples, source_count = tum_rgb_samples(directory, stride, max_frames)
    # Associate against the original mocap timestamps, never relabel GT as RGB time.
    gt_times, gt_poses = read_tum(directory / "groundtruth.txt")
    output.mkdir(parents=True)
    write_tum(output / "reference.tum", gt_times, gt_poses)
    selected = [(t, p.relative_to(directory).as_posix(), sha256_file(p)) for t, p in samples]
    (output / "inputs.json").write_text(json.dumps(selected, indent=2))
    started = time.perf_counter()
    with SlamSystem(
        frontend,
        backend,
        output / "runtime",
        config,
        asynchronous=asynchronous,
        max_pending_windows=max_pending_windows,
    ) as system:
        for index, (timestamp, path) in enumerate(samples):
            with Image.open(path) as image:
                rgb = np.array(image.convert("RGB"))
            system.push(Frame(index, timestamp, rgb))
            if index % 10 == 0:
                print(f"{directory.name}: {index + 1}/{len(samples)}", flush=True)
    elapsed = time.perf_counter() - started
    write_tum(output / "trajectory.tum", system.timestamps, system.trajectory())
    write_tum(output / "trajectory_online.tum", system.timestamps, np.stack(system.online_poses))
    np.savetxt(
        output / "latency.csv",
        np.column_stack([np.arange(system.count), system.timestamps, system.latencies]),
        delimiter=",",
        header="frame,timestamp,push_latency_ms",
        comments="",
    )
    metrics = {}
    for label, filename in [("corrected", "trajectory.tum"), ("online", "trajectory_online.tum")]:
        metrics[label] = evaluate(output / "reference.tum", output / filename)
    # Useful denominator for sampled replay, separate from dense GT coverage.
    matched = metrics["corrected"]["matched_poses"]
    statistics = system.statistics()
    statistics.update(replay_elapsed_s=elapsed, end_to_end_fps=system.count / elapsed)
    result: BenchmarkResult = {
        "sequence": directory.name,
        "source_rgb_frames": source_count,
        "sampled_frames": len(samples),
        "sampling_stride": stride,
        "max_frames": max_frames,
        "sampled_duration_s": samples[-1][0] - samples[0][0],
        "sampled_gt_match_fraction": matched / len(samples),
        "groundtruth_sha256": sha256_file(directory / "groundtruth.txt"),
        "rgb_list_sha256": sha256_file(directory / "rgb.txt"),
        "input_manifest_sha256": sha256_file(output / "inputs.json"),
        "config": asdict(config),
        "metrics": metrics,
        "statistics": statistics,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    plot_sequence(output, directory.name)
    return result


def environment_metadata() -> BenchmarkEnvironment:
    """Record concrete versions and CPU thread limits alongside results."""
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            name: version(name)
            for name in ["numpy", "scipy", "torch", "depth-anything-3", "matplotlib"]
        },
    }


def write_report(
    output: PathLike, results: list[BenchmarkResult], metadata: BenchmarkEnvironment
) -> None:
    """Build a reviewable Markdown report with links to machine-readable evidence."""
    output = Path(output)
    (output / "environment.json").write_text(json.dumps(metadata, indent=2))
    lines = [
        "# TUM RGB-D checkpoint benchmark",
        "",
        "These are measured results from our independent implementation, not published AMB3R-SLAM results.",
        "",
        "## Protocol",
        "",
        f"- Python {metadata['python']}; CPU; {metadata['torch_threads']} PyTorch threads.",
        f"- DA3-Small in both frontend/backend roles; processing resolution {metadata['resolution']} (long edge before patch-size rounding).",
        "- RGB-only input: no sensor depth, calibration, or ground-truth poses are supplied to SLAM.",
        "- Every tenth original RGB frame across each sequence; no accuracy-based frame selection.",
        "- Execution mode and queue capacity are recorded per sequence below; mapping settings are in results.json.",
        "- Each online/corrected trajectory gets its own full-trajectory Sim(3) alignment to motion-capture ground truth.",
        "- One-to-one timestamp matching within 20 ms. RPE uses consecutive matched sampled frames, not a fixed one-second interval.",
        "- FPS includes image loading, persistence, mapping, graph optimization and final flush; excludes model loading and plotting.",
        "- BLAS/OpenMP thread limits match the PyTorch thread limit; shared development container, not dedicated hardware.",
        "- One completed run per sequence; no confidence intervals or hardware real-time claim.",
        "",
        "## Results",
        "",
        "| Sequence | Frames | Matched | Online ATE (m) | Corrected ATE (m) | RPE trans. (m) | RPE rot. (deg) | FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        metrics = result["metrics"]["corrected"]
        lines.append(
            f"| {result['sequence']} | {result['sampled_frames']} | {metrics['matched_poses']} | {result['metrics']['online']['ate_rmse_m']:.4f} | {metrics['ate_rmse_m']:.4f} | {metrics['rpe_translation_rmse_m']:.4f} | {metrics['rpe_rotation_rmse_deg']:.3f} | {result['statistics']['end_to_end_fps']:.3f} |"
        )
    lines += [
        "",
        "All errors in the table are RMSE; lower is better. Sim(3) fitting removes global scale error and does not establish metric-scale accuracy.",
        "",
    ]
    for result in results:
        name = result["sequence"]
        stats = result["statistics"]
        lines += [
            f"## {name}",
            "",
            f"Sampled {result['sampled_frames']} of {result['source_rgb_frames']} RGB frames over {result['sampled_duration_s']:.2f} s; {100 * result['sampled_gt_match_fraction']:.1f}% matched ground truth. Backend: {stats['submaps']} submaps, {stats['edges']} edges. Push p50/p95: {stats['push_latency_ms_p50']:.1f}/{stats['push_latency_ms_p95']:.1f} ms.",
            "",
            f"Execution: {stats.get('execution_mode', 'sequential')}; pending-window limit: {stats.get('max_pending_windows', 1)}; peak outstanding: {stats.get('pending_windows_peak', 0)}; submission backpressure: {stats.get('backpressure_ms', 0.0):.1f} ms.",
            "",
            f"![Trajectory, absolute error and latency]({name}/diagnostics.png)",
            "",
            f"[Metrics and configuration]({name}/results.json) · [Corrected trajectory]({name}/trajectory.tum) · [Online trajectory]({name}/trajectory_online.tum) · [Latency samples]({name}/latency.csv)",
            "",
        ]
    lines += [
        "## Interpretation and limits",
        "",
        "This first real-data baseline exercises pretrained inference and the full Python SLAM path on two sequences from one standard dataset. Sampling and a small CPU model make it a reduced benchmark, not a full-rate TUM evaluation. The paper uses a stronger backend; these results cannot be compared directly with its tables. Ground-truth alignment uses the complete run and is offline. Loops are enabled but successful loop closure is not guaranteed. No held-out-training-data claim is made for the upstream pretrained checkpoint.",
        "",
        "The report retains both online and corrected metrics so backend regressions are visible. No accuracy threshold was tuned on these sequences. Automated checks enforce finite outputs, timestamp association, coverage, and consistency of saved metrics; they do not certify research-quality accuracy.",
        "",
        "## Reproduce and audit",
        "",
        "See [benchmark instructions](../../docs/BENCHMARKS.md). Exact checkpoint hashes, package versions, source hashes and invocation are in [environment.json](environment.json); each sequence includes an input list with SHA-256 image hashes. Raw datasets and model weights are downloaded separately.",
        "",
        "## References",
        "",
        "- Hengyi Wang and Lourdes Agapito. [AMB3R-SLAM: Kilometer-scale SLAM with Hierarchical Backend](https://arxiv.org/abs/2609.19518), 2026. Independent implementation; no author endorsement.",
        "- Haotong Lin et al. [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647), 2025. Pretrained checkpoints and upstream inference.",
        "- Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, Daniel Cremers. [A Benchmark for the Evaluation of RGB-D SLAM Systems](https://cvg.cit.tum.de/data/datasets/rgbd-dataset), IROS 2012. TUM RGB-D data and motion-capture ground truth (CC BY 4.0).",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
