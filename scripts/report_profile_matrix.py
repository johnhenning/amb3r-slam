"""Validate a completed four-cell diagnostic matrix and create its report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from amber_slam.contracts import JsonObject
from amber_slam.evaluation import associate, evaluate, read_tum


def validate(path: Path, result: JsonObject) -> JsonObject:
    from evo.core.metrics import APE, PoseRelation, StatisticsType
    from evo.core.trajectory import PoseTrajectory3D

    checks: JsonObject = {}
    for label, name in [("online", "trajectory_online.tum"), ("corrected", "trajectory.tum")]:
        actual = evaluate(path / "reference.tum", path / name)
        if actual != result["metrics"][label]:
            raise ValueError("Saved metrics do not match recomputation")
        tr, truth = read_tum(path / "reference.tum")
        te, predicted = read_tum(path / name)
        ir, ie = associate(tr, te)
        gt = PoseTrajectory3D(poses_se3=truth[ir], timestamps=tr[ir])
        est = PoseTrajectory3D(poses_se3=predicted[ie], timestamps=te[ie])
        est.align(gt, correct_scale=True)
        metric = APE(PoseRelation.translation_part)
        metric.process_data((gt, est))
        independent = float(metric.get_statistic(StatisticsType.rmse))
        error = abs(independent - actual["ate_rmse_m"])
        if error > 1e-8 or len(ie) != 80:
            raise ValueError("Independent ATE or complete coverage check failed")
        checks[label] = {"evo_ate_rmse_m": independent, "difference_m": error, "matched": len(ie)}
    return checks


def summarize(root: Path) -> None:
    rows: list[JsonObject] = []
    invariants: JsonObject | None = None
    validation: JsonObject = {}
    labels = [
        "default_sequential",
        "default_concurrent",
        "passive_concurrent",
        "passive_sequential",
    ]
    for label in labels:
        path = root / label / "rgbd_dataset_freiburg1_xyz"
        result = json.loads((path / "results.json").read_text())
        env = json.loads((path.parent / "environment.json").read_text())
        profile = json.loads((path / "profile/summary.json").read_text())
        invariant = {
            "inputs": result["input_manifest_sha256"],
            "groundtruth": result["groundtruth_sha256"],
            "config": result["config"],
            "checkpoint": env["checkpoint_hashes"],
            "resolution": env["resolution"],
            "threads": env["torch_threads"],
            "source": env["sources"],
        }
        if invariants is not None and invariant != invariants:
            raise ValueError("Inputs/settings/source differ across trials")
        invariants = invariant
        validation[label] = validate(path, result)
        log = (root / f"{label}.log").read_text()
        views = [int(x) for x in re.findall(r"Shape:  torch.Size\(\[(\d+), 3,", log)]
        workload = {"model_calls": len(views), "input_views": sum(views)}
        (path / "model_workload.json").write_text(json.dumps(workload, indent=2) + "\n")
        rows.append({"label": label, "result": result, "profile": profile, "workload": workload})
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (root / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    labels_display = [s.replace("_", "\n") for s in labels]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, key, title, divisor in [
        (axes[0, 0], "elapsed_s", "Replay wall time (s)", 1),
        (axes[0, 1], "cpu_total_s", "Process CPU time (core-seconds)", 1),
        (axes[1, 0], "rss_peak_bytes_sampled", "Sampled peak RSS (MiB)", 2**20),
    ]:
        ax.bar(labels_display, [r["profile"][key] / divisor for r in rows], color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    x = np.arange(4)
    for offset, stage in enumerate(["tracking", "mapping.prepare", "graph.optimize"]):
        axes[1, 1].bar(
            x + (offset - 1) * 0.25,
            [r["profile"]["stages"][stage]["wall_sum_s"] for r in rows],
            width=0.25,
            label=stage,
        )
    axes[1, 1].set_xticks(x, labels_display)
    axes[1, 1].set_title("Stage wall sums (s) — overlap; not additive")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(
        "TUM freiburg1_xyz • 80 frames • DA3-Small CPU\nOne trial per cell: diagnostic, not a variance estimate"
    )
    fig.savefig(root / "comparison.png", dpi=140)
    plt.close(fig)
    lines = [
        "# Fresh matched runtime diagnostics",
        "",
        "![Resource and stage comparison](comparison.png)",
        "",
        "One fresh process per policy/mode on the same 80 TUM freiburg1_xyz RGB frames, same pinned DA3-Small checkpoint, resolution 168, window 12, and two-thread numerical-library limits. Order: default sequential, default concurrent, passive concurrent, passive sequential. OMP_WAIT_POLICY=PASSIVE is the sole intentional policy change. Results are separate from the earlier historical six-run summary; no cross-session causal comparison is claimed.",
        "",
        "| Trial | Replay s | FPS | CPU s | CPU % | RSS MiB | USS MiB | ATE m | Online ATE m |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        p, s, m = r["profile"], r["result"]["statistics"], r["result"]["metrics"]
        lines.append(
            f"| [{r['label']}]({r['label']}/rgbd_dataset_freiburg1_xyz/results.json) | {s['replay_elapsed_s']:.3f} | {s['end_to_end_fps']:.3f} | {p['cpu_total_s']:.3f} | {p['cpu_percent_mean']:.1f} | {p['rss_peak_bytes_sampled'] / 2**20:.1f} | {p['uss_peak_bytes_sampled'] / 2**20:.1f} | {m['corrected']['ate_rmse_m']:.4f} | {m['online']['ate_rmse_m']:.4f} |"
        )
    lines += [
        "",
        "## Stage and scheduling evidence",
        "",
        "Stage times can overlap, and graph.integrate includes graph.optimize. Do not add them into an end-to-end runtime. Caller-thread CPU excludes native worker CPU. The full event spans are in each trial's profile/stages.json; resource timelines are in profile/samples.csv.",
        "",
        "| Trial | Tracking s | Mapping s | Graph optimize s | Graph integrate s | Calls/views | Throttled s | Major faults | Storage reads MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        p, st, w = r["profile"], r["profile"]["stages"], r["workload"]
        lines.append(
            f"| {r['label']} | {st['tracking']['wall_sum_s']:.3f} | {st['mapping.prepare']['wall_sum_s']:.3f} | {st['graph.optimize']['wall_sum_s']:.3f} | {st['graph.integrate']['wall_sum_s']:.3f} | {w['model_calls']}/{w['input_views']} | {p['cgroup_cpu_stat_delta'].get('throttled_usec', 0) / 1e6:.3f} | {p['major_page_faults']} | {p['storage_read_bytes'] / 2**20:.3f} |"
        )
    lines += [
        "",
        "## Scope and validation",
        "",
        "The replay timer excludes checkpoint loading/download, hashes, plotting, and evaluation. It includes RGB decoding, frame persistence, tracking, mapping, graph optimization, backpressure and final drain. Process CPU and sampled RSS/USS include both models and profiler overhead. 100% CPU means one core. Peaks use nominal 200 ms sampling; actual gaps are saved. No GPU is present. Cgroup counters cover the container, potentially including other processes; they do not measure all host scheduling/frequency effects. Source/input/checkpoint/configuration hashes match across the four runs. Independent evo checks of both trajectories are in validation.json. One trial per cell cannot establish that a policy eliminates variance.",
        "",
        "Raw results retain CPU user/system time, utilization, RSS/USS, thread count, I/O, page faults, context switches, cgroup throttling, queue backpressure, stage spans, trajectories and environment metadata. Large frame/submap caches are excluded from Git; recreate them by rerunning the commands.",
        "",
        "```bash",
        "uv sync --locked --extra da3 --extra benchmark",
        "uv run --no-sync python scripts/profile_tum_matrix.py --output reports/new_diagnostics",
        "uv run --no-sync python scripts/report_profile_matrix.py reports/new_diagnostics",
        "```",
        "",
        "The matrix downloads the standard TUM archive if needed, verifies its checksum and pins checkpoint revision e08cab65ca0ec38e7826075418411ab90cab4da3. Some proxy environments need the environment-only helper `uv pip install socksio` for Hugging Face downloads.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    print(
        json.dumps(
            [
                {
                    "label": r["label"],
                    "runtime": r["result"]["statistics"]["replay_elapsed_s"],
                    "cpu": r["profile"]["cpu_total_s"],
                }
                for r in rows
            ],
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    summarize(parser.parse_args().root)


if __name__ == "__main__":
    main()
