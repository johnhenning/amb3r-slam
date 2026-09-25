"""Validate and summarize full-route sequential/concurrent TUM experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from evo.core.metrics import APE, PoseRelation, StatisticsType
from evo.core.trajectory import PoseTrajectory3D

from amber_slam.benchmark import sha256_file, trajectory_diagnostics
from amber_slam.contracts import JsonObject
from amber_slam.evaluation import associate, evaluate, read_tum


def validate(path: Path, result: JsonObject) -> JsonObject:
    inputs = json.loads((path / "inputs.json").read_text())
    if (
        len(inputs) != result["sampled_frames"]
        or sha256_file(path / "inputs.json") != result["input_manifest_sha256"]
    ):
        raise ValueError("Input manifest mismatch")
    if result["max_frames"] is not None or len(inputs) != (result["source_rgb_frames"] + 9) // 10:
        raise ValueError("Incomplete route sampling")
    tr, truth = read_tum(path / "reference.tum")
    checks: JsonObject = {}
    for label, filename in [("online", "trajectory_online.tum"), ("corrected", "trajectory.tum")]:
        actual = evaluate(path / "reference.tum", path / filename)
        if actual != result["metrics"][label]:
            raise ValueError("Saved metrics mismatch")
        te, predicted = read_tum(path / filename)
        np.testing.assert_allclose(te, [row[0] for row in inputs], atol=1e-6, rtol=0)
        ir, ie = associate(tr, te)
        gt = PoseTrajectory3D(poses_se3=truth[ir], timestamps=tr[ir])
        est = PoseTrajectory3D(poses_se3=predicted[ie], timestamps=te[ie])
        est.align(gt, correct_scale=True)
        metric = APE(PoseRelation.translation_part)
        metric.process_data((gt, est))
        independent = float(metric.get_statistic(StatisticsType.rmse))
        if abs(independent - actual["ate_rmse_m"]) > 1e-8 or len(ie) / len(inputs) < 0.95:
            raise ValueError("Independent ATE or association coverage failed")
        checks[label] = {"evo_ate_rmse_m": independent, "matched": len(ie), "sampled": len(inputs)}
    return checks


def summarize(root: Path) -> None:
    protocol = json.loads((root / "protocol.json").read_text())
    rows: list[JsonObject] = []
    validation: JsonObject = {}
    invariants: dict[str, JsonObject] = {}
    for run in protocol["runs"]:
        directory = root / run["label"]
        path = directory / ("rgbd_dataset_" + run["sequence"])
        result = json.loads((path / "results.json").read_text())
        env = json.loads((directory / "environment.json").read_text())
        profile = json.loads((path / "profile/summary.json").read_text())
        invariant = {
            k: result[k] for k in ["input_manifest_sha256", "groundtruth_sha256", "config"]
        }
        invariant.update(
            {k: env[k] for k in ["checkpoint_hashes", "resolution", "torch_threads", "sources"]}
        )
        if run["sequence"] in invariants and invariants[run["sequence"]] != invariant:
            raise ValueError("Matched trial settings differ")
        invariants[run["sequence"]] = invariant
        validation[run["label"]] = validate(path, result)
        edges = json.loads((path / "runtime/graph_edges.json").read_text())
        corrections = json.loads((path / "corrections.json").read_text())
        loops = [e for e in edges if e["kind"] == "loop"]
        for event in corrections:
            index = event["graph"]["submaps"] - 1
            event["new_loop_edges"] = [{"i": e["i"], "j": e["j"]} for e in loops if e["j"] == index]
        (path / "loop_corrections.json").write_text(json.dumps(corrections, indent=2) + "\n")
        rows.append(
            {
                "run": run,
                "result": result,
                "profile": profile,
                "edge_counts": dict(Counter(e["kind"] for e in edges)),
                "corrections": corrections,
            }
        )
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (root / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    sequences = list(invariants)
    fig, axes = plt.subplots(
        len(sequences), 2, figsize=(12, 5 * len(sequences)), constrained_layout=True, squeeze=False
    )
    for index, sequence in enumerate(sequences):
        ax, error_ax = axes[index]
        for row in [r for r in rows if r["run"]["sequence"] == sequence]:
            run = row["run"]
            path = root / run["label"] / ("rgbd_dataset_" + sequence)
            times, truth, aligned, errors = trajectory_diagnostics(
                path / "reference.tum", path / "trajectory.tum"
            )
            name = run["mode"] + " " + str(run["repeat"])
            ax.plot(aligned[:, 0, 3], aligned[:, 1, 3], label=name)
            error_ax.plot(times - times[0], errors, label=name)
        ax.plot(truth[:, 0, 3], truth[:, 1, 3], color="black", linestyle="--", label="ground truth")
        ax.scatter(truth[0, 0, 3], truth[0, 1, 3], marker="o", color="black")
        ax.scatter(truth[-1, 0, 3], truth[-1, 1, 3], marker="x", color="black")
        ax.set(
            title=sequence + "\nCorrected trajectory, Sim(3) aligned",
            xlabel="x (m)",
            ylabel="y (m)",
            aspect="equal",
        )
        error_ax.set(
            title="Corrected position error", xlabel="Sequence time (s)", ylabel="Error (m)"
        )
        for a in [ax, error_ax]:
            a.legend(fontsize=8)
            a.grid(alpha=0.2)
    fig.savefig(root / "trajectories.png", dpi=150)
    plt.close(fig)
    lines = [
        "# Full-route indoor comparison",
        "",
        "![Trajectories and error](trajectories.png)",
        "",
        "Full route duration at every-tenth-RGB-frame sampling, without a frame-count cap. DA3-Small for both models, CPU, resolution 168, two numerical-library threads, default waiting policy. Trials run in separate processes one at a time; order is recorded in protocol.json. One pair per sequence is exploratory, not an estimate of runtime variance.",
        "",
        "| Trial | Frames / matched | Replay s | CPU core-s | Mean CPU % | RSS MiB | Online ATE m | Corrected ATE m | Loop edges |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        run, result, profile = row["run"], row["result"], row["profile"]
        metrics = result["metrics"]
        lines.append(
            f"| [{run['label']}]({run['label']}/README.md) | {result['sampled_frames']} / {metrics['corrected']['matched_poses']} | {result['statistics']['replay_elapsed_s']:.2f} | {profile['cpu_total_s']:.2f} | {profile['cpu_percent_mean']:.1f} | {profile['rss_peak_bytes_sampled'] / 2**20:.1f} | {metrics['online']['ate_rmse_m']:.4f} | {metrics['corrected']['ate_rmse_m']:.4f} | {row['edge_counts'].get('loop', 0)} |"
        )
    lines += [
        "",
        "## Interpretation and measurement scope",
        "",
        "Accepted loop edges are algorithm decisions, not ground-truth-verified closures. `loop_corrections.json` joins final graph edges to the timestamp at which each corresponding backend update reached the tracker. Multiple edges can enter one correction. Trajectories use independent full-run Sim(3) alignment; corrected poses use future information, while online poses preserve tracking-time estimates. ATE improvement alone cannot isolate the benefit of loop closure from local/long-context mapping.",
        "",
        "Downloads/model loading are outside replay time. The timer includes image decoding, persistence, all SLAM stages and final drain. CPU includes all process threads; 100% is one core. RSS and USS are sampled and read non-atomically. Each profile also retains user/system CPU, memory timeline, faults, context switches, I/O, thread count, cgroup throttling and overlapping stage spans. Source, checkpoint, settings and input hashes match within each pair. Both trajectories were independently validated with evo; see validation.json.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv sync --locked --extra da3 --extra benchmark",
        "uv run --no-sync python scripts/profile_tum_routes.py --output reports/new_routes",
        "uv run --no-sync python scripts/report_tum_routes.py reports/new_routes",
        "```",
        "",
        "Use `--repeats 3` for repeated pairs; order reverses between repeats. Dataset archives are downloaded from the official TUM server and pinned by SHA-256. Large input data/model weights/frame caches are excluded from Git.",
        "",
        "## Dataset selection",
        "",
        "- [TUM official descriptions](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download): long_office_household traverses a furnished scene and returns to the start; room traverses an office and closes a loop.",
        "- [AMB3R-SLAM paper](https://arxiv.org/html/2609.19518v1): TUM is one of nine datasets; our Small/Small CPU setting is not the paper's Small/Giant setting.",
        "- KITTI 07 then 00 remain the next scale tests. The [official KITTI downloads](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) require login. They have not been downloaded or benchmarked here.",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines))
    print(
        json.dumps(
            [
                {
                    "label": r["run"]["label"],
                    "runtime": r["result"]["statistics"]["replay_elapsed_s"],
                    "ate": r["result"]["metrics"]["corrected"]["ate_rmse_m"],
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
