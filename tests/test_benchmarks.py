"""Small offline tests plus explicitly enabled tests of external standard data.

TUM_DATA_ROOT enables real-data ingestion checks. AMBER_BENCHMARK_REPORT enables
checks of actual checkpoint-backed runs; neither setting triggers a download.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from amber_slam.benchmark import sha256_file, trajectory_diagnostics, tum_rgb_samples
from amber_slam.datasets import prepare_tum
from amber_slam.evaluation import associate, evaluate, read_tum

SEQUENCES = ("freiburg1_xyz", "freiburg1_desk")


def test_sampling_uses_original_rgb_order_without_ground_truth(tmp_path: Path) -> None:
    # No groundtruth.txt exists: availability must not influence image selection.
    (tmp_path / "rgb.txt").write_text(
        "# timestamp file\n" + "\n".join(f"{i / 30:.9f} rgb/{i}.png" for i in range(31))
    )
    samples, total = tum_rgb_samples(tmp_path, stride=10)
    assert total == 31
    assert [path.name for _, path in samples] == ["0.png", "10.png", "20.png", "30.png"]
    assert tum_rgb_samples(tmp_path, stride=10, max_frames=3)[0] == samples[:3]


@pytest.mark.parametrize("stride", [0, -1, 100])
def test_invalid_sampling_fails(tmp_path: Path, stride: int) -> None:
    (tmp_path / "rgb.txt").write_text("0 a.png\n1 b.png\n2 c.png\n")
    with pytest.raises(ValueError):
        tum_rgb_samples(tmp_path, stride)


@pytest.mark.dataset
@pytest.mark.parametrize("name", SEQUENCES)
def test_real_tum_ingestion(tmp_path: Path, name: str) -> None:
    root = os.environ.get("TUM_DATA_ROOT")
    if root is None:
        pytest.skip("Set TUM_DATA_ROOT to extracted standard TUM sequences")
    directory = Path(root) / f"rgbd_dataset_{name}"
    samples, source_count = tum_rgb_samples(directory)
    times = np.array([time for time, _ in samples])
    reference_times, reference = read_tum(directory / "groundtruth.txt")
    ir, ie = associate(reference_times, times)
    assert len(ir) / len(samples) > 0.95
    assert len(samples) == (source_count + 9) // 10
    assert all(path.is_file() for _, path in samples)
    assert np.max(np.abs(reference_times[ir] - times[ie])) <= 0.02
    # Real timestamps round-trip through the existing general dataset converter.
    manifest = tmp_path / "scene.json"
    summary = prepare_tum(
        directory,
        manifest,
        name,
        [[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]],
    )
    records = json.loads(manifest.read_text())["sequences"][0]["frames"]
    assert summary["frames"] == len(records) > source_count * 0.8
    assert np.all(np.diff([r["timestamp"] for r in records]) > 0)
    assert np.isfinite(reference).all()


@pytest.mark.dataset
@pytest.mark.parametrize("name", SEQUENCES)
def test_saved_checkpoint_benchmark_metrics(name: str) -> None:
    root = os.environ.get("AMBER_BENCHMARK_REPORT")
    if root is None:
        pytest.skip("Set AMBER_BENCHMARK_REPORT to a completed checkpoint benchmark")
    directory = Path(root) / f"rgbd_dataset_{name}"
    result = json.loads((directory / "results.json").read_text())
    assert sha256_file(directory / "inputs.json") == result["input_manifest_sha256"]
    for label, filename in [("online", "trajectory_online.tum"), ("corrected", "trajectory.tum")]:
        actual = evaluate(directory / "reference.tum", directory / filename)
        assert actual == result["metrics"][label]
        assert actual["estimate_coverage"] > 0.95
        assert actual["matched_poses"] > 30
        for value in (
            actual["ate_rmse_m"],
            actual["rpe_translation_rmse_m"],
            actual["rpe_rotation_rmse_deg"],
        ):
            assert np.isfinite(value) and value >= 0
        _, _, _, errors = trajectory_diagnostics(directory / "reference.tum", directory / filename)
        assert np.sqrt(np.mean(errors**2)) == pytest.approx(actual["ate_rmse_m"])
    timestamps, poses = read_tum(directory / "trajectory.tum")
    assert len(timestamps) == result["sampled_frames"]
    np.testing.assert_allclose(np.linalg.det(poses[:, :3, :3]), 1, atol=1e-6)
    latencies = np.loadtxt(directory / "latency.csv", delimiter=",", skiprows=1)
    assert np.isfinite(latencies).all() and np.all(latencies[:, 2] > 0)
    assert (directory / "diagnostics.png").stat().st_size > 1000


@pytest.mark.dataset
@pytest.mark.parametrize("name", SEQUENCES)
def test_ate_agrees_with_independent_evo_evaluator(name: str) -> None:
    root = os.environ.get("AMBER_BENCHMARK_REPORT")
    if root is None:
        pytest.skip("Set AMBER_BENCHMARK_REPORT to a completed checkpoint benchmark")
    pytest.importorskip("evo")
    from evo.core.metrics import APE, PoseRelation, StatisticsType
    from evo.core.trajectory import PoseTrajectory3D

    directory = Path(root) / f"rgbd_dataset_{name}"
    tr, reference = read_tum(directory / "reference.tum")
    te, estimate = read_tum(directory / "trajectory.tum")
    ir, ie = associate(tr, te)
    truth = PoseTrajectory3D(poses_se3=reference[ir], timestamps=tr[ir])
    prediction = PoseTrajectory3D(poses_se3=estimate[ie], timestamps=te[ie])
    prediction.align(truth, correct_scale=True)
    metric = APE(PoseRelation.translation_part)
    metric.process_data((truth, prediction))
    ours = evaluate(directory / "reference.tum", directory / "trajectory.tum")
    assert metric.get_statistic(StatisticsType.rmse) == pytest.approx(ours["ate_rmse_m"], abs=1e-8)
