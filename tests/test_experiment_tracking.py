"""Ensure imported profiles keep measured units and avoid cache uploads."""

from pathlib import Path

import pytest

from amber_slam.experiment_tracking import artifact_files, numeric_metrics, resource_history


def test_resource_cpu_is_derived_from_recorded_process_time(tmp_path: Path) -> None:
    path = tmp_path / "samples.csv"
    path.write_text(
        "elapsed_s,cpu_time_s,rss_bytes,uss_bytes,threads\n"
        "2,10,2097152,1048576,4\n4,15,3145728,2097152,6\n"
    )
    rows = list(resource_history(path))
    assert "resources/cpu_percent" not in rows[0]
    assert rows[1]["resources/cpu_percent"] == pytest.approx(250)
    assert rows[1]["resources/rss_mib"] == 3
    assert rows[1]["replay/seconds"] == 4
    assert numeric_metrics({"ready": True, "error": {"rmse": 0.2}}, "accuracy") == {
        "accuracy/error/rmse": 0.2
    }


def test_artifacts_exclude_large_caches_and_external_symlinks(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_text("{}")
    (tmp_path / "runtime/frames").mkdir(parents=True)
    (tmp_path / "runtime/frames/image.png").write_bytes(b"frame")
    (tmp_path / "runtime/graph_edges.json").write_text("[]")
    (tmp_path / "cache.npz").write_bytes(b"cache")
    (tmp_path / "linked.json").symlink_to(tmp_path / "results.json")
    assert {p.relative_to(tmp_path).as_posix() for p in artifact_files(tmp_path)} == {
        "results.json",
        "runtime/graph_edges.json",
    }
