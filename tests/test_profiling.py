from pathlib import Path

import pytest

from amber_slam.profiling import ReplayProfiler


def test_profile_records_stages_and_memory(tmp_path: Path) -> None:
    profile = ReplayProfiler(tmp_path, interval_s=0.01)
    with profile:
        result = profile.wrap("sum", sum)(range(10000))
    assert result == 49995000
    assert profile.summary["elapsed_s"] > 0
    assert profile.summary["rss_peak_bytes_sampled"] > 0
    assert profile.summary["stages"]["sum"]["calls"] == 1
    assert (tmp_path / "stages.json").is_file()
    assert (tmp_path / "samples.csv").is_file()
    assert not profile.thread.is_alive()


def test_profile_preserves_exception_and_joins(tmp_path: Path) -> None:
    profile = ReplayProfiler(tmp_path)
    with pytest.raises(ValueError, match="application"):
        with profile:
            raise ValueError("application")
    assert not profile.thread.is_alive()
    assert not (tmp_path / "summary.json").exists()
