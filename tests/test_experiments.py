import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from amber_slam.contracts import JsonObject
from amber_slam.experiments import Trajectory, run_experiment
from amber_slam.types import Frame


class DummyApproach:
    def __init__(self, fail: bool = False) -> None:
        self.frames: list[Frame] = []
        self.closed = False
        self.fail = fail

    def push(self, frame: Frame) -> None:
        assert frame.depth is None and frame.intrinsics is None
        if self.fail:
            raise RuntimeError("intentional failure")
        self.frames.append(frame)

    def finish(self) -> Trajectory:
        poses = np.repeat(np.eye(4)[None], len(self.frames), axis=0)
        poses[:, 0, 3] = np.arange(len(self.frames))
        poses[:, 1, 3] = np.arange(len(self.frames)) ** 2
        return Trajectory(np.array([f.timestamp for f in self.frames]), poses)

    def close(self) -> None:
        self.closed = True


@pytest.mark.parametrize("failure", [None, "push", "finish", "cleanup"])
def test_shared_replay(
    tmp_path: Path, failure: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Unit-test lifecycle/evaluation independently of host /proc availability.
    monkeypatch.setattr("amber_slam.experiments.ReplayProfiler", MagicMock())
    data = tmp_path / "data"
    data.mkdir()
    for i in range(4):
        Image.new("RGB", (8, 8)).save(data / f"{i}.png")
    (data / "rgb.txt").write_text("".join(f"{i} {i}.png\n" for i in range(4)))
    (data / "groundtruth.txt").write_text("".join(f"{i} {i} {i * i} 0 0 0 0 1\n" for i in range(4)))
    method = DummyApproach(failure in {"push", "cleanup"})
    if failure == "finish":
        monkeypatch.setattr(method, "finish", MagicMock(side_effect=RuntimeError("intentional")))
    if failure == "cleanup":

        def broken_close() -> None:
            method.closed = True
            raise ValueError("cleanup failure")

        monkeypatch.setattr(method, "close", broken_close)

    def run() -> JsonObject:
        return run_experiment(
            data,
            tmp_path / "run",
            lambda _: method,
            approach="test-only",
            revision="fixture-v1",
            config={},
            stride=1,
        )

    if failure:
        with pytest.raises(RuntimeError, match="intentional") as error:
            run()
        if failure == "cleanup":
            assert "cleanup failure" in error.value.__notes__[0]
        assert not (tmp_path / "run/results.json").exists()
    else:
        result = run()
        assert result["estimate_fraction"] == 1
        assert result["metrics"]["final"]["ate_rmse_m"] < 1e-8
        with pytest.raises(FileExistsError):
            run()
        from amber_slam.experiment_cli import main

        monkeypatch.setattr(
            "amber_slam.experiment_cli.load_factory",
            lambda spec: lambda output, config: DummyApproach(),
        )
        config_path = tmp_path / "config.json"
        config_path.write_text('{"execution_mode": "sequential"}')
        main(
            [
                "--data",
                str(data),
                "--output",
                str(tmp_path / "cli-run"),
                "--factory",
                "test:factory",
                "--config",
                str(config_path),
                "--approach",
                "fixture",
                "--revision",
                "v1",
                "--stride",
                "1",
            ]
        )
        cli_result = json.loads((tmp_path / "cli-run/results.json").read_text())
        assert cli_result["approach"] == "fixture"
        assert cli_result["statistics"]["execution_mode"] == "sequential"
    assert method.closed


def test_reject_invalid_trajectory() -> None:
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    for times in ([0, 0, 1], [0, 1, 99]):
        with pytest.raises(ValueError):
            Trajectory(np.array(times), poses).validate(np.arange(3))
    poses[0, 0, 0] = 2
    with pytest.raises(ValueError, match="rigid"):
        Trajectory(np.arange(3), poses).validate(np.arange(3))


@pytest.mark.parametrize("asynchronous", [False, True])
def test_amber_adapter_drains_and_closes(tmp_path: Path, asynchronous: bool) -> None:
    from test_pipeline import OracleModel

    from amber_slam.approaches import AmberApproach
    from amber_slam.backend import SlamConfig

    method = AmberApproach(
        tmp_path,
        OracleModel(),
        OracleModel(),
        SlamConfig(window=6, stride=1, enable_loops=False, enable_long_context=False),
        asynchronous=asynchronous,
    )
    times = np.arange(8) / 30
    try:
        for i, timestamp in enumerate(times):
            method.push(Frame(i, float(timestamp), np.zeros((32, 32, 3), np.uint8)))
        trajectory = method.finish()
        trajectory.validate(times)
        assert method.system.closed
        assert method.system.last_mapped == 7
    finally:
        method.close()
        method.close()


def test_cli_factory_validation(tmp_path: Path) -> None:
    from amber_slam.experiment_cli import load_factory, main, read_object

    with pytest.raises(ValueError, match="module:callable"):
        load_factory("missing_separator")
    with pytest.raises(ValueError, match="not callable"):
        load_factory("amber_slam.experiment_cli:__name__")
    config = tmp_path / "config.json"
    config.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        read_object(config)
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--data",
                str(tmp_path),
                "--output",
                str(tmp_path),
                "--factory",
                "missing:factory",
                "--approach",
                "test",
                "--revision",
                "v1",
                "--config",
                str(config),
            ]
        )
    assert error.value.code == 2


def test_offline_export_without_amber_metadata(tmp_path: Path) -> None:
    from amber_slam.experiment_tracking import export_run

    # Synthetic export-contract fixture, not a benchmark measurement.
    run = tmp_path / "fixture"
    (run / "profile").mkdir(parents=True)
    result = {
        "approach": "fixture",
        "approach_revision": "v1",
        "protocol": "test-only",
        "sequence": "fixture",
        "statistics": {"execution_mode": "sequential"},
        "sampled_frames": 3,
        "sampling_stride": 1,
        "max_frames": None,
        "config": {},
        "metrics": {},
        "input_manifest_sha256": "fixture",
        "groundtruth_sha256": "fixture",
    }
    (run / "results.json").write_text(json.dumps(result))
    (run / "environment.json").write_text("{}")
    (run / "profile/summary.json").write_text(
        json.dumps(
            {
                "wait_environment": {},
                "scope": "synthetic test only",
                "stages": {},
            }
        )
    )
    (run / "profile/samples.csv").write_text(
        "elapsed_s,cpu_time_s,rss_bytes,uss_bytes,threads\n1,1,1024,512,1\n"
    )
    (run / "latency.csv").write_text("frame,timestamp,push_latency_ms\n0,0,1\n")
    receipt = export_run(run, "test-only", output=tmp_path / "wandb", mode="offline")
    assert receipt["url"] is None
    assert receipt == export_run(run, "test-only", output=tmp_path / "wandb", mode="offline")
