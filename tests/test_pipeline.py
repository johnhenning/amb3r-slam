"""Oracle geometry isolates orchestration from learned-model quality."""

import numpy as np
import pytest

from amber_slam.backend import SlamConfig
from amber_slam.runtime import SlamSystem
from amber_slam.types import Frame, Reconstruction


class OracleModel:
    def reconstruct(self, frames, reference=0):
        ids = np.array([f.index for f in frames])
        poses = np.repeat(np.eye(4)[None], len(frames), 0)
        poses[:, 0, 3] = 0.03 * ids
        poses[:, 1, 3] = 0.02 * np.sin(ids * 0.4)
        poses = np.linalg.inv(poses[reference]) @ poses
        k = np.array([[32.0, 0, 15.5], [0, 32, 15.5], [0, 0, 1.0]])
        return Reconstruction(
            poses,
            np.full((len(frames), 32, 32), 3.0),
            np.ones((len(frames), 32, 32)),
            np.repeat(k[None], len(frames), 0),
        )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_streaming_span_two_and_tail_flush(tmp_path, asynchronous):
    cfg = SlamConfig(
        window=6,
        stride=1,
        enable_loops=False,
        enable_long_context=False,
        graph_iterations=2,
    )
    with SlamSystem(OracleModel(), OracleModel(), tmp_path, cfg, asynchronous) as system:
        for i in range(15):
            system.push(Frame(i, i / 30, np.zeros((32, 32, 3), np.uint8)))
    poses = system.trajectory()
    assert poses.shape == (15, 4, 4) and np.isfinite(poses).all()
    assert any(e.kind == "span-2" for e in system.backend.graph.edges)
    assert system.backend.count == 6
    np.testing.assert_allclose(poses[:, 0, 3], np.arange(15) * 0.03, atol=1e-6)
    assert len(system.store.cache) <= cfg.window * 2
    with pytest.raises(RuntimeError):
        system.push(Frame(15, 0.5, np.zeros((32, 32, 3), np.uint8)))


def test_invalid_frame_sequence(tmp_path):
    with SlamSystem(OracleModel(), OracleModel(), tmp_path, asynchronous=False) as system:
        with pytest.raises(ValueError):
            system.push(Frame(2, 0, np.zeros((32, 32, 3), np.uint8)))


def test_worker_exception_is_not_swallowed(tmp_path):
    class Broken:
        def reconstruct(self, *args, **kwargs):
            raise RuntimeError("backend failed")

    cfg = SlamConfig(window=6, stride=1, enable_loops=False, enable_long_context=False)
    with pytest.raises(RuntimeError, match="backend failed"):
        with SlamSystem(OracleModel(), Broken(), tmp_path, cfg, True) as system:
            for i in range(6):
                system.push(Frame(i, i / 30, np.zeros((32, 32, 3), np.uint8)))
