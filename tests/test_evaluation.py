from __future__ import annotations

import numpy as np
import pytest

from amber_slam.evaluation import (
    align_trajectory,
    associate,
    error_auc,
    relative_pose_errors,
)
from amber_slam.geometry import Sim3


def test_one_to_one_timestamp_matching() -> None:
    a, b = associate(np.array([0.0, 1.0, 2.0]), np.array([0.001, 0.002, 1.001, 2.001]), 0.01)
    assert len(a) == 3 and len(set(a)) == 3 and len(set(b)) == 3


def test_similarity_alignment_and_rpe() -> None:
    poses = np.repeat(np.eye(4)[None], 12, axis=0)
    t = np.arange(12) * 0.2
    poses[:, :3, 3] = np.stack([np.sin(t), np.cos(t), t * 0.2], 1)
    transform = Sim3.exp(np.array([1, 2, 3, 0.2, 0.1, -0.3, 0.5]))
    changed = np.stack([transform.pose(p) for p in poses])
    aligned, _ = align_trajectory(changed, poses)
    np.testing.assert_allclose(aligned, poses, atol=1e-8)
    translation, rotation = relative_pose_errors(aligned, poses)
    assert translation.max() < 1e-8 and rotation.max() < 1e-6
    assert error_auc([0, 0.5, 1], 1) == pytest.approx(0.5)
