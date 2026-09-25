"""Public model boundary: explicit data ownership, units and shape contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Frame:
    """One processed capture. The caller must not mutate buffers after push."""

    index: int
    timestamp: float
    rgb: np.ndarray  # HWC RGB uint8
    depth: np.ndarray | None = None  # registered z-depth in meters
    intrinsics: np.ndarray | None = None
    lidar: np.ndarray | None = None  # XYZ in CAMERA frame, meters


@dataclass
class Reconstruction:
    """Predictions in input-frame order and one shared local coordinate frame."""

    poses: np.ndarray  # V,4,4 camera -> common local world
    depth: np.ndarray  # V,H,W z-depth in same scale as translations
    confidence: np.ndarray  # V,H,W positive, comparable within same model
    intrinsics: np.ndarray  # V,3,3 at DEPTH resolution

    def validate(self, n: int) -> "Reconstruction":
        """Reject malformed predictions at the boundary rather than in geometry."""
        if self.poses.shape != (n, 4, 4) or self.intrinsics.shape != (n, 3, 3):
            raise ValueError("Invalid model pose/intrinsics shape")
        if (
            self.depth.ndim != 3
            or self.depth.shape[0] != n
            or self.confidence.shape != self.depth.shape
        ):
            raise ValueError("Invalid model depth/confidence shape")
        if not all(
            np.isfinite(a).all() for a in (self.poses, self.depth, self.confidence, self.intrinsics)
        ):
            raise ValueError("Nonfinite model prediction")
        if np.any(self.depth <= 0) or np.any(self.confidence <= 0):
            raise ValueError("Depth and confidence must be positive")
        if not np.allclose(self.poses[:, 3, :], [0, 0, 0, 1], atol=1e-4):
            raise ValueError("Pose bottom row must be homogeneous")
        rotations = self.poses[:, :3, :3]
        if not np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=2e-3):
            raise ValueError("Pose rotation must be orthonormal")
        if np.any(np.linalg.det(rotations) < 0.99):
            raise ValueError("Pose rotation must be proper")
        if np.any(self.intrinsics[:, 0, 0] <= 0) or np.any(self.intrinsics[:, 1, 1] <= 0):
            raise ValueError("Focal lengths must be positive")
        return self


class GeometryModel(Protocol):
    def reconstruct(self, frames: Sequence[Frame], reference: int = 0) -> Reconstruction: ...
