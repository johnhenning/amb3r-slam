"""Float64 geometry, column-vector transforms; tangent order [v, omega, log(s)]."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm
from scipy.spatial.transform import Rotation, Slerp

from .contracts import Array


def skew(w: ArrayLike) -> Array:
    x, y, z = np.asarray(w, dtype=float)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _V(w: ArrayLike, sigma: float) -> Array:
    a = np.zeros((6, 6))
    a[:3, :3] = skew(w) + sigma * np.eye(3)
    a[:3, 3:] = np.eye(3)
    return expm(a)[:3, 3:]


@dataclass(frozen=True)
class Sim3:
    scale: float
    rotation: Array
    translation: Array

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Sim3 scale must be finite and positive")
        if self.rotation.shape != (3, 3) or self.translation.shape != (3,):
            raise ValueError("Invalid Sim3 shapes")
        if not np.isfinite(self.rotation).all() or not np.isfinite(self.translation).all():
            raise ValueError("Nonfinite transform")

    @staticmethod
    def identity() -> Sim3:
        return Sim3(1.0, np.eye(3), np.zeros(3))

    @staticmethod
    def exp(x: ArrayLike) -> Sim3:
        x = np.asarray(x, dtype=float)
        return Sim3(
            float(np.exp(x[6])),
            Rotation.from_rotvec(x[3:6]).as_matrix(),
            _V(x[3:6], x[6]) @ x[:3],
        )

    def log(self) -> Array:
        w = Rotation.from_matrix(self.rotation).as_rotvec()
        s = np.log(self.scale)
        return np.r_[np.linalg.solve(_V(w, s), self.translation), w, s]

    def inverse(self) -> Sim3:
        r = self.rotation.T
        return Sim3(1 / self.scale, r, -r @ self.translation / self.scale)

    def __matmul__(self, other: Sim3) -> Sim3:
        return Sim3(
            self.scale * other.scale,
            self.rotation @ other.rotation,
            self.scale * self.rotation @ other.translation + self.translation,
        )

    def points(self, xyz: ArrayLike) -> Array:
        return self.scale * np.asarray(xyz) @ self.rotation.T + self.translation

    def pose(self, c2w: Array) -> Array:
        out = np.eye(4)
        out[:3, :3] = self.rotation @ c2w[:3, :3]
        out[:3, 3] = self.points(c2w[:3, 3])
        return out


def align_points(
    source: ArrayLike, target: ArrayLike, with_scale: bool = True, weights: ArrayLike | None = None
) -> Sim3:
    """Weighted Umeyama, source -> target. Reject collinear/degenerate clouds."""
    x, y = np.asarray(source, float), np.asarray(target, float)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 3 or len(x) < 3:
        raise ValueError("Need >=3 corresponding 3D points")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite correspondences")
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    if w.shape != (len(x),) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid alignment weights")
    w = w / w.sum()
    mx, my = w @ x, w @ y
    xc, yc = x - mx, y - my
    cov = (yc * w[:, None]).T @ xc
    u, d, vt = np.linalg.svd(cov)
    if d[1] < 1e-12:
        raise ValueError("Degenerate alignment")
    sign = np.ones(3)
    sign[2] = np.linalg.det(u @ vt)
    r = (u * sign) @ vt
    s = float((d * sign).sum() / (w @ (xc * xc).sum(1))) if with_scale else 1.0
    return Sim3(s, r, my - s * r @ mx)


def robust_align(
    source: ArrayLike, target: ArrayLike, with_scale: bool = True, iterations: int = 5
) -> Sim3:
    x, y = np.asarray(source), np.asarray(target)
    fit = align_points(x, y, with_scale)
    for _ in range(iterations):
        e = np.linalg.norm(fit.points(x) - y, axis=1)
        cutoff = max(1.4826 * np.median(np.abs(e - np.median(e))) * 2.5, 1e-6)
        fit = align_points(x, y, with_scale, np.minimum(1.0, cutoff / np.maximum(e, 1e-12)))
    return fit


def metric_scale(
    predicted: ArrayLike, measured: ArrayLike, min_inliers: float = 0.5, tolerance: float = 0.25
) -> float:
    p, m = np.asarray(predicted), np.asarray(measured)
    valid = np.isfinite(p) & np.isfinite(m) & (p > 1e-6) & (m > 0)
    if valid.sum() < 16:
        raise ValueError("Too few valid depth pixels")
    ratios = m[valid] / p[valid]
    s = float(np.median(ratios))
    if np.mean(np.abs(ratios / s - 1) < tolerance) < min_inliers:
        raise ValueError("Inconsistent metric depth scale")
    return s


def interpolate_poses(ids: ArrayLike, poses: Array, query: ArrayLike) -> Array:
    ids, q = np.asarray(ids), np.asarray(query)
    if len(ids) == 1:
        return np.repeat(poses[:1], len(q), axis=0)
    q = np.clip(q, ids[0], ids[-1])
    out = np.repeat(np.eye(4)[None], len(q), axis=0)
    out[:, :3, :3] = Slerp(ids, Rotation.from_matrix(poses[:, :3, :3]))(q).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(q, ids, poses[:, k, 3])
    return out


def average_poses(poses: Sequence[Array], weights: Sequence[float]) -> Array:
    w = np.asarray(weights, float)
    w /= w.sum()
    out = np.eye(4)
    out[:3, 3] = w @ np.asarray(poses)[:, :3, 3]
    out[:3, :3] = Rotation.from_matrix(np.asarray(poses)[:, :3, :3]).mean(w).as_matrix()
    return out


def unproject(depth: Array, intrinsics: Array, pose: Array, stride: int = 4) -> tuple[Array, Array]:
    h, w = depth.shape
    yy, xx = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[::stride, ::stride]
    uv = np.stack([xx, yy, np.ones_like(xx)], -1)
    xyz = (uv @ np.linalg.inv(intrinsics).T) * z[..., None]
    world = xyz @ pose[:3, :3].T + pose[:3, 3]
    return world, np.isfinite(world).all(-1) & (z > 0)
