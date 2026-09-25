"""Trajectory metrics with explicit alignment and timestamp semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

from .contracts import Array, PathLike, TrajectoryMetrics, WindowAuc
from .geometry import Sim3, align_points


def read_tum(path: PathLike) -> tuple[Array, Array]:
    rows = np.loadtxt(path, comments="#", ndmin=2)
    if rows.shape[1] != 8 or len(rows) == 0 or not np.isfinite(rows).all():
        raise ValueError("TUM format: timestamp tx ty tz qx qy qz qw")
    if np.any(np.diff(rows[:, 0]) <= 0):
        raise ValueError("Timestamps must increase")
    if np.any(np.linalg.norm(rows[:, 4:], axis=1) < 1e-10):
        raise ValueError("Zero quaternion")
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    poses[:, :3, 3] = rows[:, 1:4]
    poses[:, :3, :3] = Rotation.from_quat(rows[:, 4:]).as_matrix()
    return rows[:, 0], poses


def write_tum(path: PathLike, timestamps: ArrayLike, poses: Array) -> None:
    timestamps = np.asarray(timestamps)
    if len(timestamps) != len(poses):
        raise ValueError("Timestamp/pose length mismatch")
    rows = np.column_stack(
        [timestamps, poses[:, :3, 3], Rotation.from_matrix(poses[:, :3, :3]).as_quat()]
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt="%.9f", header="timestamp tx ty tz qx qy qz qw")


def associate(reference: Array, estimated: Array, tolerance: float = 0.02) -> Array:
    """Greedy smallest-time-difference, one-to-one association within tolerance."""
    if tolerance < 0:
        raise ValueError("Negative timestamp tolerance")
    pairs = []
    for j, t in enumerate(estimated):
        lo = np.searchsorted(reference, t - tolerance, side="left")
        hi = np.searchsorted(reference, t + tolerance, side="right")
        pairs.extend((abs(reference[i] - t), i, j) for i in range(lo, hi))
    used_ref = set()
    used_est = set()
    accepted = []
    for _, i, j in sorted(pairs):
        if i not in used_ref and j not in used_est:
            used_ref.add(i)
            used_est.add(j)
            accepted.append((i, j))
    accepted.sort()
    if not accepted:
        raise ValueError("No associated timestamps")
    return np.asarray(accepted, dtype=int).T


def align_trajectory(estimated: Array, reference: Array, mode: str = "sim3") -> tuple[Array, Sim3]:
    if mode == "none":
        return estimated.copy(), Sim3.identity()
    if mode not in {"sim3", "se3"}:
        raise ValueError("Alignment must be none, se3, or sim3")
    # Collinear/stationary position paths do not determine all rotations. Do not
    # silently hide that degeneracy with a convenient orientation alignment.
    transform = align_points(estimated[:, :3, 3], reference[:, :3, 3], with_scale=mode == "sim3")
    return np.stack([transform.pose(p) for p in estimated]), transform


def error_auc(errors: ArrayLike, threshold: float) -> float:
    """Exact integral of empirical recall on [0,threshold], normalized to [0,1]."""
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    return float(np.maximum(0.0, 1.0 - np.asarray(errors) / threshold).mean())


def relative_pose_errors(estimated: Array, reference: Array, delta: int = 1) -> tuple[Array, Array]:
    if delta < 1 or delta >= len(estimated):
        raise ValueError("Invalid RPE frame delta")
    predicted = np.linalg.inv(estimated[:-delta]) @ estimated[delta:]
    truth = np.linalg.inv(reference[:-delta]) @ reference[delta:]
    error = np.linalg.inv(truth) @ predicted
    translation = np.linalg.norm(error[:, :3, 3], axis=1)
    rotation = Rotation.from_matrix(error[:, :3, :3]).magnitude() * 180 / np.pi
    return translation, rotation


def window_auc(estimated: Array, reference: Array, length: float, ratio: float = 0.05) -> WindowAuc:
    """Our protocol: each reference-distance window is independently Sim3-aligned.

    Score all frame translation errors at ratio*length. Windows start at every
    frame; incomplete/degenerate windows are counted as skipped. This is NOT
    claimed to exactly reproduce VidMap W-AUC; use its official evaluator for
    published-table comparisons.
    """
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(reference[:, :3, 3], axis=0), axis=1))]
    values = []
    skipped = 0
    for i in range(len(reference)):
        j = int(np.searchsorted(distance, distance[i] + length))
        if j >= len(reference):
            continue
        try:
            aligned, _ = align_trajectory(estimated[i : j + 1], reference[i : j + 1], "sim3")
            errors = np.linalg.norm(aligned[:, :3, 3] - reference[i : j + 1, :3, 3], axis=1)
            values.append(error_auc(errors, length * ratio))
        except ValueError:
            skipped += 1
    return {
        "auc": float(np.mean(values)) if values else None,
        "windows": len(values),
        "skipped": skipped,
    }


def evaluate(
    reference_path: PathLike,
    estimated_path: PathLike,
    alignment: str = "sim3",
    tolerance: float = 0.02,
    delta: int = 1,
) -> TrajectoryMetrics:
    tr, reference = read_tum(reference_path)
    te, estimated = read_tum(estimated_path)
    ir, ie = associate(tr, te, tolerance)
    if len(ir) < max(3, delta + 1):
        raise ValueError("Insufficient associated poses")
    reference, estimated = reference[ir], estimated[ie]
    aligned, transform = align_trajectory(estimated, reference, alignment)
    ate = np.linalg.norm(aligned[:, :3, 3] - reference[:, :3, 3], axis=1)
    translation, rotation = relative_pose_errors(aligned, reference, delta)
    return {
        "alignment": alignment,
        "alignment_scale": transform.scale,
        "matched_poses": len(ir),
        "reference_coverage": len(ir) / len(tr),
        "estimate_coverage": len(ie) / len(te),
        "timestamp_tolerance_s": tolerance,
        "rpe_delta_frames": delta,
        "ate_rmse_m": float(np.sqrt(np.mean(ate**2))),
        "rpe_translation_rmse_m": float(np.sqrt(np.mean(translation**2))),
        "rpe_rotation_rmse_deg": float(np.sqrt(np.mean(rotation**2))),
        "translation_auc": {str(t): error_auc(ate, t) for t in [0.05, 0.1, 0.5, 1.0, 10.0]},
        "protocol": "ATE and RPE after the stated full-trajectory alignment",
    }
