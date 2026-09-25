"""Optional teacher-label generation; robust positive affine depth alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .contracts import Array, PathLike
from .data import load_frame, read_manifest
from .types import GeometryModel


def align_teacher_depth(
    prediction: Array,
    measured: Array,
    seed: int = 0,
    trials: int = 128,
    min_inlier_fraction: float = 0.4,
) -> tuple[Array, dict[str, float]]:
    valid = np.isfinite(prediction) & np.isfinite(measured) & (prediction > 0) & (measured > 0)
    x = prediction[valid].astype(float)
    y = measured[valid].astype(float)
    if len(x) < 32:
        raise ValueError("At least 32 valid pixels required for pseudo-depth alignment")
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(x), min(len(x), 10000), replace=False)
    x = x[sample]
    y = y[sample]
    threshold = max(0.02, 0.05 * np.median(y))
    best = None
    best_count = 0
    for _ in range(trials):
        i, j = rng.choice(len(x), 2, replace=False)
        if abs(x[i] - x[j]) < 1e-6:
            continue
        scale = (y[i] - y[j]) / (x[i] - x[j])
        shift = y[i] - scale * x[i]
        if scale <= 0:
            continue
        mask = np.abs(scale * x + shift - y) < threshold
        if mask.sum() > best_count:
            best = mask
            best_count = int(mask.sum())
    if best is None or best_count < len(x) * min_inlier_fraction:
        raise ValueError("Pseudo-depth alignment failed; preserve original supervision")
    scale, shift = np.linalg.lstsq(
        np.column_stack([x[best], np.ones(best.sum())]), y[best], rcond=None
    )[0]
    if scale <= 0:
        raise ValueError("Invalid pseudo-depth scale")
    aligned = (scale * prediction + shift).astype(np.float32)
    aligned[~np.isfinite(aligned) | (aligned <= 0)] = 0
    return aligned, {
        "scale": float(scale),
        "shift": float(shift),
        "inlier_fraction": best_count / len(x),
    }


def generate_labels(manifest: PathLike, model: GeometryModel, destination: PathLike) -> Path:
    data, root = read_manifest(manifest)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reports = []
    for s, sequence in enumerate(data["sequences"]):
        if sequence.get("split") not in {"train", "val"}:
            continue
        for i, record in enumerate(sequence["frames"]):
            frame = load_frame(record, root, i)
            if frame.depth is None:
                raise ValueError("Pseudo-depth generation needs sensor or reconstructed depth")
            prediction = model.reconstruct([frame]).depth[0]
            prediction = cv2.resize(prediction, (frame.depth.shape[1], frame.depth.shape[0]))
            aligned, report = align_teacher_depth(prediction, frame.depth)
            path = destination / f"{s:04d}_{i:08d}.npy"
            np.save(path, aligned)
            record["pseudo_depth"] = str(path)
            reports.append(dict(sequence=sequence["id"], frame=i, **report))
    # Resolve existing file paths because output manifest has a new parent.
    for sequence in data["sequences"]:
        for record in sequence["frames"]:
            path_keys: tuple[
                Literal["rgb", "depth", "right_rgb", "lidar", "sky_mask", "object_mask"], ...
            ] = (
                "rgb",
                "depth",
                "right_rgb",
                "lidar",
                "sky_mask",
                "object_mask",
            )
            for key in path_keys:
                if key in record:
                    record[key] = str((root / record[key]).resolve())
    (destination / "manifest.json").write_text(json.dumps(data, indent=2))
    (destination / "alignment.json").write_text(json.dumps(reports, indent=2))
    return destination / "manifest.json"
