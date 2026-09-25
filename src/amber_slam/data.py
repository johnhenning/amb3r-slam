"""Explicit scene manifests; no dataset is implicitly downloaded or split."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from PIL import Image

from .contracts import FrameRecord, Manifest, PathLike
from .types import Frame

if TYPE_CHECKING:
    from torch import Tensor


def read_manifest(path: PathLike) -> tuple[Manifest, Path]:
    path = Path(path)
    data = json.loads(path.read_text())
    if data.get("version") != 1 or not data.get("sequences"):
        raise ValueError("Expected version=1 and nonempty sequences")
    seen = set()
    for seq in data["sequences"]:
        if seq["id"] in seen:
            raise ValueError("Duplicate sequence identity across splits")
        seen.add(seq["id"])
        times = [float(f["timestamp"]) for f in seq["frames"]]
        if not times or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Sequence timestamps must be finite and strictly increasing")
    return data, path.parent


def load_frame(record: FrameRecord, root: Path, index: int) -> Frame:
    rgb = np.asarray(Image.open(root / record["rgb"]).convert("RGB"))
    depth = None
    if "depth" in record:
        p = root / record["depth"]
        depth = (
            np.load(p, allow_pickle=False)
            if p.suffix == ".npy"
            else np.asarray(Image.open(p)).astype(float)
        )
        depth = depth * record.get("depth_scale", 1.0)
    k = np.asarray(record["intrinsics"], float) if "intrinsics" in record else None
    lidar = np.load(root / record["lidar"], allow_pickle=False) if "lidar" in record else None
    if "right_rgb" in record:
        from .modalities import stereo_depth

        if k is None:
            raise ValueError("Stereo requires intrinsics")
        right = np.asarray(Image.open(root / record["right_rgb"]).convert("RGB"))
        depth = stereo_depth(rgb, right, k[0, 0], record["baseline_m"])
    return Frame(index, float(record["timestamp"]), rgb, depth, k, lidar)


class SceneDataset:
    """Dataset indices identify scenes; frame count/resolution vary by train step."""

    def __init__(self, manifest: PathLike, split: str) -> None:
        self.manifest, self.root = read_manifest(manifest)
        self.scenes = [s for s in self.manifest["sequences"] if s.get("split") == split]
        if not self.scenes:
            raise ValueError(f"No scenes in split {split}")

    def __len__(self) -> int:
        return len(self.scenes)

    def sample(
        self,
        index: int,
        views: int,
        size: tuple[int, int],
        rng: np.random.Generator,
        pseudo: bool = False,
    ) -> dict[str, Tensor]:
        import cv2
        import torch

        scene = self.scenes[index % len(self.scenes)]
        n = len(scene["frames"])
        if n < views:
            raise ValueError(f"Scene {scene['id']} has fewer than {views} views")
        # Contiguous, variable-stride windows, no crossing scene boundaries.
        stride = (
            int(rng.integers(1, max(2, min(5, (n - 1) // max(views - 1, 1) + 1))))
            if views > 1
            else 1
        )
        span = (views - 1) * stride + 1
        start = int(rng.integers(0, n - span + 1))
        indices = start + np.arange(views) * stride
        images = []
        depths = []
        poses = []
        ks = []
        skies = []
        objects = []
        h, w = size
        for i in indices:
            rec = scene["frames"][int(i)]
            if pseudo:
                if "pseudo_depth" not in rec:
                    raise ValueError("Teacher stage needs pseudo_depth on every sampled frame")
                rec = rec.copy()
                rec["depth"] = rec["pseudo_depth"]
                rec["depth_scale"] = 1.0
            frame = load_frame(rec, self.root, int(i))
            if frame.depth is None or frame.intrinsics is None or "c2w" not in rec:
                raise ValueError("Training requires z-depth, intrinsics, c2w")
            ih, iw = frame.rgb.shape[:2]
            if frame.depth.shape != (ih, iw):
                raise ValueError("Depth must be registered to RGB at same resolution")
            k = frame.intrinsics.copy()
            k[0] *= w / iw
            k[1] *= h / ih
            images.append(
                cv2.resize(frame.rgb, (w, h)).transpose(2, 0, 1).astype(np.float32) / 255.0
            )
            depths.append(
                cv2.resize(
                    frame.depth.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            poses.append(rec["c2w"])
            ks.append(k)
            mask_keys: tuple[Literal["sky_mask"], Literal["object_mask"]] = (
                "sky_mask",
                "object_mask",
            )
            for key, target in zip(mask_keys, (skies, objects)):
                if key in rec:
                    target.append(
                        cv2.resize(
                            (np.asarray(Image.open(self.root / rec[key])) > 0).astype(np.float32),
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    )
        poses = np.asarray(poses)
        poses = np.linalg.inv(poses[0]) @ poses
        out = {
            "images": np.stack(images),
            "depth": np.stack(depths),
            "poses": poses,
            "intrinsics": np.stack(ks),
        }
        if len(skies) == views:
            out["sky_mask"] = np.stack(skies)
        if len(objects) == views:
            out["object_mask"] = np.stack(objects)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in out.items()}


def synthetic_dataset(destination: PathLike, frames: int = 24, size: int = 64) -> Path:
    """Analytic textured plane, translated cameras. Pipeline fixture, not benchmark."""
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    sequences = []
    for s, split in enumerate(["train", "val", "test"]):
        folder = root / split
        folder.mkdir(exist_ok=True)
        records = []
        k = np.array([[size, 0, (size - 1) / 2], [0, size, (size - 1) / 2], [0, 0, 1.0]])
        yy, xx = np.mgrid[:size, :size]
        for i in range(frames):
            p = np.eye(4)
            p[:3, 3] = [0.4 * np.sin(i * 0.13 + s), 0.1 * np.cos(i * 0.13 + s), 0]
            depth = np.full((size, size), 3.0, np.float32)
            x = (xx - k[0, 2]) / size * 3 + p[0, 3]
            y = (yy - k[1, 2]) / size * 3 + p[1, 3]
            rgb = np.stack(
                [
                    127 + 100 * np.sin(x * 12 + s),
                    127 + 100 * np.cos(y * 14 + s),
                    127 + 100 * np.sin(x * 9 + y * 7),
                ],
                -1,
            ).astype(np.uint8)
            rgb_path = f"{split}/{i:05d}.png"
            dep_path = f"{split}/{i:05d}.npy"
            Image.fromarray(rgb).save(root / rgb_path)
            np.save(root / dep_path, depth)
            records.append(
                {
                    "timestamp": i / 30,
                    "rgb": rgb_path,
                    "depth": dep_path,
                    "intrinsics": k.tolist(),
                    "c2w": p.tolist(),
                }
            )
        sequences.append({"id": f"plane-{split}", "split": split, "frames": records})
    path = root / "manifest.json"
    path.write_text(json.dumps({"version": 1, "sequences": sequences}, indent=2))
    return path
