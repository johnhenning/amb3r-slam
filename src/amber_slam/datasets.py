"""Dataset conversion keeps image/depth files in place and writes a manifest."""

import json
from pathlib import Path

import numpy as np

from .evaluation import associate, read_tum


def _list_file(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            time, name = line.split()[:2]
            rows.append((float(time), name))
    return rows


def prepare_tum(
    directory,
    destination,
    sequence_id,
    intrinsics,
    depth_scale=0.0002,
    split="test",
    tolerance=0.02,
):
    """TUM RGB-D association; user supplies the sequence's calibrated intrinsics."""
    root = Path(directory).resolve()
    rgb = _list_file(root / "rgb.txt")
    depth = _list_file(root / "depth.txt")
    timestamps, poses = read_tum(root / "groundtruth.txt")
    image_times = np.array([t for t, _ in rgb])
    depth_times = np.array([t for t, _ in depth])
    image_ids, depth_ids = associate(image_times, depth_times, tolerance)
    gt_ids, pose_image_ids = associate(timestamps, image_times, tolerance)
    pose_lookup = dict(zip(pose_image_ids, gt_ids))
    records = []
    for i, j in zip(image_ids, depth_ids):
        if i not in pose_lookup:
            continue
        records.append(
            {
                "timestamp": rgb[i][0],
                "rgb": str(root / rgb[i][1]),
                "depth": str(root / depth[j][1]),
                "depth_scale": depth_scale,
                "intrinsics": np.asarray(intrinsics).tolist(),
                "c2w": poses[pose_lookup[i]].tolist(),
            }
        )
    if not records:
        raise ValueError("No complete RGB/depth/pose associations")
    data = {"version": 1, "sequences": [{"id": sequence_id, "split": split, "frames": records}]}
    Path(destination).write_text(json.dumps(data, indent=2))
    return {"frames": len(records), "manifest": str(destination)}


def prepare_kitti(directory, destination, sequence_id, poses_path=None, split="test"):
    """KITTI odometry image_2/image_3 with P2/P3 rectified calibration."""
    root = Path(directory).resolve()
    calibration = {}
    for line in (root / "calib.txt").read_text().splitlines():
        key, value = line.split(":", 1)
        calibration[key] = np.fromstring(value, sep=" ").reshape(3, 4)
    p2, p3 = calibration["P2"], calibration["P3"]
    baseline = abs(p2[0, 3] / p2[0, 0] - p3[0, 3] / p3[0, 0])
    times = np.loadtxt(root / "times.txt", ndmin=1)
    left = sorted((root / "image_2").glob("*.png"))
    right = sorted((root / "image_3").glob("*.png"))
    if len(left) != len(times) or len(right) != len(times):
        raise ValueError("KITTI image/time count mismatch")
    poses = None
    if poses_path:
        rows = np.loadtxt(poses_path, ndmin=2)
        if rows.shape != (len(times), 12):
            raise ValueError("KITTI pose count mismatch")
        poses = np.repeat(np.eye(4)[None], len(rows), 0)
        poses[:, :3] = rows.reshape(-1, 3, 4)
        # Ground truth is cam0. Convert to cam2 camera-to-world coordinates.
        cam0_from_cam2 = np.eye(4)
        cam0_from_cam2[0, 3] = -p2[0, 3] / p2[0, 0]
        poses = poses @ cam0_from_cam2
    frames = []
    for i, t in enumerate(times):
        record = {
            "timestamp": float(t),
            "rgb": str(left[i]),
            "right_rgb": str(right[i]),
            "baseline_m": float(baseline),
            "intrinsics": p2[:, :3].tolist(),
        }
        if poses is not None:
            record["c2w"] = poses[i].tolist()
        frames.append(record)
    Path(destination).write_text(
        json.dumps(
            {"version": 1, "sequences": [{"id": sequence_id, "split": split, "frames": frames}]},
            indent=2,
        )
    )
    return {"frames": len(frames), "manifest": str(destination)}
