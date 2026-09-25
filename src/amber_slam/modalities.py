import numpy as np
from scipy.spatial import cKDTree

from .geometry import Sim3, align_points


def stereo_depth(left, right, fx, baseline, disparities=128):
    """Inputs must be rectified; baseline >0 meters, fx in input pixels."""
    import cv2

    if baseline <= 0 or fx <= 0 or disparities <= 0 or disparities % 16:
        raise ValueError("Invalid stereo calibration/disparities")
    a = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    matcher = cv2.StereoSGBM_create(
        numDisparities=disparities,
        blockSize=5,
        P1=8 * 25,
        P2=32 * 25,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        disp12MaxDiff=1,
    )
    d = matcher.compute(a, b).astype(float) / 16
    out = np.zeros_like(d)
    valid = d > 0
    out[valid] = fx * baseline / d[valid]
    return out


def icp(source, target, initial=None, max_distance=0.5, iterations=30):
    """Trimmed point-to-point ICP, source -> target; already camera-registered."""
    x, y = np.asarray(source), np.asarray(target)
    x = x[np.isfinite(x).all(1)]
    y = y[np.isfinite(y).all(1)]
    if min(len(x), len(y)) < 6:
        raise ValueError("Too few LiDAR points")
    fit = initial or Sim3.identity()
    tree = cKDTree(y)
    for _ in range(iterations):
        transformed = fit.points(x)
        distances, idx = tree.query(transformed)
        good = distances < max_distance
        if good.sum() < 6:
            raise ValueError("ICP has insufficient correspondences")
        good &= distances <= np.quantile(distances[good], 0.9)
        delta = align_points(transformed[good], y[idx[good]], with_scale=False)
        fit = delta @ fit
        if np.linalg.norm(delta.log()) < 1e-6:
            break
    distances, _ = tree.query(fit.points(x))
    good = distances < max_distance
    return fit, float(np.sqrt(np.mean(distances[good] ** 2))), float(good.mean())


def voxel_overlap(a, b, size):
    if size <= 0:
        raise ValueError("voxel size must be positive")

    def occupied(x):
        return set(map(tuple, np.floor(x[np.isfinite(x).all(1)] / size).astype(np.int64)))

    va, vb = occupied(a), occupied(b)
    return len(va & vb) / max(1, min(len(va), len(vb)))


def project_lidar_depth(points, intrinsics, image_size):
    """Z-buffer camera-frame LiDAR into registered optical-axis depth in meters."""
    h, w = image_size
    points = np.asarray(points)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    points = points[valid]
    pixels = points @ intrinsics.T
    uv = np.rint(pixels[:, :2] / pixels[:, 2:3]).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, points = uv[inside], points[inside]
    depth = np.full((h, w), np.inf)
    np.minimum.at(depth, (uv[:, 1], uv[:, 0]), points[:, 2])
    depth[~np.isfinite(depth)] = 0
    return depth
