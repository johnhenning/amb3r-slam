"""Hierarchical reconstruction and graph optimization.

Graph state is single-owner: only its worker calls ``integrate`` or ``process``.
Local ``prepare`` may run concurrently; model calls are protected by a lock.
Submap tensors live on disk; the pose graph and trajectory metadata grow with
sequence length. This is a research CPU optimizer, not a constant-memory mapper.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import Array, GraphReport, PathLike
from .geometry import (
    Sim3,
    average_poses,
    interpolate_poses,
    metric_scale,
    robust_align,
    unproject,
)
from .graph import Edge, PoseGraph
from .modalities import icp, project_lidar_depth, voxel_overlap
from .retrieval import CandidateRetriever, LoopRetriever
from .store import FrameStore
from .types import Frame, GeometryModel, Reconstruction


@dataclass
class SlamConfig:
    """Engineering defaults, not undisclosed author hyperparameters."""

    window: int = 24
    stride: int = 3
    long_every: int = 3
    long_window: int = 6
    long_views_per_submap: int = 3
    confidence_fraction: float = 0.5
    minimum_overlap: float = 0.15
    voxel_size: float = 0.15
    loop_exclusion: int = 6
    loop_score: float = 0.12
    max_loop_rotation_deg: float = 90.0
    max_loop_translation: float = 50.0
    metric: bool = False
    lidar: bool = False
    enable_loops: bool = True
    enable_long_context: bool = True
    whiten: bool = True
    graph_iterations: int = 25

    def __post_init__(self) -> None:
        if self.window < 6 or self.window % 3:
            raise ValueError("window must be >=6 and divisible by 3")
        if self.stride < 1 or self.long_every < 1 or self.long_window < 4:
            raise ValueError("Invalid temporal configuration")
        if self.long_views_per_submap < 2:
            raise ValueError("long context requires at least two views per submap")
        if self.lidar and not self.metric:
            raise ValueError("LiDAR requires metric=True")
        if not 0 < self.minimum_overlap <= 1 or self.voxel_size <= 0:
            raise ValueError("Invalid overlap parameters")

    @property
    def step(self) -> int:
        return self.window // 3

    @classmethod
    def load(cls, path: PathLike | None) -> SlamConfig:
        return cls(**json.loads(Path(path).read_text())) if path else cls()


@dataclass
class Submap:
    index: int
    ids: Array
    poses: Array
    selected: Array
    reconstruction: Reconstruction

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            ids=self.ids,
            poses=self.poses,
            selected=self.selected,
            selected_poses=self.reconstruction.poses,
            depth=self.reconstruction.depth,
            confidence=self.reconstruction.confidence,
            intrinsics=self.reconstruction.intrinsics,
        )

    @classmethod
    def load(cls, index: int, path: Path) -> Submap:
        with np.load(path, allow_pickle=False) as z:
            reconstruction = Reconstruction(
                z["selected_poses"], z["depth"], z["confidence"], z["intrinsics"]
            )
            return cls(index, z["ids"], z["poses"], z["selected"], reconstruction)

    def cloud(self, ids: Collection[int] | None = None) -> Array:
        clouds = []
        for i, frame_id in enumerate(self.selected):
            if ids is not None and frame_id not in ids:
                continue
            r = self.reconstruction
            xyz, valid = unproject(r.depth[i], r.intrinsics[i], r.poses[i], stride=8)
            valid &= r.confidence[i, ::8, ::8] >= np.median(r.confidence[i])
            clouds.append(xyz[valid])
        return np.concatenate(clouds) if clouds else np.empty((0, 3))


def align_submaps(source: Submap, target: Submap, metric: bool = False) -> Sim3:
    """Estimate target_from_source using shared-image dense correspondences.

    If view selection has no common reconstructed images, use interpolated
    camera rotations plus robust baseline scale. This fallback also handles
    straight camera paths for which position-only Umeyama is degenerate.
    """
    shared = sorted(set(source.selected) & set(target.selected))
    source_points, target_points = [], []
    for frame_id in shared:
        a = int(np.where(source.selected == frame_id)[0][0])
        b = int(np.where(target.selected == frame_id)[0][0])
        ra, rb = source.reconstruction, target.reconstruction
        if ra.depth[a].shape != rb.depth[b].shape:
            continue
        xa, va = unproject(ra.depth[a], ra.intrinsics[a], ra.poses[a], 8)
        xb, vb = unproject(rb.depth[b], rb.intrinsics[b], rb.poses[b], 8)
        valid = va & vb
        valid &= ra.confidence[a, ::8, ::8] >= np.median(ra.confidence[a])
        valid &= rb.confidence[b, ::8, ::8] >= np.median(rb.confidence[b])
        source_points.extend(xa[valid])
        target_points.extend(xb[valid])
    if len(source_points) >= 6:
        try:
            return robust_align(source_points, target_points, not metric)
        except ValueError:
            pass  # Geometrically degenerate points; use camera orientations.
    common = sorted(set(source.ids) & set(target.ids))
    if not common:
        raise ValueError("Cannot align submaps without shared observations")
    a = source.poses[np.searchsorted(source.ids, common)]
    b = target.poses[np.searchsorted(target.ids, common)]
    rotation = (
        Rotation.from_matrix(b[:, :3, :3] @ a[:, :3, :3].transpose(0, 2, 1)).mean().as_matrix()
    )
    ratios = []
    for i in range(len(common)):
        da = np.linalg.norm(a[i + 1 :, :3, 3] - a[i, :3, 3], axis=1)
        db = np.linalg.norm(b[i + 1 :, :3, 3] - b[i, :3, 3], axis=1)
        good = (da > 1e-4) & (db > 1e-4)
        ratios.extend(db[good] / da[good])
    scale = 1.0 if metric or not ratios else float(np.median(ratios))
    translation = np.median(b[:, :3, 3] - scale * a[:, :3, 3] @ rotation.T, axis=0)
    return Sim3(scale, rotation, translation)


@dataclass
class BackendUpdate:
    poses: dict[int, Array]
    anchor_id: int
    anchor_depth: Array
    graph_report: GraphReport


class HierarchicalBackend:
    def __init__(self, model: GeometryModel, directory: PathLike, config: SlamConfig) -> None:
        self.model, self.config = model, config
        # Local mapping and graph-side loop/context inference share one model.
        # Serialize only model calls; graph optimization can overlap mapping.
        self._model_lock = Lock()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.store = FrameStore(self.directory / "frames")
        self.submap_directory = self.directory / "submaps"
        self.submap_directory.mkdir(exist_ok=True)
        self.graph = PoseGraph(metric=config.metric, whiten=config.whiten)
        self.retrieval: CandidateRetriever = LoopRetriever(config.loop_exclusion, config.loop_score)
        self.count = 0
        self.rejected = []
        (self.directory / "slam_config.json").write_text(json.dumps(asdict(config), indent=2))

    def _path(self, index: int) -> Path:
        return self.submap_directory / f"{index:06d}.npz"

    def _load(self, index: int) -> Submap:
        return Submap.load(index, self._path(index))

    def _reconstruct(
        self,
        frames: Sequence[Frame],
        scores: Sequence[float] | None = None,
        index: int = 0,
    ) -> Submap:
        ids = np.array([f.index for f in frames])
        if scores is None:
            selected = list(range(len(frames)))
        else:
            selected = []
            for group in np.unique(ids // self.config.stride):
                candidates = np.flatnonzero(ids // self.config.stride == group)
                selected.append(int(max(candidates, key=lambda i: scores[i])))
            # Endpoints prevent extrapolation; central reference reduces drift.
            selected = sorted(set(selected + [0, len(frames) // 2, len(frames) - 1]))
        reference = min(range(len(selected)), key=lambda i: abs(selected[i] - len(frames) // 2))
        chosen = [frames[i] for i in selected]
        with self._model_lock:
            reconstruction = self.model.reconstruct(chosen, reference).validate(len(chosen))
        if self.config.metric:
            predicted, measured = [], []
            for i, frame in enumerate(chosen):
                sensor_depth = frame.depth
                if sensor_depth is None and self.config.lidar:
                    if frame.lidar is None or frame.intrinsics is None:
                        raise ValueError("LiDAR metric scale requires camera calibration and scans")
                    sensor_depth = project_lidar_depth(
                        frame.lidar, frame.intrinsics, (frame.rgb.shape[0], frame.rgb.shape[1])
                    )
                if sensor_depth is None:
                    raise ValueError("Metric mode requires registered sensor depth")
                h, w = reconstruction.depth[i].shape
                measured.append(
                    cv2.resize(
                        sensor_depth.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                )
                predicted.append(reconstruction.depth[i])
            scale = metric_scale(np.stack(predicted), np.stack(measured))
            reconstruction.depth *= scale
            reconstruction.poses[:, :3, 3] *= scale
        poses = interpolate_poses(ids[selected], reconstruction.poses, ids)
        return Submap(index, ids, poses, ids[selected], reconstruction)

    def prepare(self, frames: list[Frame], scores: list[float], index: int) -> Submap:
        """Reconstruct a local window without reading or mutating graph state.

        The returned submap transfers ownership to the graph worker. Configuration
        and input frames must remain unchanged while a run is active.
        """
        return self._reconstruct(frames, scores, index)

    def _joint_edge(
        self, a: Submap, b: Submap, kind: str, context: Submap | None = None
    ) -> Edge | None:
        count = self.config.long_views_per_submap

        def sample(submap: Submap) -> list[int]:
            positions = np.linspace(
                0, len(submap.selected) - 1, min(count, len(submap.selected))
            ).astype(int)
            return submap.selected[positions].tolist()

        if context is None:
            selected = sorted(set(sample(a) + sample(b)))
            context = self._reconstruct([self.store.get(i) for i in selected])
        local_confidence = np.mean(
            [a.reconstruction.confidence.mean(), b.reconstruction.confidence.mean()]
        )
        if (
            context.reconstruction.confidence.mean()
            < self.config.confidence_fraction * local_confidence
        ):
            return None
        context_from_a = align_submaps(a, context, self.config.metric)
        context_from_b = align_submaps(b, context, self.config.metric)
        edge = context_from_a.inverse() @ context_from_b
        # Check occupancy from disjoint view groups in the JOINT reconstruction.
        cloud_a = context.cloud(set(sample(a)))
        cloud_b = context.cloud(set(sample(b)))
        if voxel_overlap(cloud_a, cloud_b, self.config.voxel_size) < self.config.minimum_overlap:
            return None
        if kind == "loop":
            predicted = self.graph.nodes[a.index].inverse() @ self.graph.nodes[b.index]
            correction = edge @ predicted.inverse()
            angle = (
                np.linalg.norm(Rotation.from_matrix(correction.rotation).as_rotvec()) * 180 / np.pi
            )
            if (
                angle > self.config.max_loop_rotation_deg
                or np.linalg.norm(correction.translation) > self.config.max_loop_translation
            ):
                return None
        return Edge(a.index, b.index, edge, kind=kind)

    def _lidar_local(self, current: Submap, previous: Submap | None) -> None:
        """Camera-frame scans and sequential ICP replace visual odometry locally.

        Reference implementation uses point-to-point ICP, not full KISS-ICP.
        """
        frames = [self.store.get(int(i)) for i in current.ids]
        if any(f.lidar is None for f in frames):
            raise ValueError("LiDAR mode needs calibrated, camera-frame point clouds")
        poses = [np.eye(4)]
        for left, right in pairwise(frames):
            assert right.lidar is not None and left.lidar is not None
            relative, _, _ = icp(right.lidar, left.lidar)
            poses.append(poses[-1] @ relative.pose(np.eye(4)))
        current.poses = np.stack(poses)
        # Fit visual local geometry into ICP coordinates using same camera poses.
        target = Submap(
            current.index,
            current.ids,
            current.poses,
            np.array([], dtype=int),
            current.reconstruction,
        )
        visual = Submap(
            current.index,
            current.ids,
            interpolate_poses(current.selected, current.reconstruction.poses, current.ids),
            np.array([], dtype=int),
            current.reconstruction,
        )
        fit = align_submaps(visual, target, metric=False)
        current.reconstruction.poses = np.stack([fit.pose(p) for p in current.reconstruction.poses])
        current.reconstruction.depth *= fit.scale

    def process(self, frames: list[Frame], scores: list[float]) -> BackendUpdate:
        """Sequential compatibility entry point."""
        return self.integrate(self.prepare(frames, scores, self.count), frames)

    def integrate(self, current: Submap, frames: list[Frame]) -> BackendUpdate:
        """Consume prepared submaps in order; exclusively owns graph/disk state."""
        if current.index != self.count:
            raise ValueError("Prepared submaps must be integrated in order")
        previous = self._load(self.count - 1) if self.count else None
        if self.config.lidar:
            self._lidar_local(current, previous)
        if previous is None:
            p = current.poses[0]
            origin = Sim3(1.0, p[:3, :3], p[:3, 3]).inverse()
        else:
            edge = align_submaps(current, previous, self.config.metric)
            origin = self.graph.nodes[-1] @ edge
        self.graph.nodes.append(origin)
        for offset in (1, 2):
            if self.count >= offset:
                earlier = self._load(self.count - offset)
                if set(earlier.ids) & set(current.ids):
                    measurement = align_submaps(current, earlier, self.config.metric)
                    self.graph.add_edge(
                        Edge(
                            earlier.index,
                            self.count,
                            measurement,
                            kind=f"span-{offset}",
                        )
                    )
        current.save(self._path(self.count))
        if self.config.enable_long_context and (self.count + 1) % self.config.long_every == 0:
            first = max(0, self.count + 1 - self.config.long_window)
            maps = [self._load(i) for i in range(first, self.count + 1)]
            if len(maps) >= 4:
                ids = set()
                for submap in maps:
                    positions = np.linspace(
                        0, len(submap.selected) - 1, self.config.long_views_per_submap
                    ).astype(int)
                    ids.update(submap.selected[positions].tolist())
                context = self._reconstruct([self.store.get(i) for i in sorted(ids)])
                for a in maps[:-3]:
                    edge = self._joint_edge(a, current, "long", context)
                    if edge is not None:
                        self.graph.add_edge(edge)
        candidates = self.retrieval.query_and_add(frames[len(frames) // 2].rgb)
        pending_loops = []
        for index in candidates if self.config.enable_loops else []:
            historical = self._load(index)
            edge = self._joint_edge(historical, current, "loop")
            if edge is None:
                self.rejected.append({"i": index, "j": self.count, "reason": "geometric_gate"})
                continue
            rmse = None
            if self.config.lidar:
                a_id, b_id = int(historical.ids[0]), int(current.ids[0])
                try:
                    source_scan = self.store.get(b_id).lidar
                    target_scan = self.store.get(a_id).lidar
                    if source_scan is None or target_scan is None:
                        raise ValueError("Loop refinement requires both LiDAR scans")
                    refined, rmse, inliers = icp(
                        source_scan,
                        target_scan,
                        initial=edge.measurement,
                    )
                except ValueError:
                    continue
                if inliers < 0.3:
                    continue
                edge.measurement = refined
            pending_loops.append((edge, rmse))
        errors = [error for _, error in pending_loops if error is not None]
        median_error = np.median(errors) if errors else 1.0
        for edge, error in pending_loops:
            if error is not None:
                edge.weight = float((median_error / max(error, 1e-6)) ** 2)
            self.graph.add_edge(edge)
        report = self.graph.optimize(self.config.graph_iterations)
        self.count += 1
        np.save(
            self.directory / "graph_nodes.npy", np.stack([node.log() for node in self.graph.nodes])
        )
        edges = [
            {
                "i": edge.i,
                "j": edge.j,
                "kind": edge.kind,
                "weight": edge.weight,
                "measurement_log": edge.measurement.log().tolist(),
            }
            for edge in self.graph.edges
        ]
        (self.directory / "graph_edges.json").write_text(json.dumps(edges, indent=2))
        corrected = self.trajectory()
        anchor = int(current.selected[-1])
        depth = current.reconstruction.depth[-1] * self.graph.nodes[-1].scale
        report.update(
            submaps=self.count,
            edges=len(self.graph.edges),
            rejected_loops=len(self.rejected),
        )
        (self.directory / "graph_report.json").write_text(json.dumps(report, indent=2))
        return BackendUpdate(corrected, anchor, depth, report)

    def trajectory(self) -> dict[int, Array]:
        contributions: dict[int, list[tuple[Array, float]]] = {}
        for i, node in enumerate(self.graph.nodes):
            submap = self._load(i)
            middle = (len(submap.ids) - 1) / 2
            for j, (frame_id, pose) in enumerate(zip(submap.ids, submap.poses)):
                weight = max(0.05, 1 - abs(j - middle) / (middle + 1))
                contributions.setdefault(int(frame_id), []).append((node.pose(pose), weight))
        return {
            i: average_poses([p for p, _ in values], [w for _, w in values])
            for i, values in contributions.items()
        }
