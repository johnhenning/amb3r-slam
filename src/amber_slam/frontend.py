"""Compact-context tracking with explicit backend correction feedback."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import cv2
import numpy as np

from .contracts import Array
from .geometry import Sim3, metric_scale
from .types import Frame, GeometryModel

if TYPE_CHECKING:
    from .backend import BackendUpdate


class Frontend:
    """Owns anchor, two recent frames, and poses in the current world gauge."""

    def __init__(self, model: GeometryModel) -> None:
        self.model = model
        self.anchor: Frame | None = None
        self.anchor_depth: Array | None = None
        self.recent: deque[Frame] = deque(maxlen=2)
        self.poses: dict[int, Array] = {}
        self.scores: dict[int, float] = {}

    def track(self, frame: Frame) -> tuple[Array, float]:
        context = {f.index: f for f in ([self.anchor] if self.anchor else []) + list(self.recent)}
        context[frame.index] = frame
        frames = list(context.values())
        prediction = self.model.reconstruct(frames, reference=0).validate(len(frames))
        local_pose = prediction.poses[-1]
        if self.anchor is None:
            self.anchor = frame
            self.anchor_depth = prediction.depth[0].copy()
            pose = np.eye(4)
        else:
            assert self.anchor_depth is not None
            h, w = prediction.depth[0].shape
            reference_depth = cv2.resize(
                self.anchor_depth.astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_NEAREST,
            )
            scale = metric_scale(prediction.depth[0], reference_depth)
            anchor_pose = self.poses[self.anchor.index]
            local_from_anchor = np.linalg.inv(prediction.poses[0]) @ local_pose
            local_from_anchor[:3, 3] *= scale
            pose = anchor_pose @ local_from_anchor
        confidence = float(np.median(prediction.confidence[-1]))
        self.poses[frame.index] = pose
        self.scores[frame.index] = confidence
        self.recent.append(frame)
        return pose.copy(), confidence

    def apply_update(self, update: BackendUpdate, anchor: Frame) -> None:
        """Correct frames newer than a delayed map, then install a new anchor.

        The similarity gauge correction carries the newest mapped pose into the
        current frontend tail. Backend depth establishes future scale. Older
        already-emitted poses remain available as revised trajectory output.
        """
        index = update.anchor_id
        old = self.poses.get(index)
        if old is not None:
            corrected = update.poses[index]
            shared = sorted(self.poses.keys() & update.poses.keys())[-12:]
            ratios = []
            for other in shared:
                old_distance = np.linalg.norm(self.poses[other][:3, 3] - old[:3, 3])
                new_distance = np.linalg.norm(update.poses[other][:3, 3] - corrected[:3, 3])
                if old_distance > 1e-4 and new_distance > 1e-4:
                    ratios.append(new_distance / old_distance)
            scale = float(np.median(ratios)) if ratios else 1.0
            rotation = corrected[:3, :3] @ old[:3, :3].T
            translation = corrected[:3, 3] - scale * rotation @ old[:3, 3]
            correction = Sim3(scale, rotation, translation)
            for key in self.poses.keys() - update.poses.keys():
                if key > index:
                    self.poses[key] = correction.pose(self.poses[key])
        self.poses.update(update.poses)
        self.anchor = anchor
        self.anchor_depth = update.anchor_depth.copy()
