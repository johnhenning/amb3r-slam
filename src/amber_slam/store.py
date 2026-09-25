"""Disk-backed frame/submap storage; retained RGB LRU is bounded."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from .contracts import Array, PathLike
from .types import Frame


class FrameStore:
    def __init__(self, path: PathLike, capacity: int = 96) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.cache: OrderedDict[int, Frame] = OrderedDict()

    def put(self, frame: Frame) -> None:
        data: dict[str, Array | float] = {"rgb": frame.rgb, "timestamp": frame.timestamp}
        for key in ["depth", "intrinsics", "lidar"]:
            value = getattr(frame, key)
            if value is not None:
                data[key] = value
        np.savez_compressed(self.path / f"{frame.index:08d}.npz", **data)
        self._cache(frame)

    def _cache(self, frame: Frame) -> None:
        self.cache[frame.index] = frame
        self.cache.move_to_end(frame.index)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def get(self, index: int) -> Frame:
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        with np.load(self.path / f"{index:08d}.npz", allow_pickle=False) as z:
            frame = Frame(
                index,
                float(z["timestamp"]),
                z["rgb"],
                *[z[k] if k in z else None for k in ["depth", "intrinsics", "lidar"]],
            )
        self._cache(frame)
        return frame
