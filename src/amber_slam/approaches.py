"""Adapters keep implementation-specific lifecycle out of experiment runners."""

from pathlib import Path

import numpy as np

from .backend import SlamConfig
from .experiments import Trajectory
from .runtime import SlamSystem
from .types import Frame, GeometryModel


class AmberApproach:
    """Reuse preloaded geometry models; create fresh SLAM state for every trial."""

    def __init__(
        self,
        output: Path,
        frontend: GeometryModel,
        backend: GeometryModel,
        config: SlamConfig,
        asynchronous: bool = True,
        max_pending_windows: int = 2,
    ) -> None:
        self.system = SlamSystem(
            frontend,
            backend,
            output,
            config,
            asynchronous=asynchronous,
            max_pending_windows=max_pending_windows,
        )
        self.closed = False

    def push(self, frame: Frame) -> None:
        self.system.push(frame)

    def finish(self) -> Trajectory:
        self.system.close()
        self.closed = True
        return Trajectory(np.asarray(self.system.timestamps), self.system.trajectory())

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            # Only finish drains/flushes. Cleanup after failed push must not map
            # additional windows or replace the original error with a worker error.
            self.system.__exit__(RuntimeError, RuntimeError("adapter cleanup"), None)
