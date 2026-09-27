"""Stale-anchor recovery must drain once, preserve validation and commit once."""

from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from test_pipeline import OracleModel

from amber_slam.frontend import TrackingScaleError
from amber_slam.runtime import SlamSystem
from amber_slam.types import Frame


@pytest.mark.parametrize("recovers", [False, True])
def test_retry_after_pending_correction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recovers: bool
) -> None:
    system = SlamSystem(OracleModel(), OracleModel(), tmp_path, asynchronous=False)
    # An unresolved message remains pending on the nonblocking collection call.
    system._pending.append((Future(), Future()))
    waits: list[bool] = []

    def collect(wait: bool = False) -> None:
        waits.append(wait)
        if wait:
            system._pending.clear()

    monkeypatch.setattr(system, "_collect", collect)
    track = MagicMock(
        side_effect=[
            TrackingScaleError("stale anchor"),
            (np.eye(4), 1.0) if recovers else TrackingScaleError("still inconsistent"),
        ]
    )
    monkeypatch.setattr(system.frontend, "track", track)
    frame = Frame(0, 0.0, np.zeros((32, 32, 3), np.uint8))
    try:
        if recovers:
            system.push(frame)
            assert system.count == 1 and len(system.timestamps) == 1
        else:
            with pytest.raises(TrackingScaleError, match="still inconsistent"):
                system.push(frame)
            assert system.count == 0 and not system.timestamps
        assert waits == [False, True]
        assert track.call_count == 2
        assert system.tracking_scale_retries == 1
    finally:
        system._shutdown()


def test_no_retry_without_pending_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with SlamSystem(OracleModel(), OracleModel(), tmp_path, asynchronous=False) as system:
        track = MagicMock(side_effect=TrackingScaleError("inconsistent"))
        monkeypatch.setattr(system.frontend, "track", track)
        with pytest.raises(TrackingScaleError):
            system.push(Frame(0, 0.0, np.zeros((32, 32, 3), np.uint8)))
        assert track.call_count == 1
