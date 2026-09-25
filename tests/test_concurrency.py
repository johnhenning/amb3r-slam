"""Event-based scheduling checks: no speed assumptions or sleep-based races."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import numpy as np
import pytest
from test_pipeline import OracleModel

from amber_slam.backend import BackendUpdate, SlamConfig, Submap
from amber_slam.runtime import SlamSystem
from amber_slam.types import Frame


def frame(index: int) -> Frame:
    return Frame(index, index / 30, np.zeros((32, 32, 3), np.uint8))


def config() -> SlamConfig:
    return SlamConfig(
        window=6, stride=1, enable_loops=False, enable_long_context=False, graph_iterations=2
    )


def test_tracking_and_mapping_overlap_blocked_graph(tmp_path: Path) -> None:
    graph_started, release_graph, second_mapping = Event(), Event(), Event()
    system = SlamSystem(OracleModel(), OracleModel(), tmp_path, config())
    integrate = system.backend.integrate
    prepare = system.backend.prepare

    def blocked_graph(current: Submap, frames: list[Frame]) -> BackendUpdate:
        if current.index == 0:
            graph_started.set()
            assert release_graph.wait(10), "test failed to release graph"
        return integrate(current, frames)

    def mapping(frames: list[Frame], scores: list[float], index: int) -> Submap:
        result = prepare(frames, scores, index)
        if index == 1:
            second_mapping.set()
        return result

    system.backend.integrate = blocked_graph
    system.backend.prepare = mapping
    try:
        for i in range(6):
            system.push(frame(i))
        assert graph_started.wait(10)
        # Tracking and local reconstruction progress while graph processing is blocked.
        system.push(frame(6))
        system.push(frame(7))
        assert second_mapping.wait(10)
        assert system.pending is not None and not system.pending.done()
        assert system.statistics()["submaps"] == 0  # completed-message snapshot
        assert system.statistics()["pending_windows_peak"] == 2
    finally:
        release_graph.set()
        system.close()
    assert system.backend.count == 2
    np.testing.assert_allclose(system.trajectory()[:, 0, 3], np.arange(8) * 0.03, atol=1e-6)


@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_bounded_pipeline_preserves_sequential_result(tmp_path: Path, capacity: int) -> None:
    trajectories = []
    edge_kinds = []
    for asynchronous in (False, True):
        with SlamSystem(
            OracleModel(),
            OracleModel(),
            tmp_path / str(asynchronous),
            config(),
            asynchronous=asynchronous,
            max_pending_windows=capacity,
        ) as system:
            for i in range(19):
                system.push(frame(i))
        trajectories.append(system.trajectory())
        edge_kinds.append([(e.i, e.j, e.kind) for e in system.backend.graph.edges])
        assert system.backend.count == 8  # seven full windows plus the tail
        assert system.statistics()["pending_windows_peak"] <= capacity
        assert system.pending is None
        system.close()  # idempotent shutdown
    np.testing.assert_allclose(trajectories[0], trajectories[1], atol=1e-6)
    assert edge_kinds[0] == edge_kinds[1]


def test_backpressure_waits_for_capacity(tmp_path: Path) -> None:
    graph_started, release_graph, two_submitted = Event(), Event(), Event()
    system = SlamSystem(OracleModel(), OracleModel(), tmp_path, config())
    integrate = system.backend.integrate

    def blocked_graph(current: Submap, frames: list[Frame]) -> BackendUpdate:
        if current.index == 0:
            graph_started.set()
            assert release_graph.wait(10)
        return integrate(current, frames)

    system.backend.integrate = blocked_graph

    def produce() -> None:
        with system:
            for i in range(10):
                system.push(frame(i))
                if i == 7:
                    two_submitted.set()

    with ThreadPoolExecutor(max_workers=1) as owner:
        future = owner.submit(produce)
        try:
            assert graph_started.wait(10) and two_submitted.wait(10)
            # The producer must remain blocked until graph capacity is released.
            with pytest.raises(TimeoutError):
                future.result(timeout=0.1)
        finally:
            release_graph.set()
        future.result(timeout=20)
    assert system.backend.count == 3
    assert system.statistics()["pending_windows_peak"] == 2
    assert system.statistics()["backpressure_waits"] == 1
    assert system.statistics()["backpressure_ms"] > 0


@pytest.mark.parametrize("stage", ["mapping", "graph"])
def test_stage_failure_shuts_down_pipeline(tmp_path: Path, stage: str) -> None:
    system = SlamSystem(OracleModel(), OracleModel(), tmp_path, config())

    def broken_mapping(frames: list[Frame], scores: list[float], index: int) -> Submap:
        raise RuntimeError("mapping failure")

    def broken_graph(current: Submap, frames: list[Frame]) -> BackendUpdate:
        raise RuntimeError("graph failure")

    if stage == "mapping":
        system.backend.prepare = broken_mapping
    else:
        system.backend.integrate = broken_graph
    with pytest.raises(RuntimeError, match=f"{stage} failure"):
        with system:
            for i in range(12):
                system.push(frame(i))
    assert system.closed and system.pending is None
    assert system.mapper is not None and system.executor is not None
    with pytest.raises(RuntimeError, match="shutdown"):
        system.mapper.submit(int, 1)
    with pytest.raises(RuntimeError, match="shutdown"):
        system.executor.submit(int, 1)


def test_owner_failure_preserves_original_exception(tmp_path: Path) -> None:
    system = SlamSystem(OracleModel(), OracleModel(), tmp_path, config())
    with pytest.raises(ValueError, match="owner failure"):
        with system:
            for i in range(8):
                system.push(frame(i))
            raise ValueError("owner failure")
    assert system.closed and system.pending is None


def test_reject_shared_model_and_unbounded_queue(tmp_path: Path) -> None:
    model = OracleModel()
    with pytest.raises(ValueError, match="separate"):
        SlamSystem(model, model, tmp_path, config())
    with pytest.raises(ValueError, match="positive"):
        SlamSystem(model, OracleModel(), tmp_path, config(), max_pending_windows=0)
