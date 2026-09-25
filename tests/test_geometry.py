import numpy as np
import pytest

from amber_slam.geometry import Sim3, align_points, interpolate_poses, metric_scale
from amber_slam.graph import Edge, PoseGraph


def test_sim3_round_trip_composition_and_action():
    rng = np.random.default_rng(3)
    for _ in range(12):
        tangent = rng.normal(size=7) * 0.2
        transform = Sim3.exp(tangent)
        np.testing.assert_allclose(transform.log(), tangent, atol=1e-10)
        np.testing.assert_allclose((transform @ transform.inverse()).log(), 0, atol=1e-10)
        points = rng.normal(size=(40, 3))
        fitted = align_points(points, transform.points(points))
        np.testing.assert_allclose(fitted.points(points), transform.points(points), atol=1e-9)


def test_depth_scale_rejects_invalid_and_recovers_known_scale():
    depth = np.linspace(1, 10, 100).reshape(10, 10)
    measured = depth * 3
    measured[:2] = np.nan
    assert metric_scale(depth, measured) == pytest.approx(3)
    with pytest.raises(ValueError):
        metric_scale(depth, np.zeros_like(depth))


def test_graph_reduces_error_and_keeps_gauge_fixed():
    graph = PoseGraph()
    truth = [Sim3.exp(np.array([i * 0.4, i * i * 0.05, 0, 0, 0, i * 0.03, 0])) for i in range(5)]
    graph.nodes = [truth[0]] + [
        t @ Sim3.exp(np.array([0.05, 0.02, 0, 0, 0, 0.03, 0.05])) for t in truth[1:]
    ]
    for i in range(4):
        graph.add_edge(Edge(i, i + 1, truth[i].inverse() @ truth[i + 1]))
    graph.add_edge(Edge(0, 4, truth[0].inverse() @ truth[4], kind="loop"))
    report = graph.optimize(50)
    assert report["final_cost"] < report["initial_cost"] * 1e-4
    np.testing.assert_allclose(graph.nodes[0].log(), truth[0].log())


def test_interpolation_includes_endpoints():
    poses = np.repeat(np.eye(4)[None], 2, 0)
    poses[1, 0, 3] = 2
    result = interpolate_poses([0, 2], poses, [0, 1, 2])
    np.testing.assert_allclose(result[:, 0, 3], [0, 1, 2])
