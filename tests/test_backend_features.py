import numpy as np
import pytest
from test_pipeline import OracleModel

from amber_slam.backend import SlamConfig
from amber_slam.geometry import Sim3
from amber_slam.modalities import icp, project_lidar_depth, voxel_overlap
from amber_slam.runtime import SlamSystem
from amber_slam.types import Frame


def test_long_context_adds_verified_edges(tmp_path):
    config = SlamConfig(
        window=6,
        stride=1,
        long_every=1,
        long_window=4,
        enable_loops=False,
        graph_iterations=2,
        minimum_overlap=0.01,
    )
    with SlamSystem(OracleModel(), OracleModel(), tmp_path, config, False) as system:
        for i in range(14):
            system.push(Frame(i, i / 30, np.zeros((32, 32, 3), np.uint8)))
    assert any(edge.kind == "long" for edge in system.backend.graph.edges)


def test_rgbd_metric_graph_has_unit_scale(tmp_path):
    config = SlamConfig(
        window=6,
        stride=1,
        metric=True,
        enable_loops=False,
        enable_long_context=False,
        graph_iterations=2,
    )
    with SlamSystem(OracleModel(), OracleModel(), tmp_path, config, False) as system:
        for i in range(10):
            system.push(Frame(i, i / 30, np.zeros((32, 32, 3), np.uint8), np.full((32, 32), 6.0)))
    assert all(node.scale == pytest.approx(1) for node in system.backend.graph.nodes)
    np.testing.assert_allclose(system.trajectory()[:, 0, 3], np.arange(10) * 0.06, atol=1e-6)


def test_icp_and_depth_projection():
    rng = np.random.default_rng(5)
    points = rng.normal(size=(300, 3))
    transform = Sim3.exp(np.array([0.02, -0.03, 0.01, 0, 0, 0.01, 0]))
    fit, error, inliers = icp(points, transform.points(points), max_distance=0.3)
    np.testing.assert_allclose(fit.points(points), transform.points(points), atol=1e-5)
    assert error < 1e-5 and inliers > 0.95
    assert voxel_overlap(points, points, 0.1) == 1
    k = np.array([[10.0, 0, 5], [0, 10.0, 5], [0, 0, 1]])
    depth = project_lidar_depth(np.array([[0, 0, 2.0], [0, 0, 3.0]]), k, (10, 10))
    assert depth[5, 5] == 2


def test_joint_loop_candidate_is_geometrically_verified(tmp_path):
    config = SlamConfig(
        window=6,
        stride=1,
        loop_exclusion=3,
        minimum_overlap=0.01,
        enable_long_context=False,
        graph_iterations=2,
    )
    with SlamSystem(OracleModel(), OracleModel(), tmp_path, config, False) as system:
        # Explicit candidate source exercises verification independently of ORB.
        class CandidateSource:
            count = 0

            def query_and_add(self, image):
                candidates = [0] if self.count >= 3 else []
                self.count += 1
                return candidates

        system.backend.retrieval = CandidateSource()
        for i in range(14):
            system.push(Frame(i, i / 30, np.zeros((32, 32, 3), np.uint8)))
    assert any(edge.kind == "loop" for edge in system.backend.graph.edges)
