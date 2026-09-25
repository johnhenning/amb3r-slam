"""Submap graph with block Huber loss (not elementwise Huber)."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .geometry import Sim3


@dataclass
class Edge:
    i: int
    j: int
    measurement: Sim3  # coordinates of j -> coordinates of i
    weight: float = 1.0
    kind: str = "local"


class PoseGraph:
    def __init__(self, metric=False, huber=1.0, baseline_floor=0.1, whiten=True):
        self.nodes = []
        self.edges = []
        self.metric, self.huber = metric, huber
        self.baseline_floor, self.whiten = baseline_floor, whiten

    def add_edge(self, edge):
        if not 0 <= edge.i < edge.j < len(self.nodes):
            raise ValueError("Invalid edge indices")
        if edge.weight <= 0 or not np.isfinite(edge.weight):
            raise ValueError("Invalid edge weight")
        if self.metric and abs(edge.measurement.scale - 1) > 1e-6:
            raise ValueError("Metric graph requires unit-scale edges")
        self.edges.append(edge)

    def residuals(self, nodes):
        out = []
        for e in self.edges:
            r = (e.measurement.inverse() @ nodes[e.i].inverse() @ nodes[e.j]).log()
            if self.whiten:
                r[:3] /= max(np.linalg.norm(e.measurement.translation), self.baseline_floor)
            r *= np.sqrt(e.weight)
            norm = np.linalg.norm(r)
            if norm > self.huber:
                # ||returned residual||^2 = Huber(||whitened residual||^2)
                r *= np.sqrt((2 * self.huber * norm - self.huber**2) / norm**2)
            out.extend(r)
        return np.asarray(out)

    def optimize(self, max_nfev=30):
        if len(self.nodes) < 2 or not self.edges:
            return {"initial_cost": 0.0, "final_cost": 0.0, "success": True}
        original = list(self.nodes)
        dof = 6 if self.metric else 7

        def decode(x):
            dx = x.reshape(-1, dof)
            return [original[0]] + [
                original[i + 1] @ Sim3.exp(np.r_[v, 0.0] if self.metric else v)
                for i, v in enumerate(dx)
            ]

        def residual(x):
            return self.residuals(decode(x))

        n = dof * (len(original) - 1)
        sparsity = lil_matrix((7 * len(self.edges), n), dtype=int)
        for k, edge in enumerate(self.edges):
            for j in (edge.i, edge.j):
                if j:
                    sparsity[7 * k : 7 * k + 7, dof * (j - 1) : dof * j] = 1
        x = np.zeros(n)
        before = float(residual(x) @ residual(x))
        result = least_squares(
            residual,
            x,
            jac_sparsity=sparsity.tocsr(),
            max_nfev=max_nfev,
            bounds=(-2.0, 2.0),
        )
        after = float(result.fun @ result.fun)
        if np.isfinite(after) and after <= before:
            self.nodes = decode(result.x)
        return {
            "initial_cost": before,
            "final_cost": after,
            "success": bool(result.success),
            "evaluations": result.nfev,
        }
