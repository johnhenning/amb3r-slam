"""Export optimized submap points. Re-run mapping first; graph nodes are required."""

import argparse
from pathlib import Path

import numpy as np

from amber_slam.backend import Submap
from amber_slam.geometry import Sim3

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run")
parser.add_argument("output")
args = parser.parse_args()
root = Path(args.run)
nodes = np.load(root / "graph_nodes.npy", allow_pickle=False)
clouds = []
for i, node in enumerate(nodes):
    submap = Submap.load(i, root / "submaps" / f"{i:06d}.npz")
    clouds.append(Sim3.exp(node).points(submap.cloud()))
cloud = np.concatenate(clouds) if clouds else np.empty((0, 3))
with open(args.output, "w") as file:
    file.write(
        f"ply\nformat ascii 1.0\nelement vertex {len(cloud)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    )
    np.savetxt(file, cloud, fmt="%.6f")
print(f"Exported {len(cloud)} points. Submap overlap may duplicate surfaces.")
