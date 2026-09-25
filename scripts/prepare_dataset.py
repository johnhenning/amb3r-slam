"""Explicit conversion of licensed local datasets; no automatic downloads."""

import argparse
import json

import numpy as np

from amber_slam.datasets import prepare_kitti, prepare_tum

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("format", choices=["tum", "kitti"])
parser.add_argument("directory")
parser.add_argument("output")
parser.add_argument("--id", required=True)
parser.add_argument("--intrinsics", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"))
parser.add_argument("--poses")
parser.add_argument("--split", choices=["train", "val", "test"], default="test")
args = parser.parse_args()
if args.format == "tum":
    if args.intrinsics is None:
        parser.error("TUM requires --intrinsics fx fy cx cy")
    fx, fy, cx, cy = args.intrinsics
    result = prepare_tum(
        args.directory,
        args.output,
        args.id,
        np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
        split=args.split,
    )
else:
    result = prepare_kitti(args.directory, args.output, args.id, args.poses, args.split)
print(json.dumps(result, indent=2))
