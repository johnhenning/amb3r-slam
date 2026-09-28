"""Fresh-process matched diagnostics: default/passive OpenMP waiting, both modes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tum"))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.threads < 1:
        parser.error("Choose a new output directory and positive thread count")
    args.output.mkdir(parents=True)
    order = [("default", False), ("default", True), ("passive", True), ("passive", False)]
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "order": order,
                "scope": "One exploratory trial per policy/mode, not a variance estimate",
                "threads": args.threads,
                "command": sys.argv,
            },
            indent=2,
        )
        + "\n"
    )
    for policy, asynchronous in order:
        label = policy + ("_concurrent" if asynchronous else "_sequential")
        env = os.environ.copy()
        for key in ["OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME"]:
            env.pop(key, None)
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
            env[key] = str(args.threads)
        if policy == "passive":
            env["OMP_WAIT_POLICY"] = "PASSIVE"
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_tum.py")),
            "--data",
            str(args.data),
            "--output",
            str(args.output / label),
            "--download",
            "--sequences",
            "freiburg1_xyz",
            "--threads",
            str(args.threads),
            "--resolution",
            "168",
            "--revision",
            "e08cab65ca0ec38e7826075418411ab90cab4da3",
            "--async-backend" if asynchronous else "--no-async-backend",
        ]
        print(f"Starting {label}", flush=True)
        with (args.output / f"{label}.log").open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {label}", flush=True)


if __name__ == "__main__":
    main()
