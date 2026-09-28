"""Matched full-route TUM comparisons in isolated, alternating-order processes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TypedDict


class Trial(TypedDict):
    sequence: str
    mode: str
    repeat: int
    label: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/tum"))
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("Choose a new output directory and positive repeats")
    args.output.mkdir(parents=True)
    runs: list[Trial] = []
    for repeat in range(args.repeats):
        for index, sequence in enumerate(["freiburg3_long_office_household", "freiburg1_room"]):
            modes = ["sequential", "concurrent"]
            if (index + repeat) % 2:
                modes.reverse()
            for mode in modes:
                runs.append(
                    {
                        "sequence": sequence,
                        "mode": mode,
                        "repeat": repeat + 1,
                        "label": f"{sequence}_{mode}_{repeat + 1}",
                    }
                )
    (args.output / "protocol.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "threads": 2,
                "sampling_stride": 10,
                "max_frames": None,
                "scope": "Complete routes at reduced temporal sampling; default waiting policy; one fresh process per trial",
                "command": sys.argv,
            },
            indent=2,
        )
        + "\n"
    )
    for run in runs:
        env = os.environ.copy()
        for key in ["OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "KMP_BLOCKTIME"]:
            env.pop(key, None)
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
            env[key] = "2"
        command = [
            sys.executable,
            str(Path(__file__).with_name("benchmark_tum.py")),
            "--data",
            str(args.data),
            "--output",
            str(args.output / run["label"]),
            "--download",
            "--sequences",
            run["sequence"],
            "--threads",
            "2",
            "--resolution",
            "168",
            "--revision",
            "e08cab65ca0ec38e7826075418411ab90cab4da3",
            "--async-backend" if run["mode"] == "concurrent" else "--no-async-backend",
        ]
        print(f"Starting {run['label']}", flush=True)
        with (args.output / (run["label"] + ".log")).open("w") as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"Completed {run['label']}", flush=True)


if __name__ == "__main__":
    main()
