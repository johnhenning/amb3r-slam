"""Publish completed benchmark reports, or prepare offline runs for later sync."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from amber_slam.experiment_tracking import export_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "amb3r-slam"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY") or None)
    parser.add_argument("--group")
    parser.add_argument("--mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--output", type=Path, default=Path("wandb"))
    args = parser.parse_args()
    paths = sorted(
        {
            p.parent
            for root in args.reports
            for p in root.rglob("results.json")
            if (p.parent / "profile/summary.json").is_file()
        }
    )
    if not paths:
        parser.error("No completed profiled runs found")
    for directory in paths:
        print(
            json.dumps(
                export_run(directory, args.project, args.entity, args.group, args.mode, args.output)
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
