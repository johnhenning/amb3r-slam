"""Run the documented CPU baseline; downloads require an explicit --download."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from amber_slam.benchmark import (
    TUM_SEQUENCES,
    environment_metadata,
    run_tum_sequence,
    sha256_file,
    write_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/tum"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--sequences", nargs="+", choices=TUM_SEQUENCES, default=list(TUM_SEQUENCES)
    )
    parser.add_argument("--checkpoint", default="depth-anything/DA3-SMALL")
    parser.add_argument(
        "--revision", help="Immutable Hugging Face revision; recorded after resolution"
    )
    parser.add_argument("--resolution", type=int, default=168)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output directory; benchmark results are never overwritten")
    if args.threads < 1 or args.resolution < 56:
        parser.error("threads must be positive and resolution >=56")
    args.data.mkdir(parents=True, exist_ok=True)
    for name in args.sequences:
        directory = args.data / f"rgbd_dataset_{name}"
        if directory.exists():
            continue
        if not args.download:
            parser.error(f"Missing {directory}; use --download or provide extracted TUM data")
        archive = args.data / f"{name}.tgz"
        if not archive.exists():
            partial = archive.with_suffix(".partial")
            with urllib.request.urlopen(TUM_SEQUENCES[name]["url"], timeout=60) as source:
                with partial.open("wb") as target:
                    shutil.copyfileobj(source, target)
            partial.replace(archive)
        if sha256_file(archive) != TUM_SEQUENCES[name]["sha256"]:
            raise ValueError(f"Dataset archive checksum mismatch: {archive}")
        with tarfile.open(archive) as source:
            source.extractall(args.data, filter="data")
    import torch
    from huggingface_hub import snapshot_download
    from threadpoolctl import threadpool_limits

    from amber_slam.adapters import DA3Model
    from amber_slam.backend import SlamConfig

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        checkpoint = Path(
            snapshot_download(
                args.checkpoint,
                revision=args.revision,
                allow_patterns=["config.json", "model.safetensors"],
            )
        )
    metadata = environment_metadata()
    metadata.update(
        checkpoint=args.checkpoint,
        checkpoint_revision=args.revision or checkpoint.name,
        checkpoint_hashes={
            name: sha256_file(checkpoint / name) for name in ["config.json", "model.safetensors"]
        },
        resolution=args.resolution,
        command=sys.argv,
        sources={
            str(path): sha256_file(path) for path in sorted(Path("src/amber_slam").glob("*.py"))
        },
    )
    frontend = DA3Model(str(checkpoint), "cpu", args.resolution)
    backend = DA3Model(str(checkpoint), "cpu", args.resolution)
    config = SlamConfig(window=12, stride=3)
    args.output.mkdir(parents=True)
    (args.output / "environment.json").write_text(json.dumps(metadata, indent=2))
    results = []
    with threadpool_limits(limits=args.threads):
        for name in args.sequences:
            directory = args.data / f"rgbd_dataset_{name}"
            results.append(
                run_tum_sequence(directory, args.output / directory.name, frontend, backend, config)
            )
    write_report(args.output, results, metadata)
    print(f"Report: {args.output / 'README.md'}")


if __name__ == "__main__":
    main()
