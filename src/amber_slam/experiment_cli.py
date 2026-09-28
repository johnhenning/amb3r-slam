"""Run a trusted Python SLAM adapter in a fresh process, without implicit downloads."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

from .contracts import JsonObject
from .experiments import SlamApproach, run_experiment


def load_factory(spec: str) -> Callable[[Path, JsonObject], SlamApproach]:
    """Import explicitly selected, trusted code; this is not a plugin sandbox."""
    module, separator, name = spec.partition(":")
    if not separator or not module or not name or ":" in name:
        raise ValueError("Factory must be module:callable")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError("Selected factory is not callable")
    return cast(Callable[[Path, JsonObject], SlamApproach], factory)


def read_object(path: Path) -> JsonObject:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Extracted TUM sequence")
    parser.add_argument("--output", type=Path, required=True, help="New run directory")
    parser.add_argument("--factory", required=True, help="Trusted module:callable(output, config)")
    parser.add_argument("--approach", required=True)
    parser.add_argument("--revision", required=True, help="Immutable implementation revision")
    parser.add_argument("--config", type=Path, required=True, help="Method configuration JSON")
    parser.add_argument("--environment", type=Path, help="Checkpoint hashes, hardware, etc. JSON")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output directory")
    if args.stride < 1 or (args.max_frames is not None and args.max_frames < 3):
        parser.error("stride must be positive; max-frames must be >=3")
    config = read_object(args.config)
    environment = read_object(args.environment) if args.environment else {}
    factory = load_factory(args.factory)
    result = run_experiment(
        args.data,
        args.output,
        lambda output: factory(output, config),
        approach=args.approach,
        revision=args.revision,
        config=config,
        stride=args.stride,
        max_frames=args.max_frames,
        environment={**environment, "factory": args.factory},
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
