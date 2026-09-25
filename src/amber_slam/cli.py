"""Thin command dispatch; algorithms live in importable library modules."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .contracts import JsonObject
from .runtime import SlamSystem
from .types import GeometryModel


def _model(kind: str, checkpoint: str, device: str, resolution: int) -> GeometryModel:
    from .adapters import DA3Model, ONNXModel, TorchModel

    if kind == "da3":
        return DA3Model(checkpoint, device, resolution)
    if kind == "torch":
        return TorchModel(checkpoint, device)
    return ONNXModel(checkpoint)


def _save_run(
    system: SlamSystem, output: Path, elapsed: float, extra: JsonObject | None = None
) -> JsonObject:
    from .evaluation import write_tum

    write_tum(output / "trajectory.tum", system.timestamps, system.trajectory())
    write_tum(
        output / "trajectory_online.tum",
        system.timestamps,
        np.stack(system.online_poses),
    )
    stats: JsonObject = dict(system.statistics())
    stats.update(total_elapsed_s=elapsed, end_to_end_fps=system.count / elapsed)
    if extra:
        stats.update(extra)
    (output / "statistics.json").write_text(json.dumps(stats, indent=2))
    return stats


def run(args: argparse.Namespace) -> JsonObject:
    from .backend import SlamConfig
    from .data import load_frame, read_manifest
    from .runtime import LatestFrameCapture, SlamSystem
    from .types import Frame

    frontend = _model(args.model, args.frontend, args.device, args.resolution)
    backend = _model(args.model, args.backend, args.device, args.resolution)
    output = Path(args.output)
    config = SlamConfig.load(args.config)
    start = time.perf_counter()
    extra = {}
    with SlamSystem(
        frontend,
        backend,
        output,
        config,
        asynchronous=args.async_backend,
        max_pending_windows=args.max_pending_windows,
    ) as system:
        (output / "invocation.json").write_text(json.dumps(vars(args), indent=2))
        if args.manifest:
            data, root = read_manifest(args.manifest)
            sequences = [s for s in data["sequences"] if s["id"] == args.sequence]
            if len(sequences) != 1:
                raise ValueError("--sequence must identify one manifest sequence")
            for i, record in enumerate(sequences[0]["frames"]):
                if args.max_frames and i >= args.max_frames:
                    break
                result = system.push(load_frame(record, root, i))
                if i % args.log_every == 0:
                    print(
                        json.dumps({"frame": i, "latency_ms": result.latency_ms}),
                        flush=True,
                    )
        else:
            capture = LatestFrameCapture(args.camera)
            age = []
            for i, (timestamp, rgb) in enumerate(capture):
                age.append((time.monotonic() - timestamp) * 1000)
                result = system.push(Frame(i, timestamp, rgb))
                if i % args.log_every == 0:
                    print(
                        json.dumps(
                            {
                                "frame": i,
                                "timestamp": timestamp,
                                "pose": result.pose.tolist(),
                                "latency_ms": result.latency_ms,
                            }
                        ),
                        flush=True,
                    )
                if args.max_frames and i + 1 >= args.max_frames:
                    break
            extra = {
                "capture_dropped_frames": capture.dropped,
                "capture_age_ms_p95": float(np.percentile(age, 95)) if age else None,
            }
    if system.count == 0:
        raise ValueError("Input produced no frames")
    return _save_run(system, output, time.perf_counter() - start, extra)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="AMB3R-SLAM independent research reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser("synthetic", help="Generate analytic smoke-test data")
    synth.add_argument("output")
    synth.add_argument("--frames", type=int, default=24)
    synth.add_argument("--size", type=int, default=64)
    train = commands.add_parser("train", help="Optional experimental training")
    train.add_argument("config")
    evaluate = commands.add_parser("evaluate", help="ATE/RPE from TUM trajectories")
    evaluate.add_argument("reference")
    evaluate.add_argument("estimated")
    evaluate.add_argument("--alignment", choices=["sim3", "se3", "none"], default="sim3")
    evaluate.add_argument("--tolerance", type=float, default=0.02)
    evaluate.add_argument("--delta", type=int, default=1)
    evaluate.add_argument("--output")
    export = commands.add_parser("export", help="Export owned checkpoint and verify ONNX parity")
    export.add_argument("checkpoint")
    export.add_argument("output")
    export.add_argument("--views", type=int, default=4)
    bench = commands.add_parser("benchmark", help="ONNX model-only CPU latency")
    bench.add_argument("model")
    bench.add_argument("--iterations", type=int, default=100)
    inference = commands.add_parser("run", help="Replay manifest or track a live camera")
    source = inference.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--camera", type=int)
    inference.add_argument("--sequence")
    inference.add_argument("--output", required=True)
    inference.add_argument("--model", choices=["da3", "torch", "onnx"], default="da3")
    inference.add_argument("--frontend", default="depth-anything/DA3-SMALL")
    inference.add_argument("--backend", default="depth-anything/DA3-GIANT")
    inference.add_argument("--device", default="cuda")
    inference.add_argument("--resolution", type=int, default=504)
    inference.add_argument("--config")
    inference.add_argument("--max-frames", type=int, default=0)
    inference.add_argument("--log-every", type=int, default=30)
    inference.add_argument(
        "--async-backend",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Concurrent tracking/mapping/graph pipeline (default); --no-async-backend for sequential",
    )
    inference.add_argument("--max-pending-windows", type=int, default=2)
    pseudo = commands.add_parser("pseudo-label", help="Optional aligned teacher depth cache")
    pseudo.add_argument("manifest")
    pseudo.add_argument("output")
    pseudo.add_argument("--checkpoint", required=True)
    pseudo.add_argument("--model", choices=["da3", "torch"], default="torch")
    pseudo.add_argument("--device", default="cpu")
    pseudo.add_argument("--resolution", type=int, default=504)
    args = parser.parse_args(argv)
    if args.command == "synthetic":
        from .data import synthetic_dataset

        result = {"manifest": str(synthetic_dataset(args.output, args.frames, args.size))}
    elif args.command == "train":
        from .training import train

        result = train(args.config)
    elif args.command == "evaluate":
        from .evaluation import evaluate

        result = evaluate(
            args.reference, args.estimated, args.alignment, args.tolerance, args.delta
        )
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2))
    elif args.command == "export":
        from .export import export_onnx

        result = export_onnx(args.checkpoint, args.output, args.views)
    elif args.command == "benchmark":
        from .export import benchmark_onnx

        result = benchmark_onnx(args.model, args.iterations)
    elif args.command == "pseudo-label":
        from .pseudo import generate_labels

        model = _model(args.model, args.checkpoint, args.device, args.resolution)
        result = {"manifest": str(generate_labels(args.manifest, model, args.output))}
    else:
        result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
