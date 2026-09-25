"""Import completed benchmark measurements into W&B outside timed replay."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from .contracts import JsonObject


def numeric_metrics(data: JsonObject, prefix: str) -> dict[str, float]:
    """Flatten measured scalar fields without treating bools as measurements."""
    values = {}
    for key, value in data.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            values.update(numeric_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values[name] = float(value)
    return values


def resource_history(path: Path) -> Iterator[dict[str, float]]:
    """Use recorded process samples, never the uploader's system utilization."""
    previous: tuple[float, float] | None = None
    with path.open() as stream:
        for row in csv.DictReader(stream):
            elapsed, cpu = float(row["elapsed_s"]), float(row["cpu_time_s"])
            values = {
                "replay/seconds": elapsed,
                "resources/rss_mib": float(row["rss_bytes"]) / 2**20,
                "resources/uss_mib": float(row["uss_bytes"]) / 2**20,
                "resources/threads": float(row["threads"]),
            }
            if previous is not None and elapsed > previous[0]:
                values["resources/cpu_percent"] = (
                    100 * (cpu - previous[1]) / (elapsed - previous[0])
                )
            previous = elapsed, cpu
            yield values


def artifact_files(directory: Path) -> list[Path]:
    """Retain diagnostics and provenance; exclude large reconstruction caches."""
    return [
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and not p.is_symlink()
        and not {"frames", "submaps"}.intersection(p.relative_to(directory).parts)
        and p.suffix not in {".npy", ".npz"}
        and p.name != "wandb-export.json"
    ]


def export_run(
    directory: Path,
    project: str,
    entity: str | None = None,
    group: str | None = None,
    mode: Literal["offline", "online"] = "offline",
    output: Path = Path("wandb"),
) -> JsonObject:
    """One measured replay per W&B run; explicit online mode opts into upload."""
    import wandb

    result_path, profile_path = directory / "results.json", directory / "profile/summary.json"
    env_path = directory.parent / "environment.json"
    result = json.loads(result_path.read_text())
    profile = json.loads(profile_path.read_text())
    environment = json.loads(env_path.read_text())
    digest = hashlib.sha256(
        result_path.read_bytes() + profile_path.read_bytes() + env_path.read_bytes()
    ).hexdigest()
    run_id = "slam-" + digest[:20]
    receipt_path = directory / "wandb-export.json"
    identity = {"run_id": run_id, "mode": mode, "entity": entity, "project": project}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if all(receipt.get(k) == v for k, v in identity.items()) and (
            mode == "online" or Path(receipt.get("directory", "")).is_dir()
        ):
            return receipt
    execution = result["statistics"]["execution_mode"]
    policy = profile["wait_environment"].get("OMP_WAIT_POLICY") or "default"
    cohort = group or directory.parent.parent.name
    config = {
        "dataset": result["sequence"],
        "execution_mode": execution,
        "wait_policy": policy,
        "sampled_frames": result["sampled_frames"],
        "sampling_stride": result["sampling_stride"],
        "max_frames": result["max_frames"],
        "resolution": environment["resolution"],
        "torch_threads": environment["torch_threads"],
        "checkpoint": environment["checkpoint"],
        "checkpoint_hashes": environment["checkpoint_hashes"],
        "sources": environment["sources"],
        "slam": result["config"],
        "input_manifest_sha256": result["input_manifest_sha256"],
        "groundtruth_sha256": result["groundtruth_sha256"],
        "environment": environment,
        "recorded_profile": True,
    }
    output.mkdir(parents=True, exist_ok=True)
    with wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        group=cohort,
        name=directory.parent.name,
        job_type="benchmark",
        mode=mode,
        resume="never" if mode == "online" else None,
        tags=[result["sequence"], execution, policy, "imported-profile"],
        config=config,
        dir=str(output),
        settings=wandb.Settings(
            disable_git=True, disable_code=True, console="off", x_disable_stats=True
        ),
    ) as run:
        run.summary.update(numeric_metrics(result["metrics"], "accuracy"))
        run.summary.update(numeric_metrics(result["statistics"], "performance"))
        run.summary.update(numeric_metrics(profile, "profile"))
        run.summary["measurement_scope"] = profile["scope"]
        run.summary["artifact_fingerprint"] = digest
        run.define_metric("replay/seconds")
        run.define_metric("resources/*", step_metric="replay/seconds")
        for record in resource_history(directory / "profile/samples.csv"):
            run.log(record)
        stages = profile["stages"]
        table = wandb.Table(
            columns=["stage", "calls", "wall_sum_s", "caller_cpu_s"],
            data=[
                [name, values["calls"], values["wall_sum_s"], values["caller_thread_cpu_sum_s"]]
                for name, values in stages.items()
            ],
        )
        run.log(
            {
                "stages/table": table,
                "stages/wall_time": wandb.plot.bar(
                    table, "stage", "wall_sum_s", title="Overlapping stage wall sums (not additive)"
                ),
                "trajectory/diagnostics": wandb.Image(str(directory / "diagnostics.png")),
            }
        )
        with (directory / "latency.csv").open() as stream:
            latency = list(csv.DictReader(stream))
        run.define_metric("tracking/frame")
        run.define_metric("tracking/push_latency_ms", step_metric="tracking/frame")
        for row in latency:
            run.log(
                {
                    "tracking/frame": float(row["frame"]),
                    "tracking/push_latency_ms": float(row["push_latency_ms"]),
                }
            )
        artifact = wandb.Artifact(
            run_id + "-results",
            type="slam-benchmark",
            metadata={
                "fingerprint": digest,
                "sequence": result["sequence"],
                "mode": execution,
                "scope": "Measured replay results; model weights and RGB/frame/submap caches excluded",
            },
        )
        for path in artifact_files(directory):
            artifact.add_file(str(path), name=path.relative_to(directory).as_posix())
        for path in [
            env_path,
            directory.parent / "README.md",
            directory.parent.with_suffix(".log"),
        ]:
            if path.is_file():
                artifact.add_file(str(path), name="provenance/" + path.name)
        run.log_artifact(artifact)
        receipt = {
            **identity,
            "group": cohort,
            "directory": str(Path(run.dir).resolve().parent),
            "url": run.url if mode == "online" else None,
        }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
