"""Shared static contracts for arrays, manifests, configuration and reports.

Array shapes and mixed NumPy dtypes remain runtime-validated at model boundaries.
``Any`` is limited to ndarray dtype parameters and open JSON export metadata;
algorithm inputs, model tensors, configuration, and benchmark metrics are typed.
"""

from pathlib import Path
from typing import Any, NotRequired, TypedDict

from numpy.typing import NDArray

Array = NDArray[Any]
PathLike = str | Path
JsonObject = dict[str, Any]


class FrameRecord(TypedDict):
    timestamp: float
    rgb: str
    depth: NotRequired[str]
    depth_scale: NotRequired[float]
    intrinsics: NotRequired[list[list[float]]]
    c2w: NotRequired[list[list[float]]]
    lidar: NotRequired[str]
    right_rgb: NotRequired[str]
    baseline_m: NotRequired[float]
    pseudo_depth: NotRequired[str]
    sky_mask: NotRequired[str]
    object_mask: NotRequired[str]


class SceneRecord(TypedDict):
    id: str
    split: str
    frames: list[FrameRecord]


class Manifest(TypedDict):
    version: int
    sequences: list[SceneRecord]


class DatasetSummary(TypedDict):
    frames: int
    manifest: str


class ModelConfig(TypedDict, total=False):
    kind: str
    preset: str
    width: int
    layers: int
    heads: int
    patch: int
    fusion: int
    checkpointing: bool


class TrainingConfig(TypedDict):
    manifest: str
    output: str
    device: NotRequired[str]
    seed: NotRequired[int]
    cpu_threads: NotRequired[int]
    model: NotRequired[ModelConfig]
    backbone_init: NotRequired[str]
    resume: NotRequired[str]
    lr: NotRequired[float]
    weight_decay: NotRequired[float]
    amp: NotRequired[bool]
    steps: NotRequired[int]
    accumulation: NotRequired[int]
    resolutions: NotRequired[list[tuple[int, int]]]
    views: NotRequired[tuple[int, int]]
    stage: NotRequired[str]
    warmup_steps: NotRequired[int]
    validate_every: NotRequired[int]
    pixels_per_step: NotRequired[int]
    batch_size: NotRequired[int]
    pseudo_start_step: NotRequired[int]
    pose_condition_probability: NotRequired[float]
    clip_grad: NotRequired[float]


class GraphReport(TypedDict):
    initial_cost: float
    final_cost: float
    success: bool
    evaluations: NotRequired[int]
    submaps: NotRequired[int]
    edges: NotRequired[int]
    rejected_loops: NotRequired[int]


class RunStatistics(TypedDict):
    frames: int
    push_latency_ms_p50: NotRequired[float]
    push_latency_ms_p95: NotRequired[float]
    push_throughput_fps: NotRequired[float]
    submaps: NotRequired[int]
    edges: NotRequired[int]
    replay_elapsed_s: NotRequired[float]
    end_to_end_fps: NotRequired[float]


class WindowAuc(TypedDict):
    auc: float | None
    windows: int
    skipped: int


class TrajectoryMetrics(TypedDict):
    alignment: str
    alignment_scale: float
    matched_poses: int
    reference_coverage: float
    estimate_coverage: float
    timestamp_tolerance_s: float
    rpe_delta_frames: int
    ate_rmse_m: float
    rpe_translation_rmse_m: float
    rpe_rotation_rmse_deg: float
    translation_auc: dict[str, float]
    protocol: str


class BenchmarkResult(TypedDict):
    sequence: str
    source_rgb_frames: int
    sampled_frames: int
    sampling_stride: int
    max_frames: int | None
    sampled_duration_s: float
    sampled_gt_match_fraction: float
    groundtruth_sha256: str
    rgb_list_sha256: str
    input_manifest_sha256: str
    config: JsonObject
    metrics: dict[str, TrajectoryMetrics]
    statistics: RunStatistics


class BenchmarkEnvironment(TypedDict):
    python: str
    platform: str
    processor: str
    torch_threads: int
    cuda_available: bool
    packages: dict[str, str]
    resolution: NotRequired[int]
    checkpoint: NotRequired[str]
    checkpoint_revision: NotRequired[str]
    checkpoint_hashes: NotRequired[dict[str, str]]
    command: NotRequired[list[str]]
    sources: NotRequired[dict[str, str]]
