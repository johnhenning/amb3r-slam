"""Fixed-shape export boundary and numerical parity gate."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contracts import JsonObject, PathLike
from .models import GeometryTransformer, UpstreamDA3, build_model

OUTPUTS = ["poses", "depth", "confidence", "intrinsics"]


@contextmanager
def portable_attention() -> Generator[None, None, None]:
    """Avoid PyTorch's fused CPU-only encoder op during legacy ONNX tracing."""
    previous = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    try:
        yield
    finally:
        torch.backends.mha.set_fastpath_enabled(previous)


class InferenceGraph(nn.Module):
    def __init__(self, model: GeometryTransformer | UpstreamDA3) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        prediction = self.model(images)
        return tuple(prediction[name] for name in OUTPUTS)


def export_onnx(
    checkpoint: PathLike,
    destination: PathLike,
    views: int = 4,
    height: int | None = None,
    width: int | None = None,
) -> JsonObject:
    """Export owned checkpoint; fail rather than claiming unverified parity."""
    import onnx
    import onnxruntime as ort

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(state["model_config"])
    model.load_state_dict(state["model"])
    model.eval()
    h, w = state["image_size"]
    h, w = height or h, width or w
    graph = InferenceGraph(model).eval()
    generator = torch.Generator().manual_seed(29)
    example = torch.randn(1, views, 3, h, w, generator=generator)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Fixed view count and resolution simplify mobile allocation and delegates.
    with portable_attention(), torch.inference_mode():
        torch.onnx.export(
            graph,
            (example,),
            str(destination),
            input_names=["images"],
            output_names=OUTPUTS,
            opset_version=17,
            dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(destination)))
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    maximum_error = {name: 0.0 for name in OUTPUTS}
    for _ in range(3):
        example = torch.randn(1, views, 3, h, w, generator=generator)
        with torch.inference_mode():
            expected = graph(example)
        values = session.run(OUTPUTS, {"images": example.numpy()})
        if not all(isinstance(value, np.ndarray) for value in values):
            raise TypeError("ONNX geometry outputs must be dense arrays")
        actual = [np.asarray(value) for value in values]
        for name, a, b in zip(OUTPUTS, actual, expected):
            b = b.numpy()
            np.testing.assert_allclose(a, b, rtol=3e-4, atol=3e-4, err_msg=name)
            maximum_error[name] = max(maximum_error[name], float(np.max(np.abs(a - b))))
    example.numpy().astype("<f4").tofile(str(destination) + ".input.f32")
    for name, array in zip(OUTPUTS, actual):
        array.astype("<f4").tofile(str(destination) + f".{name}.f32")
    metadata = {
        "schema_version": 1,
        "views": views,
        "image_size": [h, w],
        "input": "images",
        "input_shape": [1, views, 3, h, w],
        "outputs": OUTPUTS,
        "output_shapes": {name: list(a.shape) for name, a in zip(OUTPUTS, actual)},
        "dtype": "float32",
        "layout": "row-major",
        "color": "RGB",
        "normalization": {
            "scale": 255,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "poses": "camera-to-world, column vectors, first view reference",
        "depth": "camera z-depth, same arbitrary scale as pose translation",
        "parity_max_abs_error": maximum_error,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    Path(str(destination) + ".json").write_text(json.dumps(metadata, indent=2))
    return metadata


def benchmark_onnx(path: PathLike, iterations: int = 100, warmup: int = 10) -> JsonObject:
    import onnxruntime as ort

    if iterations < 1:
        raise ValueError("iterations must be positive")
    metadata = json.loads(Path(str(path) + ".json").read_text())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = {"images": np.zeros(metadata["input_shape"], np.float32)}
    for _ in range(warmup):
        session.run(None, inputs)
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, inputs)
        durations.append((time.perf_counter() - start) * 1000)
    return {
        "provider": "CPUExecutionProvider",
        "iterations": iterations,
        "p50_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "scope": "model only, excludes preprocessing, capture and SLAM backend",
    }
