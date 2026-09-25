from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .contracts import Array, PathLike
from .types import Frame, Reconstruction


def reference_order(n: int, reference: int) -> tuple[list[int], Array]:
    if not 0 <= reference < n:
        raise ValueError("Invalid reference index")
    order = [reference] + [i for i in range(n) if i != reference]
    return order, np.argsort(order)


class DA3Model:
    """Public DA3 inference API; optional local directory avoids downloads."""

    def __init__(
        self,
        checkpoint: str = "depth-anything/DA3-SMALL",
        device: str = "cuda",
        resolution: int = 504,
    ) -> None:
        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(checkpoint).to(device).eval()
        self.resolution = resolution
        self.device = device

    def reconstruct(self, frames: Sequence[Frame], reference: int = 0) -> Reconstruction:
        order, inverse = reference_order(len(frames), reference)
        prediction = self.model.inference(
            [frames[i].rgb for i in order],
            process_res=self.resolution,
            ref_view_strategy="first",
        )
        n = len(frames)
        w2c = np.repeat(np.eye(4)[None], n, axis=0)
        w2c[:, :3, :] = np.asarray(prediction.extrinsics)[:, :3, :]
        poses = np.linalg.inv(w2c)[inverse]
        poses = np.linalg.inv(poses[reference]) @ poses
        confidence = prediction.conf
        if confidence is None:
            raise ValueError("DA3 checkpoint did not return depth confidence")
        return Reconstruction(
            poses,
            np.asarray(prediction.depth)[inverse],
            np.asarray(confidence)[inverse],
            np.asarray(prediction.intrinsics)[inverse],
        ).validate(n)


class TorchModel:
    def __init__(
        self, checkpoint: PathLike, device: str = "cpu", size: tuple[int, int] | None = None
    ) -> None:
        import torch

        from .models import build_model

        self.device = device
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = build_model(state["model_config"])
        self.model.load_state_dict(state["model"])
        self.model.to(device).eval()
        height, width = size or state["image_size"]
        self.size = (int(height), int(width))

    def reconstruct(self, frames: Sequence[Frame], reference: int = 0) -> Reconstruction:
        import torch

        from .preprocess import image_tensor

        order, inverse = reference_order(len(frames), reference)
        images = image_tensor([frames[i].rgb for i in order], self.size)
        with torch.inference_mode():
            out = self.model(torch.from_numpy(images).to(self.device))
        arrays = {k: v.detach().float().cpu().numpy()[0][inverse] for k, v in out.items()}
        return Reconstruction(
            arrays["poses"], arrays["depth"], arrays["confidence"], arrays["intrinsics"]
        ).validate(len(frames))


class ONNXModel:
    def __init__(self, path: PathLike, providers: list[str] | None = None) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(path), providers=providers or ["CPUExecutionProvider"]
        )
        self.metadata = json.loads(Path(str(path) + ".json").read_text())
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != self.metadata["sha256"]:
            raise ValueError("ONNX checksum does not match its tensor contract")
        height, width = self.metadata["image_size"]
        self.size = (int(height), int(width))
        self.views = self.metadata["views"]

    def reconstruct(self, frames: Sequence[Frame], reference: int = 0) -> Reconstruction:
        from .preprocess import image_tensor

        if len(frames) > self.views:
            raise ValueError("ONNX graph has insufficient fixed view capacity")
        order, inverse = reference_order(len(frames), reference)
        ordered = [frames[i].rgb for i in order]
        ordered += [ordered[-1]] * (self.views - len(ordered))
        values = self.session.run(None, {"images": image_tensor(ordered, self.size)})
        if not all(isinstance(value, np.ndarray) for value in values):
            raise TypeError("ONNX geometry outputs must be dense arrays")
        data = {name: np.asarray(value) for name, value in zip(self.metadata["outputs"], values)}
        return Reconstruction(
            *[
                data[k][0][: len(frames)][inverse]
                for k in ["poses", "depth", "confidence", "intrinsics"]
            ]
        ).validate(len(frames))
