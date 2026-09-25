"""Trainable geometry networks. No constructor downloads model weights."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import ModelConfig


def rigid_inverse(p: torch.Tensor) -> torch.Tensor:
    r = p[..., :3, :3].transpose(-1, -2)
    t = -(r @ p[..., :3, 3:4])
    last = torch.cat(
        [
            torch.zeros_like(t[..., 0:1, :]).expand(*t.shape[:-2], 1, 3),
            torch.ones_like(t[..., 0:1, :]),
        ],
        -1,
    )
    return torch.cat([torch.cat([r, t], -1), last], -2)


def rotation_6d(x: torch.Tensor) -> torch.Tensor:
    a = F.normalize(x[..., :3], dim=-1, eps=1e-6)
    b = x[..., 3:] - (a * x[..., 3:]).sum(-1, keepdim=True) * a
    b = F.normalize(b, dim=-1, eps=1e-6)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


def rays_from_camera(
    poses: torch.Tensor, intrinsics: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=poses.device, dtype=poses.dtype),
        torch.arange(width, device=poses.device, dtype=poses.dtype),
        indexing="ij",
    )
    # Zero-skew pinhole cameras. Avoid matrix inverse for export compatibility.
    x = (xx - intrinsics[..., 0, 2, None, None]) / intrinsics[..., 0, 0, None, None]
    y = (yy - intrinsics[..., 1, 2, None, None]) / intrinsics[..., 1, 1, None, None]
    camera = torch.stack([x, y, torch.ones_like(x)], -1)
    direction = torch.einsum("bvij,bvhwj->bvhwi", poses[..., :3, :3], camera)
    origin = poses[..., :3, 3][..., None, None, :].expand_as(direction)
    return torch.cat([origin, direction], -1)


class Fusion(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(width, width, 3, padding=1),
                    nn.GELU(),
                    nn.Conv2d(width, width, 3, padding=1),
                )
                for _ in range(4)
            ]
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        x = features[-1]
        for i in range(3, -1, -1):
            x = F.interpolate(x, size=features[i].shape[-2:], mode="bilinear", align_corners=False)
            x = x + features[i]
            x = x + self.layers[i](x)
        return x


class GeometryTransformer(nn.Module):
    """Independent DA3-inspired depth/ray transformer, NOT DA3 weight compatible.

    Local first 2/3 blocks, alternating global/local last third, shared multiscale
    reassembly, separate depth/ray fusion, explicit camera tokens. Tiny defaults
    support CPU development; use UpstreamDA3 for the released DA3 architecture.
    """

    def __init__(
        self,
        width: int = 96,
        layers: int = 6,
        heads: int = 4,
        patch: int = 8,
        fusion: int = 32,
        checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if layers < 4 or width % heads:
            raise ValueError("Need >=4 layers and width divisible by heads")
        self.patch, self.width, self.checkpointing = patch, width, checkpointing
        self.embed = nn.Conv2d(3, width, patch, stride=patch)
        self.position = nn.Linear(2, width)
        self.camera_token = nn.Parameter(torch.zeros(1, 1, 1, width))
        self.reference_token = nn.Parameter(torch.randn(1, 1, 1, width) * 0.02)
        self.condition = nn.Linear(16, width)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    width,
                    heads,
                    width * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(width)
        self.reassemble = nn.ModuleList([nn.Conv2d(width, fusion, 1) for _ in range(4)])
        self.depth_fusion, self.ray_fusion = Fusion(fusion), Fusion(fusion)
        self.depth_head = nn.Conv2d(fusion, 2, 1)
        self.ray_head = nn.Conv2d(fusion, 6, 1)
        self.camera_head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 13))
        self.mask_head = nn.Conv2d(fusion, 2, 1)
        with torch.no_grad():
            camera_output = self.camera_head[-1]
            assert isinstance(camera_output, nn.Linear) and camera_output.bias is not None
            camera_output.bias.zero_()
            camera_output.bias[3:9] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])

    def forward(
        self, images: torch.Tensor, camera_condition: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        b, v, _, h, w = images.shape
        x = self.embed(images.reshape(b * v, 3, h, w))
        gh, gw = x.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, gh, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, gw, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        pos = self.position(torch.stack([xx, yy], -1).reshape(1, 1, gh * gw, 2))
        x = x.flatten(2).transpose(1, 2).reshape(b, v, gh * gw, self.width) + pos
        cam = self.camera_token.expand(b, v, 1, self.width)
        if camera_condition is not None:
            cam = cam + self.condition(camera_condition).unsqueeze(2)
        # A distinct first-view token fixes the local world gauge.
        first = torch.arange(v, device=x.device).eq(0).to(x.dtype).reshape(1, v, 1, 1)
        cam = cam + first * self.reference_token
        x = torch.cat([cam, x], 2)
        n = x.shape[2]
        taps = []
        tap_indices = [round(i * (len(self.blocks) - 1) / 3) for i in range(4)]
        for i, block in enumerate(self.blocks):
            global_attention = (
                i >= 2 * len(self.blocks) // 3 and (i - 2 * len(self.blocks) // 3) % 2 == 0
            )
            y = (
                x.reshape(b, v * n, self.width)
                if global_attention
                else x.reshape(b * v, n, self.width)
            )
            if self.checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint

                y = checkpoint(block, y, use_reentrant=False)
            else:
                y = block(y)
            x = y.reshape(b, v, n, self.width)
            if i in tap_indices:
                taps.append(
                    self.norm(x[:, :, 1:]).reshape(b * v, gh, gw, self.width).permute(0, 3, 1, 2)
                )
        features = []
        for i, (proj, tap) in enumerate(zip(self.reassemble, taps)):
            size = (max(1, gh * 2 // (2**i)), max(1, gw * 2 // (2**i)))
            features.append(
                F.interpolate(proj(tap), size=size, mode="bilinear", align_corners=False)
            )
        dfeat = self.depth_fusion(features)
        raw = F.interpolate(
            self.depth_head(dfeat), size=(h, w), mode="bilinear", align_corners=False
        )
        depth = torch.exp(raw[:, 0].clamp(-10, 10)).reshape(b, v, h, w)
        confidence = (F.softplus(raw[:, 1]) + 1e-4).reshape(b, v, h, w)
        rays = F.interpolate(
            self.ray_head(self.ray_fusion(features)),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )
        rays = rays.reshape(b, v, 6, h, w).permute(0, 1, 3, 4, 2)
        c = self.camera_head(self.norm(x[:, :, 0]))
        r = rotation_6d(c[..., 3:9])
        t = c[..., :3, None]
        bottom = torch.cat(
            [
                torch.zeros_like(t[..., :1, :]).expand(b, v, 1, 3),
                torch.ones_like(t[..., :1, :]),
            ],
            -1,
        )
        poses = torch.cat([torch.cat([r, t], -1), bottom], -2)
        poses = rigid_inverse(poses[:, :1]) @ poses
        fx = (F.softplus(c[..., 9]) + 0.1) * w
        fy = (F.softplus(c[..., 10]) + 0.1) * h
        cx = torch.sigmoid(c[..., 11]) * w
        cy = torch.sigmoid(c[..., 12]) * h
        z = torch.zeros_like(fx)
        o = torch.ones_like(fx)
        k = torch.stack([fx, z, cx, z, fy, cy, z, z, o], -1).reshape(b, v, 3, 3)
        masks = F.interpolate(
            self.mask_head(dfeat), size=(h, w), mode="bilinear", align_corners=False
        )
        return {
            "depth": depth,
            "confidence": confidence,
            "rays": rays,
            "poses": poses,
            "intrinsics": k,
            "masks": masks.reshape(b, v, 2, h, w),
        }


class UpstreamDA3(nn.Module):
    """Random-initialized released architecture, bypassing inference-only API.

    Calls backbone/head separately because upstream forward deletes ray outputs.
    Upstream package must be installed from a recorded revision. No pretrained
    download occurs; DINO initialization is an explicit separate operation.
    """

    def __init__(self, preset: str = "da3-small", **kwargs: object) -> None:
        super().__init__()
        from depth_anything_3.api import DepthAnything3

        self.net = DepthAnything3(model_name=preset).model
        self.net.gs_head = None
        self.net.gs_adapter = None
        self.preset = preset

    def forward(
        self, images: torch.Tensor, camera_condition: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri

        cam = None
        if camera_condition is not None:
            # condition stores c2w upper 3x4 + fx/fy/cx/cy in normalized pixels
            b, v = images.shape[:2]
            h, w = images.shape[-2:]
            p = torch.eye(4, device=images.device).expand(b, v, 4, 4).clone()
            p[..., :3, :] = camera_condition[..., :12].reshape(b, v, 3, 4)
            k = torch.eye(3, device=images.device).expand(b, v, 3, 3).clone()
            k[..., 0, 0] = camera_condition[..., 12] * w
            k[..., 1, 1] = camera_condition[..., 13] * h
            k[..., 0, 2] = camera_condition[..., 14] * w
            k[..., 1, 2] = camera_condition[..., 15] * h
            cam = self.net.cam_enc(rigid_inverse(p), k, (h, w))
        feats, _ = self.net.backbone(
            images, cam_token=cam, export_feat_layers=[], ref_view_strategy="first"
        )
        with torch.autocast(device_type=images.device.type, enabled=False):
            out = self.net.head(feats, *images.shape[-2:], patch_start_idx=0)
            enc = self.net.cam_dec(feats[-1][1])
            poses, k = pose_encoding_to_extri_intri(enc, images.shape[-2:])
        if poses.shape[-2] == 3:
            bottom = torch.zeros_like(poses[..., :1, :])
            bottom[..., 0, 3] = 1
            poses = torch.cat([poses, bottom], -2)
        depth = out["depth"]
        if depth.ndim == 5:
            depth = depth.squeeze(-1)
        conf = out["depth_conf"]
        if conf.ndim == 5:
            conf = conf.squeeze(-1)
        return {
            "depth": depth,
            "confidence": conf,
            "rays": out["ray"],
            "poses": poses,
            "intrinsics": k,
        }


def build_model(config: ModelConfig) -> GeometryTransformer | UpstreamDA3:
    kind = config.get("kind", "independent")
    if kind == "da3":
        return UpstreamDA3(preset=config.get("preset", "da3-small"))
    if kind == "independent":
        return GeometryTransformer(
            width=config.get("width", 96),
            layers=config.get("layers", 6),
            heads=config.get("heads", 4),
            patch=config.get("patch", 8),
            fusion=config.get("fusion", 32),
            checkpointing=config.get("checkpointing", False),
        )
    raise ValueError(f"Unknown model kind {kind}")
