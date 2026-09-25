from __future__ import annotations

import numpy as np
import torch

from amber_slam.losses import geometry_loss, normalize_targets
from amber_slam.models import GeometryTransformer


def test_model_gradient_and_shapes() -> None:
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = GeometryTransformer(width=32, layers=4, heads=4, patch=8, fusion=16)
    images = torch.randn(1, 2, 3, 32, 32)
    poses = torch.eye(4).expand(1, 2, 4, 4).clone()
    poses[:, 1, 0, 3] = 0.1
    k = torch.tensor([[32.0, 0, 15.5], [0, 32.0, 15.5], [0, 0, 1.0]]).expand(1, 2, 3, 3)
    target = normalize_targets(
        {"depth": torch.full((1, 2, 32, 32), 3.0), "poses": poses, "intrinsics": k}
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        out = model(images)
        loss, _ = geometry_loss(out, target)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert out["rays"].shape == (1, 2, 32, 32, 6)
    np.testing.assert_allclose(out["poses"][0, 0].detach(), np.eye(4), atol=1e-5)
