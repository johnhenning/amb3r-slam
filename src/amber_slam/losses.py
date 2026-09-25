import torch
import torch.nn.functional as F

from .models import rays_from_camera


def masked_mean(value, mask):
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(value)
    return torch.where(mask, value, torch.zeros_like(value)).sum() / mask.sum().clamp_min(1)


def normalize_targets(batch):
    depth = batch["depth"]
    valid = torch.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError("Batch has no valid depth")
    clean = torch.where(valid, depth, 0.0)
    rays = rays_from_camera(batch["poses"], batch["intrinsics"], *depth.shape[-2:])
    points = rays[..., :3] + clean[..., None] * rays[..., 3:]
    norms = points.norm(dim=-1)
    dims = (1, 2, 3)
    scale = (torch.where(valid, norms, 0.0).sum(dims) / valid.sum(dims).clamp_min(1)).clamp_min(
        1e-6
    )
    out = dict(batch)
    out["depth"] = clean / scale[:, None, None, None]
    out["poses"] = batch["poses"].clone()
    out["poses"][..., :3, 3] /= scale[:, None, None]
    out["rays"] = torch.cat([rays[..., :3] / scale[:, None, None, None, None], rays[..., 3:]], -1)
    out["points"] = points / scale[:, None, None, None, None]
    out["valid"] = valid
    return out


def gradient_loss(pred, target, valid):
    dx = (pred[..., 1:] - pred[..., :-1]) - (target[..., 1:] - target[..., :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    return masked_mean(dx.abs(), valid[..., 1:] & valid[..., :-1]) + masked_mean(
        dy.abs(), valid[..., 1:, :] & valid[..., :-1, :]
    )


def geometry_loss(pred, target, confidence_weight=0.2):
    d = pred["depth"].float()
    t = target["depth"]
    valid = target["valid"]
    conf = pred["confidence"].float().clamp(1e-4, 1e4)
    terms = {
        "depth": masked_mean(conf * (d - t).abs() - confidence_weight * conf.log(), valid),
        "gradient": gradient_loss(d, t, valid),
        "ray": masked_mean((pred["rays"].float() - target["rays"]).abs(), valid),
    }
    points = pred["rays"][..., :3] + d[..., None] * pred["rays"][..., 3:]
    terms["point"] = masked_mean((points - target["points"]).abs(), valid)
    # Matrix camera supervision avoids quaternion sign ambiguity. This is a
    # documented engineering choice, not the released trainer's camera loss.
    terms["camera_translation"] = (
        (pred["poses"][..., :3, 3] - target["poses"][..., :3, 3]).abs().mean()
    )
    terms["camera_rotation"] = (
        (pred["poses"][..., :3, :3] - target["poses"][..., :3, :3]).abs().mean()
    )
    h, w = d.shape[-2:]
    scale = d.new_tensor([w, h, 1.0]).reshape(1, 1, 3, 1)
    terms["intrinsics"] = ((pred["intrinsics"] - target["intrinsics"]) / scale).abs().mean()
    return sum(terms.values()), terms


def affine_depth(pred, target, valid):
    """Differentiable positive scale/shift LS, for teacher loss (not ROE)."""
    x = pred.flatten(1)
    y = target.flatten(1)
    m = valid.flatten(1).float()
    count = m.sum(1, keepdim=True).clamp_min(1)
    mx = (x * m).sum(1, keepdim=True) / count
    my = (y * m).sum(1, keepdim=True) / count
    cov = ((x - mx) * (y - my) * m).sum(1, keepdim=True)
    var = ((x - mx).square() * m).sum(1, keepdim=True).clamp_min(1e-8)
    s = (cov / var).clamp_min(1e-4)
    shift = my - s * mx
    return (s * x + shift).reshape_as(pred)


def teacher_loss(pred, batch):
    raw = batch["depth"]
    valid = torch.isfinite(raw) & (raw > 0)
    if not valid.any():
        raise ValueError("No valid teacher supervision")
    target = torch.where(valid, raw, 0.0)
    depth = affine_depth(pred["depth"], target, valid)
    # Normalize residuals per image to stop distant scenes dominating.
    norm = target.flatten(2).sum(-1) / valid.flatten(2).sum(-1).clamp_min(1)
    norm = norm[..., None, None].clamp_min(1e-6)
    terms = {
        "global": masked_mean((depth - target).abs() / norm, valid),
        "gradient": 0.5 * gradient_loss(depth / norm, target / norm, valid),
    }
    h, w = depth.shape[-2:]
    # Local affine fits complement the global fit; explicitly LS, not MoGe ROE.
    local = depth.sum() * 0
    for y in range(0, h, max(1, h // 2)):
        for x in range(0, w, max(1, w // 2)):
            sl = (
                slice(None),
                slice(None),
                slice(y, min(h, y + h // 2)),
                slice(x, min(w, x + w // 2)),
            )
            p = affine_depth(pred["depth"][sl], target[sl], valid[sl])
            local = local + masked_mean((p - target[sl]).abs() / norm, valid[sl])
    terms["local"] = local / 4
    ray = rays_from_camera(batch["poses"], batch["intrinsics"], h, w)[..., 3:]

    def normals(z):
        p = z[..., None] * ray
        a = p[..., 1:, :-1, :] - p[..., :-1, :-1, :]
        b = p[..., :-1, 1:, :] - p[..., :-1, :-1, :]
        return F.normalize(torch.cross(a, b, dim=-1), dim=-1, eps=1e-6)

    m = valid[..., 1:, :-1] & valid[..., :-1, 1:] & valid[..., :-1, :-1]
    terms["normal"] = masked_mean(1 - (normals(depth) * normals(target)).sum(-1).clamp(-1, 1), m)
    for i, key in enumerate(["sky_mask", "object_mask"]):
        if key in batch:
            if "masks" not in pred:
                raise ValueError("This architecture has no mask heads")
            terms[key] = F.mse_loss(pred["masks"][:, :, i].sigmoid(), batch[key])
    return sum(terms.values()), terms
