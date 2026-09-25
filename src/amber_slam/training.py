"""Optional from-scratch experiments; not required by the checkpoint SLAM path.

DDP, deterministic step sampling, AMP, gradient accumulation and atomic resume
checkpoints. Dataset membership is scene-disjoint in the manifest. Large-scale
DA3 recipe reproduction remains a separately documented research campaign.
"""

import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import distributed
from torch.nn.parallel import DistributedDataParallel

from .data import SceneDataset
from .losses import geometry_loss, normalize_targets, teacher_loss
from .models import build_model


def _batch(dataset, index, views, size, seed, batch_size, device, pseudo=False):
    samples = [
        dataset.sample(index + i, views, size, np.random.default_rng(seed + i), pseudo)
        for i in range(batch_size)
    ]
    keys = set.intersection(*(set(s) for s in samples))
    batch = {key: torch.stack([s[key] for s in samples]).to(device) for key in keys}
    mean = batch["images"].new_tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1)
    std = batch["images"].new_tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1)
    batch["images"] = (batch["images"] - mean) / std
    return batch


def _condition(target):
    h, w = target["depth"].shape[-2:]
    k = target["intrinsics"]
    return torch.cat(
        [
            target["poses"][..., :3, :].flatten(-2),
            torch.stack(
                [
                    k[..., 0, 0] / w,
                    k[..., 1, 1] / h,
                    k[..., 0, 2] / w,
                    k[..., 1, 2] / h,
                ],
                -1,
            ),
        ],
        -1,
    )


def train(config_path):
    config = json.loads(Path(config_path).read_text())
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if world > 1:
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
        distributed.init_process_group(backend="nccl" if device.startswith("cuda") else "gloo")
    try:
        return _train(config, world, rank, device)
    finally:
        if world > 1:
            distributed.destroy_process_group()


def _train(config, world, rank, device):
    seed = config.get("seed", 17)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("cpu_threads", 4))
    model_config = config.get("model", {})
    model = build_model(model_config).to(device)
    # Explicit optional backbone initialization; no implicit weight downloads.
    if config.get("backbone_init"):
        state = torch.load(config["backbone_init"], map_location="cpu", weights_only=True)
        backbone = model.net.backbone if hasattr(model, "net") else model
        backbone.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("lr", 2e-4),
        weight_decay=config.get("weight_decay", 0.05),
    )
    use_amp = config.get("amp", False) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    initial = 0
    if config.get("resume"):
        state = torch.load(config["resume"], map_location="cpu", weights_only=True)
        if state["model_config"] != model_config:
            raise ValueError("Resume architecture mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        initial = state["step"]
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    if world > 1:
        # Optional conditioning and teacher-only heads can be unused each step.
        model = DistributedDataParallel(
            model,
            device_ids=[int(device.split(":")[-1])] if device.startswith("cuda") else None,
            find_unused_parameters=True,
        )
    train_data = SceneDataset(config["manifest"], "train")
    val_data = SceneDataset(config["manifest"], "val")
    train_ids = {s["id"] for s in train_data.scenes}
    val_ids = {s["id"] for s in val_data.scenes}
    if train_ids & val_ids:
        raise ValueError("Training and validation scenes overlap")
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output / "config.json").write_text(json.dumps(config, indent=2))
    steps = config.get("steps", 1000)
    accumulation = config.get("accumulation", 1)
    if steps < 1 or accumulation < 1:
        raise ValueError("steps and accumulation must be positive")
    sizes = config.get("resolutions", [[64, 64]])
    min_views, max_views = config.get("views", [2, 4])
    teacher = config.get("stage", "geometry") == "teacher"
    base_lr = config.get("lr", 2e-4)
    warmup = config.get("warmup_steps", min(100, steps // 10))
    val_every = config.get("validate_every", 100)
    last_metrics = {}
    for step in range(initial, steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        schedule = np.random.default_rng(seed + step)  # Same shapes on every rank.
        size = tuple(sizes[int(schedule.integers(len(sizes)))])
        views = 1 if teacher else int(schedule.integers(min_views, max_views + 1))
        token_budget = config.get("pixels_per_step")
        batch_size = (
            max(1, token_budget // (views * size[0] * size[1]))
            if token_budget
            else config.get("batch_size", 1)
        )
        progress = max(0, (step - warmup) / max(1, steps - warmup))
        lr = (
            base_lr * (step + 1) / max(1, warmup)
            if step < warmup
            else base_lr * 0.5 * (1 + math.cos(math.pi * progress))
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        total = 0.0
        for micro in range(accumulation):
            sample_seed = seed + step * 100003 + rank * 1009 + micro * 31
            index = sample_seed % len(train_data)
            pseudo = step >= config.get("pseudo_start_step", steps + 1)
            batch = _batch(train_data, index, views, size, sample_seed, batch_size, device, pseudo)
            target = batch if teacher else normalize_targets(batch)
            condition = None
            if not teacher and schedule.random() < config.get("pose_condition_probability", 0.2):
                condition = _condition(target)
            sync = model.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
            with sync:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    prediction = model(batch["images"], condition)
                    loss, terms = (
                        teacher_loss(prediction, target)
                        if teacher
                        else geometry_loss(prediction, target)
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                scaler.scale(loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.get("clip_grad", 1.0), error_if_nonfinite=True
        )
        scaler.step(optimizer)
        scaler.update()
        if (step + 1) % val_every == 0 or step + 1 == steps:
            # Validate on unwrapped model to avoid rank-0-only DDP collectives.
            raw = model.module if world > 1 else model
            raw.eval()
            validation = []
            with torch.inference_mode():
                for index in range(len(val_data)):
                    batch = _batch(val_data, index, views, size, seed + index, 1, device)
                    target = batch if teacher else normalize_targets(batch)
                    prediction = raw(batch["images"])
                    loss, _ = (
                        teacher_loss(prediction, target)
                        if teacher
                        else geometry_loss(prediction, target)
                    )
                    validation.append(float(loss))
            last_metrics = {
                "step": step + 1,
                "train_loss": total,
                "val_loss": float(np.mean(validation)),
                "lr": lr,
            }
            if rank == 0:
                with (output / "metrics.jsonl").open("a") as log:
                    log.write(json.dumps(last_metrics) + "\n")
                state = {
                    "model_config": model_config,
                    "model": raw.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step + 1,
                    "image_size": list(size),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "config": config,
                    "metrics": last_metrics,
                }
                temporary = output / "checkpoint.tmp"
                torch.save(state, temporary)
                temporary.replace(output / "last.pt")
                print(json.dumps(last_metrics), flush=True)
            if world > 1:
                distributed.barrier()
    return last_metrics
