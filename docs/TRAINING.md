# Training: optional now, foundation reproduction later

The current path uses DA3 checkpoints. Training is outside the SLAM runtime and
is not required to run the frontend/backend.

## Implemented experimental workflow

`GeometryTransformer` is a small independent multi-view transformer with camera
tokens, within/cross-view attention, shared multiscale reassembly and depth/ray
heads. It is **not weight-compatible with DA3/DINOv2/DPT**. The tiny preset
supports quick gradient, checkpoint, inference and export tests.

`UpstreamDA3` constructs the released DA3 architecture without loading weights.
It calls backbone/heads directly because the public inference wrapper disables
gradients and removes ray outputs. This optional path requires the pinned
upstream package. Full-scale training and loss parity remain unvalidated.

The trainer provides AdamW, warmup/cosine LR, accumulation/clipping, optional
CUDA AMP, torchrun DDP, variable views/resolutions, camera conditioning,
scene-held-out validation, atomic checkpoints and resume. It writes config and
JSONL metrics. It does not acquire datasets or manage a cluster.

```bash
python -m amber_slam train configs/smoke_train.json
python -m torch.distributed.run --nproc_per_node=4 -m amber_slam train my_training.json
```

Set `resume` to an owned `last.pt` in the configuration. Checkpoints include
model, optimizer, AMP scaler, step and RNG state. Resume assumes unchanged
architecture/sampling/optimizer settings. Changed world size is not an exact
continuation guarantee. Optional `backbone_init` loads an explicitly supplied
state dict with strict keys; training constructors download no weights.

Geometry loss includes common-scale normalization, confidence-weighted depth,
rays, points, depth gradients and camera terms. Rotation uses matrix L1, an
engineering choice rather than the original trainer's camera objective.

`stage: "teacher"` uses affine depth alignment, local crop losses, gradients,
normal consistency and optional binary sky/object masks. This is experimental,
**not exact MoGe ROE or DA3 distance-weighted normal supervision**. `pseudo-label`
creates dense teacher depth aligned by positive-scale RANSAC. The
`pseudo_start_step` switch requires pseudo_depth paths on all sampled frames.

## Future reproduction work

1. Freeze DA3-Small/Giant architecture versions and evaluation protocols.
2. Choose released DINOv2 initialization (closer to DA3) or the separate,
   much larger task of reproducing DINOv2 pretraining. Random initialization
   is supported but is not the published pretrained-backbone recipe.
3. Curate synthetic teacher and multi-view geometry corpora; audit scene-level
   splits, duplicate scenes, masks, calibration and depth quality.
4. Match teacher ROE/global-local alignment, weighted normals and mask heads.
5. Cache your trained teacher's predictions and reject poor depth alignments.
6. Verify exact heads, token budgets, view/resolution sampling, conditioning,
   optimizer and normalization against a frozen target.
7. Scale to checkpointed GPU training only after short-run learning validation.
8. Replace one SLAM model at a time and compare held-out accuracy/failure rates.

Published foundation training is a large compute/data campaign. The included
smoke run proves plumbing, not reproduced foundation-model accuracy.
