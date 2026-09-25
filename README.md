# Amber SLAM

An independent Python implementation of the **AMB3R-SLAM system design**, using
**Depth Anything 3 checkpoints** for the initial frontend and backend. The
repository includes replay/live capture, evaluation, experimental training,
ONNX export, and a small C++ network-inference library.

**Status:** research implementation, not an official implementation or a claim
of reproduced benchmark results. Read [verification](docs/VERIFICATION.md) for
what was actually run and [fidelity](docs/FIDELITY.md) for deviations. The
experimental models are not interchangeable with DA3 weights.

## Review this project

Start with [the implementation plan](docs/PLAN.md), then the
[architecture and reading guide](docs/ARCHITECTURE.md). The
[paper-to-code mapping](docs/FIDELITY.md) identifies implemented methods and
remaining reproduction work. [Future training](docs/TRAINING.md) describes the
later foundation-model reproduction effort. [Mobile](docs/MOBILE.md) describes
the C++/iOS/Android boundary and acceptance gates.

## Install

Use Python 3.10–3.12 in a virtual environment. Install a matching PyTorch and
TorchVision build for your CPU/CUDA platform first, following PyTorch's install
instructions. Then, from the repository root:

```bash
python -m pip install -e '.[dev,train,export]'
python -m pytest -q
```

For the actual checkpoint-backed SLAM path, also install the pinned upstream
DA3 dependency. Its package includes substantial optional visualization and
CUDA dependencies; follow upstream installation guidance for your platform.

```bash
python -m pip install -r requirements/da3.txt
```

The base package intentionally does not install or download DA3 checkpoints.
They are fetched by DA3 on first explicit use, or you can pass local checkpoint
directories. GPU memory and runtime depend strongly on view count/resolution.
The giant backend is a desktop GPU target; no phone performance is asserted.

## Run checkpoint-backed SLAM

Prepare a [versioned manifest](docs/DATA.md), or use the included dataset
converters. Monocular runs ignore depth for geometric inference, while metric
mode uses registered sensor depth to calibrate scale.

```bash
python -m amber_slam run \
  --manifest data/my_sequence.json --sequence my-sequence \
  --frontend depth-anything/DA3-SMALL \
  --backend depth-anything/DA3-GIANT \
  --device cuda --config configs/monocular.json \
  --output runs/my-sequence
```

The default giant checkpoint is the original series for paper-oriented work.
The upstream `DA3-GIANT-1.1` checkpoint is an explicit alternative, not silently
substituted. Record the chosen model/revision when comparing results.

Live camera (a desktop with a camera and sufficient compute):

```bash
python -m amber_slam run --camera 0 --async-backend \
  --device cuda --output runs/live --max-frames 600
```

The capture queue retains only the newest frame. The mapper maintains required
submap overlap through backpressure; this is a streaming interface, **not a
hard real-time guarantee**. The Python API returns a `TrackingResult` per frame.
Both model instances may contend for one GPU; measure end-to-end timing on your
hardware. Use a new output directory for each run.

Outputs include corrected and originally emitted TUM trajectories, persisted
frames/submaps, graph nodes/edges, configuration, and timing statistics. Image
caches are bounded, but disk use, trajectory metadata and the graph grow with
the sequence. Loss of tracking raises an error rather than inventing poses.

## Evaluate and export a map

```bash
python -m amber_slam evaluate data/groundtruth.tum runs/my-sequence/trajectory.tum \
  --alignment sim3 --output runs/my-sequence/metrics.json
python scripts/export_map.py runs/my-sequence runs/my-sequence/map.ply
```

Use `--alignment se3` for metric-scale evaluation. Evaluate
`trajectory_online.tum` separately to measure the poses available at tracking
time. ATE, RPE, timestamp coverage, and translation-error AUC are reported.
The included window-AUC helper has an explicit local convention; published
VidMap W-AUC comparisons require the official protocol.

## Optional training/export smoke workflow

These commands train a **tiny independent model on analytic fixtures**, not
DA3, and test the training/deployment plumbing:

```bash
python -m amber_slam synthetic data/synthetic --frames 16 --size 32
python -m amber_slam train configs/smoke_train.json
python -m amber_slam export runs/smoke_training/last.pt runs/smoke_training/model.onnx --views 4
python -m amber_slam benchmark runs/smoke_training/model.onnx
```

Use the ONNX model as a frontend with `--model onnx` only when its fixed view
capacity covers the requested context. A four-view graph cannot serve a larger
mapping window. Export a separate graph with enough backend views or use the
PyTorch/DA3 backend through the Python API. Missing views are padded by repeating
the last frame, which can change predictions; validate that policy for a trained
mobile model.

## Reproducibility and ownership

Source repository: https://github.com/johnhenning/amb3r-slam
The downloadable archive includes `repository.bundle` containing Git history.
To restore a checkout: `git clone repository.bundle amber-slam-checkout`.
The ordinary source files are also included for immediate review.
Checkpoints, datasets and run outputs are excluded from version control. The
original code is MIT licensed; external software, checkpoints and datasets
retain their own licenses. In particular, consult the DA3 model cards before
using giant weights in a commercial app. See [sources](docs/SOURCES.md).
