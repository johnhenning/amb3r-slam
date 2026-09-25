# Amber SLAM

An independent Python implementation of the **AMB3R-SLAM system design**, using
**Depth Anything 3 checkpoints** for the initial frontend and backend. The
repository includes replay/live capture, evaluation, experimental training,
ONNX export, and a small C++ network-inference library.

**Research credit:** [AMB3R-SLAM: Kilometer-scale SLAM with Hierarchical
Backend](https://arxiv.org/abs/2609.19518) by **Hengyi Wang and Lourdes Agapito**
(UCL, 2026) provides the system design reproduced here. See the
[authors' project](https://hengyiwang.github.io/projects/amber-slam) and
[official repository](https://github.com/HengyiWang/amb3r-slam).
Geometry predictions use [Depth Anything 3](https://arxiv.org/abs/2511.10647)
by Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi,
Jiashi Feng, and Bingyi Kang (ByteDance Seed, 2025).

This repository is an independent implementation, not an official release or
affiliation with either research team. The algorithmic contributions belong to
the original researchers. See [citation entries](CITATIONS.bib),
[attribution notice](NOTICE.md), and [implementation differences](docs/FIDELITY.md).


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

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). The
project pins Python 3.12 in `.python-version`; uv manages the environment and
resolves dependencies from the committed `uv.lock`. From the repository root:

```bash
uv sync --locked --extra train --extra export --extra benchmark
uv run --no-sync pytest -q
uv run --no-sync pyrefly check
uv run --no-sync ruff format --check .
```

For the actual checkpoint-backed SLAM path, also install the pinned upstream
DA3 dependency. Its package includes substantial optional visualization and
CUDA dependencies; follow upstream installation guidance for your platform.

```bash
uv sync --locked --extra da3 --extra train --extra export
```

Development tools are in the default `dev` dependency group. `train`, `export`,
`da3`, and `benchmark` are optional extras. Commands below use `--no-sync` after the explicit
sync step so they retain the selected extras. To update dependencies deliberately,
run `uv lock --upgrade`, review the diff, then `uv sync` with your chosen extras.

The base package intentionally does not install or download DA3 checkpoints.
They are fetched by DA3 on first explicit use, or you can pass local checkpoint
directories. GPU memory and runtime depend strongly on view count/resolution.
The giant backend is a desktop GPU target; no phone performance is asserted.

## Run checkpoint-backed SLAM

Prepare a [versioned manifest](docs/DATA.md), or use the included dataset
converters. Monocular runs ignore depth for geometric inference, while metric
mode uses registered sensor depth to calibrate scale.

```bash
uv run --no-sync amber run \
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
uv run --no-sync amber run --camera 0 --async-backend \
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

## Standard-dataset results

[Read the TUM CPU report](reports/tum_cpu_2026-09-25/README.md) for measured
DA3-Small results on `freiburg1_xyz` and `freiburg1_desk`, including trajectories,
ATE/RPE and latency plots. These are sampled CPU baselines, not a reproduction
of the paper’s accuracy tables. [Reproduction commands and test protocol](docs/BENCHMARKS.md).

## Evaluate and export a map

```bash
uv run --no-sync amber evaluate data/groundtruth.tum runs/my-sequence/trajectory.tum \
  --alignment sim3 --output runs/my-sequence/metrics.json
uv run --no-sync scripts/export_map.py runs/my-sequence runs/my-sequence/map.ply
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
uv run --no-sync amber synthetic data/synthetic --frames 16 --size 32
uv run --no-sync amber train configs/smoke_train.json
uv run --no-sync amber export runs/smoke_training/last.pt runs/smoke_training/model.onnx --views 4
uv run --no-sync amber benchmark runs/smoke_training/model.onnx
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

## Citing the research

If you use this implementation in research, cite the original AMB3R-SLAM paper
and Depth Anything 3 when using its models. Machine-readable references are in
[`CITATIONS.bib`](CITATIONS.bib); [`CITATION.cff`](CITATION.cff) identifies the
original paper as the preferred research citation. Cite this repository's URL
and commit separately when identifying the specific implementation evaluated.

```bibtex
@article{wang2026amb3rslam,
  title = {{AMB3R-SLAM}: Kilometer-scale {SLAM} with Hierarchical Backend},
  author = {Wang, Hengyi and Agapito, Lourdes},
  journal = {arXiv preprint arXiv:2609.19518},
  year = {2026},
  url = {https://arxiv.org/abs/2609.19518}
}
```
