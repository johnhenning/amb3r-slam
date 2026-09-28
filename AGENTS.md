# Agent guide

## Scope and orientation

This is an independent AMB3R-SLAM system-design implementation, evolving into a
SLAM experimentation platform. It is not the authors' official implementation
and does not reproduce their published accuracy or real-time results.

Read `README.md`, `docs/ARCHITECTURE.md`, `docs/FIDELITY.md`, and
`docs/VERIFICATION.md` before algorithm changes. For experiments, read
`docs/APPROACHES.md`, `docs/PROFILING.md`, and `docs/EXPERIMENTS.md`.
Check the current Git branch, working-tree changes, CI, and run artifacts rather
than assuming a previous conversation's status is still current.

## Code map

- `src/amber_slam/runtime.py`: tracking caller plus bounded mapping/graph workers.
- `frontend.py`, `backend.py`, `geometry.py`: tracking, corrections and geometry.
- `benchmark.py`, `profiling.py`: legacy TUM replay, evaluation, resource traces.
- `experiments.py`, `approaches.py`, `experiment_cli.py`: method-neutral replay
  contract, AMB3R adapter and trusted Python factory CLI.
- `experiment_tracking.py`, `scripts/export_wandb.py`: post-replay W&B import.
- `scripts/report_interactive.py`: matched-run Plotly HTML comparison.
- `scripts/profile_tum_routes.py`: fresh-process, full-route benchmark matrix.
- `tests/`: offline contracts/regressions and opt-in external-data tests.

## Setup and verification

Use Python 3.12 and the committed `uv.lock`. From the repository root:

```bash
uv sync --locked --extra train --extra export --extra benchmark --extra tracking
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pyrefly check
uv run --no-sync pytest -q
```

Formatting also covers Python examples in Markdown. For checkpoint-backed runs,
install `--extra da3 --extra benchmark --extra tracking`. Do not upgrade the
lockfile or download datasets/checkpoints unless relevant to the task. Review
dependency changes; keep model-specific dependencies optional.

Follow `.github/workflows/test.yml` for synthetic training/ONNX smoke checks.
Label these synthetic checks; they do not establish real-data SLAM accuracy.
External-data tests require their documented environment variables. Skips are
not passes. If `/proc`, CUDA or dependencies are unavailable, report the exact
limitation; do not weaken tests or invent resource measurements to make CI green.

## Concurrency invariants

Tracking/correction application belongs to the caller; prepared mapping output
is handed off to the graph owner. Preserve bounded backpressure, required window
overlap, FIFO correction delivery, backend-model synchronization, worker-error
propagation and cleanup. Do not read worker-owned mutable graph state casually.

`TrackingScaleError` permits one retry only when pending backend corrections can
be drained. Preserve strict scale validation, propagate persistent failures, and
record retry counts. Never silently substitute ground-truth or fabricated poses.
Test sequential and concurrent modes when touching these boundaries.

## Experiment contracts

The neutral interface currently supports in-process monocular RGB with TUM-style
timestamps and Sim(3) evaluation. It is not yet a stereo, depth, IMU, subprocess
or GPU profiler. No second production SLAM method is bundled.

An adapter implements `push`, `finish` and idempotent `close`. `finish` drains
all work; cleanup must preserve the original exception. Return ordered original
sample timestamps and finite rigid camera-to-world poses. Ground truth belongs
only to the evaluator; report lost estimates and coverage rather than hiding gaps.
Factories execute trusted Python; they are not a security sandbox.

Use fresh processes/output directories and pin source/checkpoint/input hashes.
Compare the same hardware, inputs, sampling, configuration and alignment. Separate
online from corrected trajectories and accuracy from coverage. Repeat and alternate
trial order before making claims about variance or speedup. One run is descriptive.

Replay excludes model download/loading and upload. The neutral runner also excludes
SLAM-state construction, unlike the legacy runner. Do not mix their timings without
reconciling scope. CPU 100% means one core; memory peaks are sampled. Current process
profiling excludes subprocess/GPU accounting. Stage times overlap and are not additive.

## Results and credentials

W&B destination: `jlh15/slam`. Keep one measured replay per run and group matched
trials. Export after replay so logging cannot distort benchmark timings. Offline
is the local default; hosted upload requires explicit online mode and credentials.

Use `WANDB_API_KEY` only through the environment or GitHub repository secret.
Never print, commit, embed in documents, or copy credentials into workflow YAML.
Never change project visibility or share access without authorization.

Keep new generated reports, Plotly HTML, resource traces, weights, dataset archives
and reconstruction caches out of Git. Use W&B/Actions artifacts and preserve raw
reports as the portable source of truth. Old tracked baseline files are not a
precedent for committing new experiments. Never manufacture missing historical runs.

The saved-result publisher imports an existing Actions artifact without inference.
Receipts distinguish offline preparation from online success. Verify run URLs and
completion before claiming upload. An existing online ID may reject a fresh import;
do not delete or overwrite remote runs to work around that without authorization.

## Change management

Keep implementation, profiling, adapters and experiment-output changes small and
separable. Preserve unrelated working-tree edits. Add focused regressions and run
the relevant checks. Do not merge without user authorization and successful checks;
do not bypass branch rules or connector restrictions. Treat docs as dated evidence,
not a substitute for inspecting the current repository and measured artifacts.
