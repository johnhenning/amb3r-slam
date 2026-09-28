# Verification report

Date: 2026-09-25. CPU-only development environment; CUDA unavailable.

## Current validation

- Python remains **3.12** (tested interpreter 3.12.14); no 3.14 migration.
- All 160 Python functions have parameter and return annotations. Ruff `ANN`
  rules, lint and formatting checks pass. Pyrefly 1.3.1 reports zero errors in
  both the full DA3 environment and a clean fast-CI environment. Its 11 advisory
  warnings concern untyped SciPy dependencies and redundant integer conversions.
- **27 tests pass, none skipped** when the TUM dataset and report paths are set.
  Fast CI runs 21 offline tests and explicitly skips six external-data checks.
- Actual DA3-Small inference completed 80 sampled `freiburg1_xyz` frames and 62
  sampled `freiburg1_desk` frames. All matched ground truth. Corrected Sim(3) ATE
  RMSE: 0.1409 m / 0.2100 m; CPU end-to-end throughput: 0.274 / 0.376 FPS.
- Independent `evo` ATE agrees within 1e-8 m on both sequences. See the
  [report, plots and raw metrics](../reports/tum_cpu_2026-09-25/README.md) and
  [reproduction instructions](BENCHMARKS.md). These are reduced-resolution,
  every-tenth-frame CPU baselines; no published-table parity is claimed.
- After typing changes, eight-step training again reached validation loss
  1.3953187, and ONNX parity passed (maximum absolute error 1.91e-6).

## Earlier smoke verification

- 16 pytest cases pass: transform round trips, known similarity recovery,
  metric-depth validation, graph cost reduction/gauge fixing, timestamp matching,
  ATE/RPE invariance, interpolation, synchronous/asynchronous streaming,
  span-2 edges, final-window flushing, worker failures, gradient/learning smoke,
  long-context edges, RGB-D scale, ICP, projection and loop verification.
- Ruff lint and formatting pass.
- Optional tiny model trained for eight steps on generated plane fixtures;
  validation loss at step 8: 1.3953. Resume to step 10 succeeded (1.3705).
  These numbers demonstrate execution only, not useful model quality.
- Owned-checkpoint streaming replay completed for eight frames.
- ONNX export checked structurally and compared on three random inputs.
  Largest absolute difference from PyTorch: 1.91e-6 (intrinsics).
- C++17 library/executable built with GCC and ONNX Runtime 1.30.0. Native ORT
  outputs exactly matched Python ORT golden outputs on the tested input.
- Actual DA3-Small pretrained checkpoint loaded; two-view CPU inference at
  56×56 succeeded with expected pose/depth/intrinsics shapes and reference order.
  This validates integration, not standard-resolution accuracy or throughput.
- Checkpoint-backed SLAM replay with DA3-Small in both roles completed eight
  frames, two submaps and one local graph edge at resolution 56 on CPU.
  The measured end-to-end rate was 2.37 FPS; it is a tiny synthetic smoke setup,
  not the Small/Giant configuration, a mobile benchmark or a real-time claim.

## Not established

- Published AMB3R-SLAM benchmark accuracy, nine-dataset protocol parity, dynamic
  scene quality and kilometer-scale memory/latency behavior.
- DA3-Giant full-resolution inference or GPU throughput on this environment.
- Exact DBoW2, AMB3R-VO solver and VidMap W-AUC reproduction.
- Production stereo/LiDAR performance, calibrated live camera hardware behavior,
  IMU fusion, failure recovery or real relocalization reliability.
- Distributed GPU training, full DA3 training, teacher-recipe parity or useful
  independently trained checkpoints.
- Export of DA3 pretrained weights; C++ geometry/graph port; iOS/Android apps,
  accelerator delegates, quantization, thermal or battery measurements.

## Tested development versions

Python 3.12; PyTorch 2.14.0; NumPy 2.3.5; SciPy 1.17.0;
OpenCV headless 5.0.0.93; ONNX 1.23.0; ONNX Runtime 1.30.0;
pytest 9.1.1; Ruff 0.16.9. This is a development snapshot, not a universal
platform lock. DA3 declares NumPy<2 in its upstream package; use `uv sync --locked --extra da3 --extra train --extra export`
in a clean environment for a supported integration install.
The isolated development smoke used already-installed dependencies and is not
proof of compatibility across other DA3 optional features.


## Concurrent pipeline update

Validated on Python 3.12 using the locked development/train/export/benchmark
installation. Ruff lint/format and Pyrefly pass. The offline suite has **33 passing
tests**; **6 external-data/report tests are skipped** without their environment
variables. No pretrained TUM inference or GPU/mobile timing was rerun for this change.

New checks establish that tracking and next-window local mapping progress while
graph processing is explicitly blocked; outstanding work stays bounded with
backpressure; sequential and concurrent oracle trajectories/graph edges agree
across capacities 1–3; final tail mapping and idempotent shutdown work; mapping,
graph, and caller errors clean up both workers. Existing long-context, loop
verification, and RGB-D checks now run in both execution modes. These tests verify
orchestration correctness, not learned-model accuracy or hardware speedup.
