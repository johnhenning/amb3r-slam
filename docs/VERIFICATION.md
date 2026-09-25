# Verification report

Date: 2026-09-25. CPU-only development environment; CUDA unavailable.

## Completed

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
