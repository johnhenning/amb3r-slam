# Implementation and reproduction plan

## Current scope decision
Use Depth Anything checkpoints first, per the latest request. Foundation-model
reproduction is future work. The experimental training path is optional.

## Goal and boundaries
Implement the SLAM orchestration independently from arXiv:2609.19518v1.
Use DA3 pretrained weights as the geometric prior; do not pretend to retrain a
foundation model from the SLAM paper (it provides no such training recipe).
Add an explicitly experimental, trainable compact student for export research.
Do not equate passing synthetic tests with reproducing paper accuracy.

## Work packages and acceptance criteria
1. Geometry: Sim(3) exp/log/inverse, robust alignment, interpolation, metric
   scale estimation. Accept round trips, known transforms, and outlier tests.
2. Backend: overlapping local windows, span-1/2 constraints, sparse joint
   reconstruction, geometric loop verification, baseline-whitened robust graph
   optimization and correction fusion. Accept a synthetic closed-loop test and
   decreasing graph cost; document alternative retrieval and alignment choices.
3. Frontend and runtime: anchor/recent-frame inference, scale continuity,
   correction feedback, bounded image cache and capture queue, asynchronous
   backend, recording/replay and latency reports. Accept finite trajectories,
   preserved timestamps, graceful shutdown and surfaced worker failures.
4. Training: independent compact multi-view student, teacher cache generation,
   masked depth/pose/intrinsics losses, sequence-disjoint validation, checkpoints,
   deterministic seeds, resume. Accept finite gradients and a tiny overfit test.
5. Evaluation: timestamp association, SE(3)/Sim(3) alignment, ATE, RPE, error
   threshold AUC, explicit window-AUC convention, trajectory exports. Accept
   analytic metric tests and no fabricated benchmark results.
6. Deployment: fixed-shape ONNX export, Python ORT parity and latency test,
   C++ inference adapter, documented ABI and camera conventions. Accept export
   parity; native compilation and phone profiling are separate hardware gates.
7. Reproduction campaigns (requires data/weights/GPU): TUM smoke sequence, KITTI
   00/05, VBR/Oxford; then all nine benchmarks and ablations. Freeze upstream
   revisions, weights checksums, splits, calibration and configs per campaign.

## Unspecified choices to expose in configuration
Window size n, temporal stride, long-context K/M, confidence and overlap gates,
loop exclusion interval, voxel size, correction bounds, robust loss threshold,
scale inlier tolerance. Defaults are engineering choices, not author settings.
Our retrieval uses ORB bag-of-words with a user-trained vocabulary (optional),
or brute ORB matching, not the authors' DBoW2 implementation. Exact VidMap
W-AUC and original AMB3R-VO alignment must be cross-checked before table claims.

## Mobile milestone
First validate the desktop reproduction. Train/distill the compact student on
representative motion, indoor/outdoor lighting and dynamic scenes; evaluate it
separately. Port graph/geometry after golden-vector parity, then connect
AVFoundation/CameraX through a C ABI. Profile 10-minute sessions on named phones
for p50/p95 latency, frame age, memory, energy and thermal throttling. Quantize
only after float parity and calibration-set accuracy checks. The giant backend
may require a server or a substantially smaller replacement.
