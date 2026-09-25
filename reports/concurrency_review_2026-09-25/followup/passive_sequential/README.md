# TUM RGB-D checkpoint benchmark

These are measured results from our independent implementation, not published AMB3R-SLAM results.

## Protocol

- Python 3.12.14; CPU; 2 PyTorch threads.
- DA3-Small in both frontend/backend roles; processing resolution 168 (long edge before patch-size rounding).
- RGB-only input: no sensor depth, calibration, or ground-truth poses are supplied to SLAM.
- Every tenth original RGB frame across each sequence; no accuracy-based frame selection.
- Execution mode and queue capacity are recorded per sequence below; mapping settings are in results.json.
- Each online/corrected trajectory gets its own full-trajectory Sim(3) alignment to motion-capture ground truth.
- One-to-one timestamp matching within 20 ms. RPE uses consecutive matched sampled frames, not a fixed one-second interval.
- FPS includes image loading, persistence, mapping, graph optimization and final flush; excludes model loading and plotting.
- BLAS/OpenMP thread limits match the PyTorch thread limit; shared development container, not dedicated hardware.
- One completed run per sequence; no confidence intervals or hardware real-time claim.

## Results

| Sequence | Frames | Matched | Online ATE (m) | Corrected ATE (m) | RPE trans. (m) | RPE rot. (deg) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| rgbd_dataset_freiburg1_xyz | 80 | 80 | 0.1418 | 0.1409 | 0.0748 | 3.268 | 0.495 |

All errors in the table are RMSE; lower is better. Sim(3) fitting removes global scale error and does not establish metric-scale accuracy.

## rgbd_dataset_freiburg1_xyz

Sampled 80 of 798 RGB frames over 26.34 s; 100.0% matched ground truth. Backend: 18 submaps, 68 edges. Push p50/p95: 841.2/8306.2 ms.

Execution: sequential; pending-window limit: 2; peak outstanding: 0; submission backpressure: 0.0 ms.

![Trajectory, absolute error and latency](rgbd_dataset_freiburg1_xyz/diagnostics.png)

[Metrics and configuration](rgbd_dataset_freiburg1_xyz/results.json) · [Corrected trajectory](rgbd_dataset_freiburg1_xyz/trajectory.tum) · [Online trajectory](rgbd_dataset_freiburg1_xyz/trajectory_online.tum) · [Latency samples](rgbd_dataset_freiburg1_xyz/latency.csv)

## Interpretation and limits

This first real-data baseline exercises pretrained inference and the full Python SLAM path on two sequences from one standard dataset. Sampling and a small CPU model make it a reduced benchmark, not a full-rate TUM evaluation. The paper uses a stronger backend; these results cannot be compared directly with its tables. Ground-truth alignment uses the complete run and is offline. Loops are enabled but successful loop closure is not guaranteed. No held-out-training-data claim is made for the upstream pretrained checkpoint.

The report retains both online and corrected metrics so backend regressions are visible. No accuracy threshold was tuned on these sequences. Automated checks enforce finite outputs, timestamp association, coverage, and consistency of saved metrics; they do not certify research-quality accuracy.

## Reproduce and audit

See [benchmark instructions](../../docs/BENCHMARKS.md). Exact checkpoint hashes, package versions, source hashes and invocation are in [environment.json](environment.json); each sequence includes an input list with SHA-256 image hashes. Raw datasets and model weights are downloaded separately.

## References

- Hengyi Wang and Lourdes Agapito. [AMB3R-SLAM: Kilometer-scale SLAM with Hierarchical Backend](https://arxiv.org/abs/2609.19518), 2026. Independent implementation; no author endorsement.
- Haotong Lin et al. [Depth Anything 3: Recovering the Visual Space from Any Views](https://arxiv.org/abs/2511.10647), 2025. Pretrained checkpoints and upstream inference.
- Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard, Daniel Cremers. [A Benchmark for the Evaluation of RGB-D SLAM Systems](https://cvg.cit.tum.de/data/datasets/rgbd-dataset), IROS 2012. TUM RGB-D data and motion-capture ground truth (CC BY 4.0).
