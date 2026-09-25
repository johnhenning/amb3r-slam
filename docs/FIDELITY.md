# Paper-to-code mapping and reproduction limits

Primary source: https://arxiv.org/html/2609.19518v1

| Component | Code | Status / explicit choice |
|---|---|---|
| Compact frontend | `frontend.py` | Anchor plus two recent frames and incoming view; depth-ratio scale fit |
| Foundation inference | `adapters.py` | DA3 public API; Small/Giant defaults; middle reference reordered explicitly |
| Local maps | `backend.py` | n/3 scheduling, confidence selection, endpoint inclusion, pose interpolation |
| Span-2 constraints | `align_submaps`, `process` | Dense shared-image alignment; orientation/baseline fallback |
| Sparse long context | `_joint_edge`, `process` | Sparse joint inference, confidence/voxel gates, distant edges |
| Loop proposals | `retrieval.py` | ORB matching or binary vocabulary; **not DBoW2** |
| Loop verification | `_joint_edge` | Joint reconstruction, voxel occupancy and correction bounds |
| Pose graph | `graph.py` | Sim(3), fixed first node, baseline whitening, block Huber; SE(3) mode |
| Corrected poses | `trajectory`, `apply_update` | Weighted overlap fusion and delayed frontend correction |
| RGB-D/stereo | `modalities.py`, backend | Median scale, inlier gate, SGBM for rectified stereo |
| LiDAR | `modalities.py`, backend | Camera-registered scans, local ICP, loop refinement and RMSE weights; **not KISS-ICP** |
| Trajectory evaluation | `evaluation.py` | ATE/RPE/translation AUC; window-AUC helper is a documented alternative |

The first implementation follows the system design; it is not a bit-for-bit
replica. Exact robust alignment and pose weighting from AMB3R-VO, DBoW2 retrieval,
original hyperparameters, dataset preprocessing, and benchmark evaluator parity
remain reproduction checks. Independent default thresholds are in `SlamConfig`.

The frontend scale solver uses aligned anchor depth ratios, not a copied
AMB3R-VO solver. The graph runs on submap origins and propagates corrections;
the paper uses keyframe terminology that needs comparison to released source.
Long-context processing connects eligible earlier submaps to the current map,
not every possible distant pair. LiDAR scale calibration additionally projects
scans into camera depth before visual constraint construction. These choices
must be ablated rather than treated as identical to the authors' implementation.

## Validation campaign

1. Freeze DA3 source SHA, checkpoint revision/hash, Python dependencies and GPU.
2. Run a calibrated TUM sequence; inspect axes, scale and timestamp coverage.
3. Compare online and corrected trajectories. Confirm loops reduce drift without
   jumps caused by false positives; inspect edge metadata and point clouds.
4. Evaluate KITTI, VBR and Oxford Spires using the same input subsampling and
   trajectory alignment as the reference implementation.
5. Run ablations for loops, long context and whitening using identical inputs.
6. Extend to LaMAR, CroCoDL, ETH3D, EuRoC and Bonn with dataset-specific adapters
   and official benchmark protocols. The generic manifest supports those inputs;
   native converters for these datasets are not included yet.
7. Report quality, completion rate, coverage, p50/p95 latency, memory and compute
   setup. Do not compare synthetic-fixture metrics to paper tables.
