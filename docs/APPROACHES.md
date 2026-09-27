# Comparing SLAM approaches

`amber_slam.experiments.run_experiment` is an additive, method-neutral Python API.
The existing AMB3R benchmark and its detailed stage/correction traces are unchanged.
This initial protocol supports **in-process monocular RGB**, TUM timestamps and
Sim(3) evaluation. It is not yet a stereo, RGB-D, IMU or subprocess benchmark.

Implement `SlamApproach`: `push(Frame)`, `finish() -> Trajectory`, `close()`.
The factory receives only an output path, not the dataset or ground truth.
Use original sampled timestamps and rigid camera-to-world poses. Omit lost poses;
coverage is reported. Never fill gaps using ground truth. Cleanup must be idempotent.
`finish` must drain all work before returning; `close` must also handle failure.
This is an API contract, not a sandbox against a malicious implementation.

```python
from pathlib import Path
from amber_slam.approaches import AmberApproach
from amber_slam.experiments import run_experiment

# Preload frontend/backend models and config as in benchmark_tum.py.
run_experiment(
    Path("data/tum/rgbd_dataset_freiburg1_room"), Path("reports/trial-001"),
    lambda output: AmberApproach(output, frontend, backend, config),
    approach="amb3r-slam", revision="<immutable git SHA>",
    config={"execution_mode": "concurrent"},
    environment={"checkpoint_hashes": checkpoint_hashes},
)
```

Replace the factory to test another implementation; external model dependencies
belong in optional extras, not the core runtime. No second production SLAM method
is bundled yet. The test adapter is a lifecycle fixture, not an accuracy baseline.

## Command-line entry point

Expose a trusted `create(output: Path, config: dict) -> SlamApproach` callable in
your own installed module. It can preload model weights and construct an
`AmberApproach`, or return another implementation of the interface. Then run:

```bash
python -m amber_slam.experiment_cli \
  --factory my_adapters:create --approach my-method --revision IMMUTABLE_SHA \
  --config method.json --environment machine.json \
  --data data/tum/rgbd_dataset_freiburg1_room --output reports/trial-001
```

`method.json` is a JSON object passed to the factory and recorded in results.
Use `execution_mode` in this object to distinguish sequential/concurrent trials.
The optional environment JSON should include checkpoint hashes, hardware and
dependency versions. Do not include secrets in either file: both are artifacts.
The supplied revision is recorded, not automatically verified against an external
implementation. Imports execute trusted Python code; never select an untrusted
factory. Use a new output directory and a fresh CLI process for each repeat.
Dataset/model downloads and hosted uploads are never implicitly requested by the
runner; a custom factory is responsible for its own setup behavior.

Model construction/downloads are excluded from replay. RGB decode, pushes and
final drain are included. Profile CPU/RSS/USS are process-wide; child processes
and GPU memory are not measured. Run one trial per fresh process, pin source and
checkpoint hashes, alternate method order, and repeat trials. Match input hashes,
hardware, sensor inputs, alignment and sampling before comparing results. Compare
accuracy alongside estimate coverage. Do not mix these setup-excluding timings
with the older AMB3R-specific runner, which includes SLAM state construction.

Results, resource traces, input hashes and environment metadata stay outside Git.
Export with `scripts/export_wandb.py reports/trial-001 --project amb3r-slam`;
offline is the default. Add `--mode online --entity YOUR_ENTITY` and provide
`WANDB_API_KEY` through the environment for upload. Never commit credentials.
The dashboard records approach/revision/protocol and accepts methods without
DA3 metadata, online trajectories or diagnostic images. This API emits a final
trajectory only; it does not mislabel final estimates as online estimates.
