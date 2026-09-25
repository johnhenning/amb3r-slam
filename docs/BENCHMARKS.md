# Standard-dataset benchmarks

The first baseline uses two real sequences from the **TUM RGB-D benchmark**:
`freiburg1_xyz` (mostly translation) and `freiburg1_desk` (desk exploration).
These are two sequences from one dataset, not evidence across multiple datasets.
Our KITTI converter is available separately; no KITTI results are claimed here.

Read the [measured report](../reports/tum_cpu_2026-09-25/README.md). It contains
trajectory plots, position-error curves, latency plots, ATE/RPE, coverage, and
links to the underlying trajectories and JSON metrics.

## Run the CPU baseline

Use Python 3.12, the committed uv lock, and a new output directory. The DA3 extra
includes `addict`, which upstream imports but omits from its declared dependencies.
Upstream xFormers/CUDA and Gaussian-splatting warnings do not imply those paths
are used by this CPU depth/pose benchmark.

```bash
uv sync --locked --extra da3 --extra train --extra export --extra benchmark
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  uv run --no-sync scripts/benchmark_tum.py \
  --download --data data/tum --output reports/tum_cpu_new \
  --checkpoint depth-anything/DA3-SMALL \
  --revision e08cab65ca0ec38e7826075418411ab90cab4da3 \
  --resolution 168 --threads 2 --no-async-backend
```

`--download` explicitly downloads approximately 0.8 GB of compressed TUM data
from the official host. Archives are checked against recorded SHA-256 hashes and
extracted with Python's safe data filter. An existing extracted dataset can be
used without `--download`. Model weights are also cached on first use. Pass a
local checkpoint directory to run offline; config and weight hashes are saved.
Dataset images, archives, checkpoints and runtime caches are not committed.

The runner samples every tenth **original** RGB frame over the entire sequence,
without filtering by ground-truth availability. It supplies only RGB and capture
timestamps to SLAM. It keeps the original motion-capture timestamps for evaluation
and performs one-to-one association within 20 ms. It never supplies sensor depth,
calibration or ground-truth camera poses to monocular inference.

DA3-Small is used in both roles at processing resolution 168 (126×168 for these
images after patch rounding). Mapping uses window 12, view-selection stride 3,
long-context constraints and loop candidates. The archived baseline used synchronous execution, with
explicit CPU thread limits, to make backend correction scheduling repeatable.
This is a reduced CPU baseline, not the paper's Small/Giant configuration.

## Metrics and timing

- **ATE RMSE (m):** position error after a single full-trajectory Sim(3) fit.
- **RPE translation RMSE (m) / rotation RMSE (degrees):** relative pose error
  between consecutive matched sampled frames, after that same global fit.
  The interval is variable, roughly one third of a second; it is not 1-second RPE.
- **Coverage:** dense ground-truth and estimated-trajectory coverage are reported
  separately. Low dense-GT coverage is expected for a deliberately subsampled run;
  the report also gives the fraction of sampled RGB frames matched to ground truth.
- **Online versus corrected:** each trajectory is aligned independently. Online
  poses are what a consumer originally received; corrected poses include mapping
  updates and the final flush.
- **FPS:** replay time includes RGB decoding, disk caching, mapping, graph
  optimization and final flush. Model initialization, download, checksumming,
  evaluation and plotting are excluded. Push latency includes scheduled mapping,
  but final-flush work is represented only in end-to-end time.

Sim(3) removes global scale ambiguity; these metrics do not measure absolute
metric-scale recovery. Shared CPU timings are diagnostic, not mobile or real-time
claims. This first campaign uses one completed run per sequence and does not tune
accuracy thresholds on these scenes.

## Test the actual data and saved runs

```bash
TUM_DATA_ROOT=data/tum \
AMBER_BENCHMARK_REPORT=reports/tum_cpu_2026-09-25 \
  uv run --no-sync pytest tests/test_benchmarks.py -q
```

These tests validate real-file ingestion, timestamp association, complete sampled
trajectory output, rotation validity, finite metrics and latency, report artifacts,
and metric recomputation. ATE is independently cross-checked against `evo`, supplied
by the DA3 extra. Missing environment variables produce explicit skips; no network
or pretrained inference is hidden inside ordinary unit tests.

The full checkpoint run is an explicit integration benchmark. A manual GitHub
Actions workflow runs it and uploads its report; normal CI runs fast offline tests,
Ruff and Pyrefly. Numeric results are retained even when accuracy is poor. Passing
the integrity checks does not mean the published paper has been reproduced.

## Attribution

TUM RGB-D: Jürgen Sturm, Nikolas Engelhard, Felix Endres, Wolfram Burgard and Daniel
Cremers, *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012.
[Dataset and licensing](https://cvg.cit.tum.de/data/datasets/rgbd-dataset),
[sequence descriptions](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download).
Data is CC BY 4.0; derived ground-truth trajectories in reports retain that credit.
See [CITATIONS.bib](../CITATIONS.bib) for the AMB3R-SLAM and Depth Anything 3 papers.


## Concurrent versus sequential execution

The runner now defaults to concurrent tracking, local mapping, and graph processing.
Use separate output directories to compare modes on the same checkpoint and data:

```bash
uv run --no-sync python scripts/benchmark_tum.py --data data/tum \
  --output runs/tum_concurrent --resolution 168 --threads 2 --max-pending-windows 2
uv run --no-sync python scripts/benchmark_tum.py --data data/tum \
  --output runs/tum_sequential --resolution 168 --threads 2 --no-async-backend
```

`statistics` records execution mode, outstanding-window capacity/peak, and total
submission backpressure time. Compare end-to-end FPS (including final drain),
ATE/RPE, and online as well as corrected trajectories. Push latency alone omits
outstanding work. The archived TUM report remains a sequential baseline; it is
not evidence of concurrency speedup. No new pretrained-data benchmark was run
for this scheduling change.
