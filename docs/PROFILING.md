# Real-data profiling

The profiling work is stacked on the concurrency PR. It adds measurement to the
TUM benchmark runner; normal `amber run` and `SlamSystem` use have no profiler.

```bash
uv sync --locked --extra da3 --extra benchmark
uv run --no-sync python scripts/profile_tum_matrix.py --output reports/my_diagnostics
uv run --no-sync python scripts/report_profile_matrix.py reports/my_diagnostics
```

The matrix runs four fresh processes, one at a time: default sequential, default
concurrent, passive concurrent, passive sequential. It fixes PyTorch/OpenMP/BLAS/
MKL thread limits to two and resets inherited waiting overrides; passive trials
set only `OMP_WAIT_POLICY=PASSIVE`. Each run uses the same pinned DA3-Small
checkpoint, TUM freiburg1_xyz, 80 original RGB frames sampled every tenth frame,
resolution 168, window 12, view-selection stride 3 and two pending windows.
This is one trial per cell: a sensitivity check, not a variance estimate.

## Measurement contract

- Linux current process and all threads; subprocesses excluded.
- Timer starts after checkpoint download/loading and input hashing. RGB decode,
  persistence, tracking, mapping, optimization, backpressure and final drain are
  included. Plotting, metric evaluation and report writes are excluded.
- CPU seconds are user + system time; 100% CPU means one occupied core.
- RSS/USS include both model instances and allocator caches. Peaks are sampled at
  a nominal 200 ms; actual gaps are recorded. They are not allocator high-water
  marks. Concurrent RSS/USS reads are not an atomic memory snapshot.
- Resource metrics include OS threads (including idle pools), storage I/O,
  syscall counts, page faults and voluntary/involuntary context switches.
- Cgroup CPU-stat deltas measure container-wide quota throttling, not all host
  scheduling or CPU-frequency effects.
- Stage spans cover tracking, local mapping, graph integration and optimization.
  Spans overlap; integration contains optimization, so they must not be added
  into an end-to-end runtime. Stage caller-thread CPU excludes native workers.
- Profiling overhead is included in every cell. No GPU/VRAM, power, thermal or
  hardware-performance-counter measurement is claimed.

## Artifacts and checks

Each trial retains `results.json`, input/checkpoint/source hashes, trajectories,
`profile/summary.json`, `profile/samples.csv` and `profile/stages.json`.
The report generator checks identical inputs/settings/source across the matrix,
recomputes metrics and compares online/corrected ATE against independent `evo`.
Large frame/submap caches are not committed; the experiment recreates them.

[Recorded experiment and recovery disclosure](../reports/concurrency_review_2026-09-25/README.md).
The earlier six-run summary was recovered from retained tool output after
workspace maintenance removed its raw traces. It must not be treated as raw
trace evidence or combined with fresh runs to estimate variance.
