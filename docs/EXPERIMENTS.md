# Experiment tracking

Use W&B as the experiment dashboard. Git contains measurement code, configuration,
and tests; generated reports, raw profiles and SDK output are ignored. Existing
benchmark measurements can be imported without rerunning model inference.

## What each run contains

One completed replay maps to one W&B run, grouped by experiment cohort. Filter by
dataset, execution mode, waiting policy, resolution, checkpoint and thread count.
Compare replay time, process CPU seconds, mean CPU utilization, sampled RSS/USS,
online/corrected ATE and RPE, stage costs, faults, I/O and context switches.

Resource charts use the captured replay timeline, not the uploader's CPU/memory.
Tracking latency has its own frame axis. Stage wall sums overlap and are not an
additive runtime breakdown. The trajectory diagnostic image and a versioned
`slam-benchmark` artifact retain the full results, input/source/checkpoint hashes,
trajectories, stage spans, correction timeline when present, and environment.
Model weights, original RGB images and large reconstruction caches are excluded.

## Local use

```bash
uv sync --locked --extra benchmark --extra tracking
uv run --no-sync python scripts/export_wandb.py reports/my_experiment --mode offline
```

Offline is the default: this creates local W&B records without authentication or
network upload. To publish, authenticate through `wandb login` or an environment
variable and give the intended destination:

```bash
uv run --no-sync wandb login
uv run --no-sync python scripts/export_wandb.py reports/my_experiment \
  --entity YOUR_ENTITY --project amb3r-slam --mode online
```

Keep API keys out of source and command-line arguments. `WANDB_ENTITY` and
`WANDB_PROJECT` can supply defaults. Use `--group` to name a comparison cohort;
keep different machines or protocols in different groups. Source measurements
are never rewritten. Stable run IDs derive from result/profile/environment
content. A local receipt skips repeated completed exports to the same destination;
W&B rejects an existing online run ID rather than silently duplicating its data.

Logging runs after measured replay, so upload time and SDK processes do not
pollute benchmark timings. Retain the original report directories as the portable
source of truth: after moving a report archive to another machine, re-run the
exporter on the extracted reports rather than relying on relocated SDK caches.
Historical aggregate-only summaries are not synthesized into fake individual runs.

## GitHub Actions

The full-route workflow retains reports as a GitHub Actions artifact for 90 days.
It also exports each completed run to W&B when these repository settings exist:

- Secret: `WANDB_API_KEY`
- Variable: `WANDB_ENTITY`
- Optional variable: `WANDB_PROJECT` (default `amb3r-slam`)

Without those settings, it creates offline W&B records and preserves the original
reports for later import. The workflow is manually triggered to avoid rerunning
expensive experiments on every code change. It never commits result files.
Configure the project visibility/access policy in W&B before publishing.

An already-running workflow uses its original checkout; its artifact can be
imported with the same exporter after completion.
