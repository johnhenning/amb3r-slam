# Real-data concurrency and runtime-variance investigation

The original six-run CPU experiment completed, but workspace maintenance removed
its raw traces before they were committed. [The recovered summary](historical_summary.json)
contains only measurements visible in retained tool output. No missing traces or
plots are reconstructed as if they were original measurements. New, separately
labeled diagnostic runs will retain their raw artifacts here.

Original experiment: DA3-Small in both roles, TUM freiburg1_xyz, 80 RGB frames
sampled every tenth frame, resolution 168, window 12, three fresh processes per
mode, numerical-library thread limits 2, no GPU.

| Historical metric (median) | Sequential | Concurrent |
|---|---:|---:|
| Replay time | 108.836 s | 107.428 s |
| Runtime range | 104.687–120.057 s | 105.827–142.415 s |
| CPU time | 232.565 core-s | 314.098 core-s |
| CPU usage (100% = one core) | 213.669% | 282.500% |
| Sampled peak RSS | 1667.266 MiB | 1917.523 MiB |
| Sampled peak USS | 1663.188 MiB | 1913.438 MiB |
| Corrected ATE | 0.141 m | 0.141 m |
| Online ATE | 0.142 m | 0.160 m |

A 1.3% median runtime difference amid overlapping ranges does not establish a
throughput improvement. Concurrent CPU time was about 35% higher and peak RSS
about 15% higher. This does not predict GPU/mobile performance.

## Why downloading is not the explanation

The replay timer starts after checkpoint download and construction of both models.
It ends after the final backend drain, before plots and evaluation. All six runs
reported zero storage-read bytes and zero major page faults during replay.
One-time setup/downloads contributed to total job duration, not these timings.

Concurrent runs used almost the same model workload: 123 calls and 632–633 input
views. Sequential runs used 123 calls and 597 views. Delayed corrections can leave
an older anchor in the frontend context, increasing some calls from three views
to four. That explains a workload difference between modes, but not the large
variation between concurrent trials with nearly identical view counts.

Sequential runtime also varied despite identical trajectories and call/view
counts. Native thread waiting, cache/memory contention and host scheduling remain
candidate causes. The original telemetry did not isolate them. The follow-up adds
stage spans and a separate OMP_WAIT_POLICY=PASSIVE comparison; it is diagnostic,
not a parameter-tuned replacement for the historical results.

## Fresh follow-up

[Stage-level matched diagnostics](followup/README.md) use one fresh process for each
of four policy/mode combinations. These are new measurements after the workspace
restart, with full retained traces. They are not pooled with the historical runs.

### Fresh follow-up interpretation

Default concurrent replay took 121.854 s versus 159.174 s sequential (1.31x), at 391.787 versus 338.117 CPU core-seconds and 1706 versus 1542 MiB sampled peak RSS. Passive waiting changed concurrent replay to 119.246 s (-2.1%) and CPU time to 349.798 core-seconds (-10.7%); sequential replay changed to 161.483 s (+1.5%) and CPU time to 323.210 (-4.4%). This is consistent with avoidable waiting overhead, but single trials do not establish causality or reduced variance. The application default remains unchanged.

All four runs recorded zero cgroup throttling. Only the first run recorded storage reads (11.86 MiB) and major faults (8); the final sequential run was still similarly slow with neither. Downloads cannot explain replay timing because loading/downloading precedes the timer. Concurrent stages overlap but individual stage durations increase, consistent with resource contention; concurrent inference also processes 632 views versus 597 sequential despite identical 123 model-call counts. Host scheduling and frequency effects remain unmeasured. Repeated randomized warm-cache runs on an otherwise idle dedicated host would be needed to isolate the remaining variance.

Memory figures are sampled, not exact high-water marks. psutil reads RSS and USS through separate OS interfaces; concurrent allocation changes can produce a USS sample/peak above the RSS sample/peak (observed in default concurrent), so these should not be treated as a simultaneous memory accounting identity.
