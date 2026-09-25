# Sources and provenance

Reviewed on 2026-09-25:

- SLAM paper: https://arxiv.org/html/2609.19518v1
- Project: https://hengyiwang.github.io/projects/amber-slam
- Authors' repository: https://github.com/HengyiWang/amb3r-slam
  (README/.gitignore at inspection; no implementation copied).
- DA3 paper: https://arxiv.org/html/2511.10647v1
- DA3 source: https://github.com/ByteDance-Seed/Depth-Anything-3
  reviewed commit `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`.
- DA3 model cards: upstream README links checkpoint-specific licenses.
- ORT mobile: https://onnxruntime.ai/docs/get-started/with-mobile.html

No paper text, external checkpoints or external repositories are bundled.
DA3 is an installed dependency; the geometric algorithms and experimental model
are independently authored. Unspecified hyperparameters are our choices.
