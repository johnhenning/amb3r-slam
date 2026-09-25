# Research and dependency attribution

This repository independently implements ideas from **AMB3R-SLAM:
Kilometer-scale SLAM with Hierarchical Backend**, by **Hengyi Wang and
Lourdes Agapito**, University College London (2026).

- Paper: https://arxiv.org/abs/2609.19518
- Reviewed version: https://arxiv.org/html/2609.19518v1
- Authors' project: https://hengyiwang.github.io/projects/amber-slam
- Official repository: https://github.com/HengyiWang/amb3r-slam

The pretrained geometric models come from **Depth Anything 3: Recovering the
Visual Space from Any Views**, by Haotong Lin, Sili Chen, Jun Hao Liew,
Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, and Bingyi Kang (2025).

- Paper: https://arxiv.org/abs/2511.10647
- Official implementation/model links: https://github.com/ByteDance-Seed/Depth-Anything-3

The original researchers are credited for their algorithms, model architectures,
pretrained weights and published results. They are not presented as authors or
endorsers of this independent codebase. The differences and limitations of this
implementation are documented in `docs/FIDELITY.md` and `docs/VERIFICATION.md`.

The MIT license here applies to this repository's original code. External
software, model weights and datasets retain their respective licenses. No
external checkpoints or paper text are redistributed in this repository.
Citation metadata is provided in `CITATION.cff` and `CITATIONS.bib`.

TUM benchmark reports contain derived trajectories from the TUM RGB-D dataset
(Sturm et al., IROS 2012), licensed CC BY 4.0. The repository MIT license does
not replace the dataset license. See docs/BENCHMARKS.md for full attribution.
