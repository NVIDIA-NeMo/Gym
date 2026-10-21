# Multimodal RewardBench 1

[MMRB1](https://github.com/facebookresearch/multimodal_rewardbench): **4,711 pairs**, one ordering per pair. Judges evaluate saved text answers, including reasoning, alongside the question image.

Use `genrm.yaml` for GenRM (uses endpoint's chat template) or `llm_as_a_judge.yaml` for an LLM as a judge. Preparation downloads the published judging prompt from the same upstream revision as the data downloader, extracts it without executing the source, and caches it under `.cache/`.

Preparation downloads the pinned upstream script and release records, retrieves third-party images, and builds the selected input format. Downloads are cached under `.cache/` beside `prepare.py`; existing validation files are reused. The first run can be large and requires network access, Git (for PAIRS), and `uv`. The downloader uses the same Python version with isolated `datasets==3.1.0` dependencies; Gym's packages are unchanged. Conversion requires `PyYAML` in the Gym environment.

`overall/accuracy` measures agreement with human preferences; invalid verdicts score zero. Category/source breakdowns are included. No ties or style control.

VisIT-Bench uses its pinned Parquet export to avoid an upstream image-folder metadata incompatibility; record ordering is preserved.

Sources: [MMRB1 paper](https://arxiv.org/abs/2502.14191) and [Meta's reference implementation and release data](https://github.com/facebookresearch/multimodal_rewardbench). Upstream materials are [CC BY-NC 4.0](https://github.com/facebookresearch/multimodal_rewardbench/blob/main/LICENSE); third-party images and model outputs may carry additional terms. These materials are not relicensed under Gym's Apache-2.0 license.
