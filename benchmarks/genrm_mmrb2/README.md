# Multimodal RewardBench 2

[MMRB2](https://arxiv.org/abs/2512.16899), **reasoning with text-only candidate answers**: **350 pairs / 700 judgments**, using both orderings. Question inputs still contain images; candidate reasoning and final answers are evaluated. Other tasks/modalities are not supported.

Use `genrm.yaml` for GenRM (uses endpoint's chat template) or `llm_as_a_judge.yaml` for an LLM as a judge. Preparation downloads the published reasoning prompt from repository revision `7ff0552cf298441af5ebbabea634b682ef04afee`, extracts it without executing the source, and caches it under `.cache/`. The Hugging Face data revision remains pinned separately.

Preparation automatically downloads the pinned Hugging Face reasoning split and builds the selected input format. Downloads are cached; existing validation files are reused. Requires `huggingface_hub`, `pyarrow`, and `Pillow` in the Gym Python environment.

`overall/accuracy` measures agreement with human preferences; invalid verdicts score zero. Source/order breakdowns are included. No ties or style control.

Sources: [Meta's reference implementation](https://github.com/facebookresearch/MMRB2) and [Hugging Face dataset](https://huggingface.co/datasets/rl-research/multimodal-rewardbench-2). Upstream materials are [CC BY-NC 4.0](https://github.com/facebookresearch/MMRB2/blob/main/LICENSE); third-party images and model outputs may carry additional terms. These materials are not relicensed under Gym's Apache-2.0 license.
