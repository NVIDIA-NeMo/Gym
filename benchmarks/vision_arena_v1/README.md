# VisionArena-Bench v0.1

486 evaluated prompts from the 500 in [VisionArena-Bench](https://huggingface.co/datasets/lmarena-ai/vision-arena-bench-v0.1); 14 lack usable baseline answers. The prompt adapts [the paper](https://arxiv.org/pdf/2412.08687#page=19) with LMArena v3 verdicts.

Preparation downloads pinned prompts/images from Hugging Face and corresponding baseline answers from GitLab. No upstream prompts or images are stored in the GitLab artifact. Requires `huggingface_hub`, `pyarrow`, and GitLab credentials; existing prepared validation files are reused.

Win = 1, tie/both-bad = 0.5, loss = 0. No style control. Truncated and reasoning-only policy outputs score zero; failed judge pairs are excluded. Baseline self-comparisons are excluded. Output-token limits include reasoning.

Sources: [VisionArena paper](https://arxiv.org/abs/2412.08687) and [Hugging Face dataset](https://huggingface.co/datasets/lmarena-ai/vision-arena-bench-v0.1). The dataset card lists MIT and additional [dataset terms](https://huggingface.co/datasets/lmarena-ai/vision-arena-bench-v0.1#lmarena-visionarena-dataset-license-agreement), including transfer restrictions; reference-model outputs may carry additional terms. These materials are not relicensed under Gym's Apache-2.0 license.
