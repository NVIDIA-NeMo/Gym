# MedXpertQA Text

The Text subset of [MedXpertQA](https://arxiv.org/abs/2501.18362): 2,450 test questions with ten options (A-J) across
Diagnosis (1,050), Treatment (746) and Basic Science (654), split into Reasoning (1,861) and Understanding (589).
`prepare.py` downloads `Text/test.jsonl` from
[TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA) at revision `7e7c465a`, checks its
SHA-256 and row count, and keeps every native field (`id`, `question`, `options`, `label`, `medical_task`,
`body_system`, `question_type`). The dev split is not used.

The protocol and grading are the paper's; see `resources_servers/medxpertqa/README.md`.

## Reproduce a published score

The upstream harness (`eval/model/api_agent.py`) decodes with temperature 0 and seed 0, and allows 2,048 tokens per
turn (8,192 for o1, o3, QwQ, QVQ and DeepSeek-R1). For a non-reasoning model:

```bash
gym eval prepare --benchmark medxpertqa_text
gym eval run --benchmark medxpertqa_text --model-type vllm_model --split benchmark \
    --model <model> --model-url <url> --temperature 0 --max-output-tokens 2048 \
    ++policy_model.responses_api_models.vllm_model.sampling_overrides.seed=0 \
    --output results/medxpertqa_text.jsonl
```

Published scores (paper, Table 5, Text accuracy) with the default `answer_format: trigger`:
LLaMA-3.3-70B-Instruct 24.49, DeepSeek-V3 24.16, Qwen2.5-72B-Instruct 18.90, Qwen2.5-32B-Instruct 15.06; from Table 4,
GPT-4o 30.37 and GPT-4o-mini 17.84. DeepSeek-R1 (37.76) was graded with the boxed variant:
`++medxpertqa_text_medxpertqa_resources_server.resources_servers.medxpertqa.answer_format=boxed`, 8,192 tokens per
turn and no system prompt (`--prompt-config benchmarks/prompts/eval/medxpertqa/zero_shot_cot_no_system.yaml`).
