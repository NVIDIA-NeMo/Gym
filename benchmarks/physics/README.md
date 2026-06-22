# PHYSICS

Open-ended physics QA benchmark from
[`desimfj/PHYSICS`](https://huggingface.co/datasets/desimfj/PHYSICS).
Ported from NeMo Skills' `physics` benchmark.

The default split here matches Skills' default `physics:N` benchmark — the
English subset (`language=="en"`) of the upstream `test` split. Each
problem is a free-form physics question whose answer can be a number,
expression, list, set, or option label.

## Prepare data

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/physics/config.yaml]"
```

`prepare.py` downloads the dataset, applies the same flatten-and-`\boxed{}`
transformation Skills uses for the multi-part answers, and writes
`data/physics_benchmark.jsonl` with one row per problem. Each row carries
`question`, `expected_answer`, plus per-row metadata (`domain`,
`difficulty`, `answer_type`, `language`, `solution`).

## Running servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/physics/config.yaml"
ng_run "+config_paths=[$config_paths]"
```

> Reasoning-model note: start the policy vLLM server with
> `--reasoning-parser deepseek_r1` (or the model-specific parser).
> That strips `<think>…</think>` at the model edge, so `\boxed{...}`
> extraction and the judge both see clean post-think text.

## Collecting rollouts

```bash
ng_collect_rollouts \
    +agent_name=physics_physics_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/physics/data/physics_benchmark.jsonl \
    +output_jsonl_fpath=results/physics_rollouts.jsonl \
    +num_repeats=4 \
    +num_repeats_add_seed=true
```

## Verification

Two-stage, matching NeMo Skills' `physics`:

1. **Symbolic check** via `math-verify` on the `\boxed{...}` answer
   (inherited from `math_with_judge`).
2. **LLM judge fallback** when symbolic fails. The benchmark binds to the
   `physics_judge` resource server, which is a `math_with_judge` subclass
   carrying NeMo Skills' physics judge prompt
   (`nemo_skills/prompt/config/judge/physics.yaml`) verbatim. The judge is
   asked whether the model's full output is `[Correct]` or `[Incorrect]`
   against the expected answer.

The judge model is wired via
`resources_servers/physics_judge/configs/judge_openai.yaml`, which
defaults to `openai/gpt-oss-20b` on `https://integrate.api.nvidia.com/v1`
and reads its API key from `NVIDIA_API_KEY`. Override the
`judge_model.responses_api_models.openai_model.openai_*` fields on the
CLI to point it at a different OpenAI-compatible endpoint.

## Metrics

Inherits the `math_with_judge` metric set from `physics_judge`, with a
per-domain breakdown layered on top:

- `pass@1[avg-of-k]/symbolic_accuracy`, `pass@k/symbolic_accuracy`
  (math-verify pass rate)
- `pass@1[avg-of-k]/judge_accuracy`, `pass@k/judge_accuracy`
  (judge pass rate, on rollouts that fell through to the judge)
- `majority@k/...` (majority-vote accuracy, requires `extracted_answer`)
- `<domain>/pass@1[avg-of-k]/...` (per-domain pass@k — `mechanics/`,
  `thermodynamics/`, `electromagnetism/`, etc.)
