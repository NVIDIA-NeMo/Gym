# IMO ProofBench

Math benchmark from [google-deepmind/superhuman](https://github.com/google-deepmind/superhuman/) — IMO-style **proof** problems with reference solutions and per-problem grading rubrics. Ported from NeMo Skills' `imo-proofbench`.

## Prepare data

`prepare.py` downloads `proofbench.csv` from the exact pinned superhuman commit Skills uses (`c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8`) and writes `data/imo_proofbench_benchmark.jsonl`. Each row carries `problem`, `reference_solution`, `rubric`, plus `problem_id` / `category` / `level` / `expected_answer` / `source`.

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/imo_proofbench/config.yaml]"
```

## Run servers

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/imo_proofbench/config.yaml"
ng_run "+config_paths=[$config_paths]" \
    +judge_base_url=https://generativelanguage.googleapis.com/v1beta/openai \
    "+judge_api_key=$GEMINI_API_KEY" \
    +judge_model_name=gemini-2.5-pro
```

## Collect rollouts

`prepare.py` writes raw problem fields and does **not** bake
`responses_create_params.input` into each row. Pass `+prompt_config` so
`ng_collect_rollouts` materialises the chat messages from
`prompts/default.yaml` at rollout time:

```bash
ng_collect_rollouts \
    +agent_name=imo_proofbench_imo_proofbench_judge_simple_agent \
    +prompt_config=benchmarks/imo_proofbench/prompts/default.yaml \
    +input_jsonl_fpath=benchmarks/imo_proofbench/data/imo_proofbench_benchmark.jsonl \
    +output_jsonl_fpath=results/imo_proofbench_rollouts.jsonl \
    +num_repeats=4 \
    +num_repeats_add_seed=true
```

## Verification

LLM judge using the IMO 0-7 rubric prompt — byte-identical to NeMo Skills' `nemo_skills/prompt/config/judge/imo_proofbench.yaml`. The judge is asked for `<points>N out of 7</points>` and a rollout is correct iff `N >= 6` (matches Skills' `is_correct_judgement` Format-3 rule).

The benchmark binds to the `imo_proofbench_judge` resource server, which calls the judge via `/v1/chat/completions` (`use_chat_completions_for_judge: true`). Default endpoint is configurable per-run via `judge_base_url` / `judge_api_key` / `judge_model_name`. Both Google's Gemini OpenAI-compatibility layer (`generativelanguage.googleapis.com/v1beta/openai/`) and NVIDIA's `integrate.api.nvidia.com` host Gemini-2.5-Pro and work as drop-in judges.

> **Reasoning-model note:** start the policy vLLM server with `--reasoning-parser deepseek_r1` (or the model-specific parser) so `<think>…</think>` is stripped at the model edge. Without it, the judge prompt is polluted with chain-of-thought and truncated rollouts (no closing think tag) leak full reasoning into the predicted answer.

## Metrics

`compute_metrics()` returns the standard `compute_pass_majority_metrics` set plus per-category/per-level subset breakdowns:

- `pass@1[avg-of-k]/judge_correct`, `pass@k/judge_correct`, `majority@k/judge_correct` — primary parity metrics matching Skills.
- `pass@1[avg-of-k]/judge_score_{0,1,6,7}` — per-rubric-grade rate distribution.
- `pass@1[avg-of-k]/no_answer` — fraction of rollouts that never closed `</think>`.
- `<Category>/pass@1[avg-of-k]/judge_correct` — Algebra / Combinatorics / Geometry / Number theory.
- `<Level>/pass@1[avg-of-k]/judge_correct` — IMO-easy / IMO-medium / IMO-hard / pre-IMO.
