# LiveCodeBench-X

Multilingual LiveCodeBench benchmark ported from NeMo Skills'
`nemo_skills/dataset/livecodebench-x`.

## What Is Different From `livecodebench`

- Source dataset: `nvidia/Nemotron-Multilinugual-Eval-LCB`
- Languages: `de`, `es`, `fr`, `ja`
- Versions: `v5` and `v6` (same problems, different release filters; both are
  emitted into a single combined JSONL).
- Each row preserves:
  - `task_id`: LCB-canonical task id, joinable to upstream LCB.
  - `release_version`: `"v5"` or `"v6"` — used by metric stratification and
    by reviewers who want to subset to one LCB release.
  - `subset_for_metrics`: language code (mirrors Skills' field for downstream
    per-language metric breakdown).
  - `target_language`: language code (same value as `subset_for_metrics`,
    kept for symmetry with Skills' JSONL).
- Prompting mirrors Skills' `generic/default` behavior: the language-specific
  instruction prefix is baked into each row's `question`, and the prompt
  template (`benchmarks/prompts/generic_default.yaml`) is a passthrough.

## Verification

This benchmark reuses the existing `code_gen` resource server, unmodified.
`code_gen.verify()` extracts code from the model output via LCB's
`extraction_utils.extract_code(LMStyle.OpenAIChat)` and runs it against
`verifier_metadata.unit_tests` using LCB's own `testing_util.py` fork. Test
cases (public + private) are baked into each row's `verifier_metadata` at
prepare time by joining on `task_id` against the canonical
`livecodebench/code_generation_lite` (revision `refs/pr/7`) — the same data
source the existing monolingual `livecodebench/v5_2408_2502` and
`v6_2408_2505` Gym benchmarks already use.

## Data Preparation

```bash
ng_prepare_benchmark "+config_paths=[benchmarks/livecodebench-x/config.yaml]"
```

That writes `benchmarks/livecodebench-x/data/livecodebench-x_benchmark.jsonl`.

If you want English instructions instead of target-language instructions in the
prepared `question` field, run the script directly:

```bash
python benchmarks/livecodebench-x/prepare.py --prompt_language en
```

## Quickstart

```bash
ng_run "+config_paths=[benchmarks/livecodebench-x/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Then in another shell:

```bash
ng_collect_rollouts \
    +agent_name=livecodebench-x_code_gen_simple_agent \
    +input_jsonl_fpath=benchmarks/livecodebench-x/data/livecodebench-x_benchmark.jsonl \
    +output_jsonl_fpath=results/livecodebench-x/rollouts.jsonl \
    +num_repeats=4 +num_repeats_add_seed=true \
    "+responses_create_params={temperature: 1.0, top_p: 0.95, max_output_tokens: 16384}"
```
