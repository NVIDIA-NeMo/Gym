# m_arena_hard

Gym implementation of
[m-ArenaHard](https://huggingface.co/datasets/CohereLabs/m-ArenaHard),
the multilingual extension of Arena-Hard v0.1 published by CohereLabs.

## What it tests

Hard, open-ended user prompts translated into many languages. Each
candidate rollout is judged pairwise (both A↔B orderings) against a
fixed baseline answer via an LLM judge. See
[`resources_servers/arena_judge`](../../resources_servers/arena_judge/README.md)
for the judging protocol and metric details.

The HF dataset only ships `{question_id, cluster, category, prompt}`
per language config — there is **no built-in baseline answer**. To
judge end-to-end you must supply your own baseline answers via
`--baseline-file` (see below); this matches NeMo Skills, which
generates the baseline with a separate `ns generate` run and joins
it back into the prepared JSONL.

## Data

Runtime download only — benchmark JSONL is not committed. Run
[`prepare.py`](prepare.py) (or `ng_prepare_benchmark`) to populate
`data/m_arena_hard_benchmark.jsonl`. The prepare script:

1. Calls `datasets.load_dataset("CohereLabs/m-ArenaHard", lang, split="test")`
   for every language config.
2. Emits one row per `(language, question_id)` with `uid`,
   `question`, `language`, `category` (always `"hard_prompt"`),
   `original_category`, `cluster`, and `subset_for_metrics: <lang>`.
3. If `--baseline-file <path.jsonl>` is supplied, joins by
   `(language, question_id)` and emits `baseline_answer` from
   `generation`. Otherwise `baseline_answer` is set to `""` and a full
   `arena_judge` run will not be useful (the judge needs both
   answers). Use the no-baseline path only for question-set inspection
   or prepare-output parity checks.

Loading the HF dataset requires a Hugging Face token — set
`HF_TOKEN` in your environment (or `huggingface-cli login`) before
running `prepare.py`.

## Example usage

```bash
# Prepare benchmark data (no baseline -> baseline_answer is empty string)
ng_prepare_benchmark "+config_paths=[benchmarks/m_arena_hard/config.yaml]"

# Or call prepare.py directly with a baseline JSONL
python benchmarks/m_arena_hard/prepare.py --baseline-file path/to/baselines.jsonl

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/m_arena_hard/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=m_arena_hard_arena_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/m_arena_hard/data/m_arena_hard_benchmark.jsonl \
    +output_jsonl_fpath=results/m_arena_hard_rollouts.jsonl \
    +num_repeats=4
```

## Metrics

The headline number is the **Arena-Elo win-rate (%) vs baseline**,
computed by the `arena_judge` resources server as MLE logistic
regression over the pairwise battles with a 100-round bootstrap 95% CI.
Emitted keys:

- `arena_elo/score` — overall win-rate (0-100)
- `arena_elo/ci_lower` / `arena_elo/ci_upper` — bootstrap percentile CI bounds
- `arena_elo/invalid_scores` — count of judge calls that produced no
  parseable verdict

The server also emits pass@k / pass@1[avg-of-k] / majority@k for a
verdict-type decomposition (`wins`, `strict_wins`, `ties`, `losses`,
`double_wins`, `invalid_gen_base`), so a single run gives both the
Arena-Elo headline and a rollout-level verdict distribution without
extra post-processing.

## Generation sanitization

This benchmark sets `sanitize_generations: true` on the inherited
`arena_judge` resources server (see `config.yaml`) so the judge scrubs
UTF-8 surrogate halves and embedded NULs out of multilingual rollouts
before serializing them into the judge prompt. Mirrors Skills'
`++sanitize_generations=true` for the multilingual variant.

## Deferred follow-ups

- **Per-language Arena-Elo aggregation.** Rows already carry
  `subset_for_metrics: <language>`, but the current `arena_judge`
  server does not split Arena-Elo by `subset_key=language`. Adding
  per-language breakdowns is a future `arena_judge` change and is
  intentionally out of scope here.
