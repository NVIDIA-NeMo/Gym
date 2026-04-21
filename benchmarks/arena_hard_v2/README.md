# arena_hard_v2

Gym port of the [Arena Hard v2](https://github.com/lmarena/arena-hard-auto)
open-ended generation benchmark, migrated from NeMo Skills'
`nemo_skills/dataset/arena-hard-v2/`.

## What it tests

~750 hard, open-ended user prompts across two categories:

- `hard_prompt` — reasoning-heavy technical and analytical queries,
  judged against an **o3-mini** baseline
- `creative_writing` — open-ended creative tasks, judged against a
  **gemini-2.0-flash-001** baseline

Each candidate rollout is judged pairwise (both A↔B orderings) against
its category-specific baseline via an LLM judge — see
[`resources_servers/arena_judge`](../../resources_servers/arena_judge/README.md)
for the judging protocol.

## Data

Runtime download only — benchmark JSONL is not committed. Run
[`prepare.py`](prepare.py) (or `ng_prepare_benchmark`) to populate
`data/arena_hard_v2_benchmark.jsonl`. The prepare script mirrors Skills'
`arena-hard-v2/prepare.py` byte-for-byte: fetches questions and both
baselines from the arena-hard-auto GitHub repo, joins by `uid`, emits
one row per question with `question`, `baseline_answer`, `category`,
and `uid` at the top level.

## Example usage

```bash
# Prepare benchmark data
ng_prepare_benchmark "+config_paths=[benchmarks/arena_hard_v2/config.yaml]"

# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
benchmarks/arena_hard_v2/config.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts
ng_collect_rollouts \
    +agent_name=arena_hard_v2_arena_judge_simple_agent \
    +input_jsonl_fpath=benchmarks/arena_hard_v2/data/arena_hard_v2_benchmark.jsonl \
    +output_jsonl_fpath=results/arena_hard_v2_rollouts.jsonl \
    +num_repeats=4
```

## Metrics

The `arena_judge` resources server emits per-rollout pass@k on
`wins` / `strict_wins` / `ties` / `losses` / `double_wins`, plus a
per-category breakdown (hard_prompt vs creative_writing).

The Arena-Elo headline metric (MLE logistic regression + bootstrap 95%
CI over pooled pairwise battles) is **not** computed here — it lives
in the migration recipe directory (`migrate-gym-arena-hard-v2/`) as a
post-hoc script that reads both Skills and Gym rollout JSONLs. This
keeps the resource server free of the sklearn dependency and lets the
same Elo code ground-truth both pipelines at comparison time.
