# arena_judge

Pairwise LLM-judge resources server ported from
[`nemo_skills/inference/eval/arena_judge.py`](../../../nemo-skills/nemo_skills/inference/eval/arena_judge.py)
and
[`nemo_skills/evaluation/metrics/arena_metrics.py`](../../../nemo-skills/nemo_skills/evaluation/metrics/arena_metrics.py)
to preserve the upstream [arena-hard-auto](https://github.com/lmarena/arena-hard-auto)
judging protocol.

## What it does

For each rollout, the server makes **two** judge calls with the answer
order swapped to control for positional bias:

1. `gen-base` — A = candidate, B = baseline
2. `base-gen` — A = baseline, B = candidate

Each call returns a verdict drawn from
`{A>>B, A>B, A=B, B>A, B>>A}` (extracted via regex
`\[\[([AB<>=]+)\]\]`). The resource server returns:

- `reward = 1.0` if the gen-base verdict is a candidate win
  (`A>B` or `A>>B`), else `0.0`
- Raw judge outputs (`judgement_gen_base`, `judgement_base_gen`) and
  parsed labels (`verdict_gen_base`, `verdict_base_gen`) so downstream
  aggregate Arena-Elo metrics (MLE + bootstrap CI) can be recomputed
  post-hoc by the consuming benchmark's comparison scripts.

Judge prompts are category-specific: `hard_prompt` uses
[`prompts/arena.yaml`](prompts/arena.yaml) (judge writes its own answer
first), `creative_writing` uses
[`prompts/arena_creative.yaml`](prompts/arena_creative.yaml) (no
own-answer step).

## Data schema

Each JSONL row must carry the following top-level fields (pydantic
`extra="allow"` flows them through):

- `question` — the user prompt sent to the candidate model
- `baseline_answer` — reference answer to compare against
- `category` — one of `hard_prompt` / `creative_writing`
- `uid` — arena-hard-auto problem id (optional)

## Example usage

```bash
# Running servers
config_paths="responses_api_models/vllm_model/configs/vllm_model.yaml,\
resources_servers/arena_judge/configs/arena_judge.yaml"
ng_run "+config_paths=[$config_paths]"

# Collecting rollouts (5-example smoke test)
ng_collect_rollouts \
    +agent_name=arena_judge_simple_agent \
    +input_jsonl_fpath=resources_servers/arena_judge/data/example.jsonl \
    +output_jsonl_fpath=results/arena_judge_rollouts.jsonl \
    +num_repeats=1
```

## Configuration

The default config routes judge calls to
`https://inference-api.nvidia.com/v1` with
`aws/anthropic/bedrock-claude-opus-4-6`. Override via env vars:

- `NVIDIA_INFERENCE_API_KEY` — required; pulled via `${oc.env:...}`
- `ARENA_JUDGE_BASE_URL` — optional; e.g. self-hosted vLLM endpoint
- `ARENA_JUDGE_MODEL` — optional; e.g. another judge model

## Metrics

The server's `compute_metrics()` emits Tier-1 pass@k / majority@k for
`wins`, `strict_wins`, `ties`, `losses`, `double_wins`, and
`invalid_gen_base`, plus a per-category breakdown (via
`compute_subset_metrics(field="category")`). The Arena-Elo headline
metric (MLE logistic regression + bootstrap CI over pooled battles)
lives in the consuming benchmark's recipe directory, not here.
