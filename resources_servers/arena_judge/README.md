# arena_judge

Pairwise LLM-judge resources server implementing the
[arena-hard-auto](https://github.com/lmarena/arena-hard-auto) judging
protocol: two judge calls per rollout with swapped answer positions,
category-specific prompts, and an Arena-Elo score (MLE logistic
regression + 100-round bootstrap 95% CI) as the headline metric.

## What it does

For each rollout, the server makes **two** judge calls to control for
positional bias:

1. `gen-base` — A = candidate, B = baseline
2. `base-gen` — A = baseline, B = candidate

Each call returns a verdict drawn from
`{A>>B, A>B, A=B, B>A, B>>A}` (extracted via regex
`\[\[([AB<>=]+)\]\]`). The resource server returns:

- `reward = 1.0` if the gen-base verdict is a candidate win
  (`A>B` or `A>>B`), else `0.0`
- Raw judge outputs (`judgement_gen_base`, `judgement_base_gen`) and
  parsed labels (`verdict_gen_base`, `verdict_base_gen`) so
  `compute_metrics()` (and any downstream script) can rebuild the
  pairwise battles without re-invoking the judge.

Judge prompts are category-specific: `hard_prompt` uses
[`prompts/arena.yaml`](prompts/arena.yaml) (the judge writes its own
answer first), `creative_writing` uses
[`prompts/arena_creative.yaml`](prompts/arena_creative.yaml) (no
own-answer step). Both prompts are ports of arena-hard-auto's upstream
judge templates.

## Data schema

Each JSONL row must carry the following top-level fields (pydantic
`extra="allow"` flows them through):

- `question` — the user prompt sent to the candidate model
- `baseline_answer` — reference answer to compare against
- `category` — one of `hard_prompt` / `creative_writing` (rows without
  one fall back to `default_category` in the config)
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

## Configuring the judge

The judge model is an OpenAI-compatible endpoint specified by three
environment variables (resolved via `${oc.env:...}` at config-load
time):

- `ARENA_JUDGE_BASE_URL` — base URL (must support `/v1/chat/completions`)
- `ARENA_JUDGE_API_KEY` — API key; `MISSING` fallback keeps config-load
  green for judge-unrelated jobs, but live `verify()` calls will fail
  without a real key
- `ARENA_JUDGE_MODEL` — model identifier accepted by the endpoint

The judge call goes through `/v1/chat/completions` — the most widely
supported path across OpenAI-compatible providers for a simple
text-verdict judge.

## Metrics

`compute_metrics()` emits:

- **`arena_elo/score`** — headline Arena-Elo win-rate (%) vs baseline,
  computed by MLE logistic regression over the pairwise battles and
  clamped to 0-100. Bundled with a `ci_lower` / `ci_upper` pair from a
  100-round bootstrap.
- **`arena_elo/{category}/score` + CIs** — per-category breakdown (e.g.
  `arena_elo/hard_prompt/score`) for any `category` field seen on the
  input rows.
- **`arena_elo/invalid_scores`** — count of judge calls that produced
  no parseable verdict (mirrors arena-hard-auto's `num_invalid`).
- **pass@k / pass@1[avg-of-k] / majority@k** for
  `{wins, strict_wins, ties, losses, double_wins, invalid_gen_base}`
  (from `compute_pass_majority_metrics`), plus a per-category
  breakdown via `compute_subset_metrics(subset_key="category", ...)`.

`get_key_metrics()` surfaces `arena_elo/score` and its CIs as the
run's headline numbers alongside the highest-k `pass@1[avg-of-k]`
flavors.
