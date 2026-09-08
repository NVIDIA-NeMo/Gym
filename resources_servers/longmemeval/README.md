# LongMemEval

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) is a **long-term-memory
QA** benchmark: each task hands the model a multi-session chat haystack and asks
a question that can only be answered by recalling, reasoning over, updating or
abstaining from information buried in that history. This server ports the
`orig-session` / JSON-history / no-CoT configuration to NeMo Gym.

Source dataset: [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
(MIT). Upstream scoring: `src/evaluation/evaluate_qa.py`.

## How it works

Every task is **single-turn generation**. The multi-session haystack is rendered
into the prompt at dataset-prep time (`prepare_longmemeval.py`), so the model
sees one user message carrying the JSON-format session history plus the
question, and `verify()` only ever receives the question, the gold answer and
the generation. No agent fork is needed — the built-in `simple_agent` drives it.

Scoring is **LLM-as-judge**: `verify()` renders a per-question-type rubric
(verbatim from upstream `get_anscheck_prompt`), sends it to a judge model at
temperature 0, and scores 1.0 when the verdict contains "yes". Rewards are
binary.

| `question_type` | Rubric |
|-----------------|--------|
| `single-session-user` | contain |
| `single-session-assistant` | contain |
| `multi-session` | contain |
| `temporal-reasoning` | temporal (off-by-one tolerant) |
| `knowledge-update` | knowledge-update (updated answer wins) |
| `single-session-preference` | preference (rubric-based) |
| abstention | unanswerable-question |

Abstention rows are routed to the abstention rubric whenever `question_id`
contains `_abs` (or metadata sets `abstention`), **regardless** of their
`question_type`.

> **Reasoning models:** `verify()` strips `<think>…</think>` blocks (including a
> truncated, unclosed one) from the generation before rendering the rubric —
> hidden reasoning often contains the gold answer and would inflate the judge's
> verdict. When serving a reasoning model, also run the policy server with
> `--reasoning-parser <name>`.

## Judge model — required

Unlike the rule-based benchmarks here, LongMemEval **cannot score without a
judge endpoint**. Supply credentials at launch:

```bash
gym env start --resources-server longmemeval --model-type vllm_model \
    +judge_base_url=https://api.openai.com/v1 \
    +judge_api_key=$OPEN_AI_KEY \
    +judge_model_name=gpt-4o-mini
```

`gpt-4o-mini` is the upstream default and the recommended judge.

Judge knobs in `configs/longmemeval*.yaml`:

| Key | Default | Purpose |
|-----|---------|---------|
| `judge_endpoint_max_concurrency` | 32 | `asyncio.Semaphore` bound on in-flight judge calls |
| `judge_max_retries` | 5 | bounded exponential backoff on 429 / 5xx |
| `judge_retry_base_delay` | 1.0 | first backoff sleep, doubling to a 30 s cap |
| `judge_responses_create_params.max_output_tokens` | 64 | see below |

Non-retryable statuses (400/401/404/422) fail immediately rather than burning
retries. If a run reports a non-zero `n_judge_call_failed`, raise the retry
settings — rate limiting must not silently shrink the accuracy denominator.

## Differences from the original benchmark

Per-question **rubrics and verdict mapping are identical** to upstream. The
deviations are mechanical or make failures visible instead of silent:

1. **`max_output_tokens: 64` vs. upstream `max_tokens=10`.** Upstream calls Chat
   Completions; the Responses API enforces a floor of 16, and a reasoning-capable
   judge spends part of the budget before emitting visible text. The rubric still
   asks for "yes or no only", so verdicts are unchanged.
2. **No `n` parameter.** The Responses API has no `n`; a single completion is the
   default, which is what upstream's `n=1` requests.
3. **Empty generation short-circuits to 0.0** without calling the judge. A reply
   containing nothing cannot contain the answer; the row is flagged
   `empty_response` and still counts as 0.0, exactly as upstream. Only the API
   call is saved.
4. **A row with no gold `answer` is graded as 0.0**, not skipped. Upstream indexes
   its question bank directly, so a missing answer raises `KeyError` inside its
   `try` and the row vanishes from the denominator. A dataset-prep bug should
   depress the score, not shrink the population it is measured over.
5. **Bounded judge retries.** Upstream's `@backoff.on_exception` is uncapped, so
   it effectively never reaches its rate-limit skip path. A bounded retry cannot
   hang a run; exhausted retries are classified `judge_call_failed` and excluded,
   matching that skip.
6. **`<think>` stripping** (see above) — upstream has no such step because its
   policy models never emit reasoning blocks.
7. **No tokenizer-level prompt truncation.** Upstream `run_generation.py` truncates
   the rendered history to the generator's context window; that would pull in a
   tiktoken/transformers dependency purely for prompt shortening, so truncation is
   delegated to the model's own context window. Use `--topk-context` to bound the
   history by session count instead.

`tests/test_acceptance.py` asserts byte-identical prompt parity with upstream
`run_generation.py` and rubric parity with `evaluate_qa.py` (AC1–AC11).

## Error taxonomy

`verify()` always returns a binary 0.0/1.0 reward; the classification only
changes how a row is *aggregated*:

| Condition | Marker | In `accuracy` denominator? |
|-----------|--------|----------------------------|
| judge raised / non-2xx after retries | `judge_call_failed` | no (matches upstream skip) |
| unknown `question_type` | `unknown_question_type` | no (matches upstream skip) |
| empty generation | `empty_response=True` | yes, as 0.0 |
| empty judge verdict | `empty_judge_output` | yes, as 0.0 |
| missing / unusable metadata | `bad_metadata` | yes, as 0.0 |
| missing gold `answer` | (none) | yes, as 0.0 (upstream skips) |

Excluding a row drops it from `accuracy`, `count`, the per-type and the
abstention buckets. Nothing is hidden: `accuracy_strict` over *every* rewarded
row is emitted alongside, plus a counter per condition. A `judge_error` value
from outside this taxonomy (foreign or pre-taxonomy rollouts) is counted under
`n_judge_errors_other` and kept in the denominator.

## Metrics

`compute_metrics` reports:

* `accuracy` — mean reward over **scored** rows (excludes `judge_call_failed`
  and `unknown_question_type`, matching upstream's skip path)
* `accuracy_strict` — mean reward over **all** rewarded rows (full denominator)
* `count`, `question_type/<type>/accuracy`, `question_type/<type>/count`
* `abstention/accuracy`, `abstention/count`
* `n_judge_call_failed`, `n_empty_judge_output`, `n_unknown_question_type`,
  `n_bad_metadata`, `n_judge_errors_other`, `n_excluded`, `n_empty_response`

`get_key_metrics` surfaces `accuracy`, `abstention/accuracy`, `n_excluded`.

> These aggregates come from the **gym-native** metrics path (`gym eval` /
> `ng_reward_profile` / `/compute_metrics`). A driver that only means the reward
> column reports a flat per-row mean — close to `accuracy_strict`, not to
> `accuracy`, and with no per-type breakdown.

## Data

```bash
# Downloads xiaowu0162/longmemeval-cleaned into $XDG_CACHE_HOME/longmemeval
# on first run and writes data/<split>.jsonl.
python resources_servers/longmemeval/prepare_longmemeval.py                 # oracle (default)
python resources_servers/longmemeval/prepare_longmemeval.py --split s
python resources_servers/longmemeval/prepare_longmemeval.py --split m
python resources_servers/longmemeval/prepare_longmemeval.py --split s --limit 100
```

| Flag | Default | Notes |
|------|---------|-------|
| `--split` | `oracle` | `oracle` / `s` / `m` |
| `--topk-context` | `50` | keep the last N sessions; `<=0` keeps all, `1000` matches upstream's run script |
| `--limit` | `0` | writes `data/<split>_limit<N>.jsonl` so it never clobbers a full build |
| `--input` | — | convert an already-downloaded `longmemeval_*.json` instead of fetching |

Only `data/example.jsonl` (5 rows covering all question types, including an
abstention row) is committed; the full splits are gitignored — rebuild them
locally. The `--topk-context` default of 50 may drop gold evidence sessions for
entries with more sessions than that; the script warns when it does.

## Example rollouts and metrics

`data/example_rollouts.jsonl` and `data/example_reward_stats.json` are committed
and can be regenerated without any live servers or API keys — the synthetic
responses and their judge verdicts are hard-coded (4 correct, 1 incorrect,
covering all five question types in `example.jsonl`):

```bash
# Synthetic responses + judge verdicts → example_rollouts.jsonl
python resources_servers/longmemeval/generate_example_rollouts.py

# Aggregate rollouts → per-question-type summary
python resources_servers/longmemeval/generate_example_metrics.py

# Inspect
tail -n 1 resources_servers/longmemeval/data/example_rollouts.jsonl | jq .reward
cat resources_servers/longmemeval/data/example_reward_stats.json | jq .
```

Note the two distinct filenames: `example_metrics.json` is reserved for the
standard `DatasetMetrics` format produced by `gym dataset collate` (CI validates
it contains "Number of examples"), so the `compute_metrics`-style breakdown is
written to `example_reward_stats.json`.

## Run

```bash
# Full eval (resources server + judge + simple_agent + policy model)
gym env start --resources-server longmemeval --model-type vllm_model \
    +judge_base_url=... +judge_api_key=... +judge_model_name=gpt-4o-mini

# Serve-only (no policy agent/model — the caller owns generation and POSTs /verify)
gym env start --resources-server longmemeval/longmemeval_serve \
    +judge_base_url=... +judge_api_key=... +judge_model_name=gpt-4o-mini
```

A gym-native benchmark wiring also lives at `benchmarks/longmemeval/`
(`config.yaml` + `prepare.py`), which auto-builds `data/oracle.jsonl` on first
`prepare()`.

## Test

```bash
gym env test --resources-server longmemeval
```

## License

Dataset: MIT — [`xiaowu0162/longmemeval-cleaned`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned),
derived from [`github.com/xiaowu0162/LongMemEval`](https://github.com/xiaowu0162/LongMemEval)
(MIT). No upstream code is vendored — all logic is re-implemented in `app.py`
and `prepare_longmemeval.py`.
