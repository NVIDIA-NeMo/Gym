# IHEval

[IHEval](https://github.com/ytyz1307zzh/IHEval) is an **instruction-hierarchy**
benchmark: it measures whether a model respects the priority order of
system / user / tool instructions across *aligned*, *conflict*, and *reference*
settings. This server ports **all** IHEval tasks and settings to NeMo Gym —
including multi-turn rule-following and the reference cross-row concatenation.

## Tasks

Every row carries its `task` (and `domain`, `setting`, `instruction`, `answer`)
as **row top-level scalar fields** — not nested under `verifier_metadata` — so
they survive the nemo-evaluator `gym://...protocol=native` driver, which
forwards a row's top-level scalar fields but drops nested objects. The gold
`answer` (a dict/list for safety, rule-following and get-webpage) is therefore
**JSON-encoded to a string** so it too survives; `verify()` JSON-decodes it
(`_decode_answer`). `task` selects the scorer in `verify()`:

| `task` | Domain | Scorer | Reward |
|--------|--------|--------|--------|
| `verb-extract` | task-execution | word-level F1 (strict + loose) | mean |
| `translation` | task-execution | ROUGE-L f-measure (strict + loose) | mean |
| `lang-detect` | task-execution | single-key `{"language": …}` JSON match | 0/1 |
| `system-prompt-extract` | safety | TensorTrust defense check | 0/1 |
| `user-prompt-hijack` | safety | TensorTrust defense check | 0/1 |
| `slack-user` | tool-use | exact match (punctuation-stripped) | 0/1 |
| `get-webpage` | tool-use | mixed — dispatched by `answer.task` | mean / 0/1 |
| `single-turn` | rule-following | IFEval prompt/instruction × strict/loose | mean of 4 |
| `multi-turn` | rule-following | IFEval on the final turn (pre-canned history) | mean of 4 |

Both strict and loose IHEval scoring are computed where upstream does so; the
per-row reward is the mean, matching upstream's per-task `average`.

## Native tool use

Unlike the legacy nemo-evaluator BYOB port — which could not pass `tools`
through the chat-completions param and worked around it by appending the tool
schema to the system prompt — this server passes the function schema
**natively** in `responses_create_params.tools`. The canned tool-call
trajectory is pre-filled as Responses-API `function_call` /
`function_call_output` items in the input, preserving the privilege boundary
between the user instruction and the tool output (critical for the
prompt-injection *conflict* setting).

## Data

```bash
# Downloads github.com/ytyz1307zzh/IHEval and writes data/test.jsonl +
# data/example.jsonl. Set IHEVAL_REPO_DIR to use an existing checkout.
python resources_servers/iheval/prepare_iheval.py
```

`data/example.jsonl` (5 mixed rows) is committed for smoke testing;
`data/test.jsonl` (~15k rows across all eight tasks) is generated locally.

## Multi-turn rule-following

Included. Upstream's `conversation_history` (the prior user turns **and** the
fixed assistant replies) is pre-canned in the data — the model only generates
the final turn, which is scored with the same IFEval checker as `single-turn`.
So it maps to a single generation over a pre-filled multi-turn context (built
into `responses_create_params.input` by `prepare_iheval.py`), not a live
multi-turn rollout.

## Reference cross-row concatenation

Included, but reconstructed at aggregation time rather than per-row — because it
is **inherently cross-row**. Upstream (`calc_reference_score.py` /
`calc_mix_reference_score.py`) scores each data row by concatenating its
prediction with the *anchor rows'* predictions (`strong_user_instruction` /
`weak_user_instruction`) and re-scoring; the six-component `average` therefore
depends on other rollouts' generations, which a single per-row `verify()` never
sees.

So:

* **Per-row reward** for a reference row = the standalone `no_user_instruction`
  component (with the `español:` / `Verbs:` prefix stripping upstream applies) —
  a valid RL signal.
* **`compute_metrics`** collects the stashed stripped predictions + golds across
  all rollouts and reconstructs the exact upstream number, emitted as
  `reference/verb-extract/average`, `reference/translation/average`, and
  `reference/get-webpage/{verb_extract,translation,lang_detect,}/average`
  (the get-webpage overall is the length-weighted mean, matching
  `calc_mix_reference_score`).

The reconstruction has been verified to match the upstream algorithm exactly.
Note: these reference aggregates surface in the **gym-native** metrics path
(`gym eval` / `ng_reward_profile` / the `/compute_metrics` endpoint). A driver
that only averages per-row rewards (e.g. a plain nemo-evaluator mean) will
report the per-row `no_user_instruction` component instead.

## Result score (aggregate conflict)

The headline IHEval metric is the **conflict-setting** score — instruction
hierarchy is precisely what the conflict setting stresses. Following upstream
`average_final_score.py`, `compute_metrics` reports `result_score` (=
`conflict_score`) as the mean over tasks of each task's conflict score, where a
task's conflict score is the mean over its conflict-setting `average`s. Row
counts do **not** dilute it — each setting is weighted equally within a task,
and each task equally overall.

Per-setting `average` matches upstream per task type:

* verb-extract / translation / lang-detect / safety / slack-user / get-webpage —
  mean of per-row rewards (equals upstream's strict/loose mean by construction).
* single-turn / multi-turn (rule-following) — the prompt/instruction × strict/loose
  mean, with **instruction-level accuracy weighted by instruction count**
  (`sum(followed) / sum(total)`), matching `record_scores.py`.
* reference category — the cross-row concatenation `average` (see above).

Also reported: `aligned_score`, `reference_score`, the per-task
`{category}/{task}/score` breakdowns, and the `diff_aligned` / `diff_conflict`
(category − reference) deltas — the upstream "Agg." / "Diff." block.
`get_key_metrics` surfaces `result_score` first.

> Caveat: these aggregates come from the **gym-native** metrics path (`gym eval`
> / `ng_reward_profile` / `/compute_metrics`). A driver that only means per-row
> rewards over the whole dataset reports a flat all-settings mean, not the
> conflict result score — filter the dataset to conflict rows if you need the
> conflict number from a per-row-mean driver.

`compute_metrics` also reports `mean_reward` plus per-`task`, per-`domain`, and
per-`setting` breakdowns for inspection.

## Example rollouts and metrics

`data/example_rollouts.jsonl` and `data/example_metrics.json` are committed
and can be regenerated at any time without any live servers:

```bash
# Score synthetic responses against example.jsonl → example_rollouts.jsonl
python resources_servers/iheval/generate_example_rollouts.py

# Aggregate rollouts → per-task / per-domain / per-setting summary
python resources_servers/iheval/generate_example_metrics.py

# Inspect
tail -n 1 resources_servers/iheval/data/example_rollouts.jsonl | jq .reward
cat resources_servers/iheval/data/example_metrics.json | jq .
```

Note: `example.jsonl` contains only aligned-setting rows, so the headline
`result_score` (conflict score) is not present in `example_metrics.json`. Run
against `data/test.jsonl` with a full model eval for the complete IHEval metric.

## Run

```bash
# Full eval (resources server + model)
gym env start --resources-server iheval --model-type vllm_model

# Serve-only (no model needed — all scorers are rule-based)
gym env start --resources-server iheval/iheval_serve
```

Set `OPEN_ROUTER_KEY` if using an OpenRouter-backed model server.

## Test

```bash
gym env test --resources-server iheval
```

## Scoring source

The IFEval rule-following checkers under `ifeval/` are vendored from upstream
(Apache-2.0); see `ifeval/PROVENANCE.md`.
