# IHEval

[IHEval](https://github.com/ytyz1307zzh/IHEval) is an **instruction-hierarchy**
benchmark: it measures whether a model respects the priority order of
system / user / tool instructions across *aligned*, *conflict*, and *reference*
settings. This server ports **all** IHEval tasks and settings to NeMo Gym —
including multi-turn rule-following and the reference cross-row concatenation.

The gym-native metrics path reproduces the upstream **task-macro** aggregation
exactly and reports all three category scores (`aligned_score`,
`conflict_score`, `reference_score`) plus the headline `result_score`. That is
the recommended way to run this benchmark and is described first.

## Run (gym-native — the exact IHEval numbers)

All IHEval scorers are rule-based (F1 / ROUGE / IFEval / TensorTrust), so
verification needs no judge model — only the policy model generates.

**1. Start the servers** (resources server + agent + policy model):

```bash
gym env start --resources-server iheval --model-type vllm_model
```

Set `OPEN_ROUTER_KEY` if using an OpenRouter-backed model server. To bring up
just the scorer (you own generation and only call `/verify`), use the serve-only
config: `gym env start --resources-server iheval/iheval_serve`.

**2. Collect rollouts and aggregate** over the full dataset:

```bash
gym eval run --no-serve \
    --agent iheval_simple_agent \
    --input resources_servers/iheval/data/test.jsonl \
    --output results/iheval_rollouts.jsonl \
    --num-repeats 1
```

At the end you get the **task-macro** metrics (illustrative values):

```text
Key metrics for iheval_simple_agent:
{
    "result_score": 0.7217,       # headline (conflict, in the default hierarchy mode)
    "conflict_score": 0.7217,
    "aligned_score": 0.8134,
    "reference_score": 0.7660,
    "mean_reward": 0.7421,        # flat per-row mean, for reference only
    "conflict/get-webpage/score": 0.8471,
    "conflict/single-turn/score": 0.5855,
    ...
}
Aggregate metrics: results/iheval_rollouts_aggregate_metrics.json
```

Those values come from the resources server's `compute_metrics` /
`get_key_metrics` (invoked via `/aggregate_metrics` after rollout collection).
The written file `results/iheval_rollouts_aggregate_metrics.json` holds, per
agent:

* `agent_metrics` — the full `compute_metrics` dict: `result_score`,
  `aligned_score` / `conflict_score` / `reference_score`, the per-task
  `{category}/{task}/score` breakdowns, the `diff_aligned` / `diff_conflict`
  (category − reference) deltas, the reconstructed `reference/<task>/average`
  cross-row metrics, and `mean_reward` plus per-`task`/`domain`/`setting` flat
  means for inspection.
* `key_metrics` — the headline selection above (`result_score` first).

> **Run over the full `data/test.jsonl`** (all three settings — aligned,
> conflict, reference) to get all three category scores. A conflict-only subset
> can only ever produce `conflict_score`.

For sharded jobs, pass `--disable-aggregation` to `gym eval run` and compute the
global metrics once all shards finish with `gym eval aggregate`.

## Result score and the three category scores (`accuracy_mode`)

Aggregation is **task-macro**, matching the upstream headline exactly: rows are
averaged within each (task, setting) group, those group scores are averaged
across a task's settings, and those are averaged across tasks — so **every task
and setting is weighted equally regardless of how many rows it has**. Row counts
never dilute a score, and reference is a raw-task-ability baseline that never
enters `result_score`.

The headline `result_score` is selected by the resources-server config's
**`accuracy_mode`** (default `hierarchy`):

* **`hierarchy`** (default) — the **conflict-setting** score, since instruction
  hierarchy is precisely what the conflict setting stresses. Following upstream
  `average_final_score.py`, `result_score` = `conflict_score`: the mean over
  tasks of each task's conflict score, where a task's conflict score is the mean
  over its conflict-setting `average`s.
* **`hierarchy_sysprompt`** — `mean(aligned_score, conflict_score)`. Credits both
  instruction-hierarchy following (Conflict) *and* system-prompt instruction
  following (Aligned — obeying the system message when nothing conflicts).

Set it in `configs/iheval*.yaml` (or override per run, e.g.
`++iheval.resources_servers.iheval.accuracy_mode=hierarchy_sysprompt`).
`aligned_score` / `conflict_score` / `reference_score` and the aligned/conflict
− reference diffs are always reported alongside, regardless of `accuracy_mode`.

Per-setting `average` matches upstream per task type:

* verb-extract / translation / lang-detect / safety / slack-user / get-webpage —
  mean of per-row rewards (equals upstream's strict/loose mean by construction).
* single-turn / multi-turn (rule-following) — the prompt/instruction × strict/loose
  mean, with **instruction-level accuracy weighted by instruction count**
  (`sum(followed) / sum(total)`), matching `record_scores.py`.
* reference category — the cross-row concatenation `average` (see below).

## Tasks

Every row carries its `task` (and `domain`, `setting`, `instruction`, `answer`)
as **row top-level scalar fields** — not nested under `verifier_metadata`. The gold
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

This server passes the function schema **natively** in
`responses_create_params.tools`. The canned tool-call trajectory is pre-filled
as Responses-API `function_call` / `function_call_output` items in the input,
preserving the privilege boundary between the user instruction and the tool
output (critical for the prompt-injection *conflict* setting).

## Multi-turn rule-following

Included. Upstream's `conversation_history` (the prior user turns **and** the
fixed assistant replies) is pre-canned in the data — the model only generates
the final turn, which is scored with the same IFEval checker as `single-turn`.
So it maps to a single generation over a pre-filled multi-turn context (built
into `responses_create_params.input` by `prepare_iheval.py`), not a live
multi-turn rollout. Scoring matches upstream; the model just does not itself
produce the earlier turns.

## Reference cross-row concatenation

Included and reconstructed **exactly** at aggregation time — because it is
inherently cross-row. Upstream (`calc_reference_score.py` /
`calc_mix_reference_score.py`) scores each data row by concatenating its
prediction with the *anchor rows'* predictions (`strong_user_instruction` /
`weak_user_instruction`) and re-scoring; the six-component `average` therefore
depends on other rollouts' generations, which a single per-row `verify()` never
sees.

So:

* **Per-row reward** for a reference row = the standalone `no_user_instruction`
  component (with the `español:` / `Verbs:` prefix stripping upstream applies) —
  a valid RL signal. `verify()` also stashes the stripped prediction + gold.
* **`compute_metrics`** collects the stashed stripped predictions + golds across
  all rollouts and reconstructs the exact upstream number, emitted as
  `reference/verb-extract/average`, `reference/translation/average`, and
  `reference/get-webpage/{verb_extract,translation,lang_detect,}/average`
  (the get-webpage overall is the length-weighted mean, matching
  `calc_mix_reference_score`).

The reconstruction has been verified to match the upstream algorithm exactly.

## Data

```bash
# Downloads github.com/ytyz1307zzh/IHEval and writes data/test.jsonl,
# data/test_conflict.jsonl + data/example.jsonl. Set IHEVAL_REPO_DIR to use an
# existing checkout.
python resources_servers/iheval/prepare_iheval.py
```

`data/example.jsonl` (5 mixed rows) is committed for smoke testing;
`data/test.jsonl` (~19k rows across all eight tasks, all three settings) and
`data/test_conflict.jsonl` (the `conflict/*` subset) are generated locally. Use
`data/test.jsonl` for the full IHEval metric.

## Test

```bash
gym env test --resources-server iheval
```

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

## Scoring source

The IFEval rule-following checkers under `ifeval/` are vendored from upstream
(Apache-2.0); see `ifeval/PROVENANCE.md`.

---

## Appendix: driving IHEval from an external per-row driver (e.g. nel)

The per-row scoring above is identical no matter what drives the eval. But some
external drivers (e.g. nemo-evaluator / `nel`) only **mean the per-row reward
column** — they read `reward` from `/verify` and never call the gym server's
`compute_metrics` / `/aggregate_metrics`. None of the following affects the
gym-native flow above; it only matters if you read a driver's own report instead
of `results/..._aggregate_metrics.json`.

1. **Aggregation: task-macro vs. flat per-row mean.** A driver that averages the
   reward column produces a **flat per-row mean**, where larger tasks/settings
   dominate. It agrees with the task-macro only when row counts are balanced
   across groups; on the full mixed dataset the two diverge. For the exact
   upstream number, read the gym-native aggregate file.

2. **The headline is the conflict score.** The upstream headline is the conflict
   task-macro. To make a flat-mean driver approximate it directly, point the
   driver at the **conflict-only** dataset (`data/test_conflict.jsonl`) instead
   of the full mixed dataset — the conflict subset is roughly task-balanced, so
   the flat mean lands within a small fraction of a point of the exact conflict
   task-macro. This is an approximation, not the exact statistic, and it yields
   only the conflict number (not aligned/reference).

3. **The "reference" setting is inherently cross-row.** A per-row driver reports
   only the standalone `no_user_instruction` component for a reference row; the
   exact reference aggregates are reconstructed only in the gym-native path.

4. **Generation and serving parameters are the runner's responsibility.**
   Sampling parameters (temperature, top_p, max_tokens), whether reasoning is
   enabled, and the inference backend are chosen by whoever runs the eval, not
   fixed by the benchmark — differences there (not the scoring) explain most
   run-to-run gaps against a published number.
