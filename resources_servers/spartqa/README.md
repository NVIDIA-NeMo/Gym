# SpartQA Resources Server

Spatial-reasoning **answer generation** benchmark. The model is shown a
spatial-reasoning query and must return the matching answer phrase, ending with
a `Final answer: <phrase>` line. The per-sample reward is `1.0` on an
exact-or-answer-containing match against any accepted answer phrase, else `0.0`.

Source dataset: [`mteb/SpartQA`](https://huggingface.co/datasets/mteb/SpartQA)
(MTEB retrieval form — `queries` / `corpus` / `qrels` splits joined at prep
time by `prepare_spartqa.py`).

## Scoring

`verify()` extracts the model's final
answer (`_extract_answer` / `_strip_reasoning` / `_clean_candidate`), normalizes
it (`_normalize`: lowercase, strip punctuation, collapse whitespace), and
compares against every accepted phrase in `all_targets` (falling back to
`target`). A strict equality sets `exact`; a substring match sets the reward
without `exact`. Empty output scores `0.0` and never raises.

## Metrics

`compute_metrics` reports:

- `mean_reward` — mean per-sample accuracy (also the reward).
- `exact_match_rate` — fraction with a strict (exact) match.
- `parse_rate` — fraction where a non-empty answer phrase was extracted.

`get_key_metrics` surfaces `mean_reward` and `exact_match_rate`.

> **Reasoning models:** `verify()` strips a leading `<think>…</think>` block
> before extracting the answer.

## Prepare the dataset

```bash
cd gym
python resources_servers/spartqa/prepare_spartqa.py --split test
```

This joins `mteb/SpartQA` (via the HF `datasets` library) and writes the
gitignored `data/spartqa_test.jsonl`. The committed `data/example.jsonl` is a
5-row smoke-test slice.

## Example rollouts and metrics

`data/example_rollouts.jsonl` and `data/example_metrics.json` are committed
and can be regenerated at any time with the scripts below (no servers needed):

```bash
# Regenerate synthetic rollouts (rule-based scorer, no model call)
python resources_servers/spartqa/generate_example_rollouts.py

# Regenerate dataset stats summary
python resources_servers/spartqa/generate_example_metrics.py

# Inspect
tail -n 1 resources_servers/spartqa/data/example_rollouts.jsonl | jq .reward
cat resources_servers/spartqa/data/example_metrics.json | jq .
```

Note: row 2 (index 2) in the example rollouts is intentionally wrong (reward
0.0) to demonstrate a failed case; the remaining four rows score 1.0.

## Run

```bash
gym env start --resources-server spartqa --model-type vllm_model
```

No API keys are required — all scoring is rule-based.

## Test

```bash
gym env test --resources-server spartqa
```
