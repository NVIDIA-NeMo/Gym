# SpartQA Resources Server

Spatial-reasoning benchmark covering SpartQA's **CO (choose-object)** question
type. The model is shown a story, a question of the form "… X or Y?", and four
candidate answers; it must copy one of them, ending with a
`Final answer: <candidate answer>` line. The per-sample reward is `1.0` iff the
answer resolves to the gold label, else `0.0`.

Source dataset: [`mteb/SpartQA`](https://huggingface.co/datasets/mteb/SpartQA)
(MTEB retrieval form — `queries` / `corpus` / `qrels` splits joined at prep
time by `prepare_spartqa.py`).

## Task definition

Per the SpartQA paper (Mirzaee et al., NAACL 2021, Table 8 fn.), CO is a
**four-label single-choice** task:

| Label | Meaning |
|-------|---------|
| `X` | only the first object named in the question |
| `Y` | only the second object |
| `both of them` | both objects satisfy the asked relation |
| `none of them` | neither does (the paper's DK / None / `[]`) |

The retrieval encoding has no way to express "both", so a `both of them` gold
is flattened into **three** relevant qrels documents (the phrase plus each
object). `prepare_spartqa.py` (`resolve_gold`) undoes that flattening back into
one gold label, and renders all four candidates into the prompt so
`both of them` / `none of them` are reachable answers.

> Scoring the three flattened phrases as interchangeable accepted answers —
> i.e. crediting a single object for a `both of them` gold — is not the paper's
> metric and is trivially gameable: echoing the question's two options back
> ("X or Y") scores **94.9%** that way, versus **0.0%** here.

Corpus composition (`test`, 3594 rows): 1579 `both of them` (43.9%),
183 `none of them` (5.1%), 1832 a single object. The majority-class baseline is
therefore **43.9%**; always answering the first option scores **26.1%**.

## Scoring

`verify()` extracts the model's final answer (`_extract_answer` /
`_strip_reasoning` / `_clean_candidate`), then resolves it to one of the four
candidate labels (`match_label`):

1. **Verbatim match** — article- and punctuation-insensitive equality with a
   candidate (`_label_key`). Sets `exact`. Aliases (`both`, `neither`, `DK`, …)
   each map to exactly one fixed label.
2. **Unambiguous containment** — the answer contains exactly one candidate
   (when one candidate nests inside another, the most specific wins). Scores
   without `exact`.
3. Otherwise the answer resolves to **no label** and scores `0.0`. An answer
   naming two candidates is ambiguous by construction, so it never scores.

Reward is `1.0` iff the resolved label is the gold label. Empty output scores
`0.0` and never raises.

## Metrics

`compute_metrics` reports:

- `mean_reward` — mean per-sample CO accuracy (also the reward).
- `exact_match_rate` — fraction that answered the gold label verbatim.
- `parse_rate` — fraction where a non-empty answer phrase was extracted.
- `label_resolve_rate` — fraction that resolved to any candidate label; a low
  value means the model is not following the copy-a-candidate format.
- `accuracy_both_of_them` / `accuracy_none_of_them` — accuracy on those gold
  slices. A model that never answers `both of them` — 44% of the corpus —
  shows up as a near-zero slice even when the headline number looks healthy.

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
5-row slice sampled from that file, covering a `both of them` gold, a
`none of them` gold, and both single-object cases.

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

Note: row 3 (index 3) in the example rollouts is intentionally wrong (reward
0.0) — it echoes both of the question's options back, the degenerate answer the
scorer must not credit. The remaining four rows score 1.0 and cover a verbatim
match, an alias (`neither` → `none of them`), and a label recovered from a
longer sentence.

## Run

```bash
gym env start --resources-server spartqa --model-type vllm_model
```

No API keys are required — all scoring is rule-based.

## Test

```bash
gym env test --resources-server spartqa
```
