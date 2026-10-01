# ChemReason-Bench environment

[Paper](https://aclanthology.org/2026.acl-long.1535) (ACL 2026 Long Papers, pp. 33211-33248)
and [repository](https://github.com/Khadaz/ChemReason-Bench), pinned at commit
`c0b9ac2933708fcca47b1795492952cbf280e194`. Upstream publishes no tagged release and its
default branch is mutable, so the revision is pinned in `prepare.py`; an edit that preserved
the row count would otherwise move every score with nothing failing.

500 curated experimental procedures become 7,306 task instances across six families. Every
instance is text in and one JSON object out: no tools, no agent loop, no code execution. The
scorer is pure Python with no LLM judge.

## Scope

All six task families of the single `test` split are implemented, at the pinned revision:

| Task | Instances | Primary metric | What the model returns |
| --- | --- | --- | --- |
| `step_completion` | 1,483 | `step_completion_score` | An action plus minimal slots |
| `ordering` | 1,266 | `pairwise_accuracy` | Step ids in experimental order |
| `rationalization` | 1,215 | `coverage_f1` | 1-3 sentences of reasoning |
| `step_validation` | 1,148 | `f1_positive` | A score in [0, 1] |
| `condition_validation` | 1,117 | `f1_positive` | A score in [0, 1] |
| `contrastive_choice` | 1,077 | `top1_accuracy` | An option index |

`prepare.py` refuses to write anything but these counts, and counts rows that parsed rather
than bytes downloaded.

Upstream's secondary metrics are deliberately not implemented. None enters the published
headline, and `bert_score_f1` alone would pull a neural model into an otherwise
dependency-free scorer. Upstream's `range_1_400` / `range_401_500` reporting slices are not
emitted either; `benchmark_id` is carried on every row so they can be reconstructed.

## Two protocols

For the three discriminative tasks the published primary metric is the mean of two protocols
(paper appendix F.3.4, `m_t = (m_gen + m_lm) / 2`):

- **`gen`** asks for JSON, as the other three tasks do.
- **`lm`** asks for one bare decision token -- `YES`/`NO`, or an option index. It is a
  *different prompt*, not the same request with logprobs.

`prepare.py` emits both, 7,306 + 3,342 = **10,648 rows**, and `compute_metrics` reduces each
protocol separately before averaging. Scoring `gen` alone and reporting it as Primary-Overall
is not the published quantity. A dataset carrying only one protocol falls back to that
protocol rather than being halved, so a `--protocol gen` subset still scores.

## Prompting

Prompts are upstream's, not ours. The six `gen` and two `lm` user prompts are transcribed from
`predict/predict.py` and rendered at prepare time; all 10,648 are byte-identical to what that
script builds, checked row by row and re-checked after formatting touched the f-strings that
produce them. The three-line system prompt is verbatim from `predict.py`.

Upstream sends that JSON-mode system prompt on `lm` rows too, whose user prompt says "No JSON.
No extra text." That contradiction is upstream's own -- it is commented there as keeping the
layout fixed "for better vLLM parity" -- and is reproduced rather than corrected.

## Dataset format

A prepared row is flat:

| Field | Purpose |
| --- | --- |
| `task_id`, `benchmark_id` | Provenance. `benchmark_id` is the source reaction, 1-500 |
| `dataset_name`, `split` | Always `chemreason_bench` / `test`; upstream publishes one split |
| `task_type` | Selects the scorer |
| `protocol` | `gen` or `lm` |
| `question` | The rendered user prompt |
| `ground_truth` | Upstream's gold record; shape depends on `task_type` |
| `expected_step_ids` | `ordering` only: the legal step ids, in presentation order |
| `options` | `contrastive_choice` only: the option list |
| `legend` | `step_completion` only: placeholder to name map |

`expected_step_ids`, `options` and `legend` are question-side vocabulary, not gold. Upstream's
post-processors need them to canonicalize, resolve and range-check a prediction, and none is
rendered into `question`.

## Scoring

`reward` is a per-row signal in [0, 1]. **It is not the benchmark's metric.** Four of the six
published metrics are only defined over a corpus -- `f1_positive` needs the whole confusion
matrix, `step_completion_score` applies a corpus-level format-error penalty -- so
`compute_metrics` reduces per-row contributions instead of averaging rewards, and
`get_key_metrics` keeps `mean/reward` out of the headline set.

Model output is post-processed as upstream does: step tokens are canonicalized (`id2` and
`step_2` both mean `2`), repeats dropped, and once at least one legal id has matched the
unmentioned ids are appended in presentation order -- while an answer matching nothing yields
an empty list rather than a fabricated order. A contrastive choice is recovered from raw text
when the index is missing, and an out-of-range index becomes -1 rather than defaulting to
option 0. Step-completion slots additionally pass through upstream's `canonicalize_slots`,
which resolves alias keys, splits blobs like `"10 mL"`, maps reagent names to `$n$` through the
legend, and drops anything outside the whitelist.

All three of upstream's raw-text fallbacks read `obj.get("_raw", "")`, and the extractor writes
`_raw` only when JSON parsing *fails*. So a dict that parsed but lacks the requested key scores
empty rather than being scored on its own text; that is matched. One consequence is worth
recording because it looks like a porting bug: ordering's raw scan is unreachable. It runs only
when `predicted_order` is present and not a list — which means the reply parsed, so `_raw` is
absent — while a total parse failure leaves `got` at `[]`, already a list.

One deliberate departure inside those fallbacks: they receive the reply with reasoning blocks
stripped, where upstream's `_raw` is the whole answer. Scoring a reasoning model's trace
measures the trace, not the answer.

An upstream inconsistency resolved deliberately: `eval/eval_config.yaml` states step completion
as `0.5*action_em + 0.5*slot_f1`, while `eval/eval.py` and the paper both use `0.8/0.2` with a
format-error penalty. The config string is stale; code and paper agree and produced the
published numbers.

The binary tasks have one easily-inverted rule. Because the extractor returns `{"_raw": answer}`
on *every* failure, `post_binary` never sees a non-dict: its non-dict branch is dead, and an
unparseable reply falls through to the `0.5 >= 0.5` default and counts **positive**. Labelling
those negative instead moves `f1_positive` on both validation tasks.

One deliberate departure, recorded rather than hidden: when a reply contains several JSON
objects this server scores the **rightmost**, on the grounds that a later object is the
model's correction of an earlier draft. Upstream prefers a fenced block, else the span from
the first `{` to the last `}`, which on a two-object reply fails to parse at all. The
departure can only help a model that self-corrects; it is not silent, and the status field
distinguishes a parse failure from a scored answer.

### Known gap: `lm` labels

Upstream derives the `lm` label from token probabilities, never generating. Two details
matter and are not yet matched: it sums probability mass over **every vocabulary token**
whose normalised form ends in `YES` or `NO` (not a top-k argmax), and it abstains when one
side has no mass at all. This server currently takes the highest-probability decision token
among whatever alternatives it is given, reporting status `ok_logprobs` when it can — a
closer approximation than text parsing, but still not upstream's rule. Neither applies
today, because the values do not arrive at all:
`nemo_gym/responses_converter.py` constructs the output text without populating its `logprobs`
field, so chat-level logprobs are dropped on the way back into the Responses shape. Until that
changes, `lm` labels come from parsing the generated text, which agrees with upstream whenever
the reply opens with a decision token. Note that `lm` rows must not request `top_logprobs`
either: vLLM emits logprobs only when the chat-level `logprobs` flag is set, and with
`top_logprobs` set but `logprobs` unset the completion comes back empty, truncated at
`max_output_tokens` -- a full run scored 0.00 on all three `lm` tasks that way. Measured on a full Llama-3.1-8B run, 1,103 of 3,342
`lm` replies came back JSON-shaped and 20 were ambiguous or carried no decision token.

## Harness validation

Model-free checks only; no model is involved in any number here.

- **Gold as prediction.** Upstream's own answers replayed through this scorer and through
  upstream's `eval/eval.py` agree on all six primary metrics and on Primary-Overall
  (0.988983) to six decimal places. Repeated with replies delivered bare, `<think>`-wrapped
  and fenced, and across both protocols: unchanged.
- **Gold cannot reach 100 on `step_completion`.** This is an upstream data property the port
  reproduces exactly, not a defect here: 460 of 1,483 rows have empty gold slots and
  `slot_f1({}, {})` is 0 by construction, 2 rows use slot keys outside the schema (`reagents`,
  `through`), and 6 carry `duration_unit: "day"`, which upstream's own legality check rejects.

  There are two ceilings, and which one applies depends on how much of upstream's pipeline is
  in play. Scored by `eval.py` alone, gold caps at **0.9339** — that is the figure the
  six-decimal agreement above is measured against. Through the real pipeline, where
  `canonicalize_slots` runs first as it does for any model answer, gold caps at **0.9272**,
  because canonicalization alters 297 of the 1,483 gold slot sets. A model run goes through
  the second path, so 0.9272 is the ceiling that bounds a reported SC-Score.
- **Negative controls over all 7,306 rows**, not a sample. An empty prediction scores 0.00
  Primary-Overall. **A fixed constant answer scores 27.54** -- `f1_positive` pays 0.627 and
  0.729 on the two validation tasks because always-positive earns good positive-class F1 when
  46-57% of gold labels are positive. Reversing an ordering scores 0.000 on that task; adding
  one illegal unit to an otherwise perfect step-completion answer zeroes the task through the
  format-error penalty.

That 27.54 floor is the single most important number for reading this benchmark: a large part
of a weak model's headline is reachable without answering anything.

## Reproduction

Full runs through this port: all 10,648 prepared rows (7,306 `gen` + 3,342 `lm`), temperature 0,
one pass per instance, vLLM serving the policy locally. `paper` columns are the per-task rows
from the paper's appendix; `ours` and the published Primary-Overall both average the two
protocols on the three discriminative tasks, so they are directly comparable.

**Qwen2.5-7B-Instruct** — Primary-Overall **53.73** vs published 53.94 (-0.21).
27 of 10,648 replies carried no JSON, 2 no decision token.

| task | ours gen | ours lm | ours | paper gen | paper lm |
|---|---|---|---|---|---|
| ordering | 78.43 | — | 78.43 | 78.74 | — |
| contrastive_choice | 63.42 | 64.53 | 63.97 | 63.42 | 65.65 |
| step_validation | 74.68 | 68.76 | 71.72 | 75.10 | 69.35 |
| condition_validation | 80.72 | 85.19 | 82.95 | 80.85 | 85.06 |
| step_completion | 7.23 | — | 7.23 | 7.25 | — |
| rationalization | 18.09 | — | 18.09 | 17.96 | — |

**Llama-3.1-8B-Instruct** — Primary-Overall **51.68** vs published 49.45 (+2.23).
8 of 10,648 replies carried no decision token.

| task | ours gen | ours lm | ours | paper gen | paper lm |
|---|---|---|---|---|---|
| ordering | 75.31 | — | 75.31 | 73.14 | — |
| contrastive_choice | 62.12 | 63.79 | 62.95 | 62.58 | 61.37 |
| step_validation | 73.28 | 50.07 | 61.67 | 70.90 | 45.45 |
| condition_validation | 81.31 | 69.09 | 75.20 | 80.80 | 61.79 |
| step_completion | 12.21 | — | 12.21 | 9.23 | — |
| rationalization | 22.71 | — | 22.71 | 22.87 | — |

Upstream publishes neither a spread nor a run count, so no significance test is constructible
against these figures. They are reported side by side; neither pairing is a claim of a match.

Qwen lands within 0.5 of the published row on all six tasks. Llama's +2.23 is concentrated in
the two `lm` columns, where this port reads 50.07 and 69.09 against 45.45 and 61.79 — the
largest per-task gaps in either table, and the tasks most exposed to the labelling difference
described under [Known gap: `lm` labels](#known-gap-lm-labels). Read that gap as the leading
explanation for Llama's delta rather than as a model-quality difference.

## Quickstart

```bash
gym eval prepare --benchmark chemreason_bench
gym eval run --benchmark chemreason_bench --split benchmark \
  --model-type vllm_model --model Qwen/Qwen2.5-7B-Instruct \
  --temperature 0.0 --output results/rollouts.jsonl
```

Smoke subsets, neither of which is a scored population:

```bash
gym eval prepare --benchmark chemreason_bench +prepare_script_args.limit=50
# lm rows trail the gen rows, so --limit alone never reaches them:
gym eval prepare --benchmark chemreason_bench \
  +prepare_script_args.protocol=lm +prepare_script_args.limit=50
```

Upstream evaluates with deterministic decoding (`temperature = 0`) and one pass per instance.

## Tests

```bash
gym env test --resources-server chemreason_bench
```

## Licensing

Code: Apache 2.0

ChemReason-Bench data: CC BY 4.0, per upstream's `DATA_LICENSE`, which states the data may be
shared and adapted "for any purpose, including commercial use" with attribution. Upstream's
own `LICENSE` is Apache 2.0 but still carries the unfilled boilerplate
`Copyright [yyyy] [name of copyright owner]`, so no licensor is named.

Upstream provenance is a composite and is not restated in the repository. Per the paper's
section 4, ChemReason-Bench derives from two curated procedure collections, OpenExp (Liu et
al., 2024) and ChemTrans (Zeng et al., 2023), which in turn aggregate USPTO (Lowe, 2017), the
Open Reaction Database (Kearnes et al., 2021) and Organic Syntheses. Those carry their own
terms, which a downstream CC BY 4.0 card cannot unilaterally relicense. The instances are
template-rendered derivatives of canonicalized action sequences rather than verbatim
redistribution. No benchmark data is committed here; `prepare.py` downloads at run time.

Attribution, as upstream suggests: ChemReason-Bench Authors, ChemReason-Bench dataset,
licensed under CC BY 4.0.
