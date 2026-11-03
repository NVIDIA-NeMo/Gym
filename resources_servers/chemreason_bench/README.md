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

The binary tasks have one easily-inverted rule. In the JSON evaluation path `chat_json_hf` wraps
a *failed* parse as `{"_raw": answer}`, a dict with no score, so an unparseable reply falls
through to the `0.5 >= 0.5` default and counts **positive**; a reply that is valid JSON but not
an object (`[1, 2]`, `null`, `"YES"`) reaches `post_binary` as a non-dict and counts
**negative**. Getting either half backwards moves `f1_positive` on both validation tasks.

One deliberate departure, recorded rather than hidden: when a reply contains several JSON
objects this server scores the **rightmost**, on the grounds that a later object is the
model's correction of an earlier draft. Upstream prefers a fenced block, else the span from
the first `{` to the last `}`, which on a two-object reply fails to parse at all. The
departure can only help a model that self-corrects; it is not silent, and the status field
distinguishes a parse failure from a scored answer.

### `lm` labels

Upstream decides an `lm` row from token probabilities, never from text. Each `lm` row
therefore asks for them: `logprobs` is not a Responses API field and `top_logprobs` alone is
inert in vLLM, so both travel on the row's `metadata.extra_body`, and
`nemo_gym/responses_converter.py` carries the choice-level logprobs back onto the output text.
The decision follows the paper's appendix F.3.2: eq. (1), a softmax over `{YES, NO}` with
label `P(YES) >= 0.5`, and eq. (2), an argmax over the option indices. The distribution is read
at the first token of the output text; with a reasoning parser that is the first token after the
reasoning block, where upstream reads the first generated token. When only one of YES/NO is inside
the top-k window the other is given the window's smallest probability rather than abstaining,
so a confident NO is not turned into the positive default. A row with no visible
candidate takes upstream's own fallbacks (0.5, which counts positive; index -1) and is reported
as `lm_abstained`.

Two deliberate departures from `predict.py`, the code that produced the published numbers. It
matches any token *ending* in YES/NO over the full vocabulary -- 153 Llama-3.1 tokens count as
NO against 21 as YES -- a tail no top-k API can see; following the paper's rule instead costs
<1 point on Llama-3.1-8B. And it requires two visible option indices, which over a top-k window
zeroed every Phi-3-mini contrastive row; one is enough here. The window defaults to vLLM's
ceiling of 20 (`+prepare_script_args.top_logprobs`); a model reporting many `lm_abstained`
rows needs it and vLLM's `--max-logprobs` raised together (Phi-3-mini: 1000).

## Comparison with upstream

Three checks, strongest first. All runs: the full 10,648 rows (7,306 `gen` + 3,342 `lm`),
temperature 0, one pass. Gym serves the model with vLLM; upstream's harness is `predict.py` +
`eval.py` at the pinned commit, run unmodified on the same weights.

| Primary-Overall | Gym | upstream harness on same weights | paper |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 53.82 | 53.82 | 53.94 |
| Llama-3.1-8B-Instruct (paper's double BOS) | 49.35 | 49.43 | 49.45 |
| Llama-3.1-8B-Instruct (single BOS, default) | 50.43 | — | — |
| Phi-3-mini-4k-instruct | 43.55 | — | 43.51 |

### Per task

`task` is the published quantity: the mean of `gen` and `lm` for the three discriminative
tasks, the `gen` value otherwise. Primary-Overall is the unweighted mean of the six.

**Qwen2.5-7B-Instruct**

| task | Gym gen | paper gen | Gym lm | paper lm | Gym task | upstream task | paper task |
|---|---|---|---|---|---|---|---|
| ordering | 78.70 | 78.74 | — | — | 78.70 | 78.51 | 78.74 |
| contrastive_choice | 63.51 | 63.42 | 65.00 | 65.65 | 64.25 | 64.21 | 64.53 |
| step_validation | 75.04 | 75.10 | 68.90 | 69.35 | 71.97 | 72.16 | 72.23 |
| condition_validation | 80.59 | 80.85 | 85.08 | 85.06 | 82.83 | 82.95 | 82.95 |
| step_completion | 7.10 | 7.25 | — | — | 7.10 | 6.97 | 7.25 |
| rationalization | 18.05 | 17.96 | — | — | 18.05 | 18.11 | 17.96 |

**Llama-3.1-8B-Instruct**, under the paper's double-BOS prompt (see below)

| task | Gym gen | paper gen | Gym lm | paper lm | Gym task | upstream task | paper task |
|---|---|---|---|---|---|---|---|
| ordering | 73.15 | 73.14 | — | — | 73.15 | 73.13 | 73.14 |
| contrastive_choice | 62.40 | 62.58 | 61.00 | 61.37 | 61.70 | 61.65 | 61.98 |
| step_validation | 70.86 | 70.90 | 44.38 | 45.45 | 57.62 | 57.99 | 58.18 |
| condition_validation | 80.88 | 80.80 | 61.44 | 61.79 | 71.16 | 71.41 | 71.30 |
| step_completion | 9.48 | 9.23 | — | — | 9.48 | 9.43 | 9.22 |
| rationalization | 22.99 | 22.87 | — | — | 22.99 | 22.94 | 22.87 |

With a single BOS, the port's default, Llama scores 50.43: ordering 75.22, step_completion
12.20, `lm` step_validation 41.75; the other cells move by less than a point.

**Phi-3-mini-4k-instruct**

| task | Gym gen | paper gen | Gym lm | paper lm | Gym task | paper task |
|---|---|---|---|---|---|---|
| ordering | 79.76 | 79.83 | — | — | 79.76 | 79.83 |
| contrastive_choice | 59.80 | 59.24 | 50.60 | 51.07 | 55.20 | 55.15 |
| step_validation | 72.04 | 72.46 | 17.93 | 16.98 | 44.99 | 44.72 |
| condition_validation | 83.39 | 83.33 | 20.14 | 20.39 | 51.76 | 51.86 |
| step_completion | 5.27 | 5.30 | — | — | 5.27 | 5.30 |
| rationalization | 24.33 | 24.19 | — | — | 24.33 | 24.19 |

### Reading the numbers

- **Every cell is within ~1 point of the paper, and within 0.4 of upstream's harness.** Upstream
  publishes neither a spread nor a run count, so no significance test is constructible;
  run-to-run drift at temperature 0 was measured at up to 0.35 on a task.
- **The published numbers carry a tokenization bug.** `predict.py` renders the chat template to
  text, which already contains the BOS token, then calls `tokenizer(text)` with the default
  `add_special_tokens=True`. Any model whose tokenizer has `add_bos_token=true` was therefore
  scored from a double-BOS prompt, on both protocols. This port sends a single BOS by default;
  the paper's condition is one override away, no code change:
  `++policy_model.responses_api_models.vllm_model.extra_body.add_special_tokens=true`.
  Upstream's own code on our Llama weights gives 49.43, so the gap to the port's 50.43 is the
  tokenization, not the model snapshot. Qwen2.5 (no BOS token) and Phi-3-mini
  (`add_bos_token=false`) are unaffected either way.
- **Phi-3.** The paper's "Phi-3-mini 7B" is `Phi-3-mini-4k-instruct` (3.8B); the 128k variant is
  a different fine-tune and scores differently. Run with `CTX=4096`, `MAX_OUT=1024` and a
  1000-token logprob window. Its collapsed `lm` validation scores reproduce, so they are a
  property of the model, not the harness.
- **Generation caps.** Upstream caps output at 96-200 new tokens per task; the Gym runs allowed
  4096. Phi-3 was not run through upstream's harness.

### Same answers, both scorers

All 21,918 generated answers from the three Gym runs were fed through upstream's parsing and
`post_*` functions and scored by upstream's `eval.py`, and the same text through this scorer.
All 18 per-task metrics and Primary-Overall agree to two decimals. 50 rows (0.23%) differ in the
prediction record without moving a score: 49 `step_completion` actions outside the allowed set,
which upstream blanks to `""` where this port keeps the string (neither matches gold), and one
three-object contrastive reply where upstream's first-`{`-to-last-`}` span fails to parse and
its raw scan takes the first option while the rightmost-object rule takes the last (gold was a
fourth option).

## Harness validation

Model-free checks.

- **Gold as prediction.** Upstream's own answers replayed through this scorer and through
  upstream's `eval.py` agree on all six primary metrics and Primary-Overall (0.988983) to six
  decimals, with replies delivered bare, `<think>`-wrapped and fenced, across both protocols.
- **Gold cannot reach 100 on `step_completion`**, an upstream data property the port reproduces:
  460 of 1,483 rows have empty gold slots and `slot_f1({}, {})` is 0, 2 rows use slot keys
  outside the schema, 6 carry `duration_unit: "day"`, which upstream's own legality check
  rejects. Gold caps at 0.9339 under `eval.py` alone and at 0.9272 through the real pipeline,
  where `canonicalize_slots` alters 297 of the 1,483 gold slot sets.
- **Negative controls over all 7,306 rows.** An empty prediction scores 0.00. **A fixed constant
  answer scores 27.54**: `f1_positive` pays 0.627 and 0.729 on the two validation tasks because
  46-57% of gold labels are positive. That floor is the key number for reading this benchmark:
  much of a weak model's headline is reachable without answering anything. Reversing an ordering
  scores 0.000 on that task; one illegal unit zeroes a perfect step-completion answer through the
  format-error penalty.

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
redistribution.

The full dataset is not committed; `prepare.py` downloads it at run time. A small amount of
upstream `benchmark_data/` is committed under CC BY 4.0 with the attribution below:
`tests/fixtures/prompts.jsonl` and `answers.jsonl` (6 rows each, verbatim),
`tests/fixtures/golden_prompts.json` (10 prompts rendered from those rows by upstream's own
builders) and `data/example.jsonl` (5 rows restructured into the Gym row shape).

Attribution, as upstream suggests: ChemReason-Bench Authors, ChemReason-Bench dataset,
licensed under CC BY 4.0.
