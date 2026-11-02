# Long-context interleaved document QA

A long document is read in parts. A question follows each part and may be answered only
from the parts read so far:

```
[Chunk 1 + Q1][A1][Chunk 2 + Q2][A2] … [Chunk N + QN][AN]
```

Answering `Qk` may use `C1…Ck` and nothing later — a growing causal prefix. That property
is the point of the task: it is what separates this from "paste the whole document, then
ask a list of questions", which looks the same but trains something else entirely.

This server scores one answer against that prefix.

## Verification

Three stages run cheapest-first, so a pair that cannot pass never costs a judge call.

**1. Hard filters** (no API call) — in `quality.py`:

| Check | Rejects |
|---|---|
| Length caps | question > 600 chars, answer > 2500 chars |
| Degeneracy | empty, repeated word runs, duplicate trigrams, punctuation salad, soft hyphens |
| Refusal | a refusal phrase in the answer's opening 160 characters |
| Meta-reference | a question naming the *presentation* ("the chunk above", "so far") rather than the content |
| Evidence | no cited quote locatable in the prefix |
| Verbatim | `verbatim` answers that do not occur in the prefix word for word |
| Constraint | an explicit response requirement the answer breaks |
| Depth | evidence nearer than `min_evidence_distance_chars` (off by default) |

Two deliberate exceptions, both carried over from the reference implementation:

- A bare referent to the one document ("according to the text") is ordinary English, not a
  meta-reference. Only position words stay banned.
- The `unanswerable` family is *supposed* to refuse, so the refusal detector is skipped
  for it. For every other family a refusal only counts when it opens the answer, because a
  long grounded answer may legitimately note "X does not mention Y" in passing.

Quotes that the generator could not place verbatim get one fuzzy relocation pass
(`fuzzy_locate`): unicode punctuation folded, markdown emphasis dropped, whitespace
collapsed, then three-word shingle alignment at ≥ 0.8 similarity.

**2. Guessability** (optional) — set `guess_model_server` and the judge answers the
question with *no document*. If it recovers the answer, the question tests parametric
knowledge rather than long-context reading and scores 0 regardless of grounding. This
costs one cheap call and saves the expensive one.

**3. Grounding judge** — a window of `JUDGE_CONTEXT_CHARS` (1000) either side of each
located quote, plus the question and answer, go to the judge, which returns strict JSON:
`grounded`, `follows_from_quote`, `matches_type`, `rubric` (1-5), `reason`. The prefix
itself is never sent, so judge cost does not grow with context length.

## Reward

`(rubric - 1) / 4` when the verdict is grounded, supported by the quote, of the requested
family, and at or above `min_rubric`; `0.0` otherwise. So rubric 5 → 1.0, rubric 3 → 0.5,
and a rejected pair and a judged-wrong pair are both worth nothing. `rejected` and
`rejection_reasons` stay on the response either way, which is what makes
`gym eval profile` useful for diagnosing a generation run.

## Task fields

```jsonc
{
  "responses_create_params": {"input": [ ... the interleaved conversation so far ... ]},
  "question": "...",            // the question asked after the latest chunk
  "prefix": "...",              // chunks 1..k concatenated: what the model may use
  "family": "multi_hop",        // see ANSWER_SHAPE in quality.py
  "evidence": [{"quote": "..."}],  // quote(s) the answer rests on
  "answer_key": "...",          // optional: the checkable core of a prose answer
  "constraint": {"kind": "lowercase", "param": null},  // optional
  "verbatim": false             // the answer must reproduce a span of the prefix
}
```

`family` drives the expected answer shape shown to the judge. `multistep` and
`unanswerable` are *derived* families: their quotes often fail to locate, so a missing
quote is not a rejection for them.

## Run it

```bash
gym env start --resources-server lc_interleaved_qa --model-type openai_model

gym eval run --no-serve \
    --agent lc_interleaved_qa_simple_agent \
    --input resources_servers/lc_interleaved_qa/data/example.jsonl \
    --output results/lc_interleaved_qa_rollouts.jsonl
```

`data/example.jsonl` holds three hand-written tasks with short prefixes so the smoke test
is cheap; regenerate with `python create_examples.py`. Real tasks carry prefixes of tens
of thousands of characters.

## Provenance

The checks, the judge protocol and the guessability gate are adapted from an internal
long-context SDG pipeline, and cover its long-context document path only. That pipeline
also verifies SWE, graph and database sources with their own gold-task leakage checks,
and performs cross-candidate selection (diversity and per-family caps) — selection
compares candidates against each other, which `verify()` cannot do one rollout at a time,
so it belongs in the generating recipe.

# Licensing information

Code: Apache 2.0

Dependencies
- nemo_gym: Apache 2.0
