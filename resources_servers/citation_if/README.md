# `citation_if` — citation instruction-following reward checker

A resources server that scores whether a model followed a **citation instruction**: did it
cite, in the grammar it was told to use, only IDs that exist, and the ones that were
required? One `/verify` endpoint, binary reward, CPU-only string matching — no model calls,
no GPU.

The grammar regexes and verdict semantics are deliberately aligned with citation *evaluation*
suites, so a model trained against this reward is graded on the same notion of "a citation"
that it will be measured on.

## What the reward checks

**Citation behaviour only — the answer's *content* is never read.** An answer of pure
gibberish carrying the exact required citation scores 1.0. This is deliberate, not an
oversight: the reward is for instruction-following, and the data carries no gold answer to
grade against. Any `expected_answer_patterns` in row metadata is dataset build tooling and
must not be consumed here.

**But an answer must exist.** "Do not grade what the answer says" is not the same as "do not
ask whether one was written", and the gap between those was a live reward hole. A pilot
rollout once scored 1.0 on this complete output:

```
<cite>citation_7367:snippet_2</cite>
```

Right grammar, right required ID, and no answer at all. Gate 0 passed it because the raw
string is not empty — it is 36 characters of markup. **Gate 0b** closes this.

Both properties are locked by an adjacent PAIR of tests
(`test_item5_scorer_does_not_grade_answer_quality` and
`test_citation_with_no_answer_scores_zero`) so a future "fix" to one cannot silently undo the
other. If you change either, change both deliberately.

### Gate sequence — `scorer.py::score_citation_if` (cite mode)

| # | gate | fails when |
|---|---|---|
| 0 | `structural` | empty final text, or the policy emitted a tool call |
| 0b | `no_answer` | nothing but citation markup — no word character survives once citations are removed. **Presence test, not a length floor**: a one-word answer scores 1. Claim text counts, so `claim_wrap_xml` answers written entirely inside the span are fine (uses `_answer_text()`, not `_strip_spans()`) |
| 1 | `malformed_attempt` | **strict**: strip well-formed citation spans, then ANY residual ID-shaped token remains. Tag grammars also require `open == parsed == close` |
| 2 | `must_cite` | fewer than `min_valid_citations` valid citations parsed |
| 3 | `no_hallucination` | a parsed ID is not in `valid_id_set` |
| 4 | `correctness_missed_gold` / `correctness_over_cap` | (when `expected_ids` is set) a required ID was not cited, or `\|cited\| > \|expected\| + expected_slack` |

`mode=no_cite` rows invert gates 1–4: reward 1 iff zero citations of any shape. Gate 0b is
cite-mode only — abstain rows are already covered by gate 0's non-empty check.

**Why gate 1 is strict.** It replaces an existence check that fired only when *no* citation
parsed. Under that weaker rule a single well-formed citation whitelisted everything else in
the answer: raw-ID leakage, `Sources:` trailers, comma+space lists, empty markup, and raw
hallucinated IDs. Stripping spans first and then rejecting any residual ID-shaped token is
what closes it.

## Relationship to `format_verification/citation_format`

That server also grades citation markers, so the overlap is worth stating plainly. It checks
**marker presence**: every string in `expected_markers` must appear via `in text`, and any
regex-matched marker outside that set counts as spurious. Prompts are standalone formatting
instructions.

This server grades citation behaviour on a **frozen retrieval trajectory**, where the ID space
comes from the documents in the prompt. That adds checks marker matching cannot express:

| | `citation_format` | `citation_if` |
|---|---|---|
| ID validity | markers are literals from the row | parsed IDs checked against `valid_id_set` (gate 3) |
| answer must exist | no — a bare `[ref:1]` scores 1.0 | gate 0b rejects markup with no answer |
| malformed attempts | not detected | gate 1 strips spans, rejects residual ID-shaped tokens |
| recall vs precision | all expected, none extra | `expected_ids` + `expected_slack` allows over-citation headroom |
| abstention | not modelled | `mode=no_cite` inverts gates 1-4 |
| tool-calling policy | not modelled | gate 0 scores a tool call 0; catch-all endpoint ends the rollout |

The bare-marker case is the sharpest one, and it is not hypothetical: a pilot rollout here
scored 1.0 on citation markup with no answer, which is what gate 0b was written for.

Use `citation_format` for "did the output use the requested reference style". Use this server
for "did the model cite the right real sources, and actually answer".

## Row verifier schema

```json
{
  "type": "citation_if",
  "mode": "cite | no_cite",
  "grammar": "cite_xml | ascii_brackets | claim_wrap_xml | fullwidth_brackets |
              ref_colon | double_angle | web_brackets | paren_part | markdown_footnote",
  "id_kind": "full_source | snippet",
  "id_regex": "citation_[a-z0-9]{4}",
  "valid_id_set": ["citation_27bx", "..."],
  "expected_ids": ["citation_27bx"],
  "expected_slack": 1,
  "min_valid_citations": 1
}
```

`id_regex` is **authoritative** for grammar parsing — `GRAMMAR_TEMPLATES` is compiled per row
against it, so random-ID and numeric-ID schemes coexist in one pool. Rows that omit it fall
back to a numeric default keyed by `id_kind`.

### Two detectors, deliberately different widths

- **Row `id_regex`** — parses citations under the row's own grammar.
- **Broad `ATTEMPT_REGEX`** (`citation_[A-Za-z0-9_:]+`) — the gate-1 residual scan.

`ATTEMPT_REGEX` must stay a **superset** of every ID scheme, so numeric-style leakage
(`… [citation_27bx]. Sources: citation_1`) is still caught on a random-ID row. **Never narrow
it to the row's `id_regex`** — that reintroduces the leak it exists to catch.

`RESERVED_EXAMPLE_ID` (`citation_xmpl`) appears in instruction examples only and is never
valid in any row, so echoing the example always scores 0.

`match_details.cited_ids` is logged on **every** verdict, so rollouts stay re-scorable offline
under different correctness semantics without re-rolling. Do not remove it.

Two grammars — `curly_double` and `angle_pipe` — are marked `holdout: True` and must never be
assigned to training rows. They exist to detect grid overfitting: a model that has learned
"citation" rather than "these nine templates" should handle them.

## Building your own data

The training corpus this reward was developed against is internal and not published, so this
section documents the row schema and how rows are generated. `data/example.jsonl` is a working
5-row instance of everything below — read it alongside this.

### Row schema — four top-level keys

```json
{
  "responses_create_params": { "input": ["..."], "tools": ["..."], "tool_choice": "auto" },
  "verifier":  { "type": "citation_if", "mode": "cite", "grammar": "ascii_brackets" },
  "agent_ref": { "type": "responses_api_agents", "name": "citation_if_simple_agent" },
  "_gen_config": { "placement": "system", "n_documents": 3 }
}
```

| key | required | what it is |
|---|---|---|
| `responses_create_params` | yes | the frozen conversation, in Responses API form. This is the prompt. |
| `verifier` | yes | what the reward checks — full schema above under "Row verifier schema" |
| `agent_ref` | yes | must name an agent with `max_steps: 1` (see the tool-call section) |
| `_gen_config` | no | build metadata only. **The reward never reads it.** Anything here — including any `expected_answer_patterns` — is dataset tooling, not scoring input. |

### Trajectory shape

`responses_create_params.input` is a retrieval conversation that has already happened, ending at
the point where the model must answer. In order:

1. **system** — the assistant's system prompt, carrying the citation instruction if
   `placement: system`
2. **user** — the question
3. **`function_call` / `function_call_output` pairs** — one per search round. Each output carries
   the retrieved documents and their citation IDs; this is where the ID space comes from.
4. **user** — the closing turn that ends retrieval

The citation instruction can sit in the system prompt, before or after the question, or in the
final turn. Varying that placement matters: instructions far from the answer point are
substantially harder to follow, so a pool weighted to one placement measures something narrower
than it appears to.

### Generation procedure

Per row:

1. Pick the axes — grammar (9 available), `id_kind` (`full_source` or `snippet`), `mode`
   (`cite` / `no_cite`), instruction placement, and how many distractor documents to include.
2. Assemble documents — one or more gold documents that answer the question, plus distractors.
   Same-topic distractors are much harder than random ones; the example rows use unrelated
   distractors and are correspondingly easy.
3. Assign each document a citation ID matching the row's `id_regex`, and render the search
   rounds as `function_call` / `function_call_output` pairs.
4. Write the citation instruction into the chosen placement, rendering the grammar's syntax and
   using `citation_xmpl` for any example ID — `RESERVED_EXAMPLE_ID` is never valid in any row, so
   a model that copies the example scores 0.
5. Append the closing user turn.
6. Derive the verifier from what actually landed in the prompt: `valid_id_set` is every ID present
   in the retrieved documents, and `expected_ids` is the subset the answer must cite.

Step 6 is the one to get right. `valid_id_set` must be derived from the rendered prompt, not from
the source corpus — if it contains an ID the model never saw, gate 3 will fail rows for
hallucinating something that was legitimately unavailable.

## Rollout wiring and tool-call bounds

**⚠ This server does NOT cap tool calls. Whether a tool call is a failure is decided entirely by
your data.** If you build a dataset for this reward, read this section first.

What the server does on a tool call: the catch-all `POST /{tool_name}` returns
`TERMINAL_TOOL_RESPONSE` ("No further tool results are available. Provide your final answer.") and
gate 0 scores the rollout **0**. There is no `max_tool_calls` in the rows and no cap enforced in
`app.py` — the rows keep `tools` and `tool_choice="auto"`, so the policy is free to call a tool.

**What your rows must contain.** Every row has to end with a final user turn that closes
retrieval, so that a tool call at that point is unambiguously non-compliance rather than
reasonable behaviour. All five rows in `data/example.jsonl` end exactly like this — the last
entry of `responses_create_params.input`:

```json
{"role": "user", "content": "Research is complete. Provide your final answer now."}
```

A compliant model, trained or not, should not call a tool after being told that, so scoring the
call 0 is correct. The wording is not special; any instruction that unambiguously ends retrieval
works.

**Without that closing turn this reward is wrong for your data.** Scoring every tool call 0 would
teach the policy to terminate tool-call trajectories immediately — a real capability regression,
not the citation behaviour you wanted. If your rows cannot carry such a turn, bound tool use in
the request instead:

Cap the number of calls:

```json
{"responses_create_params": {"max_tool_calls": 2, "tool_choice": "auto", "tools": ["..."]}}
```

Or forbid them outright:

```json
{"responses_create_params": {"tool_choice": "none"}}
```

Both are inert in `data/example.jsonl` today: it ships `max_tool_calls: null` and
`tool_choice: "auto"`, relying entirely on the closing turn.

**Also note the step bound lives in the agent config, not in the rows.**
`configs/citation_if.yaml` sets `max_steps: 1`, so the policy never gets a second turn. Run these
rows under a multi-step agent and that bound disappears — the catch-all becomes the only backstop.
The bound is on *steps*, not individual calls: rows set `parallel_tool_calls: true`, so one step
may emit several calls at once (`extract_response_shape` returns the count; see
`test_counts_function_calls`). One or many, any call scores 0.

Tests `test_verify_tool_call_scores_zero` and `test_tool_catchall_returns_terminal_response` lock
the scoring and the catch-all respectively.

## Tests

```bash
pytest resources_servers/citation_if/tests/ -q   # 327 passing
```

- **`test_citation_if.py`** — hand-written gate tests, one or more per gate and per failure
  mode.
- **`test_reward_hack_matrix.py`** — a generated matrix of **16 known reward-hack classes ×
  9 grammars × 2 ID granularities = 265 cells**. Generated, not hand-written; the pass bar is
  100% of cells. This is permanent regression coverage — it should be green before any
  dataset built for this reward is used.
- **`test_residual_fuzz.py`** — 3 property tests × 1,000 mutations each, in **both**
  directions. Leak mutations must all score 0; benign mutations (whitespace, punctuation,
  unicode quotes) must all stay 1. An over-strict gate is as much a regression as a leaky one,
  and far harder to notice.

Two notes for anyone extending the matrix:

**Class 16 (`citation_only_no_answer`) came from a real rollout, not from anticipation** — it
is the bare-citation hack described above, added as a fixture the same day it was observed.
Its `claim_wrap_xml` cell deliberately uses a different string: a well-formed claim wrap
*contains* its claim, so a bare citation is not expressible there, and an empty claim is not a
citation at all (`_iter_spans` drops it, so it fails gate 1 on tag balance). Getting that cell
right is what distinguishes a correct gate-0b implementation from one built on
`_strip_spans`.

**Classes 8, 10 and 15 encode exact-match correctness semantics.** If those semantics are ever
revisited, re-derive the expected verdicts and regenerate the matrix — never hand-edit cells.

## The example dataset

`data/example.jsonl` holds 5 rows covering four citation grammars, both `id_kind` values,
and one `no_cite` row, so the inverted-gate path is exercised too. Each row is a complete
frozen trajectory: system instruction, question, search rounds, then a final user turn that
closes retrieval and asks for synthesis.

It is a **contract example, not a difficulty example.** The rows are ~1.7 KB with 3
documents and topically unrelated distractors. Production training rows are far larger —
tens of documents, same-topic distractors that genuinely compete, and prompts two orders of
magnitude longer. Use these rows to check wiring and scoring behaviour, not to gauge how
hard the task is.

## Layout

```
app.py                          FastAPI server (/verify + catch-all tool endpoint)
scorer.py                       pure scoring logic, no FastAPI dependency
configs/citation_if.yaml        server + citation_if_simple_agent config
data/example.jsonl              committed example rows
data/train.jsonl                gitignored build output — not in the repo
tests/                          gate tests + reward-hack fixture matrix + residual fuzz
```

## Licensing information

Code: Apache 2.0

Dependencies:
- nemo_gym: Apache 2.0

The example rows in `data/example.jsonl` are synthetic, written for this server. No external
corpus is redistributed here.
