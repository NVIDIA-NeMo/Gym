# litmus_agent Resources Server

## Overview

`litmus_agent` is a **domain-agnostic** answer verifier. It extracts a value
from a model response, parses that value according to the row schema, and
compares it with a known answer.

It does three things:

1. Extracts the model's final answer text from the rollout trajectory.
2. Pulls a value out of that text with a row-provided `output_regex`, or with the
   requested `answer_format` regex (the `fmt_00`–`fmt_30` wrapper family).
3. Scores it against `expected_answer` using a small `answer_type` taxonomy.

Reward is `1.0` on match, `0.0` otherwise (including when no value can be
extracted).

### What it deliberately is *not*

- **Not tied to any domain.** Domain-context fields ride along as *pass-through*
  fields (`extra="allow"`): accepted, preserved, and echoed back, but never
  required or interpreted by the scorer.
- **Executes tools only when configured to.** By default it is a pure verifier
  and scoring a tool-using rollout is identical to scoring a direct one (extract
  value → compare). When `sandbox_provider` is set it *additionally* hosts a
  single sandbox-backed code-execution tool — see
  [Sandbox-backed code-execution tool](#sandbox-backed-code-execution-tool).
- **Does not read the question.** The question lives in
  `responses_create_params.input` and is the model's concern; the scorer only
  sees the model's response and `expected_answer`.

## Answer Types

`answer_type` governs how the extracted text is **parsed** into a comparable
value:

| `answer_type` | Parsed as | Expected response |
|---------------|-----------|-------------------|
| `float`  | float (covers integers too; int-vs-float is a *scoring* concern, not parsing) | Number |
| `bool`   | `1.0`/`0.0` (accepts `1/0`, `true/false`, `yes/no`, `present/absent`, …) | Truthy/falsy token |
| `string` | raw captured text | Free text |

## Reward Rules

*How* the parsed value is compared is a separate, swappable concern. Each named
rule in `REWARD_RULES` scores a predicted value against the expected one and
returns a reward in `[0.0, 1.0]`:

| Rule | Params | Behavior |
|------|--------|----------|
| `exact`      | — | rounded integer exact match |
| `isclose`    | `rel_tol`, `abs_tol` | tight numeric equality (`math.isclose`) |
| `abs_window` | `abs_tol` | within an absolute tolerance |
| `rel_window` | `rel_tol` | within a relative tolerance of expected |
| `bool_eq`    | — | boolean equality |
| `string_eq`  | — | normalized string equality (case/whitespace-insensitive) |

Each `answer_type` maps to a **default rule** (`_DEFAULT_RULE`): `float` →
`isclose`, `bool` → `bool_eq`, `string` → `string_eq`. A `float` row that wants
rounded-integer matching opts in with `match={"rule": "exact"}`. The `isclose`
defaults come from the server's `float_rel_tol`/`float_abs_tol` config.

### Overriding the rule per row

A row may override the default for itself via the optional `match` field —
`{"rule": <name>, **params}`. This decouples scoring from `answer_type`, so the
same parsed type can be scored exactly in one row and within a window in another:

```jsonc
// integer answer accepted within ±2
{"answer_type": "float", "expected_answer": "100", "match": {"rule": "abs_window", "abs_tol": 2}}

// float accepted within 1% of expected
{"answer_type": "float", "expected_answer": "18.02", "match": {"rule": "rel_window", "rel_tol": 0.01}}
```

A malformed `match` (missing `rule`) or an unknown rule name fails loudly rather
than silently scoring `0.0`. The resolved rule is reported back as
`resolved_reward_rule`.

### Custom rules

Register a custom rule by adding an entry to `REWARD_RULES` (name → callable
taking `(predicted, expected, **params)` and returning a float in `[0.0, 1.0]`).
Rows then reference it by name in `match`.

### Numeric `property_type` compatibility

Rows without an explicit `answer_type` may carry `property_type` instead. These
values are mapped to the answer-type taxonomy for reporting:

| `property_type` | → `answer_type` |
|---|---|
| `float`, `count`, `fragment` | `float` |
| `bool`, `presence` | `bool` |

Compatibility rows retain their numeric scoring policy: `count`, `fragment`,
`bool`, and `presence` are parsed as numbers and default to rounded `exact`
matching, while `float` defaults to `isclose`. An explicit `answer_type` selects
the modern parsing policy, and an explicit `match` overrides either default.

A row with neither a supported `answer_type` nor a mappable `property_type`
fails loudly rather than silently scoring `0.0`.

## Answer Extraction

When present, `output_regex` is the preferred extractor and must contain exactly
one capture group. Otherwise, `answer_format` (`fmt_00`–`fmt_30`) selects the
registered regex used to locate the final answer (e.g. `fmt_00` → `((answer))`,
`fmt_07` → `\boxed{answer}`, `fmt_15` →
`<final_answer>answer</final_answer>`). The last match in the text wins, so a
self-correcting model's final value is the one scored.

Legacy rows without `answer_format` fall back to `use_box_format`:
`use_box_format: true` → boxed (`fmt_07`), `use_box_format: false` → double
parentheses (`fmt_00`). When `answer_format` is present it takes precedence.

> **Prefer delimited formats for numeric answers.** When the captured text is
> not a clean float, extraction falls back to the *last* number-like token in the
> capture. For the greedy open-ended formats (`fmt_22`–`fmt_30`, which capture to
> end of line) this can misparse — e.g. `Final Answer: 42 (my 7th attempt)`
> captures `42 (my 7th attempt)` and scores `7`, and `12.5 g/mol at 298K` scores
> `298`. For numeric answers, prefer a delimited format (`fmt_00` `((...))`,
> `fmt_07` `\boxed{...}`) so the value is bounded by the delimiters.

## Reward Signal

`reward` is the value returned by the resolved reward rule (binary rules return
`1.0`/`0.0`). When no value can be extracted from the response, or it is `NaN`,
`reward = 0.0`.

## Dataset Format

Each JSONL row:

- `responses_create_params.input`: the input messages (Responses API format)
- `responses_create_params.tools`: `[]` for direct answering; a tool spec when
  the optional code-execution route is enabled
- `expected_answer`: ground-truth value (string, int, or float)
- `answer_type`: one of `float`, `bool`, `string` (optional if a mappable legacy
  `property_type` is present)
- `output_regex`: optional row-specific regex with exactly one capture group;
  takes precedence over `answer_format`
- `answer_format`: optional key `fmt_00`–`fmt_30` selecting the extraction regex
- `match`: optional reward-rule override `{"rule": <name>, **params}`; defaults
  to the rule for the resolved `answer_type`
- `use_box_format`: optional legacy fallback when `answer_format` is absent
- any number of **pass-through** domain fields (e.g. `method`, `source_id`,
  `smiles`, `property`) — preserved and echoed back, never interpreted

`method` is read only by `compute_metrics` for grouping (e.g. `direct` vs a
tool-use method); it is otherwise a pass-through field.

See `data/example.jsonl` for concrete examples.

## Metrics

`compute_metrics` aggregates accuracy and mean reward, grouped by
`method` × resolved `answer_type`.

## Example Usage

Start the server (pure-verifier direct rows need no sandbox provider or API key):

```bash
gym env start \
    --model-type openai_model \
    --resources-server litmus_agent
```

Collect rollouts and score them:

```bash
gym eval run --no-serve \
    --agent litmus_agent_agent \
    --input resources_servers/litmus_agent/data/example.jsonl \
    --output resources_servers/litmus_agent/data/example_rollouts.jsonl
```

## Sandbox-backed code-execution tool

For rows that require tool use, `litmus_agent` can host a stateful Python
code-execution tool directly.

**Enabling it.** Set `sandbox_provider` (a single-key
[`nemo_gym.sandbox`](../../nemo_gym/sandbox) provider config, e.g.
`{opensandbox: {...}}`) and `sandbox_spec` (image, resources, ttl). The tool is
then served at `/{code_exec_tool_name}` (default `stateful_python_code_exec`),
the same name the dataset rows advertise in `responses_create_params.tools`. When
`sandbox_provider` is unset the server is a pure verifier and serves no tool.

**Lifecycle.** A sandbox is created **lazily** on a session's first tool call and
reaped when that rollout's `/verify` runs (and any stragglers on server
shutdown). Direct (no-tool) rows never create a sandbox, so a run with no
tool-using rows works even without provider credentials.

**Statefulness (replay model).** The sandbox runs one-shot commands, not a live
Python kernel. Statefulness across calls within a session is emulated by
replaying every prior **known-good** cell — with its stdout/stderr suppressed —
ahead of the newest cell, so only the newest cell's output is returned. A cell is
retained in the session's history only after it runs cleanly; a cell that raises
returns its traceback and is dropped. This is faithful for pure, deterministic
code; the tradeoff is that prior cells repeat their side effects on every call.

> **Replay cost is O(N²) and shares one timeout.** Each call replays all prior
> cells before the newest one, so a session's total exec work grows quadratically
> in cell count, and every call's replay is charged against the same
> `code_exec_timeout_s` (default 120s) as the new cell. A long session — many
> cells, or one slow/non-deterministic prior cell — can start timing out on
> *replay alone*, surfacing as a spurious failure of the newest cell. Keep tool
> sessions short (order tens of cells), and raise `code_exec_timeout_s` if cells
> are individually expensive. A replay that fails resets the session (its cell
> history is dropped) rather than scoring on desynced state.

Relevant config keys: `sandbox_provider`, `sandbox_spec`, `code_exec_tool_name`,
`code_exec_timeout_s`, `code_exec_max_output_chars`, `code_exec_user`.

## Licensing

Code: Apache 2.0
