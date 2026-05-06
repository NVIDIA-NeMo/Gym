# Harness Extension Architecture

This document describes how to extend `terminal_multi_harness` while keeping
the codebase clean and avoiding regressions to already-supported harnesses.

## Core principle

Use shared infrastructure where behavior is truly shared, but keep harness
identity explicit.

The recommended pattern is:

- one shared implementation:
  - `app.py`
  - `common/response_utils.py`
  - `common/verification_utils.py`
- one packaged RL env config per harness
- one harness-specific rulebook per harness
- one harness-specific test slice per harness when behavior differs

For this project, separate harness envs are preferred over one generic agent
name shared by all harnesses.

Example:

- Codex env:
  - config:
    - `resources_servers/terminal_multi_harness/configs/terminal_multi_harness_codex.yaml`
  - agent:
    - `terminal_multi_harness_codex_agent`
- OpenCode env:
  - config:
    - `resources_servers/terminal_multi_harness/configs/terminal_multi_harness_opencode.yaml`
  - agent:
    - `terminal_multi_harness_opencode_agent`
- Agent006 env:
  - config:
    - `resources_servers/terminal_multi_harness/configs/terminal_multi_harness_agent006.yaml`
  - agent:
    - `terminal_multi_harness_agent006_agent`

The sample row can still keep a `harness` field for provenance, but the RL env
identity should be harness-specific.

## What belongs in shared code

Shared code should only contain logic that is genuinely reusable across
harnesses.

Good shared responsibilities:

- request validation
- extracting assistant message vs tool-call structure
- generic tool-schema validation
- generic checks like:
  - actual action type must match expected action type
  - actual tool name must match expected tool name
  - actual argument keys may not exceed expected keys
- configurable string-sim threshold plumbing

Good rule of thumb:

- if two harnesses would use the same rule text in their rulebook, it can live
  in shared code

## What belongs in harness-specific policy

Harness-specific policy should be expressed outside generic code first.

Prefer to encode harness-specific behavior in:

- packaged config names
- sample-generation logic
- rulebook docs in `docs/rulebooks/`
- tests
- thin normalizers or adapters

Examples of harness-specific policy:

- what counts as completion
- whether multiple tool calls are expected
- how to align batch calls
- whether a top-level `batch` tool is a single action or fan-out should be
  compared
- whether a tool needs non-exact argument matching

## Preferred extension order

When adding a new harness:

1. add the harness-specific packaged config
2. add or update the harness-specific source-pool builder
3. add the harness rulebook in `docs/rulebooks/`
4. add tests that encode the rulebook
5. patch shared code only where the rulebook proves a new shared primitive is
   needed

Do not start by stuffing more `if harness == ...` branches into shared code.

## When to refactor shared code

Refactor shared code only if a new harness reveals a reusable abstraction.

Examples:

- a second harness also needs multiple tool-call alignment
- a second harness also needs schema-first tool-call validation
- a second harness also needs the same assistant-action classifier

If the new behavior is only used by one harness, prefer a narrow adapter or a
well-scoped conditional over a large speculative abstraction.

## How to avoid breaking existing harnesses

When extending the codebase:

- keep existing packaged configs unchanged unless there is a clear bug
- do not rename existing agent identities casually
- avoid changing generic comparator behavior without an explicit rulebook-backed
  reason
- add tests that pin the existing supported harness behavior before refactoring
- add new behavior behind new sample shapes or new harness configs

The default standard is additive change, not cross-harness rewrites.

## File-level guidance

### `configs/`

Add a new packaged config per harness.

This is where the RL env identity becomes stable for:

- `agent_ref`
- rollout dispatch
- downstream trainset provenance

### `common/response_utils.py`

Only touch this file if the new harness changes the structural extraction of
assistant actions.

Examples:

- message vs tool call
- single tool call vs multiple tool calls
- how actual model output is normalized before comparison

### `common/verification_utils.py`

Only touch this file if the new harness needs:

- a new shared action primitive
- a new shared generic gate
- a reusable comparison helper

Do not use it as a dumping ground for harness-specific special cases.
When tool-specific comparison is needed, dispatch it by `(harness, tool_name)`,
not by `tool_name` alone. That keeps same-named tools in different harnesses
from inheriting each other's match rules.

### `tests/`

Tests should mirror the rulebook.

At minimum, add tests for:

- completion behavior
- expected tool-call matching
- wrong tool name
- schema-invalid actual tool calls
- extra actual params
- any non-exact matching rule
- any multiple-tool-call behavior

### `docs/`

Every harness should have:

- a profiling note in the run directory
- a rulebook in the run directory
- reusable architectural guidance here only when the lesson generalizes

`docs/` in this package should capture the reusable process and extension
pattern, not every experiment detail.

## Recommended contract with upstream collection runs

`terminal_multi_harness` should not be the place where raw harness artifacts are
interpreted for the first time.

Upstream collection should already provide:

- normalized `expected_action`
- declared tool schemas
- stable harness tag
- stable harness-specific `agent_ref`

That keeps the verifier focused on matching logic rather than collection
reconstruction.

## Anti-patterns

Avoid these:

- inferring verifier rules from harness source alone
- mixing historical experimental assumptions directly into shared code
- sharing one generic agent identity across all harnesses
- changing token-cap semantics per harness without an explicit decision
- comparing raw runtime fan-out when the reward target is the top-level model
  action
- adding broad harness conditionals before the harness rulebook is written

## Minimal onboarding checklist for a new harness

1. create a run-specific profiling branch
2. collect real artifacts
3. write expected-answer stats
4. write and review the match-rules doc
5. add the packaged harness config
6. patch the shared code only as needed
7. add harness-specific tests
8. validate with real-data smoke
9. run the full rollout set
10. only then declare the harness supported
