# Partner IF verifier

One shared grading service for text answers, frozen function history, and captured
native-agent artifacts/events. Execution adapters remain responsible for their
tools, sessions and sandboxes. Explicit per-item verifier IDs select a code check
or the shared LLM judge; IF and optional task-completion scores stay separate.

The grading service, its LLM judge, the generic code checks and every adapter are
public here, so the harness runs on its own. A benchmark built on it can keep its
task data and a few task-specific checkers or tool emulators in a private plugin
(see [Task-specific checkers and tool backends](#task-specific-checkers-and-tool-backends)).
An internal benchmark built on it does this: the public judge and generic checks
grade most of its rubrics, and only a few additional task-specific checkers and tool
emulators are hidden.

## Contract and scoring

Each row supplies the actor's unchanged `responses_create_params` separately from
`verifier_metadata.partner_if`. The current `GradingContract` contains a stable
task ID, provenance, rubric version, audit status, rubrics and optional unscored
prerequisites. Each item has one `constraint`, a `verifier` ID/params, a
`dimension` (`if` or `completion`), and optional dependencies. Provenance retains
verbatim instructions, source roles/spans, applicability and taxonomy; none of
this grader-only metadata is appended to the actor's prompt.

- Every IF rubric earns one point. Partial score is passed IF points / total IF
  points; binary reward is 1 only when every IF rubric passes.
- Completion scores are diagnostic: failure or unavailable completion evidence
  cannot lower an independently valid IF score. IF cannot depend on completion.
- Unscored prerequisites gate their dependents without enlarging the denominator.
  Raw and effective verdicts are both retained.
- Audit-unsigned contracts require explicit `allow_provisional_rubrics: true`;
  their results remain provisional. Technical failures are masked/unscorable,
  never fabricated actor zeroes.
- The runtime accepts only the current shared contract and results. It has no
  alternate text judge, pass/fail-contract migration, or historical scoped-pass
  mode. Archived receipts belong outside the runnable benchmark.

## Code checks and capture dependencies

`checks.py` is the trusted registry. Fixed checkers take no configurable rule
expressions. The generic bounded `text_contains_v1` and `text_no_uuid_v1` checks
are built in; the unused generic YAML-rule interpreters are not. Semantic criteria
such as sentence boundaries, procedural-step coverage and citation meaning remain
LLM criteria.

Deterministic checks read their required artifact through the evidence API.
An LLM item can explicitly declare capture dependencies:

```json
{"id": "llm_judge_v1", "params": {"artifacts": ["deliverable"]}}
```

`artifacts: []` means no artifact capture is required; omission or null retains
the conservative default of requiring all supplied captures to be usable. This
declaration is about evidence availability, not redaction of other judge context.
A missing capture receipt or capture error makes only dependent items unscorable.
An explicit `absent` receipt means the actor did not produce the file and is
ordinary gradable behavior. Actor/trajectory capture failure still masks the
whole attempt. Do not declare an empty list for an item that actually needs a file.

New checks need positive, negative and boundary tests plus review against saved
answers. Determinism alone does not establish fairness. Update a task's contract
when changing its grading, and rerun/regrade consistently rather than reusing
scores from a different contract.

## Task-specific checkers and tool backends

Task-specific code lives with the benchmark that needs it, in a plugin root that
Gym searches with `--search-dir` or `NEMO_GYM_EXTRA_ROOTS`; contracts still name
checkers only by verifier ID.

- `task_check_modules` on a `partner_if` server lists importable modules whose
  `CHECKERS` map verifier IDs to `Checker` objects (usually `checks.fixed(...)`).
  Each server registers them at startup; a module cannot redefine a registered ID.
- `mock_tools_app.py` serves the `ToolBackend` class named by `backend:` as
  `module:Class`, over the configured `fixture_path` and `tasks_path`.

Relative package paths in these configs resolve under the plugin roots first, then
the Gym root.

## One grounded LLM judge

Protocol `partner-if-evidence-v5.2` exposes stable `evidence_sources` with
context/candidate origins, JSON types and numbered original lines. The judge
returns strictly boolean verdicts plus reasons and references
`{source, start_line, end_line}`; the host validates them and extracts exact
quotes. Each pass needs nonblank candidate evidence. Frozen history, tool
definitions and host metadata cannot establish new candidate behavior.

Required actions cannot pass by omission; prohibitions can pass by avoiding
forbidden behavior. There is no `unexercised` field or judge abstention.
Applicability is settled in curation, not dynamically by dropping rubric points.
Invalid envelopes, missing/duplicate IDs, bad references and transport errors
remain technical failures with the original request/reply retained where
available. A malformed completion item does not invalidate independently valid
IF items. Read the benchmark's `coverage/*` metrics alongside its rates; masked
rollouts are excluded from them.

The judge sees only visible evidence, not API reasoning. Text-answer mode rejects
new tool calls and incomplete/blank answers. `frozen_tool_history` supplies
validated recorded calls and results as context without executing them.
`tool_call_response` additionally grades newly emitted function calls without
executing task backends; completed omissions and malformed arguments are
gradable failures. Live profiles capture their actual tool events and artifacts.
Where a backend requires no final answer, a response cut off by the output-token cap
ends the episode like the step cap: its calls never ran, so the completed tool
exchanges before it are graded.
Self-contained inputs are required; hidden stored conversations are not accepted.

## Examples and validation

`data/example.jsonl` contains five synthetic wiring examples, not benchmark tasks.
`configs/smoke.yaml` exposes a validation split; `configs/partner_if.yaml`
exposes the same rows as the example split. `configs/verifier.yaml` configures
saved-output reverification. `tests/verifier_cases.jsonl` is the code-checked
verifier fixture that `gym env test` runs. `example_metrics.json` contains Gym dataset
statistics, not model accuracy; `example_rollouts.jsonl` contains genuine actor
responses and shared-judge receipts.

Inspect item-level reasons and references; valid JSON is not proof of a correct
judgment. Keep model/settings fixed for comparisons and raw scores separate from
manual adjudication. Validation sign-off remains pending (`verified: false`);
synthetic wiring checks are not a benchmark baseline or fairness certification.
