# Benchmark Review Checklist

Use this checklist for benchmark, eval, verifier, agent, or preparation changes. Review
the runtime contract first; style findings matter only after the evaluation is correct.

## 1. Establish the Intended Evaluation

- Identify the upstream benchmark version, split, task count, license, official prompt,
  official evaluator, and headline metric.
- State the evaluation unit: row, problem, subtask, test group, episode, or suite member.
- Identify whether the change reuses a Gym verifier, changes a shared verifier, adds an
  agent loop, or wraps an external harness.
- List every generated artifact and every consumer of it.
- If the PR claims parity, find the per-example comparison or reproduce it. A similar
  aggregate alone is weak evidence.

Build a contract ledger before reviewing implementation details:

| Semantic value | Source field | Prepared field | Prompt/agent consumer | Verifier consumer | Metric consumer |
| --- | --- | --- | --- | --- | --- |
| task ID | | | | | |
| question/input | | | | | |
| reference/rubric | | | | | |
| test/group selector | | | | | |
| weight/max score | | | | | |

Any blank scoring-relevant cell is a likely bug or missing test.

## 2. Code and Repository Consistency

### Scope and layout

- The change uses an existing verifier/agent when semantics already match.
- New components live in the expected top-level directory; benchmark-only data and
  composition live under `benchmarks/`.
- A single benchmark config locally declares exactly one `type: benchmark` dataset.
  Multiple benchmark declarations are treated as an eval suite.
- Instance names are isolated when a benchmark overrides shared verifier settings.
- There are no unrelated generated files or drive-by changes.
- New source files have the NVIDIA SPDX header and dependencies are license-compatible.

### Imports and dependencies

- `prepare.py` imports successfully as a repository-root module, the same way
  `gym eval prepare` imports it.
- Package-local imports are package-correct; they do not depend on running the script's
  directory as `sys.path[0]`.
- Dependencies used during preparation exist in the repository-root environment.
  A package listed only in a resources server's isolated requirements is insufficient.
- Optional heavy dependencies are lazy where appropriate, and missing optional packages
  fail with a useful message.
- Async HTTP follows `AGENTS.md`; external libraries do not silently introduce an
  incompatible HTTP/concurrency path.

### Generated artifacts

- `prepare()` returns the exact configured `jsonl_fpath` as a `Path`.
- Side artifacts such as metadata, test archives, databases, scripts, or graders are
  written to the paths used by config.
- Cross-repository outputs are resolvable through the downstream consumer's public
  name, registry, or search path; their directory layout alone is not treated as proof.
- `.gitignore` covers generated artifacts without hiding intentional fixtures.
- Preparation is deterministic and does not publish partial output over a valid file.
- Network failures, truncated sources, missing tables/columns, and upstream schema drift
  fail loudly before output publication.

## 3. Prompt and Template Contract

Check exactly one mode:

### Raw rows with `prompt_config`

- Rows do not contain a non-empty `responses_create_params.input`.
- Every prompt placeholder exists on every row and has the intended type/format.
- Literal braces are escaped for `str.format_map`.
- The template preserves the upstream problem statement, instructions, examples,
  constraints, output format, and scoring explanation in their semantic order.
- Text extraction from HTML/PDF/Markdown preserves references such as "above",
  "below", labels, footnotes, formulas, tables, and captions.

### Materialized requests with `prompt_config: null`

- `responses_create_params.input` is a valid Responses-style input.
- Tool schemas, image/audio blocks, system messages, and sampling fields match the
  intended upstream contract.
- The row is self-contained enough for the selected custom agent.
- Prompt/tool snapshots have provenance and a drift guard when copied from upstream.

For either mode, inspect rendered prompts from several structurally different tasks,
not just the first row. Include the hardest cases: images, long tables, multiple code
blocks, nested choices, or task-specific instructions.

## 4. Data, Request, and Routing Contract

- Prepared rows validate as JSON objects and contain stable unique identifiers.
- The selected agent actually receives the intended dataset. If routing is ambiguous,
  the dataset's `agent` pin names a valid connected agent.
- `agent_ref`, when present, matches a configured agent instance.
- Top-level fields and `verifier_metadata` match the selected request model exactly;
  neither location is assumed universal.
- Config paths resolve from a clean checkout and, for nested/flavored benchmarks, the
  expected `--benchmark` token is discoverable.
- Manifest-backed reuse targets a resources server that exports `VERIFIER_FIXTURE`;
  config-only/materialized integrations do not add dummy manifest fields to look valid.
- Runtime-only secrets/model settings do not block `gym eval prepare`.
- Stateful downstream calls propagate cookies and isolate task/session state.
- README commands use current CLI names and flags and work from a clean checkout.

## 5. Selector, Reward, and Metric Contract

This is the highest-risk area for grouped tests and partial credit.

### Selectors and test membership

- Every row selector exists in the verifier's canonical metadata, or an explicit tested
  mapping translates it.
- A selector runs the intended tests—not all tests, no tests, or another group's tests.
- Overlapping test membership cannot award unrelated groups accidentally.
- The evaluator only awards a group after all tests required by its aggregation rule
  have run and passed.
- Missing selectors and empty result sets fail closed with zero reward or a clear error.

### Reward

- The declared reward range matches actual responses.
- Full, partial, and zero reward conditions are unambiguous.
- A row cannot receive more than its declared maximum.
- Malformed output, compile/runtime failure, timeout, judge failure, and tool failure
  have intentional outcomes.
- If reward is binary but official scoring is fractional, both mappings are documented
  and tested.

### Aggregate metrics

- Task grouping uses stable IDs and repeat indices correctly.
- Pass@k/majority/best-of logic uses the intended per-rollout score.
- Subtask/rubric weights and caps match the official specification.
- Duplicate or overlapping test outputs are deduplicated where required.
- Missing and partial rollouts cannot inflate the aggregate.
- Suite aggregation reports per-benchmark errors/completion before any combined score.

Always test reward and aggregate metrics separately. A correct `verify()` does not prove
that `compute_metrics()` groups or weights results correctly.

## 6. Test Quality

Expect tests at the boundaries the change touches:

### Preparation tests

- Import through the package path used by the CLI.
- Replace downloads with fixtures; assert source repo/config/split/revision arguments.
- Assert exact scoring-relevant row fields, not only count or file existence.
- Cover multiple source shapes and an upstream schema-drift failure.
- Assert deterministic output and exact returned path.
- Assert a failed preparation leaves a prior output unchanged.

### Config and prompt tests

- Resolve the full config and assert agent/resources-server wiring and overridden paths.
- Verify benchmark discovery name, dataset path, prepare path, prompt mode, and repeats.
- Materialize representative rows and validate the resulting request.
- For extracted statements, snapshot or semantically assert sections whose ordering
  affects meaning.

### Verifier and metric tests

- Known-good and known-bad outputs.
- Empty/malformed output, exception, timeout, and missing-dependency behavior.
- Partial credit and boundary scores.
- Unknown and synthetic selectors.
- Overlapping groups/tests and deduplication.
- Multiple problems, subtasks, and repeats in metric aggregation.

### End-to-end evidence

- A reference solution/output passes through the real grader path.
- A deliberately wrong solution/output fails through the same path.
- External integrations show per-example parity with a pinned upstream run.
- Agent/environment changes include inspected real model rollouts, not only mocks.

Reject vacuous tests that mock away the contract under review, assert only that a
function was called, or duplicate implementation logic in the expected value.

## 7. Validation Commands

Adapt paths and names to the change:

```bash
python -m pytest benchmarks/<name>/tests -q
python -m pytest resources_servers/<verifier>/tests -q

gym list benchmarks <name>
gym eval prepare --benchmark <name>
gym env validate --benchmark <name>
gym eval run --benchmark <name> --model-type <model_type>
```

For manifest-backed entries:

```bash
gym env validate <name> --kind benchmark
gym env test <name> --kind benchmark
gym env publish <name> --kind benchmark
```

Also run focused pre-commit checks and the relevant core tests. Run the real grader or
rollout path only in an appropriate isolated environment; do not execute untrusted
benchmark harnesses directly on a workstation.

## 8. Reporting Findings

Report actionable findings first. Each finding should include:

- severity and concise title;
- the tightest relevant file/line range;
- the violated producer/consumer contract;
- a concrete input or execution path that triggers it;
- the resulting wrong reward, metric, prompt, or failure mode;
- the missing test that would have caught it, when useful.

Separate confirmed bugs from follow-up validation gaps. If no actionable bug is found,
state what was checked and which real-data, external-service, or rollout paths remain
unverified.
