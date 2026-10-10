# Benchmark Review Checklist

Use this checklist for benchmark, eval, verifier, agent, or preparation changes. Review the runtime contract before
style: a clean implementation can still evaluate the wrong task or aggregate the wrong score.

## Establish the intended evaluation

- Identify the upstream revision, canonical split, task count, license, official prompt, evaluator, and headline
  metric.
- State the evaluation unit: row, problem, subtask, test group, episode, or suite member.
- Identify whether the change reuses a scorer, changes a shared scorer, adds an agent loop, or wraps an external
  harness.
- List every generated artifact and every consumer of it.
- If the change claims parity, find the per-example comparison or reproduce it. Aggregate parity alone is weak
  evidence.

Build a contract ledger before reviewing implementation details:

| Semantic value | Source field | Prepared field | Prompt/agent consumer | Verifier consumer | Metric consumer |
| --- | --- | --- | --- | --- | --- |
| task ID | | | | | |
| question/input | | | | | |
| reference/rubric | | | | | |
| test/group selector | | | | | |
| weight/max score | | | | | |

Any blank scoring-relevant cell is a likely bug or missing test.

## Code and repository consistency

### Scope and layout

- The change uses an existing verifier or agent when its semantics match.
- New components live in the expected top-level directory; benchmark-owned composition and data live under
  `benchmarks/`.
- A discoverable benchmark config locally declares exactly one `type: benchmark` dataset. Multiple benchmark
  declarations are an eval suite, not one `--benchmark` target.
- Instance names are isolated when the benchmark overrides shared scorer settings.
- The diff contains no unrelated generated files or drive-by changes.
- New source files have the NVIDIA SPDX header and dependencies are license-compatible.

### Imports and dependencies

- `prepare.py` imports as a repository-root module, the same way `gym eval prepare` imports it.
- Package-local imports do not depend on running the script directory as `sys.path[0]`.
- Preparation dependencies exist in the repository-root environment. A package listed only in a resources server's
  isolated requirements is insufficient.
- Optional heavy dependencies are lazy where appropriate, and missing optional packages fail with a useful message.
- Async HTTP and external libraries follow the current `AGENTS.md` concurrency contract.

### Generated artifacts

- `prepare()` returns the exact configured `jsonl_fpath` as a `Path`.
- Side artifacts such as metadata, test archives, databases, scripts, or graders are written to the config's exact
  paths.
- Cross-repository outputs resolve through the downstream consumer's public name, registry, or search path; directory
  layout alone is not treated as proof.
- `.gitignore` covers generated artifacts without hiding intentional fixtures.
- Preparation is deterministic and does not replace a known-good output with a partial result.
- Network failures, truncated sources, missing tables or columns, and upstream schema drift fail before publication.

## Prompt and template contract

For the standard raw-row flow:

- Rows do not contain a non-empty `responses_create_params.input` before prompt rendering.
- Every `prompt_config` placeholder exists on every row and has the intended type and format.
- Literal braces are escaped for `str.format_map`.
- Scorer-only fields such as answers, rubrics, or private tests do not leak into the model prompt.
- The rendered prompt preserves upstream instructions, examples, constraints, output format, and scoring explanation in
  semantic order.
- HTML/PDF/Markdown extraction preserves references such as "above", "below", labels, formulas, tables, captions, and
  footnotes.

If a legacy or external integration materializes `responses_create_params.input`, confirm its current runtime contract
instead of forcing it through the raw-row pattern. It must not also configure prompt rendering, and its tool schemas,
multimodal blocks, system messages, and provenance must match the upstream request.

Inspect rendered prompts from several structurally different tasks, including the hardest cases rather than only the
first row.

## Data, request, and routing contract

- Prepared rows contain stable unique identifiers and validate against the owner `TaskData` and request models.
- The selected agent receives the intended dataset; any ambiguous route has a valid connected agent pin.
- `agent_ref`, when present, names a configured agent instance.
- Flat fields and legacy `verifier_metadata` placement match the declared schema; neither location is assumed
  universal.
- Config paths resolve from a clean checkout and nested/flavored benchmarks expose the expected `--benchmark` token.
- Manifest-backed scorer reuse targets a server that exports an inspectable `VERIFIER_FIXTURE`.
- Runtime-only secrets and model settings do not block `gym eval prepare`.
- Stateful calls propagate cookies and isolate task/session state.
- README commands use current CLI names and flags and work from a clean checkout.

## Selector, reward, and metric contract

### Selectors and test membership

- Every emitted selector exists in canonical verifier metadata, or an explicit tested mapping translates it.
- A selector runs the intended tests—not all tests, no tests, or another group's tests.
- Overlapping membership cannot award unrelated groups accidentally.
- The scorer awards a group only after every test required by its aggregation rule ran and passed.
- Unknown selectors and empty result sets fail closed with the worse reward endpoint or a clear error.

### Reward

- The manifest reward range and direction match actual responses.
- Full, partial, and worse-endpoint conditions are unambiguous.
- A row cannot receive more than its declared maximum.
- Malformed output, compile/runtime failure, timeout, judge failure, and tool failure have intentional outcomes.
- If training reward and official fractional scoring differ, both mappings are documented and tested.

### Aggregate metrics

- Task grouping uses stable task and rollout identities.
- Pass@k, majority, and best-of logic use the intended per-rollout score.
- Subtask or rubric weights and caps match the official specification.
- Duplicate or overlapping test outputs are deduplicated when required.
- Missing and partial rollouts cannot inflate the aggregate.
- Suite output reports per-benchmark errors and completion before any combined summary.

Test reward and aggregate metrics separately. A correct `verify()` does not prove that `compute_metrics()` groups or
weights results correctly.

## Test quality

### Preparation

- Import through the package path used by the CLI.
- Replace downloads with fixtures and assert source repository, config, split, and revision arguments.
- Assert exact scoring-relevant fields, not only row count or file existence.
- Cover representative source shapes and an upstream schema-drift failure.
- Assert deterministic output, exact returned path, and failure without partial publication.

### Config and prompt

- Resolve the full config and assert agent/resources-server wiring and overridden paths.
- Verify discovery name, dataset and prepare paths, prompt mode, and repeat count.
- Render representative rows and validate the resulting request.
- For extracted statements, snapshot or semantically assert sections whose ordering affects meaning.

### Verifier and metrics

- Known-good and known-bad outputs.
- Empty or malformed output, exception, timeout, and missing-dependency behavior.
- Partial credit, boundary scores, unknown selectors, overlapping groups, and deduplication.
- Multiple problems, groups, and repeats in aggregation.

### End to end

- A reference solution or output passes through the real grader path.
- A deliberately wrong solution or output fails through the same path.
- External integrations show per-example parity with a pinned upstream run.
- Agent or environment changes include inspected real model rollouts, not only mocks.

Reject tests that mock away the contract under review, assert only that a function was called, or duplicate the
implementation logic in the expected value.

## Validation commands

Adapt paths and names to the change:

```bash
python -m pytest benchmarks/<name>/tests -q
python -m pytest resources_servers/<verifier>/tests -q

gym list benchmarks <name>
gym eval prepare --benchmark <name>
gym env validate <name> --kind benchmark
gym env test <name> --kind benchmark
gym eval run --benchmark <name> --model-type <model_type>
```

Run focused pre-commit and relevant core tests. Execute an untrusted grader or rollout harness only in an appropriate
isolated environment.

## Report findings

Each actionable finding should include:

- severity and a concise title;
- the tightest relevant file and line range;
- the violated producer/consumer contract;
- a concrete input or execution path that triggers it;
- the resulting wrong prompt, reward, metric, or failure mode; and
- the missing test that would have caught it, when useful.

Separate confirmed bugs from validation gaps. If there is no actionable bug, state what was checked and which
real-data, external-service, or rollout paths remain unverified.
