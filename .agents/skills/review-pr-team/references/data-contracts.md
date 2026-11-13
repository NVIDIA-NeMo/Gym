# Data contract review

A dataset schema is a contract that code enforces. When a PR changes one, the review establishes how
the contract stays in sync across data preparation, validation, runtime parsing, and the code that
reads the data.

## When it applies

Apply this review when a PR changes any of:

- dataset rows, preparation scripts, collation, or serialization;
- Pydantic models or other schemas for task data;
- resources server or agent request models;
- fields that a verifier, agent, prompt, metric, routing step, or provenance record reads;
- compatibility normalization or a data format migration;
- tests meant to catch schema or dataset drift;
- code that passes a model or row to another library, another server, or a persisted output, or a
  new field on a model that is dumped across such a boundary (see checks.md, "Payloads at
  boundaries").

## One source of truth

Prefer one source of truth: the same typed model serves the data tooling and the runtime boundary
whenever dependencies allow it.

When the PR keeps parallel declarations, it needs a check that fails when either side changes
alone. A test that validates committed rows against one schema does not prove that a separate
request model or consumer stays in sync when only code changes. A duplicated field definition that
no automated check covers is a finding.

## Build the contract map

Trace at least one required field, one optional field, and one open-ended field through:

1. the producer or preparation script;
2. the committed dataset row;
3. normalization or migration;
4. the data schema;
5. the runtime request model;
6. the code that reads it: verifier, agent, prompt, metrics, or provenance;
7. the output or persisted provenance, when there is one.

Repeat the trace for every distinct row shape or discriminator branch, and name the source of truth
at each step.

When the reader is another library, trace an optional field into that library's code. A library can
handle an absent key differently from a key set to `None`. Check every branch that reads the field,
not only the row shape the PR targets, and ask for a test that rows without the field produce the
same output as before. For example, a resources server once passed its whole verify request to
SWE-bench's `make_test_spec`. That sent an `image_assets: None` the library's JavaScript harness
crashed on, along with an agent transcript it never reads.

## Ownership and completeness

- Every dataset declaration resolves to a schema owner.
- A resources server owns the schema of task data it reads. A self-contained agent that declares and
  reads datasets can own schemas too.
- A dangling reference, an unsupported owner type, a parse failure, or an unresolved config fails
  the check or appears in an explicit allowlist. A silent skip is a finding.
- Presence checks cover every supported owner type, not only one directory.

## Unknown and open-ended fields

Each schema states whether unknown fields are rejected, accepted but reported as schema debt, or
intentionally open. `extra="allow"` alone does not show that openness is intended. Check that strict
modes and drift tests follow the stated policy, including after a discriminated union selects its
branch.

## Migration and compatibility

- Tests use the real old and new row shapes, not a hybrid made up for the test.
- A new container key being present does not prove the row was migrated correctly.
- Normalization unpacks the new shape before validation.
- A compatibility shim keeps the exact runtime placement and precedence of duplicated fields.
- An exemption for a format that is not parsed and tested yet is removed until it is.

## What a drift check catches

A drift check fails for:

- a missing or mistyped required field;
- an undeclared field, unless its owner allows it;
- a schema change that disagrees with the runtime request model, or the reverse;
- code that reads a field the contract does not have;
- a tracked dataset that was skipped or could not be mapped;
- a malformed legacy or migrated placement.

The check uses production config resolution and ownership logic where practical. When a test
reimplements either, compare it with the production path and fail on unresolved inputs. A threshold
such as "more than N rows checked" does not prove coverage is complete.

## Reporting a finding

For each data contract finding, state:

1. which two layers of the contract disagree;
2. a concrete row, or a code-only change, that gets past validation;
3. what goes wrong at runtime;
4. whether one source of truth could remove the duplication;
5. the test or check needed if the duplication has to stay.
