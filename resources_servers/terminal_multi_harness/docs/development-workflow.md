# Development Workflow

This workflow is the recommended path for onboarding a new harness into
`terminal_multi_harness`.

The main lesson from the Codex branch is simple:

- do not infer the verifier contract from harness source code alone
- collect real trajectories first
- profile the actual expected-answer distribution
- write the match rules as a dedicated rulebook in `docs/rulebooks/`
- only then patch the env implementation

Codex is the reference example for this workflow. The in-repo packaged rulebook
is:

- `resources_servers/terminal_multi_harness/docs/rulebooks/codex-match-rules.md`

## 1. Start with real collection, not code assumptions

Before touching verifier code:

- identify how the harness actually runs inside the production collection path
- collect a real smoke dataset
- keep the raw trace artifacts that show:
  - teacher-model input
  - teacher-model output
  - trial outcome / reward
  - step order within the trial

For Codex, this meant using Harbor + Aspen and joining real backend exchanges
to trial results. The important takeaway is that the runtime path can differ
from what the open-source harness repo appears to support in theory.

Questions to answer before any verifier change:

- what is the true next-step unit for this harness
- what is the true completion shape
- are there multiple tool calls in one response
- what fields are stable enough to compare
- what fields are volatile and should not drive reward

## 2. Build a profiling pass over expected answers

Once the real collection artifact exists, compute descriptive stats over the
candidate source pool.

At minimum, profile:

- action-type counts
- tool counts
- parameter inventory per tool
- common argument-key sets
- completion-sample behavior
- schema-validity of expected tool calls
- duplicate-input structure
- batch size and mixed-tool batch behavior

The goal is not just to summarize the data. The goal is to expose hidden
assumptions before they become verifier bugs.

Examples from Codex:

- completion samples turned out to be structural `message` steps, not keyword
  matches
- batch tool calls existed, but the matching rule still needed to be decided
  from the observed structure
- some expected tool calls were schema-invalid and had to be filtered out
  before final export

## 3. Write a dedicated match-rules doc

After profiling, write one dedicated rulebook in `docs/rulebooks/` that
defines what it means for model output to match `expected_action`.

This doc should be the source of truth for the harness contract. It should be
clear, short, and normative. Implementation should follow the rulebook, not
the other way around.

Good rulebook content:

- how to classify actual model output
- what counts as completion
- the gate order for tool-call verification
- any allowed non-exact matching behavior
- any tool-specific checks
- how multiple tool calls are compared

Bad rulebook content:

- long profiling summaries
- implementation history
- rollout artifacts
- open-ended speculation

Recommended split:

- put descriptive evidence in a stats note
- put only normative rules in the match-rules note

## 4. Iterate on the rulebook before code

Do not rush into code after the first draft.

Review the match rules against real samples and tighten them until the contract
is stable enough that a teammate can implement it without guessing.

Typical review questions:

- does completion require content, or only message type
- should tool calls validate against schema
- how should extra actual parameters be handled
- when should string similarity be used instead of exact equality
- how should multiple tool calls be aligned

Codex took several iterations before the final rulebook was correct. That
iteration belongs in docs first, not only in code diffs.

## 5. Treat the rulebook as the implementation spec

Once the rulebook is stable:

- audit `response_utils.py`
- audit `verification_utils.py`
- patch `app.py` if request/response plumbing needs to change
- add or update tests for every rule that is easy to regress

Expected pattern:

- extractor changes should be minimal and structural
- comparator changes should mirror the rulebook directly
- tests should use real or representative sample shapes from the profiled data

## 6. Validate locally on real rows before large rollouts

Before full rollout, do two checks:

- direct verifier validation on real collected rows
  - self-match or replay checks against a small real subset
- a true NeMo smoke run using the packaged env
  - real agent
  - real resource server
  - real rollout path

This stage should answer:

- does the env run at all end to end
- do rewards look structurally plausible
- are completion samples behaving as intended
- are schema and tool-call assumptions compatible with the real stack

## 7. Do full rollouts, then audit coverage

After the smoke passes:

- run the full rollout set
- audit shard coverage
- classify shards as:
  - full
  - partial
  - zero-complete

Do not jump directly from “jobs submitted” to “reward profile looks odd.”
First confirm whether the rollout corpus is actually complete enough to trust.

## 8. Profile rollout rewards before exporting the final trainset

Before the final export:

- build the per-source-sample reward profile
- decide the keep band
- confirm the selected sample count makes sense
- sanity-check a few samples at the edge of the keep band

The final export should only happen after the reward profile is understood well
enough to explain why a row is kept or dropped.

## 9. Keep the token-cap rule consistent across harnesses

If a `maxseq49152` sibling is required, do not silently change the counting
basis for a new harness.

The default expectation should be:

- count prompt plus expected target length

If a harness needs a different length basis, that must be an explicit decision
and should be documented in the run plan and provenance note.

## 10. Checklist

For a new harness, the recommended sequence is:

1. collect real joined artifacts
2. profile expected answers
3. write the harness match-rules doc
4. review and tighten the rules
5. land the rulebook in `docs/rulebooks/`
6. implement and test the env changes
7. run real-data local validation
8. run a true NeMo smoke
9. run full rollouts
10. audit rollout completeness
11. build reward profile
12. export the final trainset

If these steps are followed in order, the verifier logic is much less likely to
drift from the actual harness behavior.
