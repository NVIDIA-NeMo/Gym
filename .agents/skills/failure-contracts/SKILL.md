---
name: failure-contracts
description: >
  Apply NeMo Gym's recommended failure-handling path when implementing or reviewing runtime
  failures, retries, timeouts, verification, scoring or masking, recovery, failure-report
  transport or persistence, and health consumers. Covers shared outcome policies,
  NgFailureEvent and named kinds, and evidence propagation to callers.
---

# Failure contracts

Read [Failure handling contracts](../../../fern/versions/latest/pages/observability/rollout-failures.mdx)
for the public guidance, APIs and examples. The page is the reference; keep detailed rules there.

This is a strong **SHOULD**, not a **MUST** or merge gate. NeMo Gym's recovery, observability and
health tooling increasingly relies on this path, but alternatives can still reach main. Explain
tradeoffs and consumer effects when departing from it. Existing API validation and correctness
requirements still apply to APIs the implementation uses.

## Walk through the affected path

1. **Find the producer and caller.** Identify the observing boundary, operation and originating
   stage. Prefer `NgFailureEvent` and named kinds from `nemo_gym/core/failure_kinds.py`; use existing
   names before adding new shared observations. Keep evidence separate from attribution and action.
2. **Find the decision owner.** Prefer the shared mechanism in `nemo_gym/core/outcome_policy.py`
   and the boundary's configurable defaults and supported actions. Trace who executes the decision
   and owns the retry budget, replay safety and saved output. Masking is an ungraded representation,
   not a competing action that can override `count_as_zero`.
3. **Trace upstream.** Follow failures and recovered-success reports beyond the changed function,
   through callers and transport to collection/persistence and relevant health readers. Look for
   lost events, rewritten stages, replaced causes or IDs, missing omission flags and saved decisions
   being chosen again. Forward an existing event unchanged; use `raise ... from` for a distinct
   failure at the caller's boundary.
4. **Demonstrate the affected behavior.** Select relevant checks from the page's suggested validation:
   a configuration override, retry then success, forwarding/save-reload, or exhaustion/cleanup.
   Prefer existing helpers and core tests over duplicating their implementation in each adapter.

For examples, follow the page's public-API snippets and links to policy, report and collection tests.
Check the actual consumer and configuration field before presenting an example as supported there;
not every boundary supports every action, and adoption is incremental.

## Review and handoff

State the affected path, any deliberate departure and its tooling impact, and what was exercised.
Treat golden-path adoption suggestions as non-blocking. A correctness finding needs a concrete
behavioral consequence under the applicable API or consumer contract; deviation alone is not one.
Do not turn this skill into an automatic rejection rule or require a repository-wide migration.
