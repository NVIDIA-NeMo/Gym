# Reviewing an agent server

Use this checklist for changes under `responses_api_agents/`. Start with the capability the PR
claims in the [add-harness skill](../../../../../.agents/skills/add-harness/SKILL.md). A change to model-call
capture can affect training even when evaluation produces the same answer.

## Trace a real caller

- Classify the change by entry point (agent sessions or the agent's `/run`), sandbox owner
  (resources server, agent server, or no sandbox), and claimed capability. Find subclasses that
  inherit or override the changed method.
- Follow each affected benchmark preset to the resources server, its `allowed_agents`, the
  environment server, and the agent config actually selected. Check where the model endpoint and
  tools are reachable. A successful run through a different preset can miss the changed path.
- For a relay or mini-SWE migration, check the [pairing policy](../../../../../.agents/skills/add-benchmark/SKILL.md#determine-integration-type)
  and [existing-agent migration guide](../../../../../fern/versions/latest/pages/agent-server/integrate-existing-agents.mdx).
  A single-harness exception for the relay does not extend to deprecated agent implementations.
  Existing Harbor tasks that set `[agent].user` still need their compatible implementation.

## Walk the failure through the session

- Trace seed, activation, repeated requests and close, including failure before activation. On a
  multi-worker config, check its module-level app and send a follow-up request through another
  worker. Session ownership routing supports session-local state. Look for process-local state
  outside that contract which requires a single worker or a startup refusal.
  [Agent session contract](../../../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#agent-server-and-environment-server)
- Where a harness returns a failed response or raises, trace the status and original cause through
  the [outcome and retry contract](../../../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#keep-execution-health-separate-from-reward).
  Check the actor and budget of each retry. Compare a transient 429 with a deterministic invalid
  request. Check how an ungraded episode and a scored wrong answer enter the
  [masking rules](../../../../../AGENTS.md#sandbox).
- Exercise cleanup on a deadline, cancellation, failed seed and repeated close. Distinguish the
  harness process group from a task service that detached before the stop. Under the
  [`CleanupReceipt`](../../../../../nemo_gym/agents/utils/supervisor_client.py) contract, a confirmed forced stop
  covers processes the stop reached. A detached service can remain available to verification until
  the sandbox owner tears it down. Check that a cleanup error preserves the original failure.
- For a runtime installation change, request a session smoke on the affected task image with the
  proposed runtime settings. Inspect where tools run, the version that actually starts, and whether
  the task's working directory, user and environment survive setup.
  [Installation contract](../../../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#installation-contract)

## Inspect the result and the evidence

- Match tool calls to their results and model calls to their invocations, including compaction and
  subagents. Remove a call's usage record and check that the total stays unknown with a gap.
  Preserve measured zeros and documented known zeros, including an oracle that never calls a model.
  [Output conversion](../../../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#convert-outputs-without-silent-loss)
- Use the [conformance runner](../../../../../scripts/harness_conformance/README.md) from the repository root:
  `python -m scripts.run_harness_conformance --harness <key> --output <new-dir>`.
  Compare the evidence with the capability claimed and the
  [published conformance results](../../../../../fern/versions/latest/pages/observability/harness-conformance.mdx#results). A documented evaluation-only gap, an upstream
  packaging requirement, or a runtime built on the host and executed in the sandbox can be valid.
- Read the README, warnings and config examples as a user choosing a preset. Check what it runs,
  which fields it honors or refuses, and how the user can tell whether a migration succeeded.
  Apply the [shared review checks](../checks.md) and the [PR evidence requirements](../../../../../AGENTS.md#quality-bar).
  For examples of findings with a specific affected caller, see the [published reviews](details/examples.md).
