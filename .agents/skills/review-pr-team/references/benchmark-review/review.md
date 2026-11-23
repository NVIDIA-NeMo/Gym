# Reviewing a benchmark or resources server

Use this checklist for changes under `benchmarks/`, `environments/` and `resources_servers/`.
Classify the change first. A new benchmark, a new training environment, a verifier change and a
config-only pairing each need different evidence. Use the [contribution requirements](../../../../../CONTRIBUTING.md#checks-a-new-environment-or-benchmark-must-satisfy)
and the [add-benchmark exit checks](../../../../../.agents/skills/add-benchmark/SKILL.md#step-7-exit-checks).

## Start from the user's preset

- Trace the benchmark preset to the dataset, resources server, its `allowed_agents`, and the
  environment server that orchestrates the episode. Identify which agent actually receives the
  task, who creates the sandbox, and who grades and closes it. Compare the resolved config with
  the one used for the submitted smoke. Trace shared changes to other presets and reverification.
- Apply the [pairing policy](../../../../../.agents/skills/add-benchmark/SKILL.md#determine-integration-type).
  A flat dataset can use agent sessions through `single_agent_turn_legacy`. For a relay, check
  whether the benchmark documents why it needs a single external harness or claims to support
  swapping harnesses. Existing relay configs outside the change are not migration findings.
- For [Gym rows](../../../../../fern/versions/latest/pages/environment-tutorials/resources-server-apis.mdx),
  follow one row through preparation, seed, tools and verify. Check prompt-dependent initialization
  against [`ResourcesSeedSessionRequest`](../../../../../nemo_gym/server_runtime/base_resources_server.py).
  A hook's seed model receives the optional prompt parameters only if it declares the field.
  Test re-seeding, a conflicting episode, and close after partial setup. Keep verifier-only data
  out of the model input.
- Follow image input through the **selected** harness and model backend. If Gym decodes the image,
  [`MediaFile`](../../../../../nemo_gym/core/media.py) requires supported inline data and rejects remote URLs
  and file IDs. A passthrough harness can forward a URL to a backend that accepts it.
- For [Harbor task folders](../../../../../fern/versions/latest/pages/data/task-folders/task-folder-format.mdx),
  load a real task, including its image and verifier. A single-step task uses its root instruction
  and tests or a dataset verifier. A [multi-step task](../../../../../fern/versions/latest/pages/data/task-folders/multi-step-tasks.mdx)
  uses step instructions and step or fallback root test scripts. Check a task with only step tests.

## Trace scoring and recovery

- Follow a known-good answer, a wrong answer and an infrastructure failure through scoring. Under
  the [masking rules](../../../../../AGENTS.md#sandbox), a reward of one and an infrastructure zero with only
  `evaluation_completed: false` average to 0.5. Masking the infrastructure row leaves a mean of
  one. A genuine wrong answer remains in the denominator. Check a typed ungraded failure separately.
- Match each scored response and reward to its grader using the [grading evidence paths](../../../../../.agents/skills/add-benchmark/SKILL.md#step-7b-failure-handling-conformance).
  A captured grading operation, shared verifier record, bound artifact or agent grading record
  can provide that evidence. Try a stale record from a different response.
- Use the [outcome and retry contract](../../../../../fern/versions/latest/pages/agent-server/add-or-update-a-harness.mdx#keep-execution-health-separate-from-reward)
  to trace one failure from the environment server through collector resume. For a custom
  namespaced kind, compare its recorded action, attribution and saved answer against
  [`masked_infrastructure_kind`](../../../../../nemo_gym/rollouts/rollout_recovery.py). Preserve a saved answer
  for the recovery path that owns it. For transport retries inspect the selected client
  [`request`](../../../../../nemo_gym/client/http_client.py) and its caller's bounds.
- Trace each acquired resource through tool errors, cancellation, repeated verification and
  close. Check that downstream calls carry the session cookies and that a failed stop can be
  retried by the sandbox owner. Use the [async rules](../../../../../AGENTS.md#communication--async-patterns)
  and [shared lifetime checks](../checks.md#architectural-assumptions).

## Check the evidence at the right boundary

- Resolve the affected config and inspect one real rollout, including the verifier output.
  When a reviewer identifies a particular runtime failure, follow the [review instructions](../../SKILL.md#33-draft-the-comments)
  for a smoke under its triggering conditions. For a scoring change, compare before and after
  on the same inputs. For a training environment, apply the training-specific evidence bar.
- Apply [`manifest_workloads_for_server`](../../../../../nemo_gym/environment/registry.py) to determine
  which data checks run. With referring manifests and neither legacy output, exercise every
  referring workload's verifier fixture. With either legacy output or no referring manifest,
  check the required artifacts and five rollout rows. File existence alone misses a truncated run.
- Read user instructions through the [shared review checks](../checks.md#docs-and-examples-that-a-change-makes-untrue).
  Check that the documented command still selects the advertised pairing, that the labels match the
  [glossary](../../../../../fern/versions/latest/pages/about/concepts/key-terminology.mdx), and that any score
  or compatibility claim has supporting evidence. A choice of Gym rows or Harbor tasks, or an
  unchanged supported configuration, needs no finding solely for its format.
- For examples of failures in grading, wiring and recovery, see the [published reviews](details/examples.md).
