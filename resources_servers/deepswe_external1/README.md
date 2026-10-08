# DeepSWE external tasks

DeepSWE-style coding tasks with an agent sandbox and a fresh verifier sandbox.
The server reuses Gym's DeepSWE patch collection, verifier execution and reward
handling; the official DeepSWE benchmark remains unchanged. Each JSONL row contains
the prompt, image references, resource limits, base commit, and all test/solution
file contents. No shared task directory or startup preparation is required.

## Public examples

The five examples come from [DeepSWE](https://github.com/datacurve-ai/deep-swe),
pinned to revision `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`.
The committed rows are ready to use. To regenerate them from pinned public source,
optionally run `python -m resources_servers.deepswe_external1.prepare_examples`.

```bash
gym dataset collate \
  --config resources_servers/deepswe_external1/configs/deepswe_external1_opencode.yaml \
  --output-dir resources_servers/deepswe_external1/data/cache/collated \
  --mode example_validation
gym env start \
  --config resources_servers/deepswe_external1/configs/deepswe_external1_opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type inference_provider \
  ++policy_model.responses_api_models.inference_provider.uses_reasoning_parser=true
```

Collation generates `data/example_metrics.json` with Gym's standard dataset
statistics; it does not run a model or verifier.
Use the same OpenCode config for collation and runtime so the rows route to the
running resources server. The base `deepswe_external1.yaml` config is for
standalone golden/null validation without an agent.

Configure the sandbox connection and model credentials privately. The model-server
address must be reachable from the sandbox. The OpenCode configuration is inherited
from Gym; offline images need its locally cached binary setup.
The hosted-inference example normalizes structured reasoning for Gym's chat
contract. Select the model adapter appropriate for your endpoint; training that
requires token IDs needs a compatible training model server.
The OpenCode config includes upstream's `legacy_agent` environment server for
rollout routing; it forwards requests without changing task rows or grading.

`data/example_rollouts.jsonl` contains one recorded Super/OpenCode attempt per public
example, run with inline-file provisioning on 2026-10-05. Original task fields and
Gym-converted model/tool responses are retained, including failed and budget-limited
attempts. Each row records the runtime revisions, effective settings and exported
prompt encoding; operational logs and sandbox handles are omitted.

[data/review_smoke.json](data/review_smoke.json) records these five attempts plus
two public golden/null pairs (golden 1, null 0) and a Python-free Debian bootstrap
check, including the no-network configuration guard. The resource code is identical
at the control and model-run revisions. Model runs additionally use the separately
reviewed OpenCode ripgrep fix from
[PR #3953](https://github.com/NVIDIA-NeMo/Gym/pull/3953); that agent change is not
part of this integration. These are representative integration checks, not a full
benchmark accuracy estimate. Exact revisions and cleanup evidence are in the record.
One Actionlint attempt was interrupted by a model-call timeout and retried once;
the underlying cause is unconfirmed. The record preserves the original error and
distinguishes the retry from first-attempt results.
The record also includes a separate final-follow-up check of isolated patch decoding:
one golden/null pair and one three-turn model attempt, with the exact candidate
source hash and independent cleanup checks. Earlier trajectories keep their original provenance.

For another dataset, supply rows matching `task_data.py`. Keep local training data
uncommitted. Rows are trusted controller inputs: grading files live in `files`,
not `responses_create_params.input`, and are not shown to the agent. There is no
task store, file-checksum manifest or legacy fingerprint-only row loader.

## Verification contract

Tasks use `/app` as both workdir and Git capture scope. The original task prompt
requires committed changes. Collection compares the trusted base commit with
`HEAD`, including binary changes, deletions, symlinks and executable-bit changes.
Uncommitted/untracked files, history, files outside `/app`, and runtime/package
changes are not transferred. There is no additional filename/cache exclusion list.

Only the patch crosses from agent to verifier. Trusted test files are staged
separately in the fresh verifier; its original grader applies the patch and held-out
tests. Each verifier image must provide Git and writable grading directories.
File contents are provisioned through `SandboxSpec.files`, like other SWE servers.
Candidate patches are base64-encoded for this text-only transport and decoded in B
before grading, preserving non-UTF-8 text diffs as well as Git binary patches.
Decoding uses isolated Python to avoid importing task-local modules.
Before running tests or applying the candidate patch, B checks for Python and, if missing,
installs `python3` as root through its OS package manager (APT, APK, microdnf,
DNF or Yum), with a five-minute setup timeout. Existing Python is reused; neither
the agent sandbox nor the published image is modified. Installation requires
package-repository access. The agent denies external network except the configured
model endpoint. Like upstream DeepSWE, the verifier adds no network deny policy by default; set
`enforce_verifier_no_network: true` to opt in, in which case Python must already
be present (the server does not bypass the network policy). Network policy is
controlled by these server settings, not inferred from source-task `allow_internet`
metadata. This is filesystem isolation, not proof against all grader exploits.

`is_verifying_golden_patch: true` runs the original solution in A before collection;
`is_verifying_null_patch: true` collects from an untouched A. They are mutually
exclusive. A failed collection followed by confirmation that the seeded Git
repository, base commit or HEAD is unavailable is an invalid submission: in agent
mode it remains an unmasked zero. Transport/setup failures and missing artifacts
without that evidence remain masked. Golden/null control failures are not agent
scores. Attempts retain separate logs and sandbox IDs by default; set
`clear_verifier_logs: true` to remove each attempt's local logs, patch and
`result.json` after verification. This does not remove the returned response or
the rollout collector's saved trajectories. A failed local deletion preserves the
log path and reports a cleanup error without changing the grade.
Responses include the candidate patch by default. Concurrency is controlled by
the caller, with no additional server-side cap.

## Licensing

Integration code and public task contributions: Apache-2.0. Upstream projects
retain their licenses; example rows record their source URLs and pinned revision.
