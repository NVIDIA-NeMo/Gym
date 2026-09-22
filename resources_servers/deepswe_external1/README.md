# DeepSWE external tasks

DeepSWE-style coding tasks with an agent sandbox and a fresh verifier sandbox.
The server reuses Gym's DeepSWE verifier staging and reward handling; the official
DeepSWE benchmark remains unchanged.

## Public examples

The five examples come from [DeepSWE](https://github.com/datacurve-ai/deep-swe),
pinned to revision `435ee89ec2f2e2289f33b0da4f992f0b7b7266b9`.
Prepare their checksummed task packages before starting the server:

```bash
python -m resources_servers.deepswe_external1.prepare_examples
gym env start \
  --config resources_servers/deepswe_external1/configs/deepswe_external1_opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --model-type inference_provider \
  ++policy_model.responses_api_models.inference_provider.uses_reasoning_parser=true
```

Configure the sandbox connection and model credentials privately. The model-server
address must be reachable from the sandbox. The OpenCode configuration is inherited
from Gym; offline images need its locally cached binary setup.
The hosted-inference example normalizes structured reasoning for Gym's chat
contract. Select the model adapter appropriate for your endpoint; training that
requires token IDs needs a compatible training model server.

For other prepared packages, override `tasks_dir` and `expected_task_count`, and
collect against their matching JSONL. Keep local training data and asset caches
uncommitted.

## Verification contract

Tasks use `/app` as both workdir and Git capture scope. The original task prompt
requires committed changes. Collection compares the trusted base commit with
`HEAD`, including binary changes, deletions, symlinks and executable-bit changes.
Uncommitted/untracked files, history, files outside `/app`, and runtime/package
changes are not transferred. There is no additional filename/cache exclusion list.

Only the patch crosses from agent to verifier. Trusted test files are staged
separately in the fresh verifier; its original grader applies the patch and held-out
tests. Each verifier image must already provide Git, Python and writable grading
directories. Both phases deny external network by default, except the configured
model endpoint for the agent. This is filesystem isolation, not proof against all
grader exploits.

`is_verifying_golden_patch: true` runs the original solution in A before collection;
`is_verifying_null_patch: true` collects from an untouched A. They are mutually
exclusive. Missing artifacts and setup failures are masked infrastructure errors,
not completed task failures. Attempts retain separate logs and sandbox IDs.

## Licensing

Integration code and public task contributions: Apache-2.0. Upstream projects
retain their licenses; example rows record their source URLs and pinned revision.
