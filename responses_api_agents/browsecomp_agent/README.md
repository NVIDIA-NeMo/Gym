# BrowseComp agent

The repository-native research loop runs inside a task sandbox through an agent session. Its version is the Gym source revision: seed uploads that exact source and a private Python 3.13.14 runtime with dependencies pinned to the Agent Server environment. It preserves the progress board, tool-result compaction, context resets, empty-answer retries, and Responses API model calls.

## Execution and composition

`benchmarks/browsecomp/config.yaml` runs `single_agent_turn_legacy`: resources seed → agent seed → one activation → agent close → verify → resources close. Docker is the default provider. To use Enroot or OpenSandbox, replace the included Docker provider config and set the matching sandbox images; do not merge two provider blocks under `sandbox`. Both the Model Server and granted Resources endpoints must be reachable from the agent sandbox (`use_absolute_ip: true`), and so must this agent's own token-count endpoint when `save_model_call_using_vllm_tokenize_endpoint` is on. The benchmark config sets `absolute_ip_server_types` to those three server types, so the head server, whose config endpoint returns the merged config, stays on loopback. An unknown provider or an unknown `sandbox_config` key refuses the seed with 422, since every retry would fail the same way. So does a loopback address for any of these endpoints, except under Docker: a container on the host network can reach the host's loopback, so the shared check only warns, and an activation that cannot connect then fails its reachability check as a retryable failure.

The agent borrows `sandbox_access` when supplied. Otherwise `sandbox_provider` and `sandbox_config` create an owned sandbox. Missing sandbox configuration, the unisolated Local provider, required MCP grants, and more than one direct HTTP grant are refused. No sandbox error falls back to host execution. Close stops the harness through `SandboxSession`, captures output, then disconnects a borrowed sandbox or destroys an owned one.

Enroot must provide PID isolation (upstream 4.2 or newer). Startup checks the actual exec process namespace and its `/proc` mount before staging rollout input; a probe that shows the host's namespace or a `/proc` mount of another namespace refuses the seed with 422. A probe that does not finish (a non-zero exit, a timeout or a provider error) or names no namespace fails the seed as retryable instead, because it says nothing about isolation. The resources server's shell sandbox applies the same rule. Use a complete matching Enroot installation, including its helper binaries. Merely setting `ENROOT_UNSHARE_PID` on an older installation is insufficient.

The Resources Server remains independent. A direct HTTP grant supplies its tools with per-session headers/cookies; the updated cookie jar returns at agent close. BrowseComp resources execute `bash_command` in their own sandbox, separate from the agent container. This uses two containers per BrowseComp episode and does not require a provider to serialize/connect an existing sandbox. Search credentials, ground truth and verifier code stay on the Resources Server. The model upstream API key stays on the Model Server.

For a compatible tool-free benchmark, use `resources_tool_transports: []` and change only the Environment Server's resources binding and dataset. Set `progress: false` when the pairing does not use the progress board. The adapter does not depend on BrowseComp task fields or its verifier. No SWE-bench Pro or Terminal-Bench support is claimed.

## Capability and limits

The sandbox path declares **Evaluation** capability. It returns an observation bundle even after failure, with `native_turn_evidence_unavailable` or `observation_capture_failed`. Raw Responses output and model-call captures remain available, but native invocation/turn/tool observations are not emitted. Strict evidence checks are expected to report these gaps; this is not an Observability or Training qualification. Function-calling shell conformance is applicable but optional at this capability level and has not been registered or qualified. Component tests and real pairing runs cover the resources-owned pocketshell boundary.

- Sampling controls in the request go unchanged to every model call. `max_output_tokens` is per call; `max_steps` and `rollout_timeout_s` bound the loop. `sandbox_runner_timeout_seconds` bounds the full activation, including retries (3600 seconds by default). If `rollout_timeout_s` is unset, the sandbox runner cap becomes the loop deadline; this adds a wall-clock limit to previously step-only runs.
- Token IDs/logprobs already returned by the Model Server are preserved; the sandbox path has not been qualified for training.
- With `save_model_call_using_vllm_tokenize_endpoint: true`, a session-authenticated adapter endpoint counts prompts without copying the upstream key into the sandbox. Hosted endpoints without `/tokenize` must explicitly set this option to `false`, using the existing post-call token-count reset path.
- `snap_dir` is rejected for sandbox sessions. Use captured model calls and rollout output; snapshots remain available in explicit host compatibility mode.
- Harness source, runtime and session files live under `/tmp/nemo-gym-*`, outside the task workdir. The runner deletes its staged input after loading scoped grants. The runtime needs no Python or package download access in the task image; seed builds its archives on the Agent Server.
- Agent sessions use one worker, bind the attempt-qualified rollout URL, and replay identical activation retries. A different activation returns 409. Disconnecting an HTTP waiter does not cancel the harness; close owns cancellation.

## Failure semantics and compatibility

A native rollout timeout retains partial output, sets `timed_out`, and scores zero without a judge call. Inside the sandbox the timeout counts from the runner's start, so its endpoint checks spend the rollout's budget and the supervisor's 30-second grace is left for startup and the receipt. Model HTTP errors retain their status in `ng_agent_outcome`; the Environment Server decides verification, failure or retry. Its default policy verifies 400/413/422, retries 408/425/429/5xx, and excludes other model failures from scoring. Read failure rows and coverage alongside accuracy: this differs from the old `/run` path, which counted some errors as zero-reward answers.

Host `/v1/responses` and `/run` are disabled by default. Development-only compatibility requires `allow_host_execution: true`; `/run` also requires an explicit `resources_server`. The resources server independently requires its own host-execution opt-in if its tool sandbox is disabled. These options do not provide isolation.

## License

Apache-2.0. Dependencies: NeMo Gym (Apache-2.0).
