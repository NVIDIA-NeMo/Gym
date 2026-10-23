# Pool Agent

Runs Pool CLI **v1.0.16** inside a Gym task sandbox. The Environment Server
coordinates resource preparation, agent execution, verification, and cleanup.
Pool runs its own tools and calls a Gym Model Server for inference.

[configs/pool_agent.yaml](configs/pool_agent.yaml) defines the agent defaults.
The adapter supports evaluation; token IDs and logprobs are not available for training.

## Configure and run

Configure the agent, benchmark, model, and sandbox provider independently. For
SWE-bench Pro, save this composition as `run.yaml`:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/pool_agent/configs/pool_agent.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: pool_agent
      resources_tool_transports: []

swebench_pro_resources_server:
  resources_servers:
    swebench_pro:
      allowed_agents: [pool_agent]
```

Supply a `policy_model` Gym Model Server, `policy_model_name`, and the `sandbox`
provider configuration in `model-provider.yaml`. The task sandbox must be able to
reach the Model Server. Set Pool's `context_window` to the served model's context
limit and configure generation settings on the Model Server.

Run from the repository root with Gym and the benchmark's preparation dependencies
installed:

```bash
python benchmarks/swebench/pro/prepare.py

gym env start --config run.yaml --config model-provider.yaml

gym eval run --no-serve \
  --config run.yaml --config model-provider.yaml \
  --agent pool_agent \
  -i benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  -o rollouts.jsonl --limit 3 --concurrency 3
```

The flat-row adapter `single_agent_turn_legacy` uses the Resources and Agent session
APIs. The Environment Server seeds the sessions, activates Pool once, closes the
agent, verifies the task, and closes Resources. Pool's direct `/run` endpoint is
unsupported. Pass the same configuration to server startup and collection.

For another compatible benchmark, replace its Resources configuration, reference,
and prepared input. Check its allowed agents and tool requirements before running.

## Runtime and sandbox setup

Session setup installs missing Python, Bash, curl, and tar through `apt-get` or
`apk`; installing prerequisites requires root. Images without these package
managers must already contain the prerequisites. The standalone runner requires
Python 3.8 or later. Pool is installed inside the sandbox, not on the agent-server
host.

Setup downloads the pinned release from `downloads.poolside.ai` using Poolside's
installer, including its checksum verification, and verifies the reported version.
The installation accepts Poolside's EULA. Network access to that download host is
required.

Resources can supply a direct `SandboxAccess` and an absolute task workdir. Without
that access, configure `sandbox_provider` and `sandbox_config`, including an image,
for an agent-owned sandbox. Owned sandboxes receive a TTL and provider metadata.
Agent close stops an owned sandbox or disconnects a borrowed sandbox, leaving its
task files available for verification.

Pool's state and configuration live outside the task workdir. The default shell
wrapper restores the task's HOME and XDG directories; explicit shell overrides in
Pool tool calls bypass that wrapper. `SandboxAccess.env` is forwarded, except for
Pool provider/sink overrides and OpenAI endpoint credentials that could redirect
model calls. Gym supplies the sandbox, so Pool's own container sandboxing is disabled.

Model requests preserve Gym's rollout-qualified URL and use nonstreaming responses.
A loopback proxy removes provider-specific `cache_control` hints while preserving
message content, tool schemas, and upstream HTTP errors.

## Lifecycle and results

| Setting | Default | Purpose |
|---|---:|---|
| `pool_version` | `v1.0.16` | Exact CLI release to install. |
| `context_window` | `32768` | Model context size used to configure Pool's compaction thresholds. |
| `setup_timeout` | `900` | Timeout in seconds for individual setup operations. |
| `timeout` | `900` | Agent execution budget in seconds, capped by the session's task budget. |
| `session_close_timeout_seconds` | `60` | Bound in seconds for each supervised finalization phase, including artifact collection. |

The supervisor stops the harness and confirms termination before collecting its
trajectory. Pool is configured to leave background shell services available to the
task environment. Identical activation retries share the same task. A disconnected
caller does not cancel activation or installation. Close waits for installation;
failed cleanup retains the session so a later close can retry.

A confirmed execution timeout returns an `incomplete` response with available
trajectory events and can proceed to verification. Startup, supervision, model,
and artifact-collection failures propagate as errors. Pool's explicit task-failure
exit code remains gradable when its trajectory is complete.

Trajectory conversion preserves tool-call IDs and out-of-order results. Persisted
thought events become reasoning items. Synthetic exit calls are omitted. Missing
tool results and interrupted trajectory records are reported as observation gaps.
Partial conversation evidence is available on close when it was captured successfully.

Large trajectories may exceed the artifact-collection budget. Adjust
`session_close_timeout_seconds` for the expected workload. A collection failure is
reported as an error; it is not treated as a completed task.

## Supported inputs and limitations

- One text user message, optionally preceded by system/developer text or request
  instructions. Other request controls and conversation shapes are rejected.
- Direct sandbox access; no sandbox user override, continuation, or host execution.
- Required external tool grants are rejected. Optional grants are skipped and
  reported as observation gaps.
- Root trajectory capture only. Subagent invocation trees, tool timing, and exact
  model-call ownership are unavailable. Usage is `null`, not zero; root inference
  counters would omit subagent and compaction work.

## Tests

Run the unit tests from the repository root:

```bash
python -m pytest responses_api_agents/pool_agent/tests -q -o addopts=''
```

The optional real-CLI tests use a scripted local HTTP endpoint and do not call a
hosted model. Supply a local binary matching the pinned version:

```bash
POOL_TEST_BINARY=/absolute/path/to/pool-v1.0.16 \
  python -m pytest responses_api_agents/pool_agent/tests -q -o addopts=''
```

Run the shared lifecycle tests on Linux to exercise process supervision and cleanup:

```bash
python -m pytest tests/unit_tests/test_sandbox_session.py \
  tests/unit_tests/test_sandbox_supervisor_client.py \
  tests/unit_tests/test_process_supervisor.py -q -o addopts=''
```
