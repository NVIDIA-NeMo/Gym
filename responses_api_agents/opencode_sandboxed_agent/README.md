# OpenCode Sandboxed Agent

The existing `opencode_sandboxed_agent` entrypoint supports native EnvironmentServer
sessions. OpenCode and its tools run inside the task sandbox created by Resources;
the adapter only borrows the connection. The separate local `opencode_agent` and
existing benchmark recipes keep their current behavior.

## Native EnvironmentServer sessions

Bind `single_agent_environment_server.environment_servers.single_agent.agent_server`
to this agent and `resources_server` to a Resources implementation supporting native
sessions and returning `SandboxAccess`. Bind this agent's `model_server` to a Gym
model endpoint reachable from the task sandbox. Submit episodes to **EnvironmentServer
`/run`**. Existing legacy Resources recipes do not automatically gain native lifecycle
support merely by changing their agent.

```yaml
config_paths:
  - environment_servers/single_agent/configs/single_agent.yaml
  - responses_api_agents/opencode_sandboxed_agent/configs/opencode_sandboxed_agent.yaml

single_agent_environment_server:
  environment_servers:
    single_agent:
      agent_server:
        type: responses_api_agents
        name: opencode_sandboxed_agent
      resources_server:
        type: resources_servers
        name: task_resources  # Supply a native Resources config in this run.

opencode_sandboxed_agent:
  responses_api_agents:
    opencode_sandboxed_agent:
      num_workers: 1
      model_server:
        type: responses_api_models
        name: policy_model
      opencode_version: 1.17.11
```

The agent opens `/v1/agent_sessions`, installs the pinned standalone OpenCode runtime,
and runs one `/ng-rollout/<capture_key>/v1/responses` activation. It accepts a string
or one user text message, optionally preceded by system/developer text; `instructions`
is appended through OpenCode's instruction-file configuration. Images and conversation
replay are rejected before consuming the activation. OpenCode owns tool selection.
Required Resources HTTP/MCP tools are unsupported. Native `opencode_config` accepts
only the existing `permission` and `tools` settings so provider/model overrides cannot
bypass Gym correlation.

Request `max_output_tokens`, sampling, reasoning, formatting, tool controls, and other
unused request controls are explicitly rejected. Set effective sampling and per-model-call
output limits on the Gym model server and verify its captured requests. OpenCode's model
context metadata does not establish an enforced output-token limit. Model calls use
Chat Completions through the attempt-qualified Gym URL; the adapter never rewrites the
reasoning or tool history OpenCode sends.

The supported runtime is Linux glibc x86_64 or aarch64 with Bash and Python 3.9+ (including
SQLite and `fcntl`). Setup reports failing commands, exit codes and stderr. Missing curl
and CA certificates are installed automatically only with root and apt-get; other images
must include them. Online setup also needs tar/gzip and access to GitHub releases.
`remote_opencode_binary_path` may name a pre-staged binary; the existing optional staged
installer and dual-binary configuration remain supported. Every path is inside the sandbox,
and the installed binary's version must match `opencode_version`.

The version-scoped runtime cache lives under `/tmp/nemo-gym-opencode-runtime-<version>`
and installation is serialized with a file lock. Per-session HOME, caches, instructions,
SQLite state and supervisor files live under `/tmp/nemo-gym-opencode-sessions/<id>`,
outside the task repository. Session close removes only the session directory and leaves
the reusable runtime cache for Resources to destroy with the sandbox. No host OpenCode
installation or execution occurs in native sessions.

A Linux subreaper per activation kills and reaps tool descendants, including detached
background processes. Close requires its positive cleanup receipt, successful adapter-file
removal, and disconnect before verification can proceed. Unknown launch outcomes and
unconfirmed cleanup fail closed and retain session state for retries. Successful close
retries return the same result for `session_close_retry_window_seconds` (300 seconds by
default); expired cookies continue to reject activation. State is process-local and requires
one worker. Failed sessions remain until owner recovery/process restart; there is no
cross-worker crash recovery. Process cleanup prevents verification races and is not a
security boundary against hostile task code.

The response preserves text, reasoning, tool results and partial output from OpenCode's
SQLite state. Close returns the existing session-tree, subagent and compaction observations.
Native usage restores inclusive cache/reasoning counts from all persisted assistant turns,
including subagents. Auxiliary model requests not persisted as assistant messages may remain
unaccounted for; compare against captured Gym calls before using totals for accounting.
Missing artifacts and reconciliation are explicit observation gaps. A failed model call may
still leave a correct patch: inspect episode failures and verifier outcomes separately.

Functional smoke validation is not an accuracy baseline. Training token IDs/logprobs,
multiple image families, and supervisor overhead at production concurrency need separate
validation. Pin model, prompt, limits, image, verifier and concurrency for before/after runs.


## Prerequisites

Complete [OpenSandbox access and setup](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#setup)
for sandbox credentials, endpoint configuration, and resource limits before launching.

## Legacy agent /run evaluation

From the repository root, with Gym installed and model/sandbox access configured, use the
[SWE-bench Verified recipe](../../benchmarks/swebench/verified/opencode.yaml), which binds
the agent to its resources server.

```bash
# Prepare the input before starting servers (downloads SWE-bench Verified).
gym eval prepare --config benchmarks/swebench/verified/opencode.yaml

# In terminal 1
gym env start \
    --model-type vllm_model \
    --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
    --config benchmarks/swebench/verified/opencode.yaml

# In terminal 2, with the same Gym environment activated
gym eval run --no-serve \
    --agent swebench_verified_opencode_sandboxed_agent \
    --input benchmarks/swebench/data/swebench_verified_benchmark.jsonl \
    --output results/opencode_smoke/rollouts.jsonl \
    --limit 1 \
    --num-repeats 1 \
    --concurrency 1
```

For an end-to-end evaluation, keep OpenCode execution enabled so `/run` executes
the agent and calls the SWE-bench verifier. Skipping execution limits the test to
the surrounding infrastructure.
This one-task run uses the configured timeout defaults and consumes model and sandbox resources.

## OpenCode binary: online or pre-staged

By default, the agent downloads the [OpenCode installer](https://opencode.ai/install)
and the configured version inside each task sandbox. This needs installation tools
(Bash, curl and archive extraction), a writable home directory, and network access
to OpenCode and GitHub release assets.

For sandboxes without that network access, provide a compatible installer and binary
through a mount, task image, or custom resources-server upload before the agent runs.
For S3-hosted files, arrange a mount or transfer into each task sandbox.
For the SWE-bench recipe above, configure OpenSandbox
[volume options](https://docs.nvidia.com/nemo/gym/main/infrastructure/sandbox/opensandbox#sandboxspec-provider-options)
under `swebench_verified_opencode_resources_server.resources_servers.swebench.sandbox_config.provider_options`;
the resources server creates the task sandbox.

Set both paths to existing files inside that sandbox;
setting only one leaves online installation enabled.
Save this as `offline-assets.yaml` and add `--config offline-assets.yaml` to server startup:

```yaml
swebench_verified_opencode_sandboxed_agent:
  responses_api_agents:
    opencode_sandboxed_agent:
      remote_opencode_install_script_path: /opt/gym-assets/opencode/1.17.11/install.sh
      remote_opencode_binary_path: /opt/gym-assets/opencode/1.17.11/opencode-linux-x64
      remote_opencode_musl_binary_path: null
```

The staged binary determines the installed version and must match the sandbox's
architecture and libc. Keep `remote_opencode_musl_binary_path: null` with the upstream
installer; the dual-binary mode requires a custom installer supporting
`--glibc-binary` and `--musl-binary`.
