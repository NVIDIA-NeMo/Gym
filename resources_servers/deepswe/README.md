# DeepSWE

This resources server implements the DeepSWE v1.1 two-environment evaluation contract. The coding agent works in
one sandbox; its final workspace patch is graded by the canonical DeepSWE verifier in a fresh sandbox based on the
same task image.

## Prepare pinned task assets

```bash
python -m resources_servers.deepswe.prepare \
  --source-dir /path/to/deep-swe \
  --no-download
```

The preparation step copies each immutable, versioned upstream image reference from the pinned task definition into
the Gym JSONL alongside the instruction and task ID. The resources server rejects a row whose image differs from its
pinned task definition, and both the agent and fresh verifier consume that same image. Test assets and oracle patches
remain in the resources server's gitignored control-plane cache.

Task definitions are parsed with a narrow local adapter for the pinned Pier 0.3.1 task schema, without installing
Pier's unrelated agent/model-provider stack. Agent and verifier CPU, memory, and storage requests are resolved
independently from their corresponding `task.toml` environment sections. By default, Gym requests twice the task's
declared CPU and memory for each sandbox while preserving its storage request. The `task_cpu_multiplier` and
`task_memory_multiplier` resources-server settings can override that scaling; explicit `sandbox_config.resources`
values take final precedence.

## Oracle checkpoint

Start the resources server with golden-patch mode enabled:

```bash
gym env start \
  --config resources_servers/deepswe/configs/deepswe.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  +deepswe_resources_server.resources_servers.deepswe.is_verifying_golden_patch=true
```

Then run all 113 oracle patches from another terminal:

```bash
python resources_servers/deepswe/validate_golden.py +concurrency=113
```

## Run OpenCode rollouts

`benchmarks/deepswe/opencode.yaml` runs `opencode_agent` through the `single_agent_turn_legacy` Environment Server.
For each row the Environment Server seeds this resources server, which starts the agent sandbox and lends it to the
agent as `sandbox_access`. It then opens an OpenCode session in that sandbox, runs the task prompt within a
three-hour budget, and closes the agent session, which stops OpenCode's processes and captures its transcript.
Finally it calls verify and closes the resources session, which stops the agent sandbox if verify has not.

The agent sandbox has no internet access, so the pinned OpenCode runtime cannot be downloaded into it. Point
`sandbox_runtime_binaries` at a prebuilt executable of that version that is already present inside the sandbox, for
example on a read-only mount. Launch Gym with `+use_absolute_ip=true` when OpenSandbox needs a host-routable
model-server address.

```bash
gym env start \
  --config benchmarks/deepswe/opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/<model>/configs/<model>.yaml \
  ++deepswe_opencode_agent.responses_api_agents.opencode_agent.sandbox_runtime_binaries.glibc=/mnt/opencode/opencode-linux-x64
```

Sessions are process-local, so the agent and this resources server each run one worker. Agent settings live under
`deepswe_opencode_agent.responses_api_agents.opencode_agent`. The previous `opencode_sandboxed_agent` recipe's
settings map to that block as follows:

| Previous setting | Now |
| --- | --- |
| `sandbox_timeout: 10800` | `timeout: 10800` |
| `opencode_max_context_window` | `context_window`; `max_output_tokens` follows it, as the previous output limit did |
| `remote_opencode_install_script_path`, `remote_opencode_binary_path` | `sandbox_runtime_binaries.glibc` (no install script) |
| `remote_opencode_musl_binary_path` | `sandbox_runtime_binaries.musl` |
| `opencode_config.permission` | Same command restrictions; `webfetch`, `websearch` and `skill` are disabled under `opencode_config.tools` |
| `execution_failure_reward_zero: false` (a run that ended on an OpenCode error was still verified and graded normally) | `session_return_failed_runs: true`: the run is returned with status `failed` and its error, then verified and graded normally instead of retried |
| OpenCode read the repository's `AGENTS.md` or `CLAUDE.md` and project config | `session_project_config: true` |
| `datasets` | Unchanged, on this agent block |

The session path does not support these previous controls, and the agent ignores them if they are set:

- `opencode_model_call_timeout` (one hour per model request): model requests have no per-request deadline; the
  three-hour budget bounds the run.
- Request-level `temperature` and `top_p`: configure sampling on the Gym model server.
- `token_id_capture`, `debug`, `artifacts_dir` and `local_ripgrep_binary_path`. The transcript is returned in the
  rollout's agent observations instead of an export file. OpenCode downloads ripgrep on demand, so its `glob` and
  `grep` tools need a sandbox image that already provides `rg`, as before.

Earlier reasoning reaches the model as before. Gym's vLLM model server returns it inside `<think>` tags in the
assistant content, OpenCode sends that content back in later requests, and the model server moves it into the
reasoning field before calling vLLM.

As in upstream DeepSWE v1.1, the agent must commit its work. At verification time the resources server executes the
task's pinned `[[verifier.collect]]` hook, which writes `git diff --binary <base_commit> HEAD` to
`/logs/artifacts/model.patch`. That patch is applied and graded in a fresh verifier sandbox. The untrusted agent
sandbox defaults to deny-all egress, and the OpenCode benchmark adds only the resolved Gym model-server host to that
policy. The trusted fresh verifier keeps the canonical container network stack because the current egress sidecar
disables IPv6 loopback required by upstream test suites; verifier assets and the submitted patch are supplied only
by the resources server.
