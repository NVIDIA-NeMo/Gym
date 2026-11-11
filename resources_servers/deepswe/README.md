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

A verifier attempt that produces no verdict (no `reward.json`, or a verifier sandbox that could not be created or
run) is run again on a fresh sandbox with the same collected patch, at most `inconclusive_verification_retries`
(default 2) more times; the response records `verification_attempts` and the reason of each inconclusive attempt. A
malformed verdict, a timeout and a conclusive result are not run again, and a patch that breaks the test script stays
an inconclusive, masked result after the bound as before.

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

The agent sandbox has no internet access, so the pinned OpenCode runtime cannot be downloaded into it. Set
`sandbox_executable_path` to an OpenCode 1.17.11 binary already inside the sandbox, for example on a read-only mount,
and `sandbox_executable_sha256` to its digest; session setup refuses any other file.
`benchmarks/nemotron_3.5_super/sandbox_utils.yaml` does this for its mounted copy. Pin ripgrep and OpenCode's plugin
dependencies the same way (`sandbox_ripgrep_*`, `sandbox_plugin_dependencies_*`, see the `opencode_agent` README):
without them OpenCode's `glob` and `grep` tools need an image that provides `rg`, and each start waits for a plugin
install that cannot reach npm. Launch Gym with `+use_absolute_ip=true` when OpenSandbox needs a host-routable
model-server address.

```bash
gym env start \
  --config benchmarks/deepswe/opencode.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/<model>/configs/<model>.yaml \
  ++deepswe_opencode_agent.responses_api_agents.opencode_agent.sandbox_executable_path=/mnt/opencode/opencode-linux-x64 \
  ++deepswe_opencode_agent.responses_api_agents.opencode_agent.sandbox_executable_sha256=<sha256>
```

The resources server keeps each episode session until the Environment Server closes it. A session that is never
closed expires after the agent sandbox's `ttl_s`, and shutdown stops what is left. Verify grades a session once: an
identical retry returns the same verdict, and a verify after close is masked as a lost session.

Agent settings live under `deepswe_opencode_agent.responses_api_agents.opencode_agent`. The previous
`opencode_sandboxed_agent` recipe's settings map to that block as follows:

| Previous setting | Now |
| --- | --- |
| `sandbox_timeout: 10800` | `timeout: 10800` |
| `opencode_max_context_window` | `context_window`; `max_output_tokens` follows it, as the previous output limit did |
| `remote_opencode_install_script_path`, `remote_opencode_binary_path` | `sandbox_executable_path` and `sandbox_executable_sha256` (no install script) |
| `remote_opencode_musl_binary_path` | `sandbox_executable_path` and `sandbox_executable_sha256` of the musl build, for musl images |
| `local_ripgrep_binary_path` | `sandbox_ripgrep_path` and `sandbox_ripgrep_sha256` |
| `opencode_config.permission` | Same command restrictions; `webfetch`, `websearch` and `skill` are disabled under `opencode_config.tools` |
| `execution_failure_reward_zero: false` (a run that ended on an OpenCode error was still verified and graded normally) | A run on which OpenCode reports a model error returns status `failed` and is still verified and graded. A run that fails without one (lost sandbox, nonzero exit, no assistant result) fails the episode instead of scoring 0 |
| OpenCode read the repository's `AGENTS.md` or `CLAUDE.md` and project config | `session_project_config: true` |
| `datasets` | Unchanged, on this agent block |

The session path does not support these previous controls, and the agent ignores them if they are set:

- `opencode_model_call_timeout` (one hour per model request): model requests have no per-request deadline; the
  three-hour budget bounds the run.
- `token_id_capture`, `debug` and `artifacts_dir`. The transcript is returned in the rollout's agent observations
  instead of an export file; set `export_dir` to also keep OpenCode's own exports in the sandbox.

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
