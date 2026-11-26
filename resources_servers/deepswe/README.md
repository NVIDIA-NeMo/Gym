# DeepSWE

This resources server implements the DeepSWE v1.1 two-environment evaluation contract. The coding agent works in
one sandbox; its final workspace patch is graded by the canonical DeepSWE verifier in a fresh sandbox based on the
same task image.

## Prepare pinned task assets

```bash
python -m resources_servers.deepswe.prepare \
  --source-dir /path/to/deep-swe
```

Preparation also downloads the pinned ripgrep executable that OpenCode needs into
`resources_servers/deepswe/data/cache/opencode-runtime/rg` and checks its digest. With `--no-download`, preparation
uses no network: place the pinned `rg` at that path first, or preparation stops with an error naming the missing or
differing file.

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

Tests that GNU `timeout` stops at the task's limit (`verifier.timeout_sec`) inside the verifier sandbox on the model's
patch are a confirmed deadline: the failure kind `deepswe:test_deadline`, classified as the model's, at the
`verification` stage. This server's default for that kind is `count_as_zero`, so the patch scores 0
(`verifier_timeout_s` records the limit). The default gives way to what the deployment writes: an explicit
`outcome_policy.by_stage.verification.default_action: fail` (or `by_failure_class.model: fail`) leaves the deadline
ungraded and masked, and a rule `by_failure_kind.deepswe:test_deadline` decides it over everything (`count_as_zero` or
`fail`; `retry` is refused at load, since a confirmed deadline is never run again). A `retry` as the stage's default
is for inconclusive attempts and does not apply to the deadline. `count_as_zero` is accepted only for the deadline's
own rule: the other verification rules also decide inconclusive attempts. The applied action and classification are
saved with the result (`failure_outcome_action`, `failure_outcome_attribution`). A timeout reported only by the
sandbox provider, and a golden patch past the limit, stay inconclusive.

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
`benchmarks/nemotron_3.5_super/sandbox_utils.yaml` does this for its mounted copy. The benchmark stages ripgrep
itself: `prepare.py` downloads ripgrep 15.1.0 (x86_64 musl) and checks its digest, the resources server copies it into
each agent sandbox (`agent_ripgrep_path`), and the agent pins that copy (`sandbox_ripgrep_*`) and puts its directory
on the PATH of its tool subprocesses (`sandbox_path_prefix`), so OpenCode's `glob` and `grep` tools and `rg` in the
shell work without network, as on the previous recipe. Pin OpenCode's plugin dependencies the same way
(`sandbox_plugin_dependencies_*`, see the `opencode_agent` README): without them each start waits for a plugin
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

## Run Hermes rollouts

`benchmarks/deepswe/hermes.yaml` runs `hermes_agent` on the same episode: the same agent sandbox, three-hour budget,
episode bound and verifier as the OpenCode recipe. The Agent Server stages Hermes's runtime into the sandbox, so
nothing has to be mounted and the sandbox needs no network. The recipe raises Hermes's step limit to 1000 model calls
(`max_turns`); its default of 90 ends runs on these tasks within minutes. To compare the two harnesses under the same
limit, give OpenCode `opencode_config.agent.build.steps: 1000`. Hermes keeps its own per-command timeout (180 seconds)
and toolsets, and declares the 262144-token context window; match that value to the serving window when you change it.

```bash
gym env start \
  --config benchmarks/deepswe/hermes.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/<model>/configs/<model>.yaml
```

## Run NOOA rollouts

`benchmarks/deepswe/nooa.yaml` runs `nooa_agent` (NOOA's BenchAgent, in sandboxed mode) on the same episode: the same
agent sandbox, three-hour budget, episode bound and verifier as the OpenCode recipe. The agent sandbox has no network,
so NOOA installs its runtime from an offline bundle that you build once per checkout, outside the task sandbox, on a
host with Docker. The bundle is bound to the checkout's tracked source: build it again after any change.

```bash
export NOOA_SWE_RUNTIME_WHEELHOUSE=/absolute/path/to/new-runtime-bundle
python -m responses_api_agents.nooa_agent.offline_runtime \
  --output "$NOOA_SWE_RUNTIME_WHEELHOUSE" --target x86_64/gnu
gym env start \
  --config benchmarks/deepswe/nooa.yaml \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/<model>/configs/<model>.yaml
```

NOOA has no model-call limit by default (`max_policy_calls`), and the recipe declares the 262144-token context window
(`context_window`); match that value to the serving window when you change it. A run that reaches the three-hour worker
ceiling is graded as it stands, as with OpenCode and Hermes.

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
