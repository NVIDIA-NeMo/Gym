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

A deployment without such a mount can have the resources server copy the binary into each agent sandbox instead,
with `agent_runtime_paths` (see [Run other harnesses](#run-other-harnesses)); the agent then pins the copy:

```bash
  ++deepswe_opencode_resources_server.resources_servers.deepswe.agent_runtime_paths.opencode=/path/to/opencode-linux-x64 \
  ++deepswe_opencode_agent.responses_api_agents.opencode_agent.sandbox_executable_path=/tmp/nemo-gym-deepswe/opencode \
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

## Run other harnesses

`benchmarks/deepswe/{hermes,nooa,openclaw,pi,codex}.yaml` are presets of the DeepSWE benchmark: the same tasks,
resources server and verifier, with another harness in place of OpenCode. A preset is not a benchmark of its own and
has no manifest; the resources server's `allowed_agents` (`configs/deepswe.yaml`) lists every preset's harness. Each
preset binds its own resources server, agent and `single_agent_turn_legacy` Environment Server instances
(`deepswe_<harness>_*`), narrows that resources server's `allowed_agents` to its harness, and lets the agent sandbox
reach the policy model server's host (`sandbox_model_server`) and nothing else. Each harness keeps the three-hour agent
budget and the 18,000-second episode bound of the OpenCode benchmark.

Select a harness with its preset (`--benchmark deepswe/<harness>`), not with `--agent-type` on the OpenCode benchmark.
These harnesses are session agents, which run only where a config binds them, and the preset also sets the harness's
runtime, budget and concurrency for DeepSWE.

The agent sandbox cannot download a harness runtime, so each config gets one in without network:

| Config (`--benchmark`) | Runtime in the agent sandbox | What to provide |
| --- | --- | --- |
| `deepswe/opencode` | The OpenCode binary (`sandbox_executable_path`) | A mounted copy, or `agent_runtime_paths.opencode` (above) |
| `deepswe/hermes` | Built by the Hermes agent server for the sandbox's platform and uploaded at seed | Nothing; the agent server needs network to build it once |
| `deepswe/nooa` | Installed from an offline wheelhouse the controller prepares | `NOOA_SWE_RUNTIME_WHEELHOUSE`, built with the `manylinux_2_28` builder (see the file's header) |
| `deepswe/openclaw`, `deepswe/pi`, `deepswe/codex` | Node and the harness package (`sandbox_runtime_dir`) | `DEEPSWE_OPENCLAW_RUNTIME`, `DEEPSWE_PI_RUNTIME`, `DEEPSWE_CODEX_RUNTIME` |

`agent_runtime_paths` maps a name to a file or directory on the resources server's host. The resources server copies
each into every agent sandbox, after ripgrep and before the agent session starts: a file to
`/tmp/nemo-gym-deepswe/<name>` as an executable, a directory's contents into `/tmp/nemo-gym-deepswe/<name>/`. It packs
a directory once at startup and sends it as one gzip tar, which the sandbox's `tar` unpacks. A name left unset or
blank (the environment variable is missing or empty), a path that does not exist, or a directory that holds the Gym
checkout or the DeepSWE tasks stops the server at startup. The harness then checks what it finds there against its
pinned version and refuses the seed otherwise.

Build a Node runtime with the harness's own installer on an x86_64 Linux host with network, at the version the
benchmark pins and at the path it will have in the sandbox:

```bash
bash responses_api_agents/openclaw_agent/install_openclaw_runtime.sh /tmp/nemo-gym-deepswe/openclaw 2026.6.11
bash responses_api_agents/pi_agent/install_pi_runtime.sh /tmp/nemo-gym-deepswe/pi 0.80.2
bash responses_api_agents/codex_agent/install_codex_runtime.sh /tmp/nemo-gym-deepswe/codex 0.144.4
```

The task images are Debian with glibc, so a runtime built on a glibc host runs there. One way to get such a host is
a sandbox of a task image with network, from which the directory is downloaded. Each runtime is a few hundred
megabytes, and every agent sandbox receives its copy, at most two uploads at a time per resources server, while its
episode's deadline runs. So these presets admit at most 128 episodes at once (`max_concurrent_episodes`), and the
others wait for admission before their deadline starts. At high concurrency, a runtime mounted read-only into the
sandboxes avoids that cost.

```bash
export DEEPSWE_PI_RUNTIME=/path/to/pi-runtime
gym eval run --benchmark deepswe/pi --split benchmark \
  --config nemo_gym/sandbox/providers/opensandbox/configs/opensandbox.yaml \
  --config responses_api_models/<model>/configs/<model>.yaml \
  -o results/deepswe-pi/rollouts.jsonl ++use_absolute_ip=true
```
