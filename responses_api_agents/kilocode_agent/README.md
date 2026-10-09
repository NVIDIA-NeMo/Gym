# KiloCode Agent

Runs the [Kilo Code](https://kilo.ai) CLI (`kilo run`). Kilo Code is a fork of OpenCode, so this
agent mirrors the `opencode_agent`: Kilo runs its own tools internally, and its JSON event stream
(`--format json`) is parsed into Gym format and verified by the resources server.

Minimal, meant to be extended, and currently eval-only: token IDs and logprobs are not wired up.

## Quick start

For local execution, Kilo must be on PATH (auto-installed on first invocation, or `npm install -g @kilocode/cli`). Set
`policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`; the model server started
by `--model-type` serves that backend and Kilo calls the model server.

```bash
gym env start \
  --config environments/kilocode_math/config.yaml \
  --model-type openai_model

gym eval run --no-serve --agent kilocode_math_agent \
  --input environments/kilocode_math/data/example.jsonl \
  --output kilocode_rollout.jsonl
```

Per request the agent writes `kilo.json` into an isolated run dir and runs one `kilo run --auto
--pure --format json`, then parses the streamed JSON events for the trajectory. The subprocess runs
with `KILO_NO_DAEMON=1` (fresh embedded server per run — no shared daemon), `KILO_DB=:memory:`
(ephemeral sessions), and per-run `XDG_DATA_HOME`/`XDG_CONFIG_HOME` pointed inside the run dir, so
runs don't share state and the global `~/.config/kilo` never bleeds in. `--pure` runs without
external plugins, so codebase indexing never starts. The project `kilo.json` written into the run dir
supplies the provider and permissions.

## Model server

With `model_server` set (the shipped default), Kilo's model calls go to that Gym model server rather
than to a provider directly. That is what makes requests and responses show up in Gym's capture, and
it means one config runs against vLLM, OpenAI, or an inference provider by swapping `--model-type`.
The agent writes a `nemo` provider into `kilo.json` pointed at the server's URL and passes
`-m nemo/<model>`, so `model` is the bare model name:

```yaml
model_server: {type: responses_api_models, name: policy_model}
model: ${policy_model_name}
```

`context_window`, `max_output_tokens`, and `reasoning_field` describe the served model to Kilo and
apply only on this path (see Config fields). During a `/run` the base URL carries the per-rollout
`/ng-rollout/<id>` prefix, so captured model calls are attributable to the rollout that made them.

### Sizing the output budget

The shipped `context_window` (32768) and `max_output_tokens` (8192) assume a 32k-window model server.
Both need to match whatever you actually serve, and `max_output_tokens` is the one that bites: Kilo's
system prompt and tool definitions run to roughly 10k tokens, so a large output budget pushes
`prompt + max_tokens` past `max_model_len`. vLLM rejects that with a 400, which the Gym model server
converts into an empty completion with `finish_reason: length` rather than an error. The run then
produces no assistant message and scores zero, with nothing in the CLI's own output to say why. The
agent logs a warning when it sees that shape; the fix is to lower `max_output_tokens` (or serve a
larger window), not to raise it.

### Calling a provider directly

Set `model_server: null` and declare the provider yourself in `kilo_config`. `model` is then
`<provider>/<model-name>`, where the provider is a label defined in `kilo_config` rather than a
service. This bypasses both Gym and the Kilo Gateway, so no Kilo account is needed and no model calls
are captured:

```yaml
model_server: null
model: policy/${policy_model_name}
kilo_config:
  provider:
    policy:
      npm: "@ai-sdk/openai-compatible"
      options:
        baseURL: ${policy_base_url}
        apiKey: ${policy_api_key}
```

Kilo splits `-m` on the first `/`, so `policy/Qwen/Qwen3-8B` is the model `Qwen/Qwen3-8B` under the
provider `policy`. It rejects a model that is not listed in its provider's `models` map
(`Model not found: …`), so the agent registers `model` there when it writes `kilo.json`. Only add
`models` entries by hand if you need per-model options; note that the config merge is struct-mode, so
a config that uses `_inherit_from` cannot add new keys to `models`.

## Config fields

- `model_server`: Gym model server Kilo calls; `null` to call a provider directly (see Model server)
- `concurrency`: max simultaneous `run()` calls
- `command`: the Kilo command, split on spaces so a multi-word launcher works (e.g. `npx kilo`)
- `model`: the model name, or `<provider>/<model-name>` without a model server (see Model server)
- `openai_api_key`: passed to the subprocess as `OPENAI_API_KEY`; ignored when `model_server` is set
- `openai_base_url`: passed to the subprocess as `OPENAI_BASE_URL`; ignored when `model_server` is set
- `env`: extra env vars for the subprocess
- `workspace_root`: where per-request run dirs are created and deleted
- `repo_dir`: optional persistent project dir to run in (default: ephemeral per-request dir)
- `thinking`: passes `--thinking` when true (only then are `reasoning` events emitted/captured)
- `system_prompt`: prepended to the user message
- `setup_timeout`: reserved, currently unused
- `timeout`: seconds for the `kilo run` call (the only runaway bound — Kilo has no `--max-turns`)
- `extra_args`: extra flags appended to `kilo run`
- `kilo_config`: written to `kilo.json` in the run dir (OpenCode-compatible schema)
- `context_window`: the served model's context window. Kilo measures the session against it, but only
  auto-compacts when `kilo_config` also sets `compaction.threshold_percent`; `0` turns the accounting
  off entirely. `model_server` only.
- `max_output_tokens`: per-request output budget. Kilo asks for `min(this, 32000)`, its own
  `OUTPUT_TOKEN_MAX`, so values above 32000 have no effect. Setting it too high fails silently — see
  Sizing the output budget. `model_server` only.
- `reasoning_field`: response field carrying reasoning text, written as `interleaved.field`. Gym model
  servers emit `reasoning_content`; Kilo turns interleaved reasoning off for custom OpenAI-compatible
  providers unless the field is named, so without this the reasoning channel is dropped. `null` leaves
  Kilo's default. `model_server` only.
- `kilo_version`: `@kilocode/cli` npm version installed on a clean machine (shipped pinned to
  `7.4.15`; the parser was validated against it, so treat a bump as a deliberate change — raise it,
  re-run the tests and the live eval, then commit). `null` installs `@latest`.

See `configs/kilocode_agent.yaml`.

## Task-sandbox integration (draft)

[configs/kilocode_sandboxed.yaml](configs/kilocode_sandboxed.yaml) runs Kilo inside the task
sandbox supplied by the benchmark's Resources Server. The Environment Server coordinates setup,
agent execution, verification, and cleanup; Kilo runs its own model/tool loop in the sandbox while
Gym stays on the agent-server host. The local configuration above is unchanged.

### Configure and run

The benchmark owns task data, preparation, verification, and task sandbox settings. The harness
owns its runtime and model/tool loop. The Environment Server binds the two and closes the agent
before verification. For SWE-bench Pro, save this composition as `run.yaml`:

```yaml
config_paths:
  - resources_servers/swebench_pro/configs/swebench_pro.yaml
  - responses_api_agents/kilocode_agent/configs/kilocode_sandboxed.yaml
  - environment_servers/single_agent_turn_legacy/configs/single_agent_turn_legacy.yaml

single_agent_turn_legacy:
  environment_servers:
    single_agent_turn_legacy:
      resources_server:
        name: swebench_pro_resources_server
      agent_server:
        name: kilocode_agent
      resources_tool_transports: []
```

Supply the `policy_model` Gym Model Server, `policy_model_name`, and `sandbox` provider in
`model-provider.yaml`. The sandbox must be able to reach the Model Server.

```bash
python benchmarks/swebench/pro/prepare.py

gym env start --config run.yaml --config model-provider.yaml

gym eval run --no-serve \
  --config run.yaml --config model-provider.yaml \
  --agent kilocode_agent \
  -i benchmarks/swebench/data/swebench_pro_benchmark.jsonl \
  -o rollouts.jsonl --limit 3 --concurrency 3
```

Keep `resources_tool_transports: []`: Kilo provides its own sandbox tools and rejects required
external tool grants. Collection calls the Environment Server's `/run`, never the agent's `/run`.

Size the harness for the served model and benchmark: set `context_window` / `max_output_tokens` to
the model's limits, and raise `timeout` for long-horizon tasks (the SWE-bench Pro run used 21600).
`timeout` also sets Kilo's model-stream watchdog (`chunkTimeout`), because the Gym Model Server
returns a completion only once it is finished. `webfetch` is allowed by default; deny it in
`kilo_config.permission` if a benchmark must not reach the web.

### Switch harness or benchmark

To change a compatible harness, replace its config import, the Environment Server's
`agent_server.name`, and the collection command's `--agent`. To change a compatible benchmark,
replace its Resources config/reference and prepared input, keeping the harness definition
unchanged. Check task-image support and the benchmark's `allowed_agents` before running a new pairing.

### Runtime and ownership

The task image must be Linux (x86_64 or aarch64, glibc or musl) with Python 3.8+, and the sandbox
needs network access to nodejs.org, the npm registry, and the Gym Model Server. Task images need not
ship Node: session setup runs [install_kilo_runtime.sh](install_kilo_runtime.sh), which downloads a
checksum-verified Node 22 and installs the pinned `kilo_version` into a private directory under
`/tmp/nemo-gym-kilo-sessions/<id>/`. Missing download tools (`curl`, CA certificates) are installed
with apt-get or apk when running as root; otherwise they must be in the image. Kilo runs through
that private Node, so the task's `PATH` and runtimes are untouched, and nothing is written into the
task workdir except the agent's own edits. No Kilo CLI is installed on the agent-server host for
sandbox sessions.

Kilo borrows the sandbox when Resources passes `sandbox_access`, and never destroys it. Otherwise
set `sandbox_provider` / `sandbox_config` for an agent-owned sandbox (default workdir `/app`); it
is destroyed at close, so verifiers that inspect the repository need a Resources-owned sandbox.

### Requests and results

Requests carry one text user message, optionally preceded by a system/developer message. Other
request fields, model overrides, `repo_dir`, `extra_args` and `env` are rejected. Identical
activation retries join the running task; session close owns cancellation.

CLI/API errors and nonzero exits raise, so the episode fails instead of scoring zero. As with Pi and
Codex, the CLI runs under Node, so an agent command such as `pkill -f node` also stops the harness; the
episode is then reported as an infrastructure failure. A killed run
or a `length` stop returns an `incomplete` response. Observations come from a snapshot of Kilo's
SQLite database (root and child sessions, tool timing, usage), annotated with the root outcome and
a sandbox record; if the snapshot is unavailable, a coverage gap is reported. Token IDs and
logprobs are not available.

Validated on SWE-bench Pro with the stock Resources server (Nemotron 3.5 Super, 2026-10-09): a
five-task smoke covering Ubuntu, Node, Go, Python and Alpine images completed setup and grading, and
the full run (731 tasks x 3 attempts, 256 concurrent episodes) scored 51.30% mean pass@1 (per repeat
52.39 / 50.21 / 51.30), with 2 agent-caused environment failures counted as failures.
