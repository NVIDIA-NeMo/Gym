# KiloCode Agent

Runs the [Kilo Code](https://kilo.ai) CLI (`kilo run`). Kilo Code is a fork of OpenCode, so this
agent mirrors the `opencode_agent`: Kilo runs its own tools internally, and its JSON event stream
(`--format json`) is parsed into Gym format and verified by the resources server.

Minimal, meant to be extended, and currently eval-only: token IDs and logprobs are not wired up.

## Quick start

Kilo must be on PATH (auto-installed on first start, or `npm install -g @kilocode/cli`). Set
`policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`; the model server started
by `--model-type` serves that backend and Kilo calls the model server.

```bash
gym env start \
  --resources-server math_with_judge/math_with_judge_kilocode_agent \
  --model-type openai_model

gym eval run --no-serve --agent math_with_judge_kilocode_agent \
  --input resources_servers/math_with_judge/data/example.jsonl \
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
