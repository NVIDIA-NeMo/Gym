# KiloCode Agent

Runs the [Kilo Code](https://kilo.ai) CLI (`kilo run`). Kilo Code is a fork of OpenCode, so this
agent mirrors the `opencode_agent`: Kilo runs its own tools internally, and its JSON event stream
(`--format json`) is parsed into Gym format and verified by the resources server.

Minimal, meant to be extended, and currently eval-only. Token IDs and logprobs are not wired up and
it does not use a Gym model server yet.

## Quick start

Kilo must be on PATH (auto-installed on first start, or `npm install -g @kilocode/cli`). Set
`policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

```bash
gym env start \
  --resources-server math_with_judge/math_with_judge_kilocode_agent \
  --model-type openai_model

gym eval run --no-serve --agent math_with_judge_kilocode_agent \
  --input responses_api_agents/kilocode_agent/data/example.jsonl \
  --output kilocode_rollout.jsonl --limit 5
```

Per request the agent writes `kilo.json` into an isolated run dir and runs one `kilo run --auto
--pure --format json`, then parses the streamed JSON events for the trajectory. The subprocess runs
with `KILO_NO_DAEMON=1` (fresh embedded server per run — no shared daemon), `KILO_DB=:memory:`
(ephemeral sessions), and per-run `XDG_DATA_HOME`/`XDG_CONFIG_HOME` pointed inside the run dir, so
runs don't share state and the global `~/.config/kilo` never bleeds in. `--pure` runs without
external plugins, so codebase indexing never starts. The project `kilo.json` written into the run dir
supplies the provider and permissions.

## Model id

`model` is `<provider>/<model-name>`. To bypass the Kilo Gateway and use a custom OpenAI-compatible
endpoint (e.g. a Gym-served vLLM), define the provider in `kilo_config` (written to `kilo.json`) and
reference it here:

```yaml
model: nvinf/nvidia/qwen/qwen3-next-80b-a3b-instruct
kilo_config:
  provider:
    nvinf:
      npm: "@ai-sdk/openai-compatible"
      options:
        baseURL: ${policy_base_url}
        apiKey: ${policy_api_key}
      models:
        nvidia/qwen/qwen3-next-80b-a3b-instruct: {}
```

## Config fields

- `concurrency`: max simultaneous `run()` calls
- `command`: the Kilo command, split on spaces so a multi-word launcher works (e.g. `npx kilo`)
- `model`: `<provider>/<model-name>` (see Model id)
- `openai_api_key`: passed to the subprocess as `OPENAI_API_KEY`
- `openai_base_url`: passed to the subprocess as `OPENAI_BASE_URL`
- `env`: extra env vars for the subprocess
- `workspace_root`: where per-request run dirs are created and deleted
- `repo_dir`: optional persistent project dir to run in (default: ephemeral per-request dir)
- `thinking`: passes `--thinking` when true (only then are `reasoning` events emitted/captured)
- `system_prompt`: prepended to the user message
- `setup_timeout`: reserved, currently unused
- `timeout`: seconds for the `kilo run` call (the only runaway bound — Kilo has no `--max-turns`)
- `extra_args`: extra flags appended to `kilo run`
- `kilo_config`: written to `kilo.json` in the run dir (OpenCode-compatible schema)
- `kilo_version`: `@kilocode/cli` npm version installed on a clean machine (shipped pinned to
  `7.4.15`; the parser was validated against it, so treat a bump as a deliberate change — raise it,
  re-run the tests and the live eval, then commit). `null` installs `@latest`.

See `configs/kilocode_agent.yaml`.
