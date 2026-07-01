# OpenCode Agent

Runs the OpenCode CLI (`opencode run`). OpenCode runs its own tools internally. 
The run's sqlite session is read for the full trajectory (including tool
calls) and converts it into Gym format and uses resources server to verify.

Minimal, meant to be extended, and currently eval-only. 
Token IDs and logprobs are not wired up and
it does not use a Gym model server yet.

## Quick start

OpenCode must be on PATH (auto-installed on first start, or `npm install -g opencode-ai`). Set
`policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

```bash
gym env start \
  --resources-server math_with_judge/math_with_judge_opencode_agent \
  --model-type openai_model

gym eval run --no-serve --agent math_with_judge_opencode_agent \
  --input responses_api_agents/opencode_agent/data/example.jsonl \
  --output opencode_rollout.jsonl --limit 5
```

Per request the agent writes `opencode.json` into an isolated run dir, runs one `opencode run` with
its own `XDG_DATA_HOME`, then reads the sqlite session for the trajectory.

## Model id

`model` is `<provider>/<model-name>`. For a custom OpenAI-compatible endpoint, define the provider
in `opencode_config` (written to `opencode.json`) and reference it here:

```yaml
model: nvinf/nvidia/qwen/qwen3-next-80b-a3b-instruct
opencode_config:
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
- `command`: the OpenCode command, split on spaces so a multi-word launcher works (e.g. `npx opencode`)
- `model`: `<provider>/<model-name>` (see Model id)
- `openai_api_key`: passed to the subprocess as `OPENAI_API_KEY`
- `openai_base_url`: passed to the subprocess as `OPENAI_BASE_URL`
- `env`: extra env vars for the subprocess
- `workspace_root`: where per-request run dirs are created and deleted
- `thinking`: passes `--thinking` when true
- `system_prompt`: prepended to the user message
- `setup_timeout`: reserved, currently unused
- `timeout`: seconds for the `opencode run` call
- `extra_args`: extra flags appended to `opencode run`
- `opencode_config`: written to `opencode.json` in the run dir
- `opencode_version`: npm version to pin on install (null means latest)

See `configs/opencode_agent.yaml`.