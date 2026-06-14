# Pi Agent

Runs the [pi](https://github.com/earendil-works/pi) CLI (`pi --print --mode json --no-session`). 
pi runs its own tools internally. Resources server for verifier.

Minimal, meant to be modified if needed, and currently eval-only. Token IDs and logprobs are not wired up and
it does not use a Gym model server yet.

## Quick start

pi must be on PATH (auto-installed on first start, or `npm install -g @earendil-works/pi-coding-agent`).
Put `policy_base_url`, `policy_api_key`, and `policy_model_name` in `env.yaml`.

```bash
ng_run "+config_paths=[resources_servers/math_with_judge/configs/math_with_judge_pi_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"

ng_collect_rollouts +agent_name=math_with_judge_pi_agent \
  +input_jsonl_fpath=responses_api_agents/pi_agent/data/example.jsonl \
  +output_jsonl_fpath=pi_rollout.jsonl +limit=5
```

Per request the agent writes `models.json` into an isolated `HOME`, runs one `pi` invocation with
stdin from `/dev/null`, then parses the jsonl `message_end` events. Example rollouts are in `data/`.

## Model id

`model` is `<provider>/<model-id>`. Define the provider in `models_config` (written to
`~/.pi/agent/models.json`) and reference it here:

```yaml
model: nvinf/nvidia/qwen/qwen3-next-80b-a3b-instruct
models_config:
  providers:
    nvinf:
      baseUrl: ${policy_base_url}
      api: openai-completions
      apiKey: ${policy_api_key}
      models:
      - id: nvidia/qwen/qwen3-next-80b-a3b-instruct
        reasoning: false
```

## Config fields

- `concurrency`: max simultaneous `run()` calls
- `command`: the pi command, split on spaces so a multi-word launcher works
- `model`: `<provider>/<model-id>` (see Model id)
- `env`: extra env vars for the subprocess (e.g. provider API keys)
- `workspace_root`: where per-request HOMEs are created and deleted
- `thinking`: passed to `--thinking` (off, minimal, low, medium, high, xhigh)
- `system_prompt`: appended via `--append-system-prompt`
- `timeout`: seconds for the `pi` run
- `extra_args`: extra flags appended to the `pi` command
- `models_config`: written to `~/.pi/agent/models.json`
- `pi_version`: npm version to pin on install (null means latest)

See `configs/pi_agent.yaml`.