# OpenClaw Agent

Runs OpenClaw CLI (`openclaw agent --local --json`).
OpenClaw runs its own tools internally.
Resources server is used for verifier.

Minimal, meant to be extended, and currently eval-only. 

## Quick start

OpenClaw must be installed (or it is auto-installed on first start). 
Make sure `env.yaml` is also set.

```bash
gym env start --resources-server math_with_judge/math_with_judge_openclaw_agent

gym eval run --no-serve --agent math_with_judge_openclaw_agent \
  --input resources_servers/math_with_judge/data/example.jsonl \
  --output openclaw_rollout.jsonl --limit 3
```

## Model id

OpenClaw drops the leading `<provider>/` to form the upstream id,
so we include an extra prefix, such as for `nvidia/...` ids:

```yaml
model: nvinf/nvidia/meta/llama-3.3-70b-instruct
openclaw_config:
  models:
    providers:
      nvinf:
        api: openai-completions
        baseUrl: ${policy_base_url}
        apiKey: ${policy_api_key}
        models:
        - {id: nvidia/meta/llama-3.3-70b-instruct, name: nvidia/meta/llama-3.3-70b-instruct, api: openai-completions}
```

## Config fields

- `concurrency`: max simultaneous `run()` calls
- `command`: the OpenClaw command, split on spaces so a multi-word launcher works (e.g. `npx openclaw`)
- `model`: `<provider>/<model-name>` (see Model id)
- `workspace_root`: where per-request workspaces are created and deleted
- `openclaw_agent_id`: passed to `--agent`
- `thinking`: passed to `--thinking` (off, low, medium, high, ...)
- `system_prompt`: prepended to the user message
- `setup_timeout`: seconds for `openclaw setup`
- `timeout`: seconds for the `openclaw agent` run
- `extra_args`: extra flags appended to `openclaw agent`
- `env`: extra env vars for the subprocess (e.g. provider API keys)
- `openclaw_config`: deep-merged into the generated `openclaw.json`
- `openclaw_version`: npm version to pin on install (null means latest)

See `configs/openclaw_agent.yaml`.