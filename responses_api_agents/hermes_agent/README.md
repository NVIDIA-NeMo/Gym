# Hermes Agent

# Quick start

## Create env.yaml in Gym/

```
policy_base_url: https://api.openai.com/v1
policy_api_key: sk...
policy_model_name: gpt-4o
```

## Launch nemo gym servers

```bash
ng_run "+config_paths=[resources_servers/math_with_judge/configs/math_with_judge_hermes_agent.yaml,responses_api_models/openai_model/configs/openai_model.yaml]"
```

## Collect rollouts

```bash
ng_collect_rollouts \
    +agent_name=math_with_judge_hermes_agent \
    +input_jsonl_fpath=resources_servers/math_with_judge/data/example.jsonl \
    +output_jsonl_fpath=hermes_agent_rollout.jsonl \
    +limit=1
```

5 example math rollouts are at `responses_api_agents/hermes_agent/data/` with statistics:

```
Collecting rollouts: 100%|█████████████████████████████████████| 5/5 [00:24<00:00,  4.87s/it]
Sorting results to ensure consistent ordering
Computing aggregate metrics
INFO:     127.0.0.1:9672 - "GET /global_config_dict_yaml HTTP/1.1" 200 OK

Key metrics for math_with_judge_hermes_agent:
{
    "mean/reward": 0.2,
    "mean/turns_used": 1.6,
    "mean/finished_naturally": 1.0,
    "mean/library_reward": 0.2,
    "mean/input_tokens": 0.0,
    "mean/output_tokens": 0.0,
    "mean/total_tokens": 0.0
}
Finished rollout collection! View results at:
Fully materialized inputs: responses_api_agents/hermes_agent/data/example_math_rollouts_materialized_inputs.jsonl
Rollouts: responses_api_agents/hermes_agent/data/example_math_rollouts.jsonl
Aggregate metrics: responses_api_agents/hermes_agent/data/example_math_rollouts_aggregate_metrics.json
```

Example training reward for small multi environment test is shown [here](https://github.com/NVIDIA-NeMo/Gym/pull/1033#issuecomment-4399509664).

## Description

Runs [hermes-agent](https://github.com/NousResearch/hermes-agent) in a nemo gym agent server via the `run_agent.AIAgent` entrypoint, which matches the hermes-agent CLI and user experience. Can be used for benchmarks with hermes agent, or training in the harness.

## Setup

`hermes-agent` is pinned in `requirements.txt` to a fork branch with patches for token id tracking, chat template, and sampling parameters needed for training.

For agent integrations like this, the agent must point at Gym's model server, it must include prompt and generation token id in requests for Nemo RL and other trainer integration on policy token id correction, it must not override sampling parameters like temperature and top p, and it must not do non-monotonic things like dropping past reasoning content or context compaction.

## Resources server compatibility

Works with any resources server based verifier, but does not work for resources server tools or other endpoints out of the box. Hermes Agent ships its own toolset (terminal, file, code_execution, web, etc.), so it does not rely on tools defined in the dataset. It may work with Gymnasium style resources servers, though. In testing, only the resources server's task data and `verify` are used. This means existing benchmarks (math, code, reasoning_gym, mcqa, instruction_following, ...) can be used as-is by adding a `<server>_hermes_agent` config.

## Configuration example

```yaml
hermes_agent:
  responses_api_agents:
    hermes_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: my_verifier
      model_server:
        type: responses_api_models
        name: policy_model
      enabled_toolsets: [terminal, file, code_execution]
      max_turns: 30
      concurrency: 32
      temperature: 1.0
      system_prompt: |
        your system prompt here.
```

| field | default | description |
|-------|---------|-------------|
| `enabled_toolsets` | `null` (all) | forwarded to `AIAgent(enabled_toolsets=...)` |
| `disabled_toolsets` | `null` | forwarded to `AIAgent(disabled_toolsets=...)` |
| `max_turns` | `30` | maps to `AIAgent.max_iterations` |
| `concurrency` | `32` | max simultaneous `run()` calls |
| `temperature` | `1.0` | sampling temperature passed to `AIAgent` |
| `terminal_backend` | `local` | sets `TERMINAL_ENV` (process-global); `local`, `docker`, `daytona`, `modal`, `ssh` |
| `terminal_timeout` | `60` | sets `TERMINAL_TIMEOUT` (process-global); per-command wall-clock seconds |
| `system_prompt` | `null` | passed as `system_message` to `run_conversation`; falls back to any system item in `body.input` |

The model-server url is resolved at request time and passed to `AIAgent(base_url=..., api_key="gym")`. <!-- pragma: allowlist secret -->