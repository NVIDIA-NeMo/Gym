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

## Description

Runs [hermes-agent](https://github.com/NousResearch/hermes-agent) in a nemo gym agent server via the `run_agent.AIAgent` entrypoint, which matches the hermes-agent CLI and user experience. The point is to teach the model to operate inside the hermes-agent harness on real tasks.

## Setup

`hermes-agent` is pinned to a hermes agent fork and commit in `requirements.txt` with patches for token id tracking, chat template, and sampling parameter set for training.

Notably, for agent integrations like this, the agent must point at Gym's model server, it must include prompt and generation token id fields for Nemo RL and other trainer's on policy token id correction, it must not override sampling parameters like temperature and top p, and it must not do non-monotonic things like dropping past reasoning content or context compaction.

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
| `system_prompt` | `null` | passed as `system_message` to `run_conversation`; falls back to any system item in `body.input` |

The model-server url is resolved at request time and passed to `AIAgent(base_url=..., api_key="gym")`.