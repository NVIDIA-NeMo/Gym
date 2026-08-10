# Terminus-2 Agent

Runs [Harbor's Terminus-2](https://github.com/harbor-framework/harbor/tree/main/src/harbor/agents/terminus_2)
as a NeMo Gym Responses API agent.

Terminus-2 executes terminal commands in the agent process working directory. Use
`anyterminal_agent` when each rollout should run inside a Terminal Bench task container.

## AnyTerminal

```bash
gym eval run \
  --config responses_api_agents/anyterminal_agent/configs/anyterminal_terminus_2.yaml \
  --config nemo_gym/sandbox/providers/docker/configs/docker.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  --agent anyterminal_terminus_2 \
  --input responses_api_agents/anyterminal_agent/data/terminal_bench.jsonl \
  --output terminus_2_rollouts.jsonl
```

Use `configs/terminus_2_agent.yaml` to connect the standalone agent to another
resources server. The working directory must provide `bash`, `tmux`, and `script`.

The agent converts the Terminus-2 trajectory to Responses API messages and tool
calls. Sampling parameters are forwarded to the configured model server.
