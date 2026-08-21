# LangGraph Agent

LangGraph agent adapter. 

Examples here include a iterative reflection agent, subagent orchestrator agent, parallel thinking agent, and rewoo agent. Most of these are based on langgraph examples: https://github.com/langchain-ai/langgraph/tree/main/examples

Please note that agents such as parallel thinking which produce non-monotonically increasing trajectories will not work with NeMo RL training by default, as NeMo RL expects monotonically increasing trajecories. These can be used for rollouts or evaluations, or used in research experiments in developing approaches to train on non-monotonic agent trajectories.

## Quick Start

```bash
gym env start \
    --resources-server reasoning_gym/reflection_agent \
    --model-type vllm_model
```

```bash
gym eval run --no-serve \
    --agent reasoning_gym_reflection_agent \
    --input resources_servers/reasoning_gym/data/example.jsonl \
    --output example_rollouts.jsonl \
    --limit 1
```
