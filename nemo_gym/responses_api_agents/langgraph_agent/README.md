# LangGraph Agent

LangGraph agent adapter. 

Examples here include a iterative reflection agent, subagent orchestrator agent, parallel thinking agent, and rewoo agent. Most of these are based on langgraph examples: https://github.com/langchain-ai/langgraph/tree/main/examples

Please note that agents such as parallel thinking which produce non-monotonically increasing trajectories will not work with NeMo RL training by default, as NeMo RL expects monotonically increasing trajecories. These can be used for rollouts or evaluations, or used in research experiments in developing approaches to train on non-monotonic agent trajectories.

## Quick Start

```bash
ng_run "+config_paths=[nemo_gym/resources_servers/reasoning_gym/configs/reflection_agent.yaml,nemo_gym/responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_reflection_agent \
    +input_jsonl_fpath=nemo_gym/resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=example_rollouts.jsonl \
    +limit=1
```
