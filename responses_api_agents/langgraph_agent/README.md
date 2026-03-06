# LangGraph Agent

LangGraph agent adapter. 

Reflection agent example: generate, critique, revise loop

## Quick Start

```bash
ng_run "+config_paths=[resources_servers/reasoning_gym/configs/reasoning_gym.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
    +agent_name=reasoning_gym_langgraph_agent \
    +input_jsonl_fpath=resources_servers/reasoning_gym/data/example.jsonl \
    +output_jsonl_fpath=example_rollouts.jsonl \
    +limit=1
```
