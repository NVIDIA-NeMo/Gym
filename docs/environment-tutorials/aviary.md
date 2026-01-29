(environment-aviary)=

# Aviary

Integration with [Future-House/aviary](https://github.com/Future-House/aviary), a gymnasium for defining custom language agent RL environments.

Aviary is a framework for building custom RL environments with tool use and multi-step reasoning. Environments built in Aviary can be ran through NeMo Gym for training and inference. The library features pre-existing environments on math, general knowledge, biological sequences, scientific literature search, and protein stability.

---

## Available Environments

The integration includes several pre-built Aviary environments:

- **GSM8K** (`gsm8k_app.py`) - Grade school math problems with calculator tool
- **HotPotQA** (`hotpotqa_app.py`) - Multi-hop question answering
- **BixBench** (`notebook_app.py`) - Jupyter notebook execution for scientific tasks
- **Client/Proxy** (`client_app.py`) - Generic interface to remote Aviary dataset servers

---

## Example Usage

### GSM8K Environment

Run the GSM8K Aviary resources server with a model config:

```bash
ng_run "+config_paths=[resources_servers/aviary/configs/gsm8k_aviary.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Collect rollouts:

```bash
ng_collect_rollouts \
    +agent_name=gsm8k_aviary_agent \
    +input_jsonl_fpath=resources_servers/aviary/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/aviary/data/example_rollouts.jsonl
```

---

## Reference

- [Aviary GitHub](https://github.com/Future-House/aviary) - Official Aviary repository
- [Aviary Paper](https://arxiv.org/abs/2412.21154) - Training language agents on challenging scientific tasks
- `resources_servers/aviary/` - NeMo Gym resources server implementations
- `responses_api_agents/aviary_agent/` - NeMo Gym aviary agent integration
