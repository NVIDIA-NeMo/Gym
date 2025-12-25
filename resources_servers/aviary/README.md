> Keywords: Tool Use, Multi-step Reasoning, Environment Interaction, Scientific Tasks

This resource server adapts [**Aviary** environments](https://github.com/Future-House/aviary) into the NeMo Gym resources-server interface, so NeMo Gym agents can interact with Aviary `Environment`s via **OpenAI-style function tools** (tool schemas + tool calls + tool outputs). This allows one to implement tool and environment logic in Aviary, and deploy the environment for inference or training with Gym.

### Implemented servers in this folder

- **GSM8K**: `gsm8k_app.py`
  - Meant primarily as an example, this implements [GSM8k](https://arxiv.org/abs/2110.14168) as a set of environments equipped with a calculator too.
- **HotPotQA**: `hotpotqa_app.py`
  - The HotPotQA environment asks agents to perform multi-hop question answering on the [HotPotQA dataset](https://aclanthology.org/D18-1259/)
- **BixBench**: `notebook_app.py`
  - Implements the [BixBench dataset](https://arxiv.org/abs/2503.00096) as a set of environments that allow execution of a Jupyter notebook.
  - Also serves as an example for how to implement notebook-backed environments for other computational scientific tasks.
- **Client/proxy to a remote Aviary dataset server**: `client_app.py`
  - A generic interface to an Aviary `TaskDatasetServer`. Can be used to interact with any Aviary environments being served remotely.


# Example usage

Run the GSM8K Aviary resources server together with a model config:

```bash
config_paths="resources_servers/aviary/configs/gsm8k_aviary.yaml,\
responses_api_models/vllm_model/configs/vllm_model.yaml"
ng_run "+config_paths=[$config_paths]"
```

Then collect rollouts:

```bash
ng_collect_rollouts \
    +agent_name=gsm8k_aviary_agent +input_jsonl_fpath=resources_servers/aviary/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/aviary/data/example_rollouts.jsonl
```

# Licensing information
Code: Apache 2.0

Data: MIT (GSM8k),  Apache 2.0 (BixBench)

Dependencies
- nemo_gym: Apache 2.0
- aviary:  Apache 2.0