# Verifiers Integration for NeMo Gym

Integration of [verifiers](https://github.com/primeintellect-ai/verifiers) environments with NeMo Gym.

From verifiers' readme: "Verifiers is a library for creating RL environments for LLMs." It includes many community implementations on Prime Intellect's [Environments Hub](https://app.primeintellect.ai/dashboard/environments).

## Installation

Install verifiers and an environment from the Environments Hub:

```bash
uv add verifiers
uv tool install prime 
prime env install primeintellect/reverse-text
prime env install primeintellect/math-python
```

Currently the environment package must also be listed in the `requirements.txt` files for both the resource server and agent, since they run in isolated venvs. For example:

```
--extra-index-url https://hub.primeintellect.ai/primeintellect/simple/
reverse-text
```
i
## Dataset Prep

**Create dataset from verifiers environment:**
```bash
python resources_servers/verifiers/scripts/create_dataset.py \
    --env-id reverse-text \
    --size 100 \
    --output resources_servers/verifiers/data/reverse_text_train.jsonl
```

## Rollout Collection

```bash
ng_run "+config_paths=[resources_servers/verifiers/configs/verifiers.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

```bash
ng_collect_rollouts \
    +agent_name=verifiers_agent \
    +input_jsonl_fpath=resources_servers/verifiers/data/example.jsonl \
    +output_jsonl_fpath=results/verifiers_rollouts.jsonl \
    +limit=5
```

## Available Environments

Any text-based environment from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments):
