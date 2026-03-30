(pypi-install)=

# Using NeMo Gym from PyPI

NeMo Gym is available as a PyPI package. This page covers how to run existing environments and create new ones without cloning the repository.

## Install

```bash
pip install nemo-gym
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install nemo-gym
```

## Using an Existing Environment

NeMo Gym ships with many environments that can be used out of the box. The built-in `simple_agent` supports LLM with tools, which is sufficient for most use cases.

Create a working directory and configure your model endpoint:

```bash
mkdir myproject && cd myproject

cat > env.yaml <<EOF
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: your-model-name
EOF
```

Start a built-in environment:

```bash
ng_run "+config_paths=[resources_servers/arc_agi/configs/arc_agi.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Every environment ships with 5 example tasks so you can collect rollouts without any local dataset setup. Full training datasets are available on the [NVIDIA NeMo Gym HuggingFace collection](https://huggingface.co/collections/nvidia/nemo-gym).

```bash
mkdir -p results
ng_collect_rollouts \
    +agent_name=arc_agi_simple_agent \
    +input_jsonl_fpath=resources_servers/arc_agi/data/example.jsonl \
    +output_jsonl_fpath=results/arc_agi_rollouts.jsonl \
    +limit=1
```

Config paths and data paths are resolved against the package install location automatically. If the same path exists in your current directory, the local version takes priority.

## Creating a New Environment

The most common way to create a new environment is by adding a resources server. If your problem can be framed as an LLM with tools and a verify function, you only need a new resources server. The built-in `simple_agent` handles the rest.

### Scaffold a resources server

The entrypoint must be under a `resources_servers/` directory:

```bash
ng_init_resources_server +entrypoint=resources_servers/hello_world
```

This generates `app.py`, a config YAML, tests, `requirements.txt`, and a `data/` directory. Edit `app.py` to implement your `verify()` method and any tools.

Add at least one example to `data/example.jsonl`:

```bash
echo '{"responses_create_params":{"input":[{"role":"user","content":"Hello!"}]},"verifier_metadata":{}}' \
    > resources_servers/hello_world/data/example.jsonl
```

### Run your server

```bash
ng_run "+config_paths=[resources_servers/hello_world/configs/hello_world.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Your local `resources_servers/hello_world/` takes priority over the package install. Built-in servers like `simple_agent` and `vllm_model` are resolved from the package automatically.

### Collect rollouts

```bash
mkdir -p results
ng_collect_rollouts \
    +agent_name=hello_world_simple_agent \
    +input_jsonl_fpath=resources_servers/hello_world/data/example.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl \
    +limit=1
```

### Creating a custom agent

If the built-in `simple_agent` is not sufficient, use it as a starting point by subclassing it directly, or making a copy of it.

```python
from responses_api_agents.simple_agent.app import SimpleAgent

class MyAgent(SimpleAgent):
    ...
```
