(pypi-install)=

# Using NeMo Gym from PyPI

NeMo Gym is available as a PyPI package. This page covers how to run existing environments and create new ones using the PyPI package rather than cloning the repository and installing locally.

## Install

```bash
pip install nemo-gym
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv pip install nemo-gym
```

## Using an Existing Environment

Environments in NeMo Gym are composed of an agent server typically paired with one or more resources servers. NeMo Gym ships with many environments that can be used out of the box after isntalling the package. The simple agent supports LLM with tools, which is sufficient for many use cases. A new agent is required for logic other than LLM with tools, or for external agent framework integrations. 

Create a working directory and add an `env.yaml` with your model endpoint:

```bash
mkdir myproject && cd myproject

cat > env.yaml <<EOF
policy_base_url: http://localhost:8000/v1
policy_api_key: EMPTY
policy_model_name: your-model-name
EOF
```

Start a built-in environment. Config paths are resolved against the package install location automatically:

```bash
ng_run "+config_paths=[resources_servers/arc_agi/configs/arc_agi.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

Every environment ships with 5 example tasks so you can run rollouts without any local dataset setup. Full training datasets are available on the [NVIDIA NeMo Gym HuggingFace collection](https://huggingface.co/collections/nvidia/nemo-gym). Collect rollouts using the package-relative path directly:

```bash
mkdir -p results
ng_collect_rollouts \
    +agent_name=arc_agi_simple_agent \
    +input_jsonl_fpath=resources_servers/arc_agi/data/example.jsonl \
    +output_jsonl_fpath=results/arc_agi_rollouts.jsonl \
    +limit=1
```

## Creating a New Environment

The most common way to make a new environment is by creating a new resources server. The simple agent in NeMo Gym is a LLM with tools implementation, so if your problem can be framed as an LLM with tools, and a verify function, then you only need a new resources server. 

### Scaffold a resources server

Run `ng_init_resources_server` from your project directory. The entrypoint must be under a `resources_servers/` directory so that `ng_run` can locate it:

```bash
ng_init_resources_server +entrypoint=resources_servers/hello_world
```

This creates the following structure:

```
myproject/
├── env.yaml
└── resources_servers/
    └── hello_world/
        ├── app.py
        ├── configs/
        │   └── hello_world.yaml
        ├── tests/
        │   └── test_app.py
        ├── data/
        ├── requirements.txt
        └── README.md
```

Edit `app.py` to implement your `verify()` method and any tools or other environment logic such as state management, and update `configs/hello_world.yaml` as needed.

Add at least one example, to `data/example.jsonl`:

```bash
echo '{"responses_create_params":{"input":[{"role":"user","content":"Hello!"}]},"verifier_metadata":{}}' \
    > resources_servers/hello_world/data/example.jsonl
```

### Run your server

From `myproject/`, pass your local config alongside the built-in agent and model configs. Make sure `env.yaml` is in your current directory so it is discoverable by the CLI.

```bash
ng_run "+config_paths=[resources_servers/hello_world/configs/hello_world.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]"
```

`ng_run` checks your current directory first for `resources_servers/hello_world/`, so your local implementation takes priority over anything in the package. Built-in servers like `responses_api_agents/simple_agent` and `responses_api_models/vllm_model` are resolved from the package install automatically.

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
