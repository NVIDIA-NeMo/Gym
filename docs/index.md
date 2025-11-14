(gym-home)=

# NeMo Gym Documentation

NeMo Gym is a framework for building reinforcement learning (RL) training environments. Gym is used to create data for RL training, and is especially tailored for agentic model training.

At the core of NeMo Gym are three server concepts: **Responses API Model** servers are model endpoints, **Resources** servers contain tool implementations and verification logic, and **Agent** servers orchestrate the interaction between models and resources.

## Quickstart

Run a simple agent and start collecting rollouts for training in under 5 minutes.

::::{tab-set}

:::{tab-item} 1. Set Up

```bash
# Clone and install dependencies
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv --python 3.12 && source .venv/bin/activate
uv sync --extra dev --group docs

# Configure your model API access
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-4.1-2025-04-14" > env.yaml
```

:::

:::{tab-item} 2. Start Servers

**Terminal 1** (start servers):

```bash
# Start servers (this will keep running)
config_paths="resources_servers/example_simple_weather/configs/simple_weather.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**Terminal 2** (interact with agent):

```bash
# In a NEW terminal, activate environment
cd Gym && source .venv/bin/activate

# Interact with your agent
python responses_api_agents/simple_agent/client.py
```

:::

:::{tab-item} 3. Collect Rollouts

**Terminal 2** (keep servers running in Terminal 1):

```bash
# Create a simple dataset with one query
echo '{"responses_create_params":{"input":[{"role":"developer","content":"You are a helpful assistant."},{"role":"user","content":"What is the weather in Seattle?"}]}}' > weather_query.jsonl

# Collect verified rollouts
ng_collect_rollouts \
    +agent_name=simple_weather_simple_agent \
    +input_jsonl_fpath=weather_query.jsonl \
    +output_jsonl_fpath=weather_rollouts.jsonl

# View the result
cat weather_rollouts.jsonl | python -m json.tool
```

This generates training data with verification scores!

:::

:::{tab-item} 4. Clean Up Servers

**Terminal 1** (or any terminal with venv activated):

```bash
# Stop all servers and clean up Ray processes
ray stop
```

You can also use `Ctrl+C` in Terminal 1 to stop the `ng_run` process, then run `ray stop` to clean up.

:::
::::

```{toctree}
:caption: Get Started
:hidden:
:maxdepth: 1

about/glossary.md
tutorials/01-concepts.md
get-started/setup-installation.md
about/concepts/task-verification.md
tutorials/offline-training-w-rollouts.md
about/concepts/rollout-collection-fundamentals.md
about/concepts/configuration-system.md
how-to-faq.md
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 1

tutorials/offline-training-w-rollouts
```


```{toctree}
:caption: Development
:hidden:


apidocs/index.rst
```
