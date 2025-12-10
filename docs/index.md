(gym-home)=

# NeMo Gym Documentation

[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) is a library for building reinforcement learning (RL) training environments for large language models (LLMs). It provides infrastructure to develop environments, scale rollout collection, and integrate seamlessly with your preferred training framework.

A training environment consists of three server components: **Agents** orchestrate the rollout lifecycle—calling models, executing tool calls via resources, and coordinating verification. **Models** provide stateless text generation using LLM inference endpoints. **Resources** define tasks, tool implementations, and verification logic.

## Quickstart

Run a training environment and start collecting rollouts for training in under 5 minutes.

::::{tab-set}

:::{tab-item} 1. Set Up

**Install NeMo Gym**

Get NeMo Gym installed and ready to use:

```bash
# Clone the repository
git clone git@github.com:NVIDIA-NeMo/Gym.git
cd Gym

# Install UV (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create virtual environment
uv venv --python 3.12
source .venv/bin/activate

# Install NeMo Gym
uv sync --extra dev --group docs
```

**Configure Your API Key**

Create an `env.yaml` file that contains your OpenAI API key and the {term}`Policy Model` you want to use. Replace `your-openai-api-key` with your actual key. This file helps keep your secrets out of version control while still making them available to NeMo Gym.

```bash
echo "policy_base_url: https://api.openai.com/v1
policy_api_key: your-openai-api-key
policy_model_name: gpt-4.1-2025-04-14" > env.yaml
```

> **Note:** We use GPT-4.1 in this quickstart because it provides low latency (no reasoning step) and works reliably out-of-the-box. NeMo Gym is **not limited to OpenAI models**—you can use self-hosted models via vLLM or any OpenAI-compatible inference server that supports function calling. See the setup guide for details.

:::

:::{tab-item} 2. Start Servers

**Terminal 1** (start servers):

```bash
# Start servers (this will keep running)
config_paths="resources_servers/example_simple_weather/configs/simple_weather.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**Terminal 2** (interact with agent):

```bash
# In a NEW terminal, activate environment
source .venv/bin/activate

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

**Terminal 1** with the running servers: Ctrl+C to stop the `ng_run` process.

:::
::::

---

## What's Next?

Now that you can generate rollouts, choose your path:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Use an Existing Training Environment
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse the available resource servers to find a training-ready environment that matches your goals.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Custom Training Environment
:link: tutorials/creating-resource-server
:link-type: doc

Implement or integrate existing tools and define task verification logic.
:::

::::

```{toctree}
:hidden:
Home <self>
```

```{toctree}
:caption: About
:hidden:
:maxdepth: 2

about/index.md
Concepts <about/concepts/index>
Ecosystem <about/ecosystem>
```

```{toctree}
:caption: Get Started
:hidden:
:maxdepth: 1

Overview <get-started/index>
get-started/setup-installation.md
get-started/rollout-collection.md
```


```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 1

tutorials/index.md
tutorials/creating-resource-server
tutorials/offline-training-w-rollouts
tutorials/rl-training-with-nemo-rl
how-to-faq.md
```

```{toctree}
:caption: Reference
:hidden:
:maxdepth: 1

reference/cli-commands.md
```

```{toctree}
:caption: Development
:hidden:

apidocs/index.rst
```
