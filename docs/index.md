(gym-home)=

# NeMo Gym Documentation

NeMo Gym is a framework for building reinforcement learning training environments. It provides the infrastructure for creating high-quality training data through agentic model interactions. Use NeMo Gym to generate rollouts for RL training, collect trajectories for supervised fine-tuning, or create preference pairs for alignment.

At the core of NeMo Gym are three server concepts: **Model** servers provide LLM inference capabilities, **Resources** servers expose tools and environments that agents interact with, and **Agent** servers orchestrate the interaction between models and resources to generate verified training data.

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

:::{tab-item} 2. Run Agent

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

````{div} sd-d-flex-row

```{button-ref} tutorials/index
:ref-type: doc
:color: secondary
:class: sd-rounded-pill sd-mr-3

Browse Tutorials
```

```{button-link} https://github.com/NVIDIA-NeMo/Gym
:color: primary
:class: sd-rounded-pill

View on GitHub
```
````

## Introduction to Gym

Learn more about Gym, how it works at a high-level, and the key features.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` About NeMo Gym
:link: about-overview
:link-type: ref
Overview of NeMo Gym and its approach to scalable rollout collection.
+++
{bdg-secondary}`target-users` {bdg-secondary}`core-components`
:::

:::{grid-item-card} {octicon}`light-bulb;1.5em;sd-mr-1` Concepts
:link: about-concepts
:link-type: ref
Core concepts behind models, resources, agents, and verification.
+++
{bdg-secondary}`mental-models` {bdg-secondary}`abstractions`
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Architecture
:link: about-architecture
:link-type: ref
How NeMo Gym components work together and interact.
+++
{bdg-secondary}`system-design` {bdg-secondary}`deployment`
:::

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` Ecosystem
:link: about-ecosystem
:link-type: ref
NeMo Gym's place in the NVIDIA NeMo Framework and ecosystem.
+++
{bdg-secondary}`nemo-framework` {bdg-secondary}`positioning`
:::

::::

## Get Started

New to NeMo Gym? Follow our guided tutorial path to build your first agent.

::::{grid} 1 1 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` 1. Setup and Installation
:link: get-started/setup-installation
:link-type: doc
Get NeMo Gym installed and servers running with your first successful agent interaction.
+++
{bdg-secondary}`environment` {bdg-secondary}`first-run`
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` 2. Your First Agent
:link: get-started/first-agent
:link-type: doc
Understand how your weather agent works and learn to interact with it.
+++
{bdg-secondary}`workflow` {bdg-secondary}`tools`
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` 3. Verifying Agent Results
:link: get-started/verifying-agent-results
:link-type: doc
Understand how NeMo Gym evaluates agent performance and what verification means for training.
+++
{bdg-secondary}`rewards` {bdg-secondary}`scoring`
:::

:::{grid-item-card} {octicon}`iterations;1.5em;sd-mr-1` 4. Collecting Rollouts
:link: get-started/collecting-rollouts
:link-type: doc
Generate your first batch of rollouts and understand how they become training data.
+++
{bdg-secondary}`training-data` {bdg-secondary}`scale`
:::

::::

---

:::{toctree}
:hidden:
Home <self>
:::

```{toctree}
:caption: About
:hidden:
:maxdepth: 2

Overview <about/index>
Concepts <about/concepts/index>
Architecture <about/architecture>
Ecosystem <about/ecosystem>
<!-- Key Features <about/features> -->
Release Notes <about/release-notes/index>
```

```{toctree}
:caption: Get Started
:hidden:
:maxdepth: 1

get-started/index
get-started/setup-installation
get-started/first-agent
get-started/verifying-agent-results
get-started/collecting-rollouts
```

```{toctree}
:caption: Tutorials
:hidden:
:maxdepth: 1

Overview <tutorials/index>
tutorials/offline-training-w-rollouts
tutorials/separate-policy-and-judge-models
```


```{toctree}
:caption: Development
:hidden:

apidocs/index.rst
```
