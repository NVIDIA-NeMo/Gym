(gs-collecting-rollouts)=

# Rollout Collection

In the previous tutorial, you set up NeMo Gym and ran your first agent interaction. But to train an agent with reinforcement learning, you need hundreds or thousands of these interactions—each one scored and saved. That's what rollout collection does.

:::{card}

**Goal**: Generate your first batch of rollouts and understand how they become training data.

^^^

**In this tutorial, you will**:

1. Run batch rollout collection
2. Examine results with the rollout viewer
3. Learn key parameters for scaling

:::

:::{button-ref} detailed-setup
:color: secondary
:outline:
:ref-type: doc

← Previous: Detailed Setup Guide
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed [Detailed Setup Guide](detailed-setup.md)
- ✅ Servers still running (or ready to restart them)
- ✅ `env.yaml` configured with your OpenAI API key
- ✅ Virtual environment activated

**What's in a rollout?** A complete record of a task execution: the input, the model's reasoning and tool calls, the final output, and a verification score.

---

## 1. Inspect the Data

Look at the example dataset included with the Simple Weather resource server:

```bash
head -1 resources_servers/example_single_tool_call/data/example.jsonl | python -m json.tool
```

Each line contains a `responses_create_params` object with:

- **input**: The conversation messages (user query)
- **tools**: Available tools the agent can use

## 2. Verify Servers Are Running

If you still have servers running from the [Detailed Setup Guide](detailed-setup.md) tutorial, proceed to the next step.

If not, start them again:

```bash
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**✅ Success Check**: You should see 3 Gym servers running including the `example_single_tool_call_simple_agent`, along with the head server.

## 3. Generate Rollouts

In a separate terminal, run:

```bash
ng_collect_rollouts +agent_name=example_single_tool_call_simple_agent \
    +input_jsonl_fpath=resources_servers/example_single_tool_call/data/example.jsonl \
    +output_jsonl_fpath=results/example_single_tool_call_rollouts.jsonl \
    +limit=5 \
    +num_repeats=2 \
    +num_samples_in_parallel=3
```

```{list-table} Parameters
:header-rows: 1
:widths: 35 15 50

* - Parameter
  - Type
  - Description
* - `+agent_name`
  - `str`
  - Which agent to use (required)
* - `+input_jsonl_fpath`
  - `str`
  - Path to input JSONL file (required)
* - `+output_jsonl_fpath`
  - `str`
  - Path to output JSONL file (required)
* - `+limit`
  - `int`
  - Max examples to process (default: `null` = all)
* - `+num_repeats`
  - `int`
  - Rollouts per example (default: `null` = 1)
* - `+num_samples_in_parallel`
  - `int`
  - Concurrent requests (default: `null` = unlimited)
```

**✅ Success Check**: You should see:

```text
Collecting rollouts: 100%|████████████████| 10/10 [00:08<00:00,  1.67s/it]
```

## 4. View Rollouts

Launch the rollout viewer:

```bash
ng_viewer +jsonl_fpath=results/example_single_tool_call_rollouts.jsonl
```

The viewer starts on port 7860 and accepts requests only from localhost by default. Visit <http://127.0.0.1:7860> in your browser.

:::{tip}
**Configuring Network Access**

By default, the viewer accepts requests only from localhost (`server_host=127.0.0.1`). To make it accessible from a different machine:

```bash
# Accept requests from anywhere (e.g., for remote access)
ng_viewer +jsonl_fpath=results/example_single_tool_call_rollouts.jsonl +server_host=0.0.0.0

# Use a custom port
ng_viewer +jsonl_fpath=results/example_single_tool_call_rollouts.jsonl +server_port=8080
```
:::

The viewer shows each rollout with:

- **Input**: The original query and tools
- **Response**: Tool calls and agent output
- **Reward**: Verification score (0.0–1.0)

:::{important}
**Where Do Reward Scores Come From?**

Scores come from the `verify()` function in your resource server. Each rollout is automatically sent to the `/verify` endpoint during collection. The default returns 1.0, but you can implement custom logic to score based on tool usage, response quality, or task completion.
:::

---

## Rollout Generation Parameters

::::{tab-set}

:::{tab-item} Essential

```bash
ng_collect_rollouts \
    +agent_name=your_agent_name \              # Which agent to use
    +input_jsonl_fpath=input/tasks.jsonl \     # Input dataset
    +output_jsonl_fpath=output/rollouts.jsonl  # Where to save results
```

:::

:::{tab-item} Data Control

```bash
    +limit=100 \                    # Limit examples processed (null = all)
    +num_repeats=3 \                # Rollouts per example (null = 1)
    +num_samples_in_parallel=5      # Concurrent requests (null = default)
```

:::

:::{tab-item} Model Behavior

```bash
    +responses_create_params.max_output_tokens=4096 \     # Response length limit
    +responses_create_params.temperature=0.7 \            # Randomness (0-1)
    +responses_create_params.top_p=0.9                    # Nucleus sampling
```
::::

---

## What's Next?

Congratulations! You now have a working NeMo Gym installation and understand how to generate rollouts. Choose your path based on your goals:

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Use an Existing Training Environment
:link: https://github.com/NVIDIA-NeMo/Gym#-available-resource-servers

Browse the available resource servers on GitHub to find a training-ready environment that matches your goals.
+++
{bdg-secondary}`github` {bdg-secondary}`resource-servers`
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Build a Custom Training Environment
:link: ../tutorials/creating-resource-server
:link-type: doc

Implement or integrate existing tools and define task verification logic.
+++
{bdg-secondary}`tutorial` {bdg-secondary}`custom-tools`
:::

::::
