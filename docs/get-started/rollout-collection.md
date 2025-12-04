(gs-collecting-rollouts)=

# Collecting Rollouts

In the previous tutorial, you set up NeMo Gym and ran your first agent interaction. But to train an agent with reinforcement learning, you need hundreds or thousands of these interactions—each one scored and saved. That's what rollout collection does.

:::{card}

**Goal**: Generate your first batch of rollouts and understand how they become training data.

^^^

**In this tutorial, you will**:

1. Run batch rollout collection
2. Examine results with the rollout viewer
3. Learn key parameters for scaling

:::

:::{button-ref} setup-installation
:color: secondary
:outline:
:ref-type: doc

← Previous: Setup and Installation
:::

---

## Before You Begin

Make sure you have:

- ✅ Completed [Setup and Installation](setup-installation.md)
- ✅ Servers still running (or ready to restart them)
- ✅ `env.yaml` configured with your OpenAI API key
- ✅ Virtual environment activated

**What's in a rollout?** A complete record of a task execution: the input, the model's reasoning and tool calls, the final output, and a verification score.

---

## 1. Inspect the Data

Look at the example dataset included with the Simple Weather resource server:

```bash
head -1 resources_servers/example_simple_weather/data/example.jsonl | python -m json.tool
```

Each line contains a `responses_create_params` object with:

- **input**: The conversation messages (user query)
- **tools**: Available tools the agent can use

## 2. Verify Servers Are Running

If you still have servers running from the [Setup and Installation](setup-installation.md) tutorial, proceed to the next step.

If not, start them again:

```bash
config_paths="resources_servers/example_simple_weather/configs/simple_weather.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**✅ Success Check**: You should see 3 servers running including the `simple_weather_simple_agent`.

## 3. Generate Rollouts

In a separate terminal, run:

```bash
ng_collect_rollouts +agent_name=simple_weather_simple_agent \
    +input_jsonl_fpath=resources_servers/example_simple_weather/data/example.jsonl \
    +output_jsonl_fpath=results/simple_weather_rollouts.jsonl \
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
Collecting rollouts: 100%|████████████████| 5/5 [00:08<00:00,  1.67s/it]
```

## 4. View Rollouts

Launch the rollout viewer:

```bash
ng_viewer +jsonl_fpath=results/simple_weather_rollouts.jsonl
```

Then visit <http://127.0.0.1:7860>

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

:::

::::

---

## Next Steps

You've completed the get-started tutorials. Your `simple_weather_rollouts.jsonl` file is training data ready for RL, SFT, or DPO pipelines.

From here, explore the [Tutorials](../tutorials/index.md) for advanced topics or [Concepts](../about/concepts/index.md) for deeper understanding.
