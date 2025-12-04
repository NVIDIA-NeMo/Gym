(gs-collecting-rollouts)=

# Collecting Rollouts

In the previous tutorial, you set up NeMo Gym and ran your first agent interaction. But to train an agent with reinforcement learning, you need hundreds or thousands of these interactions—each one scored and saved. That's what rollout collection does.

:::{card}

**Goal**: Generate your first batch of rollouts and understand how they become training data.

^^^

**In this tutorial, you will**:

1. Understand what rollouts capture
2. Run batch rollout collection
3. Examine complete interaction records with the rollout viewer
4. Learn key parameters for scaling rollout generation

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

---

## What Are Rollouts?

A rollout is a complete record of a task instance execution that captures:

- **Input**: What the model was asked to do
- **Reasoning**: How the model processed the request internally
- **Tool usage**: What tools were called and their responses
- **Verification**: How well the task was achieved (reward scores)
- **Output**: The final response to the user

When you ran a single interaction in [Setup and Installation](setup-installation.md), that was **one rollout**. Now you will generate **many rollouts** systematically—this is the foundation of RL training data.

---

## Why Generate Rollouts?

Reinforcement learning needs:

- **Volume**: Hundreds or thousands of examples
- **Diversity**: Different inputs and agent behaviors
- **Structured data**: Consistent format for training pipelines
- **Quality signals**: Verification scores for every interaction

NeMo Gym's rollout collection automates this process—you provide the inputs, and it handles running the agent, capturing everything, and scoring each interaction.

---

## Generate Your First Rollouts

Now that you have servers running from the previous tutorial, let's generate rollouts using the **Simple Weather** resource server you already set up.

::::{tab-set}

:::{tab-item} 1. Inspect Data

First, look at the example dataset included with the Simple Weather resource server:

```bash
head -1 resources_servers/example_simple_weather/data/example.jsonl | python -m json.tool
```

**What this dataset contains**: Simple weather queries where agents must use the `get_weather` tool to provide weather information.

Each line in the input JSONL file follows this schema:

```json
{
    "responses_create_params": {
        "input": [
            {
                "content": "what's it like in sf?",
                "role": "user"
            }
        ],
        "tools": [
            {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": ""
                        }
                    },
                    "required": [
                        "city"
                    ],
                    "additionalProperties": false
                },
                "strict": true,
                "type": "function",
                "description": ""
            }
        ]
    }
}
```

**Key components**

- **responses_create_params**: Original task and available tools (required)
- **input**: The conversation messages including system prompt and user query
- **tools**: Available tools the agent can use (in this case, `get_weather`)

:::

:::{tab-item} 2. Verify Servers

If you still have servers running from the [Setup and Installation](setup-installation.md) tutorial, you're ready to proceed to the next step.

If not, start them again:

```bash
config_paths="resources_servers/example_simple_weather/configs/simple_weather.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**✅ Success Check**: You should see 3 servers running including the `simple_weather_simple_agent`.

:::

:::{tab-item} 3. Generate Rollouts

In a separate terminal, run:

```bash
ng_collect_rollouts +agent_name=simple_weather_simple_agent \
    +input_jsonl_fpath=resources_servers/example_simple_weather/data/example.jsonl \
    +output_jsonl_fpath=results/simple_weather_rollouts.jsonl \
    +limit=5 \
    +num_repeats=2 \
    +num_samples_in_parallel=3
```

**What these parameters mean**

- `+agent_name`: Which agent to use (your weather agent)
- `+input_jsonl_fpath`: Input dataset with weather queries
- `+output_jsonl_fpath`: Where to save results
- `+limit=5`: Process only the first 5 examples (for quick testing)
- `+num_repeats=2`: Generate 2 rollouts per example (10 total rollouts)
- `+num_samples_in_parallel=3`: Process 3 requests simultaneously

**✅ Success Check**: You should see output like:

```text
Collecting rollouts: 100%|████████████████| 5/5 [00:08<00:00,  1.67s/it]
```

:::

:::{tab-item} 4. View Rollouts

Launch the rollout viewer:

```bash
ng_viewer +jsonl_fpath=results/simple_weather_rollouts.jsonl
```

Then visit <http://127.0.0.1:7860>

**What you'll see**: An interactive viewer showing tool calls and verification scores for each rollout.

:::

::::

:::{important}
**Where Do Reward Scores Come From?**

The rollout viewer shows a reward score for each interaction. These scores come from the `verify()` function in your resource server.

Each rollout is automatically sent to your weather resource server's `/verify` endpoint, which scores the interaction. The default verification returns 1.0 for all responses, but you can implement custom logic to score based on:

- Tool usage (did the agent call the right tools?)
- Response quality (did the response mention relevant information?)
- Task completion (did the agent accomplish the goal?)

This is the power of NeMo Gym: verification happens automatically during rollout collection!
:::

---

## Understanding Rollout Structure

Each rollout in the output file contains three main sections:

### 1. Input (`responses_create_params`)

The exact input provided—the query and available tools:

```json
{
    "responses_create_params": {
        "input": [
            {
                "content": "You are a helpful personal assistant...",
                "role": "developer",
                "type": "message"
            },
            {
                "content": "what's it like in sf?",
                "role": "user",
                "type": "message"
            }
        ]
    }
}
```

### 2. Complete Response (`response`)

Everything the agent did—tool calls, tool outputs, and final message:

```json
{
    "response": {
        "output": [
            {
                "arguments": "{\"city\":\"San Francisco\"}",
                "call_id": "call_zuJigUcshS8H02NTWrsI4fcH",
                "name": "get_weather",
                "type": "function_call",
                "status": "completed"
            },
            {
                "call_id": "call_zuJigUcshS8H02NTWrsI4fcH",
                "output": "{\"city\":\"San Francisco\",\"weather_description\":\"The weather in San Francisco is cold.\"}",
                "type": "function_call_output"
            },
            {
                "content": [
                    {
                        "text": "The weather in San Francisco is currently cold. If you need more specific details or a forecast, just let me know!",
                        "type": "output_text"
                    }
                ],
                "role": "assistant",
                "type": "message"
            }
        ]
    }
}
```

### 3. Verification Score (`reward`)

The score from the resource server's verification function:

```json
{
    "reward": 1.0
}
```

**This complete record is a rollout**: input + agent behavior + verification score.

---

## What Makes This Training Data?

Each rollout has:

- ✅ What the agent was asked to do
- ✅ How the agent responded
- ✅ Whether the response was good (reward score)

This is exactly what RL algorithms need:

- **States**: User inputs
- **Actions**: Agent responses (tool calls + messages)
- **Rewards**: Verification scores

Your `simple_weather_rollouts.jsonl` file is now **training data** that could be fed into an RL algorithm to improve the agent's performance.

---

## Comparing Individual vs. Batch Processing

```{list-table} Individual vs. Batch Processing
:header-rows: 1
:widths: 50 50

* - **Setup Tutorial** (Individual)
  - **Rollout Collection** (Batch)
* - Ran one interaction manually
  - Ran five interactions automatically
* - Saw one response
  - Generated ten complete rollouts
* - Scored one at a time
  - All scored and saved
* - Useful for testing and understanding
  - Ready for training at scale
```

The same pattern works for 50, 500, or 5,000 examples—that is the power of rollout collection.

---

## Rollout Generation Parameters

Here's a quick reference for common parameters:

### Essential

```bash
ng_collect_rollouts \
    +agent_name=your_agent_name \              # Which agent to use
    +input_jsonl_fpath=input/tasks.jsonl \     # Input dataset
    +output_jsonl_fpath=output/rollouts.jsonl  # Where to save results
```

### Data Control

```bash
    +limit=100 \                    # Limit examples processed (null = all)
    +num_repeats=3 \                # Rollouts per example (null = 1)
    +num_samples_in_parallel=5      # Concurrent requests (null = default)
```

### Model Behavior

```bash
    +responses_create_params.max_output_tokens=4096 \     # Response length limit
    +responses_create_params.temperature=0.7 \            # Randomness (0-1)
    +responses_create_params.top_p=0.9                    # Nucleus sampling
```

---

## What You've Learned

You now have hands-on experience with:

- Understanding rollout structure (input + behavior + score)
- Running batch rollout collection with `ng_collect_rollouts`
- Using the rollout viewer (`ng_viewer`) to inspect results
- Key parameters for scaling data generation

**Key insight**: Rollout collection bridges the gap between individual agent interactions (which you understand) and training data (which RL needs). It's the systematic way to capture agent behavior at scale.

---

## Next Steps

You've completed the get-started tutorial series! You can now:

1. **Scale up**: Use larger datasets (hundreds or thousands of examples)
2. **Integrate with training**: Feed rollouts into RL, SFT, or DPO pipelines
3. **Build custom agents**: Apply these patterns to your own domains
4. **Design verification**: Create sophisticated scoring for complex tasks

:::{admonition} Congratulations!
:class: tip

You've built a solid foundation in NeMo Gym. From here, explore the [Tutorials](../tutorials/index.md) section for advanced topics, or dive into [Concepts](../about/concepts/index.md) for deeper understanding of the framework.
:::
