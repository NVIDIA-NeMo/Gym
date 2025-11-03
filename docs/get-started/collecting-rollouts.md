(gs-collecting-rollouts)=

# Collecting Rollouts

In the previous tutorials, you ran individual agent interactions and learned how verification scores each response. But to train an agent with reinforcement learning, you need hundreds or thousands of these interactions—each one scored and saved. That's what rollout collection does.

:::{card}

**Goal**: Generate your first batch of rollouts and understand how they become training data.

^^^

**In this tutorial, you will**:

1. Create a simple input dataset with weather queries
2. Run batch rollout collection
3. Examine the complete interaction records
4. Understand how rollouts connect to training

:::

:::{button-ref} verifying-agent-results
:color: secondary
:outline:
:ref-type: doc

← Previous: Verifying Agent Results
:::

:::{tip}
**Going deeper**: For comprehensive coverage of **rollout collection strategies**, **dataset management**, and **production workflows**, refer to [Rollout Collection Fundamentals](concepts-rc-fundamentals).
:::

---

## What Are Rollouts?

**Rollouts are complete records of agent interactions**—from the user's question through the agent's reasoning, tool calls, and final response, all the way to the verification score.

Think of them as "interaction transcripts with grades":

- **Input**: User question or task
- **Process**: Agent reasoning and tool usage
- **Output**: Final response to user
- **Score**: Verification reward (0.0–1.0)

When you ran a single interaction in tutorial 1, that was **one rollout**. Now you'll generate **many rollouts** systematically—this is the foundation of RL training data.

---

## Why Generate Rollouts?

In the verification tutorial, you scored individual responses. But reinforcement learning needs:

- **Volume**: Hundreds or thousands of examples
- **Diversity**: Different inputs and agent behaviors
- **Structured data**: Consistent format for training pipelines
- **Quality signals**: Verification scores for every interaction

NeMo Gym's rollout collection automates this process—you provide the inputs, and it handles running the agent, capturing everything, and scoring each interaction.

---

## Create Your Input Dataset

Let's create a simple dataset of weather-related queries. Create a file called `weather_queries.jsonl` in your Gym directory:

```bash
cd /path/to/Gym

cat > weather_queries.jsonl << 'EOF'
{"responses_create_params": {"input": [{"role": "developer", "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has."}, {"role": "user", "content": "What's the weather like in Seattle?"}], "tools": [{"type": "function", "name": "get_weather", "description": "Get weather information for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"], "additionalProperties": false}, "strict": true}]}}
{"responses_create_params": {"input": [{"role": "developer", "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has."}, {"role": "user", "content": "I'm going out in Boston tonight"}], "tools": [{"type": "function", "name": "get_weather", "description": "Get weather information for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"], "additionalProperties": false}, "strict": true}]}}
{"responses_create_params": {"input": [{"role": "developer", "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has."}, {"role": "user", "content": "Should I bring an umbrella in Chicago?"}], "tools": [{"type": "function", "name": "get_weather", "description": "Get weather information for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"], "additionalProperties": false}, "strict": true}]}}
{"responses_create_params": {"input": [{"role": "developer", "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has."}, {"role": "user", "content": "How's the weather in New York?"}], "tools": [{"type": "function", "name": "get_weather", "description": "Get weather information for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"], "additionalProperties": false}, "strict": true}]}}
{"responses_create_params": {"input": [{"role": "developer", "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has."}, {"role": "user", "content": "What should I wear in San Francisco today?"}], "tools": [{"type": "function", "name": "get_weather", "description": "Get weather information for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}}, "required": ["city"], "additionalProperties": false}, "strict": true}]}}
EOF
```

**What this dataset contains**:
- **5 weather-related queries** (Seattle, Boston, Chicago, New York, San Francisco)
- **Same system message** you've been using throughout the tutorials
- **Same tool definition** (get_weather function)
- **JSONL format**: One JSON object per line

Each line has a `responses_create_params` field that matches what you sent to the agent in tutorial 2.

---

## Start Your Servers

Make sure your NeMo Gym servers are running (Same as [Tutorial 1: Setup and Installation](setup-installation.md#step-3-start-the-servers)):

```bash
# If servers aren't already running
config_paths="resources_servers/simple_weather/configs/simple_weather.yaml,responses_api_models/openai_model/configs/openai_model.yaml"

ng_run "+config_paths=[${config_paths}]"
```

Keep this terminal open with the servers running.

---

## Run Rollout Collection

Open a **new terminal** and run the rollout collection command:

```bash
# Navigate to project directory
cd /path/to/Gym

# Activate virtual environment
source .venv/bin/activate

# Run rollout collection
ng_collect_rollouts \
    +agent_name=simple_weather_simple_agent \
    +input_jsonl_fpath=weather_queries.jsonl \
    +output_jsonl_fpath=weather_rollouts.jsonl \
    +limit=null
```

**What these parameters mean**:

- `+agent_name=simple_weather_simple_agent`: Which agent to use (your weather agent)
- `+input_jsonl_fpath=weather_queries.jsonl`: Input dataset you just created
- `+output_jsonl_fpath=weather_rollouts.jsonl`: Where to save results
- `+limit=null`: Process all examples (in this case, 5)

**✅ Success Check**: You should see output like:

```
Collecting rollouts: 100%|████████████████| 5/5 [00:08<00:00,  1.67s/it]
```

This means all 5 queries were processed successfully!

---

## Understand the Output

Let's examine what was captured. Open `weather_rollouts.jsonl` or view it in the terminal:

```bash
# View one complete rollout (the first line)
head -n 1 weather_rollouts.jsonl | python -m json.tool
```

You'll see a JSON object with three main sections:

### 1. Input (`responses_create_params`)

```json
{
  "responses_create_params": {
    "input": [
      {
        "role": "developer",
        "content": "You are a helpful personal assistant..."
      },
      {
        "role": "user",
        "content": "What's the weather like in Seattle?"
      }
    ],
    "tools": [...]
  }
}
```

This is the exact input you provided—the query and available tools.

### 2. Complete Response (`response`)

```json
{
  "response": {
    "output": [
      {
        "type": "function_call",
        "name": "get_weather",
        "arguments": "{\"city\":\"Seattle\"}"
      },
      {
        "type": "function_call_output",
        "output": "{\"city\": \"Seattle\", \"weather_description\": \"The weather in Seattle is cold.\"}"
      },
      {
        "type": "message",
        "content": [{
          "text": "The weather in Seattle is cold. You might want to bring a jacket!"
        }]
      }
    ]
  }
}
```

This captures **everything the agent did**:
- What tool it called (`get_weather` with `"Seattle"`)
- What the tool returned (weather data)
- The final response to the user

### 3. Verification Score (`reward`)

```json
{
  "reward": 1.0
}
```

This is the score from your verification function—the same one you built in tutorial 3. The agent got a perfect score (1.0) because it called the weather tool and mentioned the weather in its response.

**This complete record is a rollout**: input + agent behavior + verification score.

---

## What Makes This Training Data?

Look at all 5 rollouts together:

```bash
# Count total rollouts
wc -l weather_rollouts.jsonl
# Output: 5

# View all verification scores
jq '.reward' weather_rollouts.jsonl
```

Each rollout has:
- ✅ What the agent was asked to do
- ✅ How the agent responded
- ✅ Whether the response was good (reward score)

This is exactly what RL algorithms need:
- **States**: User inputs
- **Actions**: Agent responses (tool calls + messages)
- **Rewards**: Verification scores

Your `weather_rollouts.jsonl` file is now **training data** that could be fed into an RL algorithm to improve the agent's performance.

---

## Comparing Individual vs. Batch Processing

```{list-table} Individual vs. Batch Processing
:header-rows: 1
:widths: 50 50

* - **Tutorial 1-3** (Individual)
  - **Now** (Batch)
* - Ran one interaction manually
  - Ran 5 interactions automatically
* - Saw one response
  - Generated 5 complete rollouts
* - Scored one at a time
  - All scored and saved
* - Useful for testing and understanding
  - Ready for training at scale
```

The same pattern works for 50, 500, or 5,000 examples—that's the power of rollout collection.

---

## What You've Learned

You now have hands-on experience with:

- ✓ Creating input datasets in JSONL format
- ✓ Running batch rollout collection
- ✓ Understanding rollout structure (input + behavior + score)
- ✓ Generating structured training data

**Key insight**: Rollout collection bridges the gap between individual agent interactions (which you understand) and training data (which RL needs). It's the systematic way to capture agent behavior at scale.

---

## Next Steps

You've completed the get-started tutorial series! You can now:

1. **Scale up**: Use larger datasets (hundreds or thousands of examples)
2. **Integrate with training**: Feed rollouts into RL, SFT, or DPO pipelines
3. **Build custom agents**: Apply these patterns to your own domains
4. **Design verification**: Create sophisticated scoring for complex tasks

---

:::{admonition} Congratulations!
:class: tip

You've built a solid foundation in NeMo Gym. From here, explore the [Tutorials](../tutorials/index.md) section for advanced topics, or dive into [Concepts](../about/concepts/index.md) for deeper understanding of the framework.
:::