# Rollout Collection

A rollout is complete record of a task instance execution that captures:
- What the model was asked to do (input)
- How the model reasoned (internal processing)
- What tools were used (tool calls and tool responses)
- How well the task was achieved (verification scores)
- The final response (output to user)


## Generating Your First Rollouts

Now that you have servers running from the previous tutorial, let's generate rollouts using the **Simple Weather** resource server you already set up.

::::{tab-set}

:::{tab-item} 1. Inspect data
```bash
head -1 resources_servers/example_simple_weather/data/example.jsonl | python -m json.tool
```

**What this dataset contains**: Simple weather queries where agents must use the `get_weather` tool to provide weather information.

Each line in the input JSONL file follows the schema below.

**Key components**:
- **responses_create_params**: Original task and available tools. Required
- **input**: The conversation messages including system prompt and user query
- **tools**: Available tools the agent can use (in this case, `get_weather`)

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

:::
:::{tab-item} 2. Verify servers are running

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

**What's happening**:
- `limit=5`: Process only the first 5 examples (for quick testing)
- `num_repeats=2`: Generate 2 rollouts per example (10 total rollouts)
- `num_samples_in_parallel=3`: Process 3 requests simultaneously

:::

:::{tab-item} 4. View rollouts

Launch the rollout viewer
```bash
ng_viewer +jsonl_fpath=results/simple_weather_rollouts.jsonl
```

Then visit http://127.0.0.1:7860

**What you'll see**: An interactive viewer showing tool calls and verification scores for each rollout.

**Key components**:
- **reward**: Verification score from the resource server. Required on output
- **response**: Complete output conversation including tool calls and responses

```json
{
    "responses_create_params": {
        "input": [
            {
                "content": "You are a helpful personal assistant that aims to be helpful and reduce any pain points the user has.",
                "role": "developer",
                "type": "message"
            },
            {
                "content": "what's it like in sf?",
                "role": "user",
                "type": "message"
            }
        ],
    },
    "response": {
        "output": [
            {
                "arguments": "{\"city\":\"San Francisco\"}",
                "call_id": "call_zuJigUcshS8H02NTWrsI4fcH",
                "name": "get_weather",
                "type": "function_call",
                "id": "fc_026df8ad0671316700692f58eb22cc8193bdd92b0524f0c66c",
                "status": "completed"
            },
            {
                "call_id": "call_zuJigUcshS8H02NTWrsI4fcH",
                "output": "{\"city\":\"San Francisco\",\"weather_description\":\"The weather in San Francisco is cold.\"}",
                "type": "function_call_output",
                "id": null,
                "status": null
            },
            {
                "id": "msg_026df8ad0671316700692f58edf44881938cb000ea88577cae",
                "content": [
                    {
                        "annotations": [],
                        "text": "The weather in San Francisco is currently cold. If you need more specific details or a forecast, just let me know!",
                        "type": "output_text",
                        "logprobs": []
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message"
            }
        ],
    },
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
            "description": null
        }
    ],
    "reward": 1.0
}
```

:::
::::


## Rollout Generation Parameters

Essential
```bash
ng_collect_rollouts \
    +agent_name=your_agent_name \              # Which agent to use
    +input_jsonl_fpath=input/tasks.jsonl \     # Input dataset
    +output_jsonl_fpath=output/rollouts.jsonl  # Where to save results
```

Data Control
```bash
    +limit=100 \                    # Limit examples processed (null = all)
    +num_repeats=3 \                # Rollouts per example (null = 1)
    +num_samples_in_parallel=5      # Concurrent requests (null = default)
```

Model Behavior
```bash
    +responses_create_params.max_output_tokens=4096 \     # Response length limit
    +responses_create_params.temperature=0.7 \            # Randomness (0-1)
    +responses_create_params.top_p=0.9                    # Nucleus sampling
```
