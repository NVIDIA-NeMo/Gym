# Rollout Collection

A {term}`rollout <Rollout / Trajectory>` is complete record of a task instance execution that captures:
- What the model was asked to do (input)
- How the model reasoned (internal processing)  
- What tools were used (tool calls and tool responses)
- How well the task was achieved (verification scores)
- The final response (output to user)


## Generating Your First Rollouts

Let's generate rollouts using the **Example Multi Step** resource server, which tests reading comprehension across long documents.

::::{tab-set}

:::{tab-item} 1. Inspect data
```bash
head -1 resources_servers/example_multi_step/data/example.jsonl | python -m json.tool
```

**What this dataset contains**: Complex reading comprehension tasks where agents must find specific information ("needles") within long documents ("haystacks").

Each line in the input JSONL file follows the schema below.

**Key components**:
- **responses_create_params**: Original task and available tools. Required
- **metadata** (e.g. `expected_synonyms`, `minefield_label`, etc): Additional metadata used by the resources server to either setup or perform verification

```json
{
    "responses_create_params": {
        "input": [
            {
                "role": "user",
                "content": "What factors contribute to a region experiencing extremely high temperatures, and how do these factors interact?"
            }
        ]
    },
    "expected_synonyms": [
        "Blazing",
        "Warm"
    ],
    "minefield_label": "Hot"
}
```

:::
:::{tab-item} 2. Start servers
Start the example_multi_step agent server
```bash
config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"
ng_run "+config_paths=[${config_paths}]"
```

**✅ Success Check**: You should see 3 servers running including the `example_multi_step_simple_agent`.

:::

:::{tab-item} 3. Generate Rollouts

In a separate terminal, run:
```bash
ng_collect_rollouts +agent_name=example_multi_step_simple_agent \
    +input_jsonl_fpath=resources_servers/example_multi_step/data/example.jsonl \
    +output_jsonl_fpath=results/example_multi_step_rollouts.jsonl \
    +limit=5 \
    +num_repeats=2 \
    +num_samples_in_parallel=3 \
    +responses_create_params.max_output_tokens=8192
```

**What's happening**:
- `limit=5`: Process only the first 5 examples (for quick testing)
- `num_repeats=2`: Generate 2 rollouts per example (10 total rollouts)
- `num_samples_in_parallel=3`: Process 3 requests simultaneously
- `max_output_tokens=8192`: Allow longer responses for complex reasoning

:::

:::{tab-item} 4. View rollouts

Launch the rollout viewer
```bash
ng_viewer +jsonl_fpath=results/example_multi_step_rollouts.jsonl
```

Then visit http://127.0.0.1:7860

**What you'll see**: An interactive viewer showing reasoning, tool calls, and verification scores for each rollout.

**Key components**:
- **{term}`reward <Reward / Reward Signal>`**: Verification score from the resource server. Required on output
- **response**: Complete output conversation including tool calls and responses
- **metadata** (`parsed_synonym_values`, `set_overlap`, etc): Additional metrics for analysis

```json
{
    "responses_create_params": {
        "input": [
            {
                "content": "What factors contribute to a region experiencing extremely high temperatures, and how do these factors interact?",
                "role": "user",
                "type": "message"
            }
        ]
    },
    "response": {
        "output": [
            {
                "arguments": "{\"synonym\":\"Blazing\"}",
                "name": "get_synonym_value",
                "type": "function_call",
            },
            "..."
        ]
    },
    "reward": 1.0,
    "parsed_synonym_values": [
        711,
        407
    ],
    "accuracy": true,
    "set_overlap": 1.0,
    "original_term_minefield_hit": false,
    "order_instruction_following_failure": false
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
