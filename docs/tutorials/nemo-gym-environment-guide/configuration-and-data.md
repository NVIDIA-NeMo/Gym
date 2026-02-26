(building-environments-configuration)=

# Configuration and Training Data

Every environment needs a YAML configuration file and JSONL training data. This section covers both.

---

## Configuration: YAML Files

Every environment needs a configuration file that defines:
1. The Resource Server itself
2. The Agent that uses the Resource Server
3. Training/validation datasets

**File: `resources_servers/example_multi_step/configs/example_multi_step.yaml`**

```yaml
# simplified
# Define the Resource Server
example_multi_step_resources_server:
  resources_servers:
    example_multi_step:
      entrypoint: app.py                    # Entry point script
      domain: instruction_following         # Domain classification
      verified: false                       # Verification status
      description: Multi-step tool calling  # Human-readable description

# Define the Agent that uses this Resource Server
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_multi_step_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
        - name: train
          type: train
          jsonl_fpath: resources_servers/example_multi_step/data/train.jsonl
          gitlab_identifier:
            dataset_name: example_multi_step
            version: 0.0.1
            artifact_fpath: train.jsonl
          license: Apache 2.0
        - name: validation
          type: validation
          jsonl_fpath: resources_servers/example_multi_step/data/validation.jsonl
          gitlab_identifier:
            dataset_name: example_multi_step
            version: 0.0.1
            artifact_fpath: validation.jsonl
          license: Apache 2.0
        - name: example
          type: example
          jsonl_fpath: resources_servers/example_multi_step/data/example.jsonl
```

---

## Available Domains

From `nemo_gym/config_types.py`:

```python
# simplified
class Domain(str, Enum):
    MATH = "math"
    CODING = "coding"
    AGENT = "agent"
    KNOWLEDGE = "knowledge"
    INSTRUCTION_FOLLOWING = "instruction_following"
    LONG_CONTEXT = "long_context"
    SAFETY = "safety"
    GAMES = "games"
    TRANSLATION = "translation"
    E2E = "e2e"
    OTHER = "other"
```

---

## Server References

To link servers together in configuration:

```yaml
# simplified
# Reference a model server
model_server:
  type: responses_api_models
  name: policy_model

# Reference a resource server
resources_server:
  type: resources_servers
  name: my_resources_server

# Reference an agent server
agent_server:
  type: responses_api_agents
  name: my_agent
```

---

## Training Data Format: JSONL

Training data is stored in JSONL (JSON Lines) format where each line is a complete JSON object representing one training sample.

### Structure

```json
{
  "id": 0,
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are an extraction agent..."},
      {"role": "user", "content": "What factors contribute to high temperatures?"}
    ],
    "tools": [
      {
        "type": "function",
        "name": "get_synonym_value",
        "description": "Get the synonym value for a synonym.",
        "parameters": {
          "properties": {"synonym": {"type": "string"}},
          "type": "object",
          "required": ["synonym"],
          "additionalProperties": false
        },
        "strict": true
      },
      {
        "type": "function",
        "name": "extract_synonym_values",
        "description": "Extract the synonym values you retrieved.",
        "parameters": {
          "properties": {
            "synonym_values": {
              "items": {"type": "integer"},
              "type": "array"
            }
          },
          "type": "object",
          "required": ["synonym_values"],
          "additionalProperties": false
        },
        "strict": true
      }
    ],
    "parallel_tool_calls": false
  },
  "expected_synonyms": ["Blazing", "Warm"],
  "expected_synonym_values": [711, 407],
  "minefield_label": "Hot",
  "minefield_label_value": 299
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `id` | Unique sample identifier |
| `responses_create_params` | OpenAI Responses API-compatible input |
| `responses_create_params.input` | Conversation messages (system, user, assistant) |
| `responses_create_params.tools` | Available tools/functions for the agent |
| `expected_*` | Ground truth fields for reward computation in `verify()` |

### Tool Definition Format

Each tool follows the OpenAI function calling schema:

```json
{
  "type": "function",
  "name": "tool_name",
  "description": "What this tool does",
  "parameters": {
    "type": "object",
    "properties": {
      "param1": {"type": "string", "description": "..."},
      "param2": {"type": "integer", "description": "..."}
    },
    "required": ["param1"],
    "additionalProperties": false
  },
  "strict": true
}
```

---

> **Previous**: {ref}`Real-World Environment <building-environments-real-world>` | **Next**: {ref}`Infrastructure, Training, and Next Steps <building-environments-infrastructure>`
