# NeMo Gym Rollout Collection Guide

## Quick Start

### 1. Activate the uv environment

```bash
source /home/shashankv/bignlp/multistep-tool-rl-gym/RL/3rdparty/Gym-workspace/Gym/.venv/bin/activate
```

### 2. Terminal 1 - Start NeMo Gym servers

```bash
cd /home/shashankv/bignlp/multistep-tool-rl-gym/RL/3rdparty/Gym-workspace/Gym

ng_run \
    "+config_paths=[responses_api_models/openai_model/configs/openai_model.yaml,resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +policy_base_url=https://integrate.api.nvidia.com/v1 \
    +policy_api_key=$NVIDIA_API_KEY \
    +policy_model_name=openai/gpt-oss-120b
```

### 3. Terminal 2 - Collect rollouts

```bash
cd /home/shashankv/bignlp/multistep-tool-rl-gym/RL/3rdparty/Gym-workspace/Gym
source .venv/bin/activate

ng_collect_rollouts \
    +agent_name=workplace_assistant_simple_agent \
    +input_jsonl_fpath=/home/shashankv/bignlp/multistep-tool-rl-gym/DataDesigner/docs/colab_notebooks/workplace_assistant/workplace_assistant_train-gpt-oss-fixed.jsonl \
    +output_jsonl_fpath=rollouts_output.jsonl \
    +limit=5
```

Remove `+limit=5` to process all records.

### 4. View results

```bash
ng_viewer rollouts_output.jsonl
```

---

## LLM Endpoint Configuration

The system uses three configurable parameters:

| Parameter | Description |
|-----------|-------------|
| `policy_base_url` | `https://integrate.api.nvidia.com/v1` |
| `policy_api_key` | Your `$NVIDIA_API_KEY` |
| `policy_model_name` | `openai/gpt-oss-120b` |

### Available Models

- `openai/gpt-oss-120b` - Large model for high-quality generation
- `openai/gpt-oss-20b` - Smaller, faster model

---

## Rollout Collection Parameters

```bash
ng_collect_rollouts \
    +agent_name=workplace_assistant_simple_agent \
    +input_jsonl_fpath=input.jsonl \
    +output_jsonl_fpath=output.jsonl \
    +limit=100 \                          # Max examples (optional)
    +num_repeats=4 \                      # Repeat each example N times (optional)
    +num_samples_in_parallel=10 \         # Concurrent requests (optional)
    '+responses_create_params={temperature: 0.7, max_tokens: 2048}'  # Overrides (optional)
```

---

## Input Data Format (JSONL)

Each line must be a JSON object with this structure:

```json
{
  "id": 0,
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "Today's date is Thursday, 2026-01-29..."},
      {"role": "user", "content": "Send an email to John about the meeting"}
    ],
    "tools": [
      {
        "type": "function",
        "name": "email_send_email",
        "description": "Sends an email to the specified recipient.",
        "parameters": {
          "type": "object",
          "properties": {
            "recipient": {"type": "string", "description": "Email address"},
            "subject": {"type": "string", "description": "Subject line"},
            "body": {"type": "string", "description": "Body content"}
          },
          "required": ["recipient", "subject", "body"],
          "additionalProperties": false
        },
        "strict": false
      }
    ],
    "parallel_tool_calls": false,
    "temperature": 1.0
  },
  "ground_truth": [
    {"name": "company_directory_find_email_address", "arguments": "{\"name\": \"John\"}"},
    {"name": "email_send_email", "arguments": "{\"recipient\": \"john@company.com\", ...}"}
  ],
  "category": "workplace_assistant_email",
  "environment_name": "workplace_assistant"
}
```

**Important:** Tool schemas must NOT include extra fields like `database` or `operation_type`. Use the fixed JSONL file:
```
workplace_assistant_train-gpt-oss-fixed.jsonl
```

---

## CLI Commands Reference

| Command | Description |
|---------|-------------|
| `ng_run` | Start NeMo Gym servers |
| `ng_collect_rollouts` | Collect rollouts from agents |
| `ng_viewer` | View rollout results |
| `ng_test` | Run tests |
| `ng_status` | Check server status |
| `ng_help` | Display help |
| `ng_version` | Show version |
| `ng_dump_config` | Dump configuration |

---

## Key File Paths

```
# Rollout collection script
nemo_gym/rollout_collection.py

# OpenAI model config
responses_api_models/openai_model/configs/openai_model.yaml

# Workplace assistant config
resources_servers/workplace_assistant/configs/workplace_assistant.yaml

# Example data
resources_servers/workplace_assistant/data/example.jsonl

# Generated training data (from DataDesigner)
# See examples/multistep_toolcalling_datagen/ to generate your own training JSONL
resources_servers/workplace_assistant/data/train.jsonl
```

---

## Workflow Overview

1. **Start servers** - LLM endpoint + workplace_assistant environment
2. **Load input data** - Read JSONL file with queries and tool definitions
3. **Collect rollouts** - For each example:
   - Send to `workplace_assistant_simple_agent`
   - Agent loops up to max_steps (6) calling tools and LLM
   - Responses API returns structured output with tool calls
4. **Verify & score** - Workplace assistant verifies against ground_truth
5. **Write results** - Each rollout written to output JSONL with reward scores

---

## Troubleshooting

### Error: 422 Unprocessable Entity

**Cause:** Tool schemas have extra fields (`database`, `operation_type`).

**Fix:** Use the fixed JSONL file: `workplace_assistant_train-gpt-oss-fixed.jsonl`

### Error: Workspace member missing `pyproject.toml`

**Cause:** Git submodules not initialized.

**Fix:**
```bash
cd /home/shashankv/bignlp/multistep-tool-rl-gym/RL
git submodule update --init 3rdparty/Automodel-workspace/Automodel
```
