(data-prepare-validate)=
# Prepare and Validate Data

Format and validate JSONL datasets for NeMo Gym training using `ng_prepare_data`.

:::{card}

**Goal**: Validate data format and prepare datasets for training.

^^^

**In this guide, you will**:

1. Validate datasets with `ng_prepare_data`
2. Generate training and validation splits
3. Understand the JSONL data format

:::

**Prerequisites**:
- NeMo Gym installed ({doc}`/get-started/detailed-setup`)
- Familiarity with resources servers ({doc}`/resources-server/index`)

---

## Quick Start

From the repository root:

```bash
ng_prepare_data \
    "+config_paths=[resources_servers/example_multi_step/configs/example_multi_step.yaml]" \
    +output_dirpath=data/test \
    +mode=example_validation
```

Success output:

```text
####################################################################################################
#
# Finished!
#
####################################################################################################
```

This generates two types of output:
- **Per-dataset metrics**: `resources_servers/example_multi_step/data/example_metrics.json` (alongside source JSONL)
- **Aggregated metrics**: `data/test/example_metrics.json` (in output directory)

---

## Data Format

NeMo Gym uses JSONL files. Each line requires a `responses_create_params` field following the [OpenAI Responses API schema](https://platform.openai.com/docs/api-reference/responses/create).

### Minimal Format

```json
{"responses_create_params": {"input": [{"role": "user", "content": "What is 2+2?"}]}}
```

### With Verification Fields

Most resources servers add fields for reward computation:

```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "What is 15 * 7? Put your answer in \\boxed{}."}]
  },
  "question": "What is 15 * 7?",
  "expected_answer": "105"
}
```

:::{tip}
Check `resources_servers/<name>/README.md` for required fields specific to each resources server.
:::

### Key Properties

| Property | Type | Description |
|----------|------|-------------|
| `input` | string or list | **Required.** User query or message list |
| `tools` | list | Tool definitions for function calling |
| `parallel_tool_calls` | bool | Allow parallel tool calls (default: `true`) |
| `temperature` | float | Sampling temperature |
| `max_output_tokens` | int | Maximum response tokens |

### Message Roles

| Role | Use |
|------|-----|
| `user` | User queries |
| `assistant` | Model responses (multi-turn) |
| `developer` | System instructions (preferred) |
| `system` | System instructions (legacy) |

---

## Preprocess Raw Datasets

If your dataset doesn't have `responses_create_params`, you need to preprocess it before using `ng_prepare_data`.

**When to preprocess**:
- Downloaded datasets without NeMo Gym format
- Custom data needing system prompts
- Need to split into train/validation sets

### Add `responses_create_params`

The `responses_create_params` field wraps your input in the Responses API format. This typically includes a system prompt and the user content.

::::{dropdown} Preprocessing script (preprocess.py)
:icon: code
:open:

Save this script as `preprocess.py`. It reads a raw JSONL file, adds `responses_create_params`, and splits into train/validation:

```python
import json
import os

# Configuration — customize these for your dataset
INPUT_FIELD = "problem"  # Field containing the input text (e.g., "problem", "question", "prompt")
FILENAME = "raw_data.jsonl"
SYSTEM_PROMPT = "Your task is to solve a math problem. Put the answer inside \\boxed{}."
TRAIN_RATIO = 0.999  # 99.9% train, 0.1% validation

dirpath = os.path.dirname(FILENAME) or "."
with open(FILENAME, "r", encoding="utf-8") as fin, \
    open(os.path.join(dirpath, "train.jsonl"), "w", encoding="utf-8") as ftrain, \
    open(os.path.join(dirpath, "validation.jsonl"), "w", encoding="utf-8") as fval:
    
    lines = list(fin)
    split_idx = int(len(lines) * TRAIN_RATIO)
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        row = json.loads(line)
        
        # Remove fields not needed for training (optional)
        row.pop("generated_solution", None)
        row.pop("problem_source", None)
        
        # Add responses_create_params
        row["responses_create_params"] = {
            "input": [
                {"role": "developer", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row.get(INPUT_FIELD, "")},
            ]
        }
        
        out = json.dumps(row) + "\n"
        (ftrain if i < split_idx else fval).write(out)
```

:::{important}
You must customize these variables for your dataset:
- `INPUT_FIELD`: The field name containing your input text. Common values: `"problem"` (math), `"question"` (QA), `"prompt"` (general), `"instruction"` (instruction-following)
- `SYSTEM_PROMPT`: Task-specific instructions for the model
- `TRAIN_RATIO`: Train/validation split ratio
:::

::::

Run and verify:

```bash
uv run preprocess.py
wc -l train.jsonl validation.jsonl
```

### Create Config for Custom Data

After preprocessing, create a config file to point `ng_prepare_data` at your local files.

::::{dropdown} Example config: custom_data.yaml
:icon: file-code

```yaml
custom_resources_server:
  resources_servers:
    custom_server:
      entrypoint: app.py
      domain: math  # math | coding | agent | knowledge | other
      description: Custom math dataset
      verified: false

custom_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: custom_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: train.jsonl
        license: Creative Commons Attribution 4.0 International
      - name: validation
        type: validation
        jsonl_fpath: validation.jsonl
        license: Creative Commons Attribution 4.0 International
```

::::

Run data preparation:

```bash
config_paths="responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,custom_data.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" +mode=train_preparation +output_dirpath=data
```

This validates your data and adds the `agent_ref` field to each row, routing samples to your resource server.

---

## Validation Modes

| Mode | Purpose | Validates |
|------|---------|-----------|
| `example_validation` | PR submission | `example` datasets |
| `train_preparation` | Training prep | `train`, `validation` datasets |

### Example Validation

```bash
ng_prepare_data "+config_paths=[resources_servers/example_multi_step/configs/example_multi_step.yaml]" \
    +output_dirpath=data/example_multi_step \
    +mode=example_validation
```

### Training Preparation

```bash
ng_prepare_data "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true
```

### CLI Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `+config_paths` | Yes | YAML config paths |
| `+output_dirpath` | Yes | Output directory |
| `+mode` | Yes | `example_validation` or `train_preparation` |
| `+should_download` | No | Download missing datasets (default: `false`) |
| `+data_source` | No | `huggingface` (default) or `gitlab` |

---

## Troubleshooting

| Issue | Symptom | Fix |
|-------|---------|-----|
| Missing `responses_create_params` | Sample silently skipped | Add field with valid `input` |
| Invalid JSON | Sample skipped | Fix JSON syntax |
| Invalid role | Sample skipped | Use `user`, `assistant`, `system`, or `developer` |
| Missing dataset file | `AssertionError` | Create file or set `+should_download=true` |

:::{warning}
Invalid samples are silently skipped. If metrics show fewer examples than expected, check your data format.
:::

::::{dropdown} Find invalid samples
:icon: code
:open:

```python
import json

def validate_sample(line: str) -> tuple[bool, str]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}"
    
    if "responses_create_params" not in data:
        return False, "Missing 'responses_create_params'"
    
    if "input" not in data["responses_create_params"]:
        return False, "Missing 'input' in responses_create_params"
    
    return True, "OK"

with open("your_data.jsonl") as f:
    for i, line in enumerate(f, 1):
        valid, msg = validate_sample(line)
        if not valid:
            print(f"Line {i}: {msg}")
```

::::

---

## Validation Process

`ng_prepare_data` performs these steps:

1. **Load configs** — Parse server configs, identify datasets
2. **Check files** — Verify dataset files exist
3. **Validate samples** — Parse each line, validate against schema
4. **Compute metrics** — Aggregate statistics
5. **Collate** — Combine samples with agent references

### Output Locations

Metrics files are written to two locations:
- **Per-dataset**: `{dataset_jsonl_path}_metrics.json` — alongside each source JSONL file
- **Aggregated**: `{output_dirpath}/{type}_metrics.json` — combined metrics per dataset type

### Re-Running

- **Output files** (`train.jsonl`, `validation.jsonl`) are overwritten in `output_dirpath`
- **Metrics files** (`*_metrics.json`) are compared — delete them if your data changed

### Generated Metrics

| Metric | Description |
|--------|-------------|
| Number of examples | Valid sample count |
| Number of tools | Tool count stats (avg/min/max/stddev) |
| Number of turns | User messages per sample |
| Temperature | Temperature parameter stats |

::::{dropdown} Example metrics file
:icon: file

```json
{
    "name": "example",
    "type": "example",
    "jsonl_fpath": "resources_servers/example_multi_step/data/example.jsonl",
    "Number of examples": 5,
    "Number of tools": {
        "Total # non-null values": 5,
        "Average": 2.0,
        "Min": 2.0,
        "Max": 2.0
    }
}
```

::::

---

## Dataset Configuration

Define datasets in your server's YAML config:

```yaml
datasets:
  - name: train
    type: train
    jsonl_fpath: resources_servers/my_server/data/train.jsonl
    license: Apache 2.0
  - name: validation
    type: validation
    jsonl_fpath: resources_servers/my_server/data/validation.jsonl
    license: Apache 2.0
  - name: example
    type: example
    jsonl_fpath: resources_servers/my_server/data/example.jsonl
```

| Type | Purpose | Required for |
|------|---------|--------------|
| `example` | Small sample (~5 rows) for format checks | PR submission |
| `train` | Training data | RL training |
| `validation` | Evaluation during training | RL training |

---

## Next Steps

:::{card} {octicon}`play;1.5em;sd-mr-1` Collect Rollouts
:link: /get-started/rollout-collection
:link-type: doc

Generate training examples by running your agent on prepared data.
:::

:::{card} {octicon}`book;1.5em;sd-mr-1` NeMo RL Integration
:link: /tutorials/nemo-rl-grpo/index
:link-type: doc

Use validated data with NeMo RL for GRPO training.
:::
