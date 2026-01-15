(data-prepare-validate)=
# Prepare and Validate Data

Format and validate JSONL datasets for NeMo Gym training using `ng_prepare_data`.

**Goal**: Validate data format and prepare datasets for training.

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

This generates `data/test/example_metrics.json` with dataset statistics.

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

**Key behavior**: Invalid samples are silently skipped. If metrics show fewer examples than expected, check your data.

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

### Re-Running

- **Output files** (`train.jsonl`, `validation.jsonl`) are overwritten
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
