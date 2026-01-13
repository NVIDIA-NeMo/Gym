(data-prepare-validate)=
# Prepare and Validate Data

```{note}
This page is a stub. Content is being developed.
```

Learn how to prepare and validate your data for use with NeMo Gym training environments.

:::{card}

**Goal**: Format your data for NeMo Gym and validate it before training.

^^^

**What you'll learn**:

1. Data format requirements for NeMo Gym
2. How to convert existing datasets
3. Validation techniques and common issues

:::

---

## Data Format

NeMo Gym uses JSONL files where each line is a JSON object containing task data.

:::::{tab-set}

::::{tab-item} Basic Format

Each line must contain `responses_create_params` with an `input` field:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "Your task here"}]}}
```

::::

::::{tab-item} Training Format

For training, include verification metadata:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Calculate 15 * 7"}
    ]
  },
  "expected_answer": "105",
  "metadata": {"source": "math_dataset", "difficulty": "easy"}
}
```

::::

:::::

### Required Fields

| Field | Required | Description |
|-------|----------|-------------|
| `responses_create_params` | Yes | Container for input |
| `responses_create_params.input` | Yes | List of messages |
| `expected_answer` | For training | Ground truth for verification |

### Message Roles

Valid roles for messages in `input`:

- `user` — The user's query or task
- `assistant` — Model responses (for multi-turn)
- `developer` — System instructions (preferred)
- `system` — System instructions (legacy)

---

## Converting Data

:::::{tab-set}

::::{tab-item} From Custom Datasets

```python
import json

def convert_to_nemo_gym(examples):
    for example in examples:
        yield {
            "responses_create_params": {
                "input": [{"role": "user", "content": example["question"]}]
            },
            "expected_answer": example["answer"]
        }

with open("data.jsonl", "w") as f:
    for item in convert_to_nemo_gym(my_data):
        f.write(json.dumps(item) + "\n")
```

::::

::::{tab-item} Adding System Prompts

Include a `developer` message before the user message:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "developer", "content": "You are a math tutor. Show your work."},
      {"role": "user", "content": "Solve: 3x + 5 = 20"}
    ]
  }
}
```

::::

::::{tab-item} From Hugging Face

For datasets on Hugging Face Hub, see {doc}`download-huggingface` for download commands and conversion examples.

::::

:::::

---

## Validation

### Common Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Missing `input` field | `KeyError` during rollout collection | Ensure `responses_create_params.input` exists |
| Invalid role | Unexpected model behavior | Use `user`, `assistant`, `developer`, or `system` |
| Malformed JSON | Parse errors | Validate each line with `json.loads()` |
| Empty input list | No model response | Include at least one message |

### Debugging Tips

::::{dropdown} Validate a single line
:icon: code

```python
import json

with open("data.jsonl") as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            assert "responses_create_params" in data
            assert "input" in data["responses_create_params"]
        except Exception as e:
            print(f"Line {i}: {e}")
```

::::

::::{dropdown} Test with a small sample
:icon: terminal

```bash
head -5 data.jsonl > sample.jsonl
ng_collect_rollouts +agent_name=my_agent \
    +input_jsonl_fpath=sample.jsonl \
    +output_jsonl_fpath=test_output.jsonl
```

::::

---

## Next Steps

Once your data is prepared and validated:

:::{card} {octicon}`play;1.5em;sd-mr-1` Collect Rollouts
:link: /get-started/rollout-collection
:link-type: doc

Generate training examples by running your agent on the prepared data.
:::
