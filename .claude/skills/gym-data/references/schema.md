# NeMo Gym JSONL Data Schema

Self-contained reference for the data format used across all NeMo Gym benchmarks.

---

## Input JSONL

Each line is a JSON object with these top-level fields:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Solve this problem..."}
    ],
    "tools": [],
    "parallel_tool_calls": false
  },
  "verifier_metadata": { ... },
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "my_benchmark_simple_agent"
  }
}
```

### responses_create_params

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | list[dict] | Yes | Messages in OpenAI format (role + content) |
| `tools` | list[dict] | No | Tool definitions for tool-calling benchmarks |
| `parallel_tool_calls` | bool | No | Whether the model may call multiple tools in one turn |

#### input messages

Follow the OpenAI message format:

```json
[
  {"role": "system", "content": "System instructions..."},
  {"role": "user", "content": "The task or question..."}
]
```

Roles: `system`, `user`, `assistant`. Multi-turn conversations alternate user/assistant.

#### tools (for tool-calling benchmarks)

```json
"tools": [
  {
    "type": "function",
    "function": {
      "name": "search_web",
      "description": "Search the web for information",
      "parameters": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
      }
    }
  }
]
```

### agent_ref

Optional. Routes the row to a specific agent at rollout collection time:

```json
"agent_ref": {
  "type": "responses_api_agents",
  "name": "my_benchmark_simple_agent"
}
```

If omitted, the `+agent_name=` CLI argument is used instead. Required for multi-agent datasets where different rows target different agents.

### verifier_metadata

Opaque dict passed through to the resources server's `verify()` method. Structure varies by benchmark domain.

#### By domain

**Code generation:**
```json
{
  "expected_output": "42",
  "test_cases": [{"input": "6 7", "expected": "42"}],
  "language": "python",
  "function_name": "multiply"
}
```

**Math / algebra:**
```json
{
  "expected_answer": 42,
  "solution_steps": ["6 * 7 = 42"]
}
```

**SQL:**
```json
{
  "db_id": "concert_singer",
  "gold_sql": "SELECT count(*) FROM singer",
  "question": "How many singers are there?",
  "ignore_order": true,
  "condition_cols": ["name"]
}
```

**Safety / jailbreak:**
```json
{
  "category": "harmful_content",
  "attack_type": "role_play",
  "expected_verdict": "UNSAFE"
}
```

**Judge-based (equivalence_llm_judge):**
```json
{
  "expected_answer": "42",
  "template_metadata": {
    "output_regex": "(?:Answer|ANSWER)[:\\s]*(.+)"
  },
  "extraction_length_threshold": 100
}
```

**Search / tool-use:**
```json
{
  "expected_tool": "search_web",
  "expected_args": {"query": "population of France"},
  "ground_truth_answer": "approximately 68 million"
}
```

---

## Output JSONL

Output from `ng_collect_rollouts`. Each line is one rollout:

```json
{
  "reward": 1.0,
  "response": {
    "output_text": "The answer is 42."
  },
  "task_index": 0,
  "prompt_token_ids": [1, 2, 3],
  "generation_token_ids": [4, 5, 6],
  "generation_log_probs": [-0.1, -0.2, -0.05]
}
```

Additional fields depend on the resources server's `VerifyResponse` class (e.g., `extracted_model_code`, `failure_reason`, `judge_evaluations`).

---

## Data leakage checks

Before uploading:

1. **Expected answers must NOT appear in system or user prompts.** The model should not be able to extract the answer from context.
2. **Gold SQL / gold code should not appear in the prompt.** Only the question/task description.
3. **verifier_metadata should not be referenced in the prompt.** It's for the verifier, not the model.

A quick check:
```python
import json
for line in open("data.jsonl"):
    d = json.loads(line)
    prompts = " ".join(m["content"] for m in d["responses_create_params"]["input"])
    answer = str(d["verifier_metadata"].get("expected_answer", ""))
    if answer and answer in prompts:
        print(f"LEAK: expected_answer '{answer}' found in prompt")
```

---

## Dataset types

| Type | Where it lives | Committed to git? | Size |
|------|---------------|-------------------|------|
| `example` | `data/example.jsonl` | Yes | 5 entries |
| `train` | HuggingFace → local `data/train.jsonl` | No | Full dataset |
| `validation` | HuggingFace → local `data/validation.jsonl` | No | Subset |

### .gitignore patterns (auto-generated)

```
*train.jsonl
*validation.jsonl
*train_prepare.jsonl
*validation_prepare.jsonl
*example_prepare.jsonl
```

If your filename doesn't match these patterns (e.g. `my_eval.jsonl`), add a custom pattern.

---

## Dataset registry

### Upload to HuggingFace

```bash
ng_upload_dataset_to_hf \
    +dataset_name=my_benchmark \
    +input_jsonl_fpath=resources_servers/my_benchmark/data/my_dataset.jsonl \
    +resource_config_path=resources_servers/my_benchmark/configs/my_benchmark.yaml
```

Requires `hf_token`, `hf_organization`, `hf_collection_name`, and `hf_collection_slug` in `env.yaml`.

### YAML wiring

Both `jsonl_fpath` and `huggingface_identifier` coexist:

```yaml
datasets:
- name: my_dataset
  type: train
  jsonl_fpath: resources_servers/my_benchmark/data/train.jsonl
  huggingface_identifier:
    repo_id: my-org/my-benchmark-dataset
    artifact_fpath: train.jsonl
  license: Apache-2.0
```

- `jsonl_fpath`: local download destination
- `huggingface_identifier`: where to fetch from (use `gitlab_identifier` only for NVIDIA-internal datasets not yet on HuggingFace)
- `license`: required for train/validation datasets

### Validation

```bash
# Validate example data (for PR submission)
ng_prepare_data "+config_paths=[...]" +output_dirpath=/tmp/prepare +mode=example_validation

# Download and prepare train/validation from HuggingFace (default)
ng_prepare_data "+config_paths=[...]" +output_dirpath=data/ +mode=train_preparation +should_download=true
```
