---
name: gym-data
description: >
  Prepare, validate, and register datasets for NeMo Gym benchmarks. Use when converting
  source data to Gym JSONL format, generating example.jsonl files, uploading to
  HuggingFace, validating with ng_prepare_data, or wiring dataset identifiers
  into YAML configs. Covers the full data lifecycle from raw source to registered dataset.
license: Apache-2.0
compatibility: Requires Python 3.12+, uv. HuggingFace operations require hf_token in env.yaml.
metadata:
  author: nvidia-nemo-gym
  version: "1.0"
allowed-tools: Bash(python:*) Bash(ng_*) Read Write Edit Grep Glob
---

# NeMo Gym Data Preparation

## Step 1: Understand the target schema

Every line in a Gym JSONL file must have this structure:

```json
{
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "System prompt here"},
      {"role": "user", "content": "Problem statement here"}
    ]
  },
  "verifier_metadata": {
    // Task-specific fields used by verify()
  },
  "agent_ref": {
    "type": "responses_api_agents",
    "name": "my_benchmark_simple_agent"
  }
}
```

- `responses_create_params.input` follows OpenAI message format
- `verifier_metadata` is opaque to the framework — define whatever fields your benchmark's `verify()` method needs (test cases, expected answers, task IDs, etc.)
- `agent_ref` routes the row to a specific agent. Optional if you pass `+agent_name=` on the CLI, but required for multi-agent datasets where different rows target different agents

## Step 2: Convert source data

If converting from another format:

1. **Write the conversion script in the source repo**, not in NeMo Gym. Prompt files also belong in the source repo. Exception: when there is no external source repo.
2. Map source fields to `responses_create_params.input` messages and `verifier_metadata`
3. System prompts go in the first message with `role: system`
4. Validate every line is valid JSON and has the required top-level keys

### Tool definitions in input
For benchmarks involving tool use, `responses_create_params` can include a `tools` array alongside `input`:
```json
{
  "responses_create_params": {
    "input": [...],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
        }
      }
    ],
    "parallel_tool_calls": true
  },
  "verifier_metadata": {...}
}
```
Tools and `parallel_tool_calls` are passed through to the model server. Fill in tool `description` fields — models perform significantly better with descriptive tool definitions.

### verifier_metadata patterns by domain
The `verifier_metadata` structure varies by benchmark type. Study the existing server's `verify()` to know which fields it reads:

| Domain | Common verifier_metadata fields |
|--------|-------------------------------|
| Code generation | `test_cases` [{input, expected_output}], `function_name`, `language` |
| Math | `expected_answer`, `solution_type` (numeric, symbolic, proof) |
| SQL | `db_id`, `gold_sql`, `ignore_order`, `condition_cols` |
| Safety/Jailbreak | `adversarial_prompt`, `attack_type` |
| LLM-as-Judge | `expected_answer`, `template_metadata` {`output_regex`} |
| Search/QA | `ground_truth`, `question` |

### Data leakage check
Before finalizing data, verify that:
- The expected answer does NOT appear verbatim in the system or user prompt
- The `verifier_metadata` doesn't contain fields that could leak through to the model (only `responses_create_params.input` reaches the model; `verifier_metadata` stays server-side)
- For judge-based benchmarks, the judge prompt template doesn't inadvertently reveal the expected answer format

## Step 3: Generate example.jsonl

Create `data/example.jsonl` with exactly 5 entries. These are committed to git and used for smoke testing.

Selection criteria:
- Pick entries that exercise different code paths in `verify()`
- Include at least one "easy" case (should always get reward 1.0 from a capable model)
- Include at least one edge case (unusual input format, boundary condition)
- Keep entries small — example data should load instantly

## Step 4: Validate data

### Example validation (required before PR submission)

```bash
ng_prepare_data "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml]" \
    +output_dirpath=/tmp/prepare +mode=example_validation
```

### Train preparation (download and prepare train/validation datasets)

```bash
ng_prepare_data "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml]" \
    +output_dirpath=data/my_benchmark +mode=train_preparation +should_download=true
```

By default, `data_source` is `huggingface`. Use `+data_source=gitlab` only for NVIDIA-internal datasets that haven't been migrated to HuggingFace yet.

### What to check

- Every line parses as valid JSON
- `responses_create_params.input` is a non-empty list of messages
- Each message has `role` and `content` fields
- `verifier_metadata` fields match what `verify()` expects
- `agent_ref` is present if the dataset is used without `+agent_name=` on the CLI, or if different rows target different agents

## Step 5: Upload to dataset registry

Train and validation datasets must NOT be committed to git. Upload them to HuggingFace:

```bash
ng_upload_dataset_to_hf \
    +dataset_name=my_benchmark \
    +input_jsonl_fpath=resources_servers/my_benchmark/data/my_dataset.jsonl \
    +resource_config_path=resources_servers/my_benchmark/configs/my_benchmark.yaml
```

Requires HuggingFace credentials in `env.yaml`:
```yaml
hf_token: <your-hf-token>
hf_organization: <your-hf-org>
hf_collection_name: <collection-name>
hf_collection_slug: <collection-slug>
```

After upload, verify download works:
```bash
ng_prepare_data "+config_paths=[resources_servers/my_benchmark/configs/my_benchmark.yaml]" \
    +output_dirpath=data/my_benchmark +mode=train_preparation +should_download=true
```

## Step 6: Wire YAML config

Add dataset entries to the server's YAML config. Train/validation datasets need both `jsonl_fpath` (local path) and a remote identifier:

```yaml
datasets:
- name: my_dataset
  type: train
  jsonl_fpath: resources_servers/my_benchmark/data/my_dataset.jsonl
  huggingface_identifier:
    repo_id: my-org/my-benchmark-dataset
    artifact_fpath: train.jsonl
  license: MIT
- name: example
  type: example
  jsonl_fpath: resources_servers/my_benchmark/data/example.jsonl
```

- `jsonl_fpath` is the local download destination
- `huggingface_identifier` tells the system where to fetch from (use `gitlab_identifier` only for NVIDIA-internal datasets not yet on HuggingFace)
- `huggingface_identifier.artifact_fpath` is optional — omit it for structured dataset discovery
- `example` datasets don't need a remote identifier — they're committed to git
- `num_repeats` (default: 1) — repeats the dataset N times during training data preparation
- `license` is required for train and validation datasets. Valid values:
  - `Apache 2.0`
  - `MIT`
  - `Creative Commons Attribution 4.0 International`
  - `Creative Commons Attribution-ShareAlike 4.0 International`
  - `GNU General Public License v3.0`
  - `TBD` (placeholder — replace before merge)

## Step 7: Fix .gitignore

Check `data/.gitignore`. The scaffold generates default patterns:
```
*train.jsonl
*validation.jsonl
*train_prepare.jsonl
*validation_prepare.jsonl
*example_prepare.jsonl
```

If your filename doesn't match (e.g. `my_eval.jsonl`), add a custom pattern. If data was previously tracked:
```bash
git rm --cached <file>
```
