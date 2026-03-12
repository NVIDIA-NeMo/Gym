(data-prompt-config)=
# Prompt Config

Apply YAML-based prompt templates at rollout time to build `responses_create_params.input` on the fly. This enables prompt sweeps without re-preparing JSONL data.

:::{card}

**Goal**: Use prompt configs to separate prompt templates from dataset preparation.

**Time**: ~5 minutes

^^^

**In this guide, you will**:

1. Write a prompt config YAML file
2. Apply it during rollout collection with `ng_collect_rollouts`
3. Optionally materialize prompts into JSONL with `ng_materialize_prompts`

:::

**Prerequisites**:
- NeMo Gym installed ({doc}`/get-started/detailed-setup`)
- A JSONL dataset with raw fields (e.g. `question`, `expected_answer`)

---

## Overview

A prompt config is a YAML file with a required `user` field and an optional `system` field. Placeholders like `{question}` are filled from each data row's top-level fields during rollout collection.

Prompt configs and pre-populated `responses_create_params.input` in the JSONL data are **mutually exclusive**. Use one or the other. If any row already contains `responses_create_params.input` when a prompt config is specified, an error is raised.

---

## Prompt Config Format

### Minimal (user message only)

```yaml
# The {question} placeholder is filled from each row's "question" field.
user: "{question}"
```

### With system message

```yaml
# Math chain-of-thought prompt with system message.
# Expects rows with a "question" field.
system: "You are a helpful math assistant. Think step by step and put your final answer in \\boxed{{}}."
user: "{question}"
```

### Multiple fields

```yaml
# Expects rows with "question" and "context" fields.
system: "Answer the question using the provided context."
user: |
  Context: {context}

  Question: {question}
```

:::{note}
Literal braces must be doubled (`{{` / `}}`). For example, `\\boxed{{}}` produces `\boxed{}` in the output.
:::



---

## Usage

### At rollout time

Pass `+prompt_config=<path>` to `ng_collect_rollouts`:

```bash
ng_collect_rollouts \
    +agent_name=my_agent \
    +input_jsonl_fpath=data/raw_problems.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl \
    +prompt_config=/path/to/my_prompt.yaml \
    +num_repeats=5 \
    "+responses_create_params={max_output_tokens: 16384, temperature: 1.0}"
```

The `+prompt_config` path must be either an absolute path or a path relative to the Gym repository root.

The input JSONL should contain raw fields (e.g. `question`, `expected_answer`) **without** `responses_create_params.input`. The prompt config builds the input messages during rollout collection.

### Standalone materialization

Use `ng_materialize_prompts` to write a prompt template into JSONL without running rollouts:

```bash
ng_materialize_prompts \
    +input_jsonl_fpath=data/raw_problems.jsonl \
    +prompt_config=/path/to/my_prompt.yaml \
    +output_jsonl_fpath=data/materialized.jsonl
```

This produces a new JSONL file with `responses_create_params.input` populated from the template. This is useful for inspection or passing to other tools that expect pre-populated input.

---

## Input Data Format

When using prompt configs, your input JSONL should have the placeholder fields at the top level:

```json
{"question": "What is 2+2?", "expected_answer": "4"}
{"question": "What is 3*5?", "expected_answer": "15"}
```

Other fields in `responses_create_params` (such as `tools` and `temperature`) are preserved. Only `input` is built from the template.

---

## CLI Parameters

### `ng_collect_rollouts`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `+prompt_config` | No | Path to a prompt YAML file. Mutually exclusive with pre-populated `responses_create_params.input` in the JSONL data. |

See {doc}`/reference/cli-commands` for the full list of `ng_collect_rollouts` parameters.

### `ng_materialize_prompts`

| Parameter | Required | Description |
|-----------|----------|-------------|
| `+input_jsonl_fpath` | Yes | Raw JSONL data (no `responses_create_params.input`). |
| `+prompt_config` | Yes | Path to prompt YAML file to apply. |
| `+output_jsonl_fpath` | Yes | Output path for materialized JSONL with populated prompts. |

---

## How It Works

1. The prompt YAML is loaded and validated (must have a `user` key)
2. All rows are checked for conflicts. If any row already has `responses_create_params.input`, an error is raised
3. For each row, placeholders in `system` and `user` are filled from the row's fields
4. The resulting messages are set as `responses_create_params.input`


