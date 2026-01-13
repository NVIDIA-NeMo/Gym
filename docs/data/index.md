(data-index)=
# Data

Data management is critical for RL training with NeMo Gym. This section covers how to prepare, validate, and source datasets for your training environments.

## Data Format

NeMo Gym uses JSONL files where each line contains a task:

```json
{"responses_create_params": {"input": [{"role": "user", "content": "What is 2+2?"}]}}
```

For training datasets, include metadata and expected answers:

```json
{
  "responses_create_params": {"input": [...]},
  "expected_answer": "4",
  "metadata": {"difficulty": "easy"}
}
```

## Dataset Types

| Type | Purpose | License Required |
|------|---------|------------------|
| `example` | Testing and development | No |
| `train` | RL training data | Yes |
| `validation` | Evaluation during training | Yes |

## Data Workflow

1. **Prepare** — Format data as JSONL with required fields
2. **Validate** — Check format and completeness
3. **Collect Rollouts** — Generate training examples with `ng_collect_rollouts`
4. **Train** — Use rollouts for RL training

## Guides

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Prepare and Validate
:link: prepare-validate
:link-type: doc
Format and validate your training data.
+++
{bdg-secondary}`data-prep` {bdg-secondary}`validation`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Download from Hugging Face
:link: download-huggingface
:link-type: doc
Use datasets from Hugging Face Hub.
+++
{bdg-secondary}`huggingface` {bdg-secondary}`datasets`
:::

::::

## Configuration

Define datasets in your server configuration:

```yaml
datasets:
  - name: my_train_data
    type: train
    jsonl_fpath: data/train.jsonl
    license: Apache 2.0
```

See {doc}`/reference/configuration` for complete dataset configuration.
