(data-index)=
# Data

NeMo Gym datasets use JSONL format for reinforcement learning (RL) training. Each dataset connects to an agent server—the component that orchestrates agent-environment interactions during training.

## Prerequisites

- **NeMo Gym installed**: See {doc}`/get-started/detailed-setup`
- **Repository cloned** (for built-in datasets):
  ```bash
  git clone https://github.com/NVIDIA-NeMo/Gym.git
  cd Gym
  ```

:::{note}
NeMo Gym uses OpenAI-compatible schemas for model server compatibility. **No OpenAI account required**—local servers like vLLM use the same format.
:::

## Data Format

Each JSONL line requires a `responses_create_params` field following the [OpenAI Responses API schema](https://platform.openai.com/docs/api-reference/responses/create):

```json
{"responses_create_params": {"input": [{"role": "user", "content": "What is 2+2?"}]}}
```

Additional fields like `expected_answer` vary by resources server—the component that provides tools and reward signals.

**Source**: `nemo_gym/base_resources_server.py:35-36`

### Example Data

```json
{"responses_create_params": {"input": [{"role": "user", "content": "What is 2+2?"}]}, "expected_answer": "4"}
{"responses_create_params": {"input": [{"role": "user", "content": "What is 3*5?"}]}, "expected_answer": "15"}
{"responses_create_params": {"input": [{"role": "user", "content": "What is 10/2?"}]}, "expected_answer": "5"}
```

## Quick Start

Run this command from the repository root:

```bash
ng_prepare_data \
    "+config_paths=[resources_servers/example_multi_step/configs/example_multi_step.yaml]" \
    +output_dirpath=data/test \
    +mode=example_validation
```

**Success**: `Finished!` message and `data/test/example_metrics.json` created.

## Dataset Types

| Type | Purpose | License |
|------|---------|---------|
| `example` | Testing and development | Not required |
| `train` | RL training data | Required |
| `validation` | Evaluation during training | Required |

**Source**: `nemo_gym/config_types.py:352`

## Configuration

Define datasets in your agent server's YAML config:

```yaml
datasets:
  - name: train
    type: train
    jsonl_fpath: resources_servers/workplace_assistant/data/train.jsonl
    huggingface_identifier:
      repo_id: nvidia/Nemotron-RL-agent-workplace_assistant
      artifact_fpath: train.jsonl
    license: Apache 2.0
```

**Source**: `resources_servers/workplace_assistant/configs/workplace_assistant.yaml:21-27`

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Dataset identifier |
| `type` | Yes | `example`, `train`, or `validation` |
| `jsonl_fpath` | Yes | Path to data file |
| `license` | Train/validation | See valid values below |
| `huggingface_identifier` | No | Remote download location |
| `num_repeats` | No | Repeat count (default: `1`) |

### Valid Licenses

`Apache 2.0` · `MIT` · `GNU General Public License v3.0` · `Creative Commons Attribution 4.0 International` · `Creative Commons Attribution-ShareAlike 4.0 International` · `TBD` · `NVIDIA Internal Use Only, Do Not Distribute`

**Source**: `nemo_gym/config_types.py:363-372`

## Workflow

```{mermaid}
flowchart LR
    A[Create JSONL] --> B[Add to config]
    B --> C[Run ng_prepare_data]
    C -->|Pass| D[Train with NeMo RL]
    C -->|Fail| E[Fix and retry]
```

## Validation Modes

| Mode | Scope | Use Case |
|------|-------|----------|
| `example_validation` | `example` datasets | Format check before contributing |
| `train_preparation` | `train` + `validation` | Full prep for RL training |

To prepare training data with auto-download:

```bash
ng_prepare_data \
    "+config_paths=[resources_servers/workplace_assistant/configs/workplace_assistant.yaml]" \
    +output_dirpath=data/workplace_assistant \
    +mode=train_preparation \
    +should_download=true
```

:::{tip}
HuggingFace downloads require authentication. Set `hf_token` in `env.yaml` or export `HF_TOKEN`.
:::

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `JSON parse error at line N` | Invalid JSON | Check quotes, commas, brackets at line N |
| `ValidationError: responses_create_params` | Missing field | Add `responses_create_params.input` |
| `A license is required` | Missing license | Add `license` to dataset config |
| `Missing local datasets` | File not found | Check path or add `+should_download=true` |

## Guides

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Prepare and Validate
:link: prepare-validate
:link-type: doc
Full data preparation workflow.
+++
{bdg-secondary}`data-prep`
:::

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Download from Hugging Face
:link: download-huggingface
:link-type: doc
Fetch datasets from HuggingFace Hub.
+++
{bdg-secondary}`huggingface`
:::

::::

## CLI Commands

| Command | Description |
|---------|-------------|
| `ng_prepare_data` | Validate and generate metrics |
| `ng_download_dataset_from_hf` | Download from HuggingFace |
| `ng_viewer` | View dataset in Gradio UI |

See {doc}`/reference/cli-commands` for details.

## Large Datasets

- Validation streams line-by-line (memory-efficient)
- Single-threaded; >100K samples may take minutes
- Use `num_repeats` instead of duplicating JSONL lines
