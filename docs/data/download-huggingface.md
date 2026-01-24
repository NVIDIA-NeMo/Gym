(data-download-huggingface)=
# Download from Hugging Face

Download JSONL datasets from Hugging Face Hub for NeMo Gym training.

**Goal**: Download a dataset from Hugging Face Hub in JSONL format for training.

**Prerequisites**: NeMo Gym installed ({doc}`/get-started/detailed-setup`)

---

## Quick Start

```bash
ng_download_dataset_from_hf \
    +repo_id=nvidia/Nemotron-RL-math-OpenMathReasoning \
    +split=train \
    +output_fpath=./data/train.jsonl
```

```text
[Nemo-Gym] - Downloaded train split to: ./data/train.jsonl
```

:::{note}
NeMo Gym uses [Hydra](https://hydra.cc/) for configuration. Arguments use `+key=value` syntax.
:::

---

## Options

| Option | Description |
|--------|-------------|
| `repo_id` | **Required.** Hugging Face repository (e.g., `nvidia/Nemotron-RL-math-OpenMathReasoning`) |
| `output_dirpath` | Output directory. Files named `{split}.jsonl`. **Use this OR `output_fpath`.** |
| `output_fpath` | Exact output file path. Requires `split` or `artifact_fpath`. **Use this OR `output_dirpath`.** |
| `artifact_fpath` | Download a specific file from the repo (raw file mode) |
| `split` | Dataset split: `train`, `validation`, or `test`. Omit to download all. |
| `hf_token` | Authentication token for private/gated repositories |

---

## Download Methods

:::::{tab-set}

::::{tab-item} Structured Dataset (Recommended)
Downloads using the `datasets` library and converts to JSONL.

**Use when**: Repository uses Hugging Face's standard dataset format.

**All splits**:

```bash
ng_download_dataset_from_hf \
    +repo_id=nvidia/Nemotron-RL-knowledge-mcqa \
    +output_dirpath=./data/
```

```text
[Nemo-Gym] - Downloaded train split to: ./data/train.jsonl
[Nemo-Gym] - Downloaded validation split to: ./data/validation.jsonl
```

**Single split**:

```bash
ng_download_dataset_from_hf \
    +repo_id=SWE-Gym/SWE-Gym \
    +split=train \
    +output_fpath=./data/train.jsonl
```

::::

::::{tab-item} Raw File
Downloads a specific file directly without conversion.

**Use when**: Repository contains pre-formatted JSONL files.

```bash
ng_download_dataset_from_hf \
    +repo_id=nvidia/nemotron-RL-coding-competitive_coding \
    +artifact_fpath=opencodereasoning_filtered_25k_train.jsonl \
    +output_fpath=./data/train.jsonl
```

```text
[Nemo-Gym] - Downloaded opencodereasoning_filtered_25k_train.jsonl to: ./data/train.jsonl
```

::::

:::::

---

## Troubleshooting

::::{dropdown} Authentication Failed (401)
:open:

```text
huggingface_hub.utils.HfHubHTTPError: 401 Client Error
```

**Fix**: Verify your token is valid. For gated datasets, accept the license on Hugging Face first.
::::

::::{dropdown} Repository Not Found (404)
:open:

```text
huggingface_hub.utils.HfHubHTTPError: 404 Client Error
```

**Fix**: Check `repo_id` format is `organization/dataset-name`. Verify the repository exists and is public (or you have access).
::::

::::{dropdown} Validation Error: Output Path

```text
ValueError: Either output_dirpath or output_fpath must be provided
```

**Fix**: Add `+output_dirpath=./data/` or `+output_fpath=./data/train.jsonl`.
::::

::::{dropdown} Validation Error: Conflicting Options

```text
ValueError: Cannot specify both artifact_fpath and split
```

**Fix**: Use `artifact_fpath` for raw files OR `split` for structured datasets—not both.
::::

---

## Private Repositories

:::{warning}
Avoid passing tokens on the command line—they appear in shell history.
:::

**Recommended** — Use environment variable:

```bash
export hf_token=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
ng_download_dataset_from_hf \
    +repo_id=my-org/private-dataset \
    +output_dirpath=./data/
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Use a **read-only** token.

:::{dropdown} Alternative: Pass token directly
:icon: alert

Not recommended for shared systems:

```bash
ng_download_dataset_from_hf \
    +repo_id=my-org/private-dataset \
    +hf_token=hf_xxxxxxxxxxxxxxxxxxxxxxxxx \
    +output_dirpath=./data/
```
:::

---

## NVIDIA Datasets

| Dataset | Repository |
|---------|-----------|
| OpenMathReasoning | `nvidia/Nemotron-RL-math-OpenMathReasoning` |
| Competitive Coding | `nvidia/nemotron-RL-coding-competitive_coding` |
| Workplace Assistant | `nvidia/Nemotron-RL-agent-workplace_assistant` |
| Structured Outputs | `nvidia/Nemotron-RL-instruction_following-structured_outputs` |
| MCQA | `nvidia/Nemotron-RL-knowledge-mcqa` |

---

:::{dropdown} Automatic Downloads During Data Preparation
:icon: download

NeMo Gym can automatically download missing datasets during data preparation. Configure `huggingface_identifier` in your resources server config:

```yaml
datasets:
  - name: train
    type: train
    jsonl_fpath: resources_servers/code_gen/data/train.jsonl
    huggingface_identifier:
      repo_id: nvidia/nemotron-RL-coding-competitive_coding
      artifact_fpath: opencodereasoning_filtered_25k_train.jsonl
    license: Apache 2.0
```

Run with download enabled:

```bash
config_paths="resources_servers/code_gen/configs/code_gen.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=./data/prepared \
    +mode=train_preparation \
    +should_download=true \
    +data_source=huggingface
```

If `jsonl_fpath` doesn't exist locally, NeMo Gym downloads from `huggingface_identifier` before processing.
:::

:::{dropdown} Caching Behavior
:icon: database

Downloads use Hugging Face's cache at `~/.cache/huggingface/`.

- **Structured datasets**: Reads from cache (fast), overwrites output file
- **Raw files**: Uses cached copy, then copies to output path

To force fresh download:

```bash
rm -rf ~/.cache/huggingface/hub/datasets--<org>--<dataset>
```
:::

:::{dropdown} Source References
:icon: code

| Section | Source |
|---------|--------|
| Config schema | `nemo_gym/config_types.py:306-349` |
| Download logic | `nemo_gym/hf_utils.py:57-115` |
| Validation rules | `nemo_gym/config_types.py:334-349` |
| Auto-download | `nemo_gym/train_data_utils.py:476-494` |
:::

## Related

- {doc}`prepare-validate` — Validate downloaded datasets
- {doc}`/reference/cli-commands` — Full CLI reference
