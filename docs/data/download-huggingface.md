(data-download-huggingface)=
# Download from Hugging Face

```{warning}
This article has not been reviewed by a developer SME. Content may change.
```

Download and use datasets from Hugging Face Hub for NeMo Gym training.

---

## CLI Command

Use `ng_download_dataset_from_hf` to download datasets:

```bash
ng_download_dataset_from_hf \
    +repo_id=NVIDIA/my-dataset \
    +output_dirpath=data/ \
    +split=train
```

## Command Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `repo_id` | `str` | Required | Hugging Face repository ID (e.g., `NVIDIA/NeMo-Gym-Math`) |
| `artifact_fpath` | `str` | `null` | Path to specific file in repo for raw JSONL download |
| `output_dirpath` | `str` | `null` | Directory to save downloaded files (files named `{split}.jsonl`) |
| `output_fpath` | `str` | `null` | Exact file path for single file download |
| `split` | `"train"` \| `"validation"` \| `"test"` | `null` | Dataset split to download |
| `hf_token` | `str` | `null` | Authentication token for private datasets |

## Download Methods

NeMo Gym supports three methods for downloading datasets:

| Method | When to Use | Parameters |
|--------|-------------|------------|
| **Raw File** | Download a specific JSONL file | `artifact_fpath` + `output_fpath` |
| **Single Split** | Download one split from structured dataset | `split` + (`output_fpath` or `output_dirpath`) |
| **All Splits** | Download all available splits | `output_dirpath` only |

## Constraints

The following constraints apply to command options:

- **Output path**: Specify either `output_dirpath` or `output_fpath`, not both
- **Download method**: Specify either `artifact_fpath` (raw file) or `split` (structured dataset), not both
- **Single file output**: When using `output_fpath` without `artifact_fpath`, you must specify `split`

:::{tip}
Use `output_dirpath` without `split` to download all available splits at once.
:::

## Examples

:::::{tab-set}

::::{tab-item} Download All Splits

```bash
ng_download_dataset_from_hf \
    +repo_id=NVIDIA/NeMo-Gym-Math-example_multi_step-v1 \
    +output_dirpath=data/
```

::::

::::{tab-item} Download Specific Split

```bash
ng_download_dataset_from_hf \
    +repo_id=NVIDIA/NeMo-Gym-Math-example_multi_step-v1 \
    +split=train \
    +output_fpath=data/train.jsonl
```

::::

::::{tab-item} Download Raw File

```bash
ng_download_dataset_from_hf \
    +repo_id=NVIDIA/NeMo-Gym-Math-example_multi_step-v1 \
    +artifact_fpath=train.jsonl \
    +output_fpath=data/train.jsonl
```

::::

:::::

## Private Datasets

For private repositories, provide your Hugging Face token:

```bash
ng_download_dataset_from_hf \
    +repo_id=my-org/private-dataset \
    +hf_token=hf_xxxxx \
    +output_dirpath=data/
```

Get your token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## NVIDIA Datasets

NVIDIA publishes NeMo Gym datasets to Hugging Face under the [NVIDIA organization](https://huggingface.co/NVIDIA).

Example datasets:

- `NVIDIA/NeMo-Gym-Math-example_multi_step-v1` — Multi-step math problems

## Source Code

The download functionality is implemented in `nemo_gym/hf_utils.py`. See `DownloadJsonlDatasetHuggingFaceConfig` in `nemo_gym/config_types.py` for the full configuration schema.

## Related

- {doc}`prepare-validate` — Convert datasets to NeMo Gym format
- {doc}`/reference/configuration` — Configure datasets in your server
- [NVIDIA Hugging Face Datasets](https://huggingface.co/NVIDIA) 🔗 — Browse available datasets