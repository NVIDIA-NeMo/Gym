(data-download-huggingface)=
# Download from Hugging Face

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

| Option | Description |
|--------|-------------|
| `repo_id` | Hugging Face repository ID (e.g., `NVIDIA/NeMo-Gym-Math`) |
| `artifact_fpath` | Path to specific file in repo (e.g., `train.jsonl`) |
| `output_dirpath` | Directory to save downloaded files (files named `{split}.jsonl`) |
| `output_fpath` | Exact file path for single file download |
| `split` | Dataset split: `train`, `validation`, `test` |
| `hf_token` | Authentication token for private datasets |

## Constraints

The following constraints apply to command options:

- **Output path**: Specify either `output_dirpath` or `output_fpath`, not both
- **Download method**: Specify either `artifact_fpath` (raw file) or `split` (structured dataset), not both
- **Single file output**: When using `output_fpath` without `artifact_fpath`, you must specify `split`

```{tip}
Use `output_dirpath` without `split` to download all available splits at once.
```

## Examples

### Download All Splits

```bash
ng_download_dataset_from_hf \
    +repo_id=NVIDIA/NeMo-Gym-Math-example_multi_step-v1 \
    +output_dirpath=data/
```

### Download Specific Split

```bash
ng_download_dataset_from_hf \
    +repo_id=NVIDIA/NeMo-Gym-Math-example_multi_step-v1 \
    +split=train \
    +output_fpath=data/train.jsonl
```

### Download Raw File

```bash
ng_download_dataset_from_hf \
    +repo_id=NVIDIA/NeMo-Gym-Math-example_multi_step-v1 \
    +artifact_fpath=train.jsonl \
    +output_fpath=data/train.jsonl
```

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

## Related

- {doc}`prepare-validate` — Convert datasets to NeMo Gym format
- {doc}`/reference/configuration` — Configure datasets in your server
