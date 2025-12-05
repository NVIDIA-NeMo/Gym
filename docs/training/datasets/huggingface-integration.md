(huggingface-dataset-integration)=

# HuggingFace Dataset Integration

Upload and download datasets from HuggingFace Hub for sharing and collaboration.

---

## Configuration

Add your HuggingFace credentials to `env.yaml`:

```yaml
hf_token: {your huggingface token}
hf_organization: {your huggingface org}
hf_collection_name: {your collection}
hf_collection_slug: {your collection slug}  # alphanumeric string at end of collection URI

# Optional
hf_dataset_prefix: str  # Override default "NeMo-Gym" prefix
```

---

## Naming Convention

Datasets follow this naming pattern:

```
{hf_organization}/{hf_dataset_prefix}-{domain}–{resource_server_name}-{your dataset name}
```

**Example**:
```
NVIDIA/NeMo-Gym-Math-math_with_judge-dapo17k
```

When uploading, you only provide `{your dataset name}` — everything else is populated from your config.

---

## Upload to HuggingFace

### Basic Upload

```bash
resource_config_path="resources_servers/multineedle/configs/multineedle.yaml"

ng_upload_dataset_to_hf \
    +dataset_name={your dataset name} \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl \
    +resource_config_path=${resource_config_path}
```

The resource server config path is required because `domain` is used in the HuggingFace dataset naming.

### Upload and Delete from GitLab

```bash
ng_upload_dataset_to_hf \
    +dataset_name={your dataset name} \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl \
    +resource_config_path=${resource_config_path} \
    +delete_from_gitlab=true
```

You'll see a confirmation dialog:
```
[Nemo-Gym] - Dataset uploaded successful
[Nemo-Gym] - Found model 'fs-test' in the registry. Are you sure you want to delete it from Gitlab? [y/N]:
```

### Alternative: Upload and Delete in One Command

```bash
ng_gitlab_to_hf_dataset \
    +dataset_name={your dataset name} \
    +input_jsonl_fpath=data/multineedle_benchmark.jsonl \
    +resource_config_path=${resource_config_path}
```

---

## Delete from GitLab Only

If you've already uploaded to HuggingFace and want to clean up GitLab:

```bash
ng_delete_dataset_from_gitlab \
    +dataset_name={your dataset name}
```

:::{important}
GitLab model names are **case sensitive**. Models named `My_Model` and `my_model` can exist simultaneously. Ensure your HuggingFace dataset name matches the GitLab casing exactly.
:::

---

## Download from HuggingFace

```bash
ng_download_dataset_from_hf \
    +repo_id=NVIDIA/NeMo-Gym-Instruction_Following-multineedle-{your dataset name} \
    +artifact_fpath=multineedle_benchmark.jsonl \
    +output_fpath=data/multineedle_benchmark_hf.jsonl
```

---

## Command Reference

```{list-table}
:header-rows: 1
:widths: 35 65

* - Command
  - Description
* - `ng_upload_dataset_to_hf`
  - Upload dataset to HuggingFace (optionally delete from GitLab)
* - `ng_gitlab_to_hf_dataset`
  - Upload to HuggingFace and delete from GitLab
* - `ng_delete_dataset_from_gitlab`
  - Delete dataset from GitLab registry
* - `ng_download_dataset_from_hf`
  - Download dataset from HuggingFace
```


