# CLI Commands

This page documents all available NeMo Gym CLI commands.

:::{note}
Each command has both a short form (such as `ng_run`) and a full form (such as `nemo_gym_run`). They are functionally identical.
:::

## Quick Reference

```bash
# Display help
ng_help

# Get detailed help for any command
ng_run +help=true
ng_test +h=true
```

---

## Server Management

Commands for running, testing, and managing NeMo Gym servers.

### `ng_run` / `nemo_gym_run`

Start NeMo Gym servers for agents, models, and resources.

This command reads configuration from YAML files specified via `+config_paths` and starts all configured servers. The configuration files should define server instances with their entrypoints and settings.

**Configuration Parameter**

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Parameter
  - Type
  - Description
* - `config_paths`
  - List[str]
  - Paths to YAML configuration files. Specify using Hydra: `+config_paths="[file1.yaml,file2.yaml]"`
```

**Example**

```bash
# Start servers with specific configs
config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_run "+config_paths=[${config_paths}]"
```

---

### `ng_test` / `nemo_gym_test`

Test a specific server module by running its pytest suite and optionally validating example data.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 25 10 65

* - Parameter
  - Type
  - Description
* - `entrypoint`
  - str
  - Entrypoint for this command. Must be a relative path with two parts (such as `responses_api_agents/simple_agent`).
* - `should_validate_data`
  - bool
  - Whether to validate the example data (examples, metrics, rollouts, and so on) for this server. Default: `False`.
```

**Example**

```bash
ng_test +entrypoint=resources_servers/example_single_tool_call
```

---

### `ng_test_all` / `nemo_gym_test_all`

Run tests for all server modules in the project.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 35 10 55

* - Parameter
  - Type
  - Description
* - `fail_on_total_and_test_mismatch`
  - bool
  - Fail if the number of server modules does not match the number with tests. Default: `False`.
```

**Example**

```bash
ng_test_all
```

---

### `ng_dev_test` / `nemo_gym_dev_test`

Run core NeMo Gym tests with coverage reporting. Runs pytest with the `--cov` flag.

**Example**

```bash
ng_dev_test
```

---

### `ng_init_resources_server` / `nemo_gym_init_resources_server`

Initialize a new resources server with template files and directory structure.

**Example**

```bash
ng_init_resources_server +entrypoint=resources_servers/my_server
```

---

## Data Collection

Commands for collecting verified rollouts for RL training.

### `ng_collect_rollouts` / `nemo_gym_collect_rollouts`

Perform a batch of rollout collection.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Type
  - Description
* - `agent_name`
  - str
  - The agent to collect rollouts from.
* - `input_jsonl_fpath`
  - str
  - The input data source to use to collect rollouts, in the form of a file path to a JSONL file.
* - `output_jsonl_fpath`
  - str
  - The output data JSONL file path.
* - `limit`
  - Optional[int]
  - Maximum number of examples to load and take from the input dataset.
* - `num_repeats`
  - Optional[int]
  - The number of times to repeat each example to run. Useful if you want to calculate mean@k, such as mean@4 or mean@16.
* - `num_samples_in_parallel`
  - Optional[int]
  - Limit the number of concurrent samples running at once.
* - `responses_create_params`
  - Dict
  - Overrides for the `responses_create_params`, such as `temperature` and `max_output_tokens`.
```

**Example**

```bash
ng_collect_rollouts \
    +agent_name=example_single_tool_call_simple_agent \
    +input_jsonl_fpath=weather_query.jsonl \
    +output_jsonl_fpath=weather_rollouts.jsonl \
    +limit=100 \
    +num_repeats=4 \
    +num_samples_in_parallel=10
```

---

## Data Management

Commands for preparing and viewing training data.

### `ng_prepare_data` / `nemo_gym_prepare_data`

Prepare and validate training data, generating metrics and statistics for datasets.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 20 30 50

* - Parameter
  - Type
  - Description
* - `output_dirpath`
  - str
  - Directory path where processed datasets and metrics will be saved.
* - `mode`
  - Literal["train_preparation", "example_validation"]
  - Processing mode. Use `train_preparation` to prepare train and validation datasets for training, or `example_validation` to validate example data for PR submission.
* - `should_download`
  - bool
  - Whether to automatically download missing datasets from remote registries. Default: `False`.
```

**Example**

```bash
config_paths="resources_servers/example_multi_step/configs/example_multi_step.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"
ng_prepare_data "+config_paths=[${config_paths}]" \
    +output_dirpath=data/example_multi_step \
    +mode=example_validation
```

---

### `ng_viewer` / `nemo_gym_viewer`

Launch a Gradio interface to view and explore dataset rollouts interactively.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 20 10 70

* - Parameter
  - Type
  - Description
* - `jsonl_fpath`
  - str
  - Filepath to a local JSONL file to view.
* - `server_host`
  - str
  - Network address where the viewer accepts requests. Defaults to `"127.0.0.1"` (localhost only). Set to `"0.0.0.0"` to accept requests from anywhere.
* - `server_port`
  - int
  - Port where the viewer accepts requests. Defaults to `7860`. If the specified port is unavailable, Gradio will search for the next available port.
```

**Examples**

```bash
# Launch viewer with default settings (accessible from localhost only)
ng_viewer +jsonl_fpath=weather_rollouts.jsonl

# Accept requests from anywhere (e.g., for remote access)
ng_viewer +jsonl_fpath=weather_rollouts.jsonl +server_host=0.0.0.0

# Use a custom port
ng_viewer +jsonl_fpath=weather_rollouts.jsonl +server_port=8080
```

---

## Dataset Registry - GitLab

Commands for uploading, downloading, and managing datasets in GitLab Model Registry.

### `ng_upload_dataset_to_gitlab` / `nemo_gym_upload_dataset_to_gitlab`

Upload a local JSONL dataset artifact to GitLab.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 20 10 70

* - Parameter
  - Type
  - Description
* - `dataset_name`
  - str
  - The dataset name.
* - `version`
  - str
  - The version of this dataset. Must be in the format `x.x.x`.
* - `input_jsonl_fpath`
  - str
  - Path to the JSONL file to upload.
```

**Example**

```bash
ng_upload_dataset_to_gitlab \
    +dataset_name=example_multi_step \
    +version=0.0.1 \
    +input_jsonl_fpath=data/train.jsonl
```

---

### `ng_download_dataset_from_gitlab` / `nemo_gym_download_dataset_from_gitlab`

Download a JSONL dataset from GitLab Model Registry.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 20 10 70

* - Parameter
  - Type
  - Description
* - `dataset_name`
  - str
  - The dataset name.
* - `version`
  - str
  - The version of this dataset. Must be in the format `x.x.x`.
* - `artifact_fpath`
  - str
  - The filepath to the artifact to download.
* - `output_fpath`
  - str
  - Path where the downloaded dataset will be saved.
```

**Example**

```bash
ng_download_dataset_from_gitlab \
    +dataset_name=example_multi_step \
    +version=0.0.1 \
    +artifact_fpath=train.jsonl \
    +output_fpath=data/train.jsonl
```

---

### `ng_delete_dataset_from_gitlab` / `nemo_gym_delete_dataset_from_gitlab`

Delete a dataset from GitLab Model Registry. Prompts for confirmation.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 20 10 70

* - Parameter
  - Type
  - Description
* - `dataset_name`
  - str
  - Name of the dataset to delete from GitLab.
```

**Example**

```bash
ng_delete_dataset_from_gitlab +dataset_name=old_dataset
```

---

## Dataset Registry - HuggingFace

Commands for uploading and downloading datasets to/from HuggingFace Hub.

### `ng_upload_dataset_to_hf` / `nemo_gym_upload_dataset_to_hf`

Upload a JSONL dataset to HuggingFace Hub with optional GitLab deletion after successful upload.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 25 15 60

* - Parameter
  - Type
  - Description
* - `hf_token`
  - str
  - HuggingFace API token for authentication.
* - `hf_organization`
  - str
  - HuggingFace organization name where the dataset will be uploaded.
* - `hf_collection_name`
  - str
  - HuggingFace collection name for organizing datasets.
* - `hf_collection_slug`
  - str
  - Alphanumeric collection slug found at the end of the collection URI.
* - `dataset_name`
  - str
  - Name of the dataset. Will be combined with domain and resource server name.
* - `input_jsonl_fpath`
  - str
  - Path to the local JSONL file to upload.
* - `resource_config_path`
  - str
  - Path to resource server config file. Used to extract domain for naming convention.
* - `hf_dataset_prefix`
  - str
  - Prefix prepended to dataset name. Default: `NeMo-Gym`.
* - `delete_from_gitlab`
  - Optional[bool]
  - Delete the dataset from GitLab after successful upload to HuggingFace. Default: `False`.
```

**Example**

```bash
resource_config_path="resources_servers/example_multi_step/configs/example_multi_step.yaml"
ng_upload_dataset_to_hf \
    +dataset_name=my_dataset \
    +input_jsonl_fpath=data/train.jsonl \
    +resource_config_path=${resource_config_path} \
    +delete_from_gitlab=true
```

---

### `ng_download_dataset_from_hf` / `nemo_gym_download_dataset_from_hf`

Download a JSONL dataset from HuggingFace Hub to local filesystem.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 20 10 70

* - Parameter
  - Type
  - Description
* - `output_fpath`
  - str
  - Local file path where the downloaded dataset will be saved.
* - `hf_token`
  - str
  - HuggingFace API token for authentication.
* - `artifact_fpath`
  - str
  - Name of the artifact file to download from the repository.
* - `repo_id`
  - str
  - HuggingFace repository ID in format `organization/dataset-name`.
```

**Example**

```bash
ng_download_dataset_from_hf \
    +repo_id=NVIDIA/NeMo-Gym-Math-example_multi_step-v1 \
    +artifact_fpath=train.jsonl \
    +output_fpath=data/train.jsonl
```

---

### `ng_gitlab_to_hf_dataset` / `nemo_gym_gitlab_to_hf_dataset`

Upload a JSONL dataset to HuggingFace Hub and automatically delete from GitLab after successful upload.

This command always deletes the dataset from GitLab after uploading to HuggingFace. Use `ng_upload_dataset_to_hf` if you want optional deletion control.

**Parameters**

Same as `ng_upload_dataset_to_hf` but `delete_from_gitlab` is not available. This command always deletes.

**Example**

```bash
resource_config_path="resources_servers/example_multi_step/configs/example_multi_step.yaml"
ng_gitlab_to_hf_dataset \
    +dataset_name=my_dataset \
    +input_jsonl_fpath=data/train.jsonl \
    +resource_config_path=${resource_config_path}
```

---

## Configuration & Help

Commands for debugging configuration and getting help.

### `ng_dump_config` / `nemo_gym_dump_config`

Display the resolved Hydra configuration for debugging purposes.

**Example**

```bash
ng_dump_config "+config_paths=[<config1>,<config2>]"
```

---

### `ng_help` / `nemo_gym_help`

Display a list of available NeMo Gym CLI commands.

**Example**

```bash
ng_help
```

---

### `ng_version` / `nemo_gym_version`

Display NeMo Gym version and system information.

**Parameters**

```{list-table}
:header-rows: 1
:widths: 20 10 70

* - Parameter
  - Type
  - Description
* - `json_format`
  - bool
  - Output in JSON format for programmatic use. Default: `False`. Can be specified with `+json=true`.
```

**Example**

```bash
# Display version information
ng_version

# Output as JSON
ng_version +json=true
```

---

## Getting Help

For detailed help on any command, run it with `+help=true` or `+h=true`:

```bash
ng_run +help=true
ng_collect_rollouts +h=true
```

This will display all available configuration parameters and their descriptions.
