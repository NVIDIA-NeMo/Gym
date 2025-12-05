(prepare-training-data)=

# Prepare Training Data

Use `ng_prepare_data` to validate, format, and prepare datasets for NeMo RL or other training frameworks.

---

## How It Works

The `ng_prepare_data` command:

1. Validates dataset format and schema
2. Computes aggregate statistics
3. Adds `agent_ref` routing information for Gym
4. Collates datasets into train/validation splits

---

## Dataset Configuration

When you initialize a resource server, the config includes dataset definitions:

```yaml
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: example_multi_step_resources_server
      model_server:
        type: responses_api_models
        name: policy_model
      datasets:
      - name: train
        type: train
        license: Apache 2.0
        jsonl_fpath: resources_servers/example_multi_step/data/train.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: example_multi_step
          version: 0.0.1
          artifact_fpath: example_multi_step/train.jsonl
      - name: validation
        type: validation
        license: Apache 2.0
        jsonl_fpath: resources_servers/example_multi_step/data/validation.jsonl
        num_repeats: 1
        gitlab_identifier:
          dataset_name: example_multi_step
          version: 0.0.1
          artifact_fpath: example_multi_step/validation.jsonl
      - name: example
        type: example
        jsonl_fpath: resources_servers/example_multi_step/data/example.jsonl
        num_repeats: 1
```

### Dataset Object Schema

```{list-table}
:header-rows: 1
:widths: 20 15 65

* - Field
  - Required
  - Description
* - `name`
  - Yes
  - Identifier for the dataset
* - `type`
  - Yes
  - `train`, `validation`, or `example`
* - `jsonl_fpath`
  - Yes
  - Local path to JSONL file
* - `num_repeats`
  - No
  - Repeat each row N times (default: 1)
* - `license`
  - Train/Val
  - Dataset license (required for train/validation)
* - `gitlab_identifier`
  - Train/Val
  - Remote path in GitLab registry
* - `start_idx`
  - No
  - Slice start index
* - `end_idx`
  - No
  - Slice end index
```

### Dataset Types

- **train**: Training data for NeMo RL or other frameworks
- **validation**: Validation data for evaluation during training
- **example**: First 5 rows of train data for sanity checks (required for PRs)

---

## Preparation Modes

### Example Validation Mode

Use for PR submissions and format sanity checks:

```bash
config_paths="resources_servers/example_multi_step/configs/example_multi_step.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/example_multi_step \
    +mode=example_validation
```

### Train Preparation Mode

Use when actually preparing data for training:

```bash
config_paths="resources_servers/math_with_judge/configs/bytedtsinghua_dapo17k.yaml,\
responses_api_models/openai_model/configs/openai_model.yaml"

ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/bytedtsinghua_dapo17k \
    +mode=train_preparation \
    +should_download=true
```

---

## What ng_prepare_data Does

1. **Load datasets** — Attempts to load all specified datasets from disk. Reports missing files before processing.

2. **Validate format** — Reads each example and validates against Responses API schema. Reports file paths and indices of invalid examples.
   - Only requirement: each example has a `responses_create_params` key with valid Responses API schema

3. **Compute statistics** — Calculates and saves aggregate statistics:
   - Number of examples
   - Avg/max/min number of tools
   - Input length (OpenAI tokens)
   - Avg/max/min number of turns
   - Number of unique create params
   - Avg/max/min temperature and sampling params
   - Number of unique user messages

4. **Verify consistency** — Checks that statistics match any existing aggregate statistics files.

5. **Collate datasets** — Combines all datasets into final train and validation JSONL files at the output directory.

6. **Add routing info** — Adds `agent_ref` property to each example, telling Gym which agent server handles that example.

---

## Parameters

```{list-table}
:header-rows: 1
:widths: 30 70

* - Parameter
  - Description
* - `+config_paths`
  - Comma-separated list of config files (same configs used for `ng_run`)
* - `+output_dirpath`
  - Directory to save prepared data
* - `+mode`
  - `example_validation` or `train_preparation`
* - `+should_download`
  - Download dataset if not present locally (default: false)
```

---

## Integration with Training

The prepared datasets work directly with NeMo RL:

```bash
# Prepare data
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=data/my_dataset \
    +mode=train_preparation

# Use same config paths for training
# The output train/validation files are ready for NeMo RL consumption
```

:::{tip}
Use the **same config paths** for `ng_prepare_data` and your training framework. There's no special pre/post processing — what you see is what you get.
:::


