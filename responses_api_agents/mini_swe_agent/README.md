# Mini-SWE-Agent Environment

A NeMo Gym responses API agent that integrates the [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent) harness for evaluating language models on software engineering tasks using the SWE-Bench dataset.

## Table of Content
- [Mini-SWE-Agent Environment](#mini-swe-agent-environment)
  - [Table of Content](#table-of-content)
  - [Overview](#overview)
  - [Reward Profiling](#reward-profiling)
    - [Model - Qwen/Qwen3-Coder-30B-A3B-Instruct](#model---qwenqwen3-coder-30b-a3b-instruct)
  - [Dataset Information](#dataset-information)
  - [Configuration](#configuration)
    - [Agent Configuration](#agent-configuration)
  - [Usage](#usage)
    - [Server](#server)
    - [Trajectory Collection Script](#trajectory-collection-script)
  - [Contributing](#contributing)
  - [Licensing Information](#licensing-information)
    - [Dependencies](#dependencies)

## Overview

The Mini-SWE-Agent environment provides an interface for training models on solving real-world software engineering problems. 
It leverages the SWE-Gym dataset of GitHub issues and uses containerized environments (Docker/Singularity) to execute code modifications and validate solutions.

## Reward Profiling

### Model - Qwen/Qwen3-Coder-30B-A3B-Instruct
```md
Accuracy: 0.10
Resolved: 241
Total Instances: 2401
Average Turns: 34.45
Median Turns: 27
FAIL_TO_PASS_SUCCESS: 1675
FAIL_TO_PASS_FAILURE: 21706
PASS_TO_PASS_SUCCESS: 798361
PASS_TO_PASS_FAILURE: 919090
Average Tokens: 14181.87
Median Tokens: 12249
Max Tokens: 91690
Min Tokens: 2198
```
## Dataset Information

- Training data - [SWE-Gym/SWE-Gym](https://huggingface.co/datasets/SWE-Gym/SWE-Gym) contains 2438 instances sourced from 11 Python repos, following SWE-Bench data collection procedure.
- Validation data - [princeton-nlp/SWE-bench_Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified) SWE-bench Verified is a subset of 500 samples from the SWE-bench test set, which have been human-validated for quality. SWE-bench is a dataset that tests systems’ ability to solve GitHub issues automatically. See this post for more details on the human-validation process.

## Configuration

### Agent Configuration 

Path - `resources_servers/mini_swe_resource/configs/mini_swe_resource.yaml
```yaml
mini_swe_resource_resources_server:
  resources_servers:
    mini_swe_resource:
      entrypoint: app.py
mini_swe_main_agent:
  responses_api_agents:
    mini_swe_agent:
      entrypoint: app.py
      resources_server:
        type: resources_servers
        name: mini_swe_resource_resources_server
      model_server:
        type: responses_api_models
        name: openai_model
      datasets:
      - name: train
        type: train
        jsonl_fpath: resources_servers/mini_swe_resource/data/train.jsonl
        gitlab_identifier:
          dataset_name: mini_swe_agent
          version: 0.0.1
          artifact_fpath: train.jsonl
        license: MIT
      - name: validation
        type: validation
        jsonl_fpath: resources_servers/mini_swe_resource/data/validation.jsonl
        gitlab_identifier:
          dataset_name: mini_swe_agent
          version: 0.0.1
          artifact_fpath: validation.jsonl
        license: MIT
      - name: example
        type: example
        jsonl_fpath: resources_servers/mini_swe_resource/data/example.jsonl
      concurrency: 16 # number of instances to run concurrently
      env: singularity 
      cache_dir_template: ??? # The cache dir path where singularity images are stored
      run_golden: False # If set to true, run the golden patch
      step_timeout: 600 # Timeout for each agent step
      eval_timeout: 1800 # Timeout for running the evaluation (unit tests)
      skip_if_exists: False # If set to true, skip all instances already processed for the model
```


## Usage

### Server

```bash
# Download swe-gym data
ng_download_dataset_from_gitlab \
            +dataset_name=mini_swe_agent \
            +version=0.0.1 \
            +artifact_fpath=train.jsonl \
            +output_fpath=data/train.jsonl

# Start server
CONFIG_PATHS="resources_servers/mini_swe_resource/configs/mini_swe_resource.yaml,responses_api_models/openai_model/configs/openai_model.yaml"
ng_run +config_paths=[$CONFIG_PATHS] \
        '+mini_swe_main_agent.responses_api_agents.mini_swe_agent.cache_dir_template=/lustre/fsw/portfolios/llmservice/users/igitman/images/swe-bench/xingyaoww_sweb.eval.x86_64.\{instance_id\}.sif' \
        +mini_swe_main_agent.responses_api_agents.mini_swe_agent.run_golden=False \
        +mini_swe_main_agent.responses_api_agents.mini_swe_agent.skip_if_exists=True \
        +mini_swe_main_agent.responses_api_agents.mini_swe_agent.concurrency=16 \
        +mini_swe_main_agent.responses_api_agents.mini_swe_agent.step_timeout=300 \
        +mini_swe_main_agent.responses_api_agents.mini_swe_agent.eval_timeout=900 &

# Collect rollouts
ng_collect_rollouts +agent_name=mini_swe_main_agent \
            +input_jsonl_fpath=data/train.jsonl \
            +output_jsonl_fpath=results/mini_swe_agent_swe_gym.jsonl

# View trajectories
ng_viewer +jsonl_fpath=results/mini_swe_agent_swe_gym.jsonl
```

### Trajectory Collection Script
```bash
sbatch scripts/mini_swe_agent/trajectory_collection.slurm
```

## Contributing

Please refer to the main NeMo Gym documentation for contributing guidelines.

## Licensing Information

- **Code**: Apache 2.0
- **SWE-GYM**: MIT

### Dependencies
- **nemo_gym**: Apache 2.0
- **mini-swe-agent**: MIT
- **SWE-Bench-Package**: MIT
