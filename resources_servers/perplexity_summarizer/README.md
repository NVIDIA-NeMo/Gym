# Description

Search-augmented QA evaluation with 7 Perplexity datasets using Perplexity Search API and LLM-as-a-judge.

## Task & Dataset Overview

Pre-baked trajectory datasets (model receives full trajectory, generates summary with `tool_choice=none`):


| Dataset                          | Eval Focus              | Judge         | max_tool_calls | Trajectory Source     |
| -------------------------------- | ----------------------- | ------------- | -------------- | --------------------- |
| `perplexity_user_if`             | Instruction following   | LLM (IF)      | 0              | GPT-5.1               |
| `perplexity_abstention`          | Abstention behavior     | LLM (IF)      | 0              | GPT-5.1               |
| `perplexity_language_mismatch`   | Language consistency    | LLM (IF)      | 0              | GPT-5.1               |
| `perplexity_search`              | Search quality          | Reward Model* | 0              | Qwen3 (pplx-internal) |
| `perplexity_chat`                | Chat quality (no tools) | Reward Model* | 0              | Qwen3 (pplx-internal) |


Fresh rollout datasets (model starts from scratch, makes its own tool calls):


| Dataset                      | Eval Focus          | Judge | max_tool_calls |
| ---------------------------- | ------------------- | ----- | -------------- |
| `perplexity_frames`          | Multi-hop reasoning | LLM   | 3              |
| `perplexity_facts_grounding` | Factual grounding   | LLM   | 3              |


*Reward model judge is stubbed (`NotImplementedError`). Use `judge_type: llm` until implemented.

## Agent configuration:


| Parameter           | Default | Description                                                                                                                                                                                                                                   |
| ------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `max_tool_calls`    | None    | Tool calls before forcing text response via `tool_choice=none`. None = unlimited.                                                                                                                                                             |
| `temperature`       | None    | Overrides JSONL value when set                                                                                                                                                                                                                |
| `top_p`             | None    | Overrides JSONL value when set                                                                                                                                                                                                                |
| `max_output_tokens` | None    | Overrides JSONL value when set                                                                                                                                                                                                                |
| `bad_words`         | None    | Token strings to suppress via vLLM when `tool_choice="none"`. Injected by the agent only when forcing text response (after `max_tool_calls` reached). Lives on agent config, not model server, because it's conditional on tool_choice state. |


Recommended policy hparams: `temperature=0.6`, `top_p=1.0`, `max_output_tokens=8192`.

Thinking mode is controlled via the model server config, not the agent:

```yaml
policy_model:
  responses_api_models:
    vllm_model:
      chat_template_kwargs:
        enable_thinking: false  # or true
```

CLI override: `"++policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=true"`

## Judge configuration:


| Datasets                                  | Reference Judge                  | Hparams                                                                |
| ----------------------------------------- | -------------------------------- | ---------------------------------------------------------------------- |
| user_if, abstention, language_mismatch    | gpt-5.1 (Responses API)         | is_responses_native=true, reasoning_effort=medium, max_output_tokens=32768 |
| frames, facts_grounding                   | gpt-4.1 (Chat Completions)      | is_responses_native=false, temperature=0.0, seed=0 (via extra_body)    |
| search, chat                              | Reward model (stubbed)           | N/A                                                                    |


Judge prompts use free-text output: "followed: yes/no" for IF datasets, "correct: yes/no" for correctness datasets.

Resource server hparams: `search_max_concurrency=20`, `search_rate_limit_qps=45`.

## Dataset creation

```bash
# Preprocess raw datasets to Gym format
python resources_servers/perplexity_summarizer/preprocess_to_gym.py \
    --input /path/to/raw.jsonl --output /path/to/output.jsonl \
    --dataset_name perplexity_user_if

# Available dataset names:
# perplexity_user_if, perplexity_abstention, perplexity_language_mismatch,
# perplexity_search, perplexity_chat, perplexity_frames, perplexity_facts_grounding

# Upload to GitLab
ng_upload_dataset_to_gitlab \
    +dataset_name=perplexity_user_if \
    +version=0.0.4 \
    +input_jsonl_fpath=resources_servers/perplexity_summarizer/data/perplexity_user_if.jsonl

# Current dataset versions:
# abstention v0.0.4 | user_if v0.0.4 | language_mismatch v0.0.1
# facts_grounding v0.0.5 | frames v0.0.5
```

## Usage

### env.yaml setup

All credentials and endpoints are configured via `env.yaml` at the nemo-gym repo root:

```yaml
# Model endpoint (vLLM)
policy_base_url: https://your-vllm-endpoint/v1
policy_api_key: your-api-key
policy_model_name: your-model-name

# Judge endpoint (NV Inference)
judge_base_url: https://inference-api.nvidia.com/v1
judge_api_key: your-nv-inference-api-key
judge_model_name: openai/openai/gpt-5.1  # IF judge; override to us/azure/openai/gpt-4.1 for correctness judge

# Perplexity Search via NV Inference (required for FACTS/FRAMES fresh rollout)
perplexity_api_key: your-nv-inference-api-key  # same key as judge_api_key
pplx_search_url: https://inference-api.nvidia.com/v1/search/perplexity-search

# W&B experiment tracking
wandb_api_key: your-wandb-api-key
wandb_project: your-wandb-project

# GitLab dataset registry (only needed for ng_upload/download_dataset_to_gitlab)
mlflow_tracking_uri: https://gitlab-master.nvidia.com/api/v4/projects/YOUR_PROJECT_ID/ml/mlflow
mlflow_tracking_token: your-gitlab-api-token
```

The `perplexity_api_key` and `pplx_search_url` are used by FACTS and FRAMES configs (`use_nv_search: true`). Pre-baked datasets (abstention, user_if, language_mismatch) do not use search — they have search results already embedded in the JSONL.

### NV Inference Perplexity Search

Fresh rollout datasets (FACTS, FRAMES) require a search API. The configs use NV Inference's Perplexity Search endpoint:

- `use_nv_search: true` in the YAML config enables the NV Inference path
- `pplx_search_url` in env.yaml points to the search endpoint
- `perplexity_api_key` in env.yaml authenticates requests (same key as the judge)

When `use_nv_search: false` (default), the server uses the `perplexityai` Python SDK instead.

### Running pre-baked trajectory evals (abstention, user_if, language_mismatch)

```bash
config_paths="responses_api_agents/perplexity_summarizer_agent/configs/perplexity_summarizer_agent.yaml,resources_servers/perplexity_summarizer/configs/perplexity_summarizer_abstention.yaml"

# Download data from GitLab registry
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=resources_servers/perplexity_summarizer/data \
    +mode=train_preparation +should_download=true +data_source=gitlab

# Spin up servers
ng_run "+config_paths=[$config_paths]"

# Collect rollouts (smoke test with example data)
ng_collect_rollouts +agent_name=perplexity_summarizer \
    +input_jsonl_fpath=resources_servers/perplexity_summarizer/data/perplexity_abstention_example.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl +num_repeats=1
```

### Running fresh rollout evals (FACTS, FRAMES)

Fresh rollout datasets require the correctness judge (gpt-4.1) instead of the default IF judge (gpt-5.1). Override `judge_model_name` at runtime:

```bash
config_paths="responses_api_agents/perplexity_summarizer_agent/configs/perplexity_summarizer_agent.yaml,resources_servers/perplexity_summarizer/configs/perplexity_summarizer_facts_grounding.yaml"

ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    ++output_jsonl_fpath=results/facts_grounding.jsonl \
    ++split=validation \
    ++num_repeats=1 \
    ++num_samples_in_parallel=4 \
    ++judge_model_name=us/azure/openai/gpt-4.1
```

## E2E Rollout Collection with W&B

Uses `policy_base_url`, `policy_api_key`, and `policy_model_name` from `env.yaml` by default.

```bash
WANDB_PROJECT=my-project
EXPERIMENT_NAME=perplexity_summarizer/model-name
config_paths="responses_api_agents/perplexity_summarizer_agent/configs/perplexity_summarizer_agent.yaml,resources_servers/perplexity_summarizer/configs/perplexity_summarizer_abstention.yaml"

ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    +wandb_project=$WANDB_PROJECT \
    +wandb_name=$EXPERIMENT_NAME \
    ++output_jsonl_fpath=results/$EXPERIMENT_NAME.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++resume_from_cache=true \
    ++split=validation \
    ++num_repeats=3 \
    ++num_samples_in_parallel=4
```

To override model endpoint and thinking mode:

```bash
ng_e2e_collect_rollouts \
    "+config_paths=[${config_paths}]" \
    +wandb_project=$WANDB_PROJECT \
    +wandb_name=$EXPERIMENT_NAME \
    ++output_jsonl_fpath=results/$EXPERIMENT_NAME.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++resume_from_cache=true \
    ++split=validation \
    ++num_repeats=3 \
    ++num_samples_in_parallel=4 \
    ++policy_base_url=https://your-vllm-endpoint/v1 \
    ++policy_api_key=your-api-key \
    ++policy_model_name=your-model-name \
    ++policy_model.responses_api_models.vllm_model.chat_template_kwargs.enable_thinking=true
```

Key parameters:
- `num_repeats`: Number of independent runs per example (use 3 for reproducible results)
- `num_samples_in_parallel`: Concurrency (reduce to 4 if endpoints are rate-limited)
- `enable_thinking`: Set to `true` or `false` to control thinking mode (default: `false` in agent config)

# Licensing information

Code: LicenseRef-NvidiaProprietary
Data: NVIDIA Internal Use Only, Do Not Distribute

Dependencies

- nemo_gym: Apache 2.0
- perplexityai: MIT

