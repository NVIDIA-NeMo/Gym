# Description

Search-augmented QA evaluation with 6 Perplexity datasets using Perplexity Search API and LLM-as-a-judge.

## Task & Dataset Overview
Pre-baked trajectory datasets (model receives full trajectory, generates summary with `tool_choice=none`):

| Dataset | Eval Focus | Judge | max_tool_calls | Trajectory Source |
|---------|-----------|-------|----------------|-------------------|
| `perplexity_user_if` | Instruction following | LLM | 0 | GPT-5.1 |
| `perplexity_search` | Search quality | Reward Model* | 0 | Qwen3 (pplx-internal) |
| `perplexity_chat` | Chat quality (no tools) | Reward Model* | 0 | Qwen3 (pplx-internal) |
| `perplexity_abstention` | Abstention behavior | LLM | 0 | GPT-5.1 |

Fresh rollout datasets (model starts from scratch, makes its own tool calls):

| Dataset | Eval Focus | Judge | max_tool_calls |
|---------|-----------|-------|----------------|
| `perplexity_frames` | Multi-hop reasoning | LLM | 3 |
| `perplexity_facts_grounding` | Factual grounding | LLM | 3 |

*Reward model judge is stubbed (`NotImplementedError`). Use `judge_type: llm` until implemented.

## Agent configuration:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tool_calls` | None | Tool calls before forcing text response via `tool_choice=none`. None = unlimited. |
| `temperature` | None | Overrides JSONL value when set |
| `top_p` | None | Overrides JSONL value when set |
| `max_output_tokens` | None | Overrides JSONL value when set |

Recommended policy hparams: `temperature=1.0`, `top_p=1.0`, `max_output_tokens=32768`.

## Judge configuration:

| Datasets | Reference Judge | Hparams |
|----------|----------------|---------|
| user_if, abstention | gpt-5.1 (Responses API) | reasoning_effort=medium, max_output_tokens=32768, temp/top_p=default |
| frames, facts_grounding | gpt-4.1 (Chat Completions) | temperature=0.0, max_output_tokens=8192, no reasoning |
| search, chat | Reward model (stubbed) | N/A |

Judge prompts use free-text output: "followed: yes/no" for IF datasets, "correct: yes/no" for correctness datasets.

Resource server hparams: `search_max_concurrency=20`, `search_rate_limit_qps=45`.

## Dataset creation
```bash
# Preprocess raw datasets to Gym format
python resources_servers/perplexity_summarizer/preprocess_to_gym.py \
    --input /path/to/raw.jsonl --output /path/to/output.jsonl \
    --dataset_name perplexity_user_if

# Upload to GitLab
ng_upload_dataset_to_gitlab \
    +dataset_name=perplexity_user_if \
    +version=0.0.1 \
    +input_jsonl_fpath=resources_servers/perplexity_summarizer/data/perplexity_user_if.jsonl
```

## Usage
```bash
# Put your API keys in env.yaml
echo "perplexity_api_key: pplx-xxx
policy_base_url: https://inference-api.nvidia.com/v1
policy_api_key: your-key
policy_model_name: your-model
judge_base_url: https://inference-api.nvidia.com/v1
judge_api_key: your-key
judge_model_name: openai/gpt-5-nano" > env.yaml

# Download data
config_paths="resources_servers/perplexity_summarizer/configs/perplexity_summarizer_user_if.yaml"
ng_prepare_data "+config_paths=[$config_paths]" \
    +output_dirpath=resources_servers/perplexity_summarizer/data \
    +mode=train_preparation +should_download=true +data_source=gitlab

# Spin up servers (include model config for policy_model)
ng_run "+config_paths=[$config_paths,responses_api_models/vllm_model/configs/vllm_model.yaml]"

# Collect rollouts
ng_collect_rollouts +agent_name=perplexity_summarizer \
    +input_jsonl_fpath=resources_servers/perplexity_summarizer/data/perplexity_user_if_example.jsonl \
    +output_jsonl_fpath=results/rollouts.jsonl +num_repeats=1
```

# Licensing information
Code: LicenseRef-NvidiaProprietary
Data: NVIDIA Internal Use Only, Do Not Distribute

Dependencies
- nemo_gym: Apache 2.0
- perplexityai: MIT
