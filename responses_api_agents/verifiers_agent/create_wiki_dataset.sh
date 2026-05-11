#!/usr/bin/env bash
set -euo pipefail

# Load project secrets so NVIDIA_API_KEY is in os.environ when wiki_search reads it.
set -a
source ~/Gym/.env
set +a

# Mirror the kwargs in configs/wiki-search.yaml's vf_env_args.
# *_api_key_var holds the NAME of the env var that contains the key (read via os.environ[...]).
VF_ENV_ARGS='{
  "judge_model": "gcp/google/gemini-3.1-flash-lite-preview",
  "judge_base_url": "https://inference-api.nvidia.com/v1",
  "judge_api_key_var": "NVIDIA_API_KEY",
  "embed_model": "azure/openai/text-embedding-3-small",
  "embed_base_url": "https://inference-api.nvidia.com/v1",
  "embed_api_key_var": "NVIDIA_API_KEY"
}'

# Full dataset
# python3 scripts/create_dataset.py \
#     --env-id wiki-search \
#     --env-args "$VF_ENV_ARGS" \
#     --output data/wiki-search.jsonl

# # Mock training set (total datset has 478 examples, fixed seed)
# python3 scripts/create_dataset.py \
#     --env-id wiki-search \
#     --env-args "$VF_ENV_ARGS" \
#     --size 300 \
#     --seed 42 \
#     --output data/wiki-search-mock-train.jsonl

# # Mock validation set (total datset has 478 examples, fixed seed)
python3 scripts/create_dataset.py \
    --env-id wiki-search \
    --env-args "$VF_ENV_ARGS" \
    --size 50 \
    --seed 123 \
    --output data/wiki-search-mock-val.jsonl

# Tiny smoke set (3 examples)
# python3 scripts/create_dataset.py \
#     --env-id wiki-search \
#     --env-args "$VF_ENV_ARGS" \
#     --size 3 \
#     --seed 123 \
#     --output data/wiki-search-mock.jsonl
