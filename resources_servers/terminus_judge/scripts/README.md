# Terminus Judge HF Data Pipeline

Use a single entrypoint:

- `prepare.py` for conversion, split, and smoke rollout collection.

This pipeline uses the public `open-thoughts/OpenThoughts-Agent-v1-SFT` dataset as an example input and can be adapted to similar trajectory-format datasets.

## Prerequisites

1. Python env with:
   - `datasets`
   - `openapi-schema-validator`
2. For smoke stage:
   - `gym env start`, `gym env status`, `gym eval run --no-serve` on `PATH`
   - reachable policy/model endpoint
3. If HF cache is read-only:
   - `HF_HOME=/tmp/hf_home`
   - `HF_DATASETS_CACHE=/tmp/hf_home/datasets`

## Stage 1: Convert HF Trajectories to Samples

Local parquet:

```bash
python resources_servers/terminus_judge/scripts/prepare.py convert \
  --hf_parquet_glob "datasets/openthoughts/data/*.parquet" \
  --split train \
  --dataset_name openthoughts_agent_v1_sft \
  --output_dir resources_servers/terminus_judge/data/openthoughts_agent_v1_sft \
  --max_rows 50 \
  --threshold 0.95
```

HF Hub streaming:

```bash
python resources_servers/terminus_judge/scripts/prepare.py convert \
  --hf_dataset open-thoughts/OpenThoughts-Agent-v1-SFT \
  --split train \
  --dataset_name openthoughts_agent_v1_sft \
  --output_dir resources_servers/terminus_judge/data/openthoughts_agent_v1_sft \
  --max_rows 50 \
  --threshold 0.95
```

## Stage 2: Deduplicated Stratified Split

```bash
python resources_servers/terminus_judge/scripts/prepare.py split \
  --input resources_servers/terminus_judge/data/openthoughts_agent_v1_sft/samples.jsonl \
  --output_dir resources_servers/terminus_judge/data/openthoughts_agent_v1_sft \
  --train_size 0 \
  --val_per_bucket 5 \
  --max_per_group 50 \
  --seed 42
```

## Stage 3: Smoke Rollout Collection

By default this uses the canonical 5-row examples already committed in:

- `resources_servers/terminus_judge/data/example.jsonl`
- `resources_servers/terminus_judge/data/example_rollouts.jsonl`

```bash
python resources_servers/terminus_judge/scripts/prepare.py smoke \
  --policy_base_url "http://<model-host>:<port>/v1" \
  --policy_api_key "<key>" \
  --policy_model_name "<model-name>"
```
