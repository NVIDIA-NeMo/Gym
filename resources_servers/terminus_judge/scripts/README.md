# Terminus Judge HF Data Pipeline

This directory contains a self-contained 3-stage pipeline for building
`terminus_judge` training/eval data from HF trajectory datasets.

## Files

- `convert_hf_traj_to_samples.py`: Stage 1 conversion from trajectories to per-turn samples.
- `make_train_validation_stratified.py`: Stage 2 deduplicated, stratified split builder.
- `run_smoke_rollouts.sh`: Stage 3 local rollout smoke runner (`ng_run` + `ng_collect_rollouts`).

## Prerequisites

1. Run with a Python environment that has:
   - `datasets`
   - `openapi-schema-validator`
2. If your default HF cache path is read-only, set writable cache env vars:
   - `HF_HOME=/tmp/hf_home`
   - `HF_DATASETS_CACHE=/tmp/hf_home/datasets`
3. For Stage 3, ensure `ng_run`, `ng_status`, and `ng_collect_rollouts` are available on `PATH`.
4. Stage 3 also requires a reachable policy model endpoint (or equivalent model-server config).
5. If Ray startup fails with `pthread_create ... Resource temporarily unavailable`,
   set low-thread Ray env vars before Stage 3:
   - `RAY_num_server_call_thread=1`
   - `RAY_num_grpc_internal_threads=1`
   - `RAY_enable_worker_prestart=0`
   - `RAY_worker_maximum_startup_concurrency=1`
   - `RAY_gcs_server_rpc_server_thread_num=1`
   - `RAY_gcs_server_rpc_client_thread_num=1`

## Stage 1: Convert Trajectories To Samples

Example using local parquet:

```bash
HF_HOME=/tmp/hf_home \
HF_DATASETS_CACHE=/tmp/hf_home/datasets \
python resources_servers/terminus_judge/scripts/convert_hf_traj_to_samples.py \
  --hf_parquet_glob "datasets/openthoughts/data/*.parquet" \
  --split train \
  --dataset_name openthoughts_agent_v1_sft \
  --output_dir resources_servers/terminus_judge/data/openthoughts_agent_v1_sft \
  --max_rows 50 \
  --threshold 0.95
```

Example using HF Hub streaming:

```bash
HF_HOME=/tmp/hf_home \
HF_DATASETS_CACHE=/tmp/hf_home/datasets \
python resources_servers/terminus_judge/scripts/convert_hf_traj_to_samples.py \
  --hf_dataset open-thoughts/OpenThoughts-Agent-v1-SFT \
  --split train \
  --dataset_name openthoughts_agent_v1_sft \
  --output_dir resources_servers/terminus_judge/data/openthoughts_agent_v1_sft \
  --max_rows 50 \
  --threshold 0.95
```

Output:

- `resources_servers/terminus_judge/data/openthoughts_agent_v1_sft/samples.jsonl`

## Stage 2: Build Deduplicated Stratified Train/Validation Splits

```bash
python resources_servers/terminus_judge/scripts/make_train_validation_stratified.py \
  --input resources_servers/terminus_judge/data/openthoughts_agent_v1_sft/samples.jsonl \
  --output_dir resources_servers/terminus_judge/data/openthoughts_agent_v1_sft \
  --train_size 0 \
  --val_per_bucket 5 \
  --max_per_group 50 \
  --seed 42
```

Outputs:

- `resources_servers/terminus_judge/data/openthoughts_agent_v1_sft/train.jsonl`
- `resources_servers/terminus_judge/data/openthoughts_agent_v1_sft/validation.jsonl`

## Stage 3: Smoke Rollout Collection

```bash
POLICY_BASE_URL="http://<model-host>:<port>/v1" \
POLICY_API_KEY="<key>" \
POLICY_MODEL_NAME="<model-name>" \
bash resources_servers/terminus_judge/scripts/run_smoke_rollouts.sh
```

By default this:

1. Copies first `SMOKE_LIMIT` rows from validation to smoke input.
2. Starts `ng_run`.
3. Waits for 3 healthy services in `ng_status`.
4. Runs `ng_collect_rollouts`.

Outputs:

- `resources_servers/terminus_judge/data/openthoughts_agent_v1_sft/smoke/smoke_input.jsonl`
- `resources_servers/terminus_judge/data/openthoughts_agent_v1_sft/smoke/smoke_rollouts.jsonl`

## Notes

- The provided `dev/Gym/AGENTS.md` ends at `Sampling:` in Stage 2. For this implementation,
  Stage 2 uses:
  - fixed-per-bucket validation sampling (`--val_per_bucket`)
  - remaining rows for train when `--train_size 0`
  - otherwise proportional stratified sampling across buckets
- All required reference files listed in `AGENTS.md` were present and inspected.
