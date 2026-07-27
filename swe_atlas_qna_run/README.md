# SWE-Atlas QN/A dfw run scripts

Run from this directory on dfw:

```bash
cd /lustre/fsw/portfolios/llmservice/users/wasiuddina/Gym/swe_atlas_qna_run
```

Edit `config.env` first, especially `SLURM_ACCOUNT`, `SLURM_PARTITION`, and judge credentials if `NVIDIA_API_KEY` is not already exported.

Suggested single-allocation order:

1. `./00_setup_repo_env.sh` on the login node
2. `./01_prepare_smoke_slice.sh` on the login node
3. `./02_create_nemotron_config.sh` on the login node
4. `./03_request_allocation.sh` to get a GPU-backed Slurm allocation shell
5. `./04_check_compute_env_and_judge.sh` inside that allocation shell
6. `bash ./05_start_vllm.sh` inside that allocation shell
7. `./05_start_servers.sh` inside that allocation shell
8. `./06_check_status.sh` until healthy, `./06_check_status.sh --tail-vllm` for vLLM logs, or `./06_check_status.sh --tail` for Gym logs
9. `./07_run_smoke_eval.sh`
10. `./08_inspect_smoke_result.sh`
11. `./09_run_full_eval_and_profile.sh` after the smoke run is healthy
12. `./10_stop_servers.sh` when done

By default, `RUN_GYM_IN_CONTAINER=false` runs Gym on the host allocation shell while vLLM runs in `VLLM_IMAGE`. This avoids nesting Gym's Apptainer sandbox provider inside `GYM_IMAGE`, which can trigger user namespace errors. If Apptainer reports fakeroot/user namespace errors, keep `SANDBOX_USER_NULL=true` in `config.env` and restart servers. If `/sandbox/answer.txt` cannot be written, set `APPTAINER_WRITABLE_TMPFS=true` and restart servers.

The policy model uses an external OpenAI-compatible vLLM server. Edit `VLLM_EXTRA_ARGS` in `config.env` to pass model-specific flags to `vllm serve`; Gym points at that server via `POLICY_BASE_URL`.

## Sharded full evaluation

Use this when one 4-hour allocation is not enough. The sharded flow splits the 124 SWE-Atlas QnA tasks into `NUM_SHARDS` independent input files, runs one shard per single-node job, then merges the rollout JSONLs and computes global aggregate metrics.

Prepare shards once on the login node:

```bash
export NUM_SHARDS=4
./11_prepare_full_shards.sh
```

This writes files like:

```text
swe_atlas_qna_run/shards/swe_atlas_qna_4/input_shard_0_of_4.jsonl
```

Each input row is stamped with a global `_ng_task_index`, so the final aggregate/profile can align tasks correctly across shards.

For each shard, request a normal one-node allocation and run the usual server startup flow inside that allocation:

```bash
export NUM_SHARDS=4
export SHARD_INDEX=0

./04_check_compute_env_and_judge.sh
bash ./05_start_vllm.sh
./05_start_servers.sh
./06_check_status.sh
./12_run_shard_eval.sh "${SHARD_INDEX}"
./10_stop_servers.sh --all
```

Repeat with `SHARD_INDEX=1`, `2`, and `3`. Each shard writes to:

```text
results/swe_atlas_qna_shards_4/shard_<index>_of_4.jsonl
results/swe_atlas_qna_shards_4/shard_<index>_of_4_materialized_inputs.jsonl
```

If multiple shard jobs run concurrently from the same shared checkout, give each job separate PID/log files before starting servers so they do not overwrite each other:

```bash
export SHARD_INDEX=0
source ./config.env
export SERVER_LOG="${RUN_DIR}/gym_servers_shard_${SHARD_INDEX}.log"
export SERVER_PID_FILE="${RUN_DIR}/gym_servers_shard_${SHARD_INDEX}.pid"
export VLLM_LOG="${RUN_DIR}/vllm_server_shard_${SHARD_INDEX}.log"
export VLLM_PID_FILE="${RUN_DIR}/vllm_server_shard_${SHARD_INDEX}.pid"
```

After all shard rollout files exist, start Gym servers once more and aggregate:

```bash
./05_start_servers.sh
./13_aggregate_shards.sh
./10_stop_servers.sh --gym
```

The aggregate script verifies that all `NUM_SHARDS` outputs exist, merges the materialized input sidecars, runs `gym eval aggregate`, then runs `gym eval profile`. Final outputs are:

```text
results/swe_atlas_qna_full.jsonl
results/swe_atlas_qna_full_materialized_inputs.jsonl
results/swe_atlas_qna_full_aggregate_metrics.json
```
