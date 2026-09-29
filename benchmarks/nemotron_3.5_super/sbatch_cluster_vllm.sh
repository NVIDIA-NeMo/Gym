#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Same independent P/D recipe as sbatch_external_vllm.sh, owned by cluster.py.
set -euo pipefail
# Slurm executes a spool copy of this file, so preserve the submit-side root.
ROOT=${GYM_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}
export GYM_SOURCE_ROOT="$ROOT"
SCRIPT="$ROOT/benchmarks/nemotron_3.5_super/sbatch_cluster_vllm.sh"
PYTHON=${GYM_PYTHON:-$ROOT/.venv/bin/python}
HELPER="$ROOT/benchmarks/nemotron_3.5_super/launcher_comparison.py"
cd "$ROOT"

case "${1:-}" in
    --router-proxy)
        shift
        command=(vllm-router "$@")
        if [[ "${1:-}" == --version ]]; then
            command=(python3 -c 'from importlib.metadata import version; print("vllm-router " + version("vllm-router"))')
        fi
        exec srun --overlap --exact --nodes=1 --ntasks=1 --gpus=0 \
            --cpus-per-task="$SLURM_CPUS_ON_NODE" --nodelist="$ROUTER_NODE" \
            --container-image="$CONTAINER" --container-name="launcher-router-$SLURM_JOB_ID" \
            --container-mounts="$MOUNTS" --no-container-mount-home --no-container-entrypoint \
            "${command[@]}"
        ;;
    --eval)
        shift
        source /opt/Gym_venv/bin/activate
        cd /opt/Gym
        export PYTHONPATH=/opt/Gym NEMO_GYM_EXTRA_ROOTS=/opt/Gym PYTHONDONTWRITEBYTECODE=1
        export NEMO_GYM_RUN_ID="$SLURM_JOB_ID" NEMO_GYM_USER="${NEMO_GYM_USER:-$SLURM_JOB_USER}"
        source "$VLLM_CONFIG"
        gym eval prepare "$@" +use_cached_prepared_benchmarks=true
        exec gym eval run "$@" \
            --config benchmarks/nemotron_3.5_super/sandbox_utils.yaml \
            --config benchmarks/nemotron_3.5_super/policy_model_override.yaml \
            +uv_venv_dir=/opt/uv_venvs +skip_venv_if_present=true \
            "+nemo_gym_log_dir=$RUN_DIR/logs" \
            "++output_jsonl_fpath=$RUN_DIR/rollouts.jsonl" \
            ++overwrite_metrics_conflicts=true ++split=benchmark \
            ++use_absolute_ip=true ++reuse_existing_data_preparation=true \
            "++policy_base_url=http://$ROUTER_IP:8000/v1" \
            ++policy_api_key=dummy_api_key "++policy_model_name=${MODEL_NAME:-$MODEL}" \
            ++upload_rollouts=false ++wandb_project=null \
            ++global_aiohttp_connector_limit_per_host=16384 \
            ++port_range_low=63000 ++port_range_high=64000 \
            ++num_samples_in_parallel=null \
            ++rollout_collection_driver=benchmarks.rollout_timing:run \
            "${GYM_MODEL_PARAMS[@]}"
        ;;
    --cleanup)
        shift
        unset PYTHONPATH
        exec "$PYTHON" nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py \
            --connection-config "$ROOT/env.yaml" --run-id "$1" --user "$2" --reap
        ;;
    --batch)
        shift
        export RUN_DIR="$RESULTS_ROOT/slurm_job_id_$SLURM_JOB_ID"
        mkdir -p "$RUN_DIR"
        export PYTHONPATH="$ROOT" NEMO_GYM_EXTRA_ROOTS="$ROOT" PYTHONDONTWRITEBYTECODE=1
        export RAY_TMPDIR=/tmp LITELLM_LOCAL_MODEL_COST_MAP=True
        if [[ $(ulimit -Hn) == unlimited ]] || (( $(ulimit -Hn) >= 65535 )); then ulimit -Sn 65535; fi
        mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
        export ROUTER_NODE=${nodes[0]}
        export ROUTER_IP
        ROUTER_IP=$(getent ahostsv4 "$ROUTER_NODE" | awk 'NR==1 {print $1}')
        eval_node=${nodes[1]}
        node_args=()
        for node in "${nodes[@]}"; do
            ip=$(getent ahostsv4 "$node" | awk 'NR==1 {print $1}')
            node_args+=(--node "$node=$ip")
        done
        printf '#!/usr/bin/env bash\nexec bash %q --router-proxy "$@"\n' "$SCRIPT" > "$RUN_DIR/router-command"
        chmod 700 "$RUN_DIR/router-command"
        "$PYTHON" "$HELPER" build --output "$RUN_DIR" --router "$RUN_DIR/router-command" "${node_args[@]}"
        date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/startup-started.txt"
        "$PYTHON" -m responses_api_models.local_vllm_model.cluster \
            --config "$RUN_DIR/cluster-config.json" --output "$RUN_DIR/cluster" > "$RUN_DIR/controller.log" 2>&1 &
        controller_pid=$!
        eval_pid=""
        cleanup() {
            status=$?
            trap - EXIT INT TERM
            set +e
            if [[ -n "$eval_pid" ]]; then kill "$eval_pid" 2>/dev/null; wait "$eval_pid"; fi
            kill "$controller_pid" 2>/dev/null
            wait "$controller_pid"
            printf '%s\n' "$status" > "$RUN_DIR/batch-exit-code.txt"
            exit "$status"
        }
        trap cleanup EXIT
        trap 'exit 130' INT
        trap 'exit 143' TERM
        deadline=$((SECONDS + 2100))
        until [[ -f "$RUN_DIR/cluster/gym-connection.private.json" ]]; do
            if ! kill -0 "$controller_pid" 2>/dev/null; then
                echo "Cluster launcher failed; see $RUN_DIR/controller.log" >&2
                exit 1
            fi
            if (( SECONDS >= deadline )); then echo 'Cluster startup timed out' >&2; exit 1; fi
            sleep 2
        done
        "$PYTHON" "$HELPER" smoke --run-dir "$RUN_DIR" > "$RUN_DIR/inference-smoke.log" 2>&1
        date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/startup-ready.txt"
        srun --overlap --exact --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_ON_NODE" \
            --nodelist="$eval_node" --gpus=0 \
            --container-image="$CONTAINER" --container-name="launcher-eval-$SLURM_JOB_ID" \
            --container-mounts="$MOUNTS" --no-container-mount-home --no-container-entrypoint \
            bash "$SCRIPT" --eval "$@" > "$RUN_DIR/eval.log" 2>&1 &
        eval_pid=$!
        completed_pid=""
        status=0
        wait -n -p completed_pid "$controller_pid" "$eval_pid" || status=$?
        if [[ "$completed_pid" == "$controller_pid" ]]; then
            echo 'Cluster controller exited before evaluation finished' >&2
            exit 1
        fi
        eval_pid=""
        if [[ -n "${BASELINE_JSONL:-}" && -f "$RUN_DIR/rollouts.jsonl" ]]; then
            "$PYTHON" "$HELPER" compare --baseline "$BASELINE_JSONL" \
                --candidate "$RUN_DIR/rollouts.jsonl" --output "$RUN_DIR/comparison.json" \
                > "$RUN_DIR/comparison.log" 2>&1 || { echo 'Comparison failed; inspect comparison.log' >&2; exit 1; }
        fi
        exit "$status"
        ;;
    --help|-h)
        cat <<'EOF'
MODEL=/shared/checkpoint CONTAINER=/shared/vllm.sqsh \
SBATCH_ACCOUNT=your_account SBATCH_PARTITION=batch SBATCH_QOS=interactive \
BASELINE_JSONL=/shared/previous/date_run.jsonl \
bash benchmarks/nemotron_3.5_super/sbatch_cluster_vllm.sh

Defaults: nemotron_3.5_super.sh, P2D2, DP1/TP4, SWE Verified/OpenCode 500x3,
concurrency 1500, original 3-hour agent timeout. MODEL and CONTAINER are required.
Optional positional arguments replace the two default Gym --config arguments.
No W&B upload. Exact serving-option parity is checked before starting workers.
EOF
        exit 0
        ;;
esac

: "${MODEL:?Set MODEL to the same checkpoint as the external baseline}"
: "${CONTAINER:?Set CONTAINER to the same shared image as the external baseline}"
export MODEL CONTAINER
export VLLM_CONFIG=${VLLM_CONFIG:-benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super.sh}
export NUM_PREFILL_NODES=${NUM_PREFILL_NODES:-2} NUM_DECODE_NODES=${NUM_DECODE_NODES:-2}
export GPUS_PER_NODE=${GPUS_PER_NODE:-4}
export MOUNTS=${MOUNTS:-/lustre:/lustre,$ROOT:/opt/Gym}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-opencode_swe_verified/cluster-super35-BF16-conc1500-p2d2-dp1tp4}
export RESULTS_ROOT="$ROOT/results/$EXPERIMENT_NAME"
export NEMO_GYM_USER=${NEMO_GYM_USER:-$USER}
[[ -f "$CONTAINER" && -f "$VLLM_CONFIG" && -x "$PYTHON" ]]
if [[ -n "${BASELINE_JSONL:-}" ]]; then
    [[ -f "$BASELINE_JSONL" && -f "${BASELINE_JSONL%.jsonl}.timings.jsonl" ]]
    export BASELINE_JSONL
fi
if (( $# == 0 )); then
    set -- --config responses_api_models/vllm_model/configs/vllm_model.yaml \
        --config benchmarks/swebench/verified/opencode.yaml
fi
mkdir -p "$RESULTS_ROOT" "$ROOT/slurm-logs"
job=$(sbatch --parsable --nodes="$((NUM_PREFILL_NODES + NUM_DECODE_NODES))" \
    --ntasks-per-node=1 --gres="gpu:$GPUS_PER_NODE" --exclusive \
    --time="${SBATCH_TIME:-04:00:00}" --segment="$((NUM_PREFILL_NODES + NUM_DECODE_NODES))" \
    --comment="${SLURM_COMMENT:-}" --job-name=gym-cluster-comparison \
    --output="$ROOT/slurm-logs/%j-cluster-comparison.log" "$SCRIPT" --batch "$@")
job=${job%%;*}
echo "Submitted cluster benchmark job $job"
if cleanup_job=$(env -u SBATCH_RESERVATION sbatch --parsable --dependency="afterany:$job" \
    --partition=cpu --qos=cpu-short --gres=none --nodes=1 --ntasks=1 --cpus-per-task=1 \
    --mem=512M --time=00:15:00 --job-name="gym-cluster-cleanup-$job" \
    --output="$ROOT/slurm-logs/%j-cluster-cleanup.log" "$SCRIPT" --cleanup "$job" "$NEMO_GYM_USER"); then
    printf 'main=%s\ncleanup=%s\n' "$job" "${cleanup_job%%;*}" > "$RESULTS_ROOT/submission-$job.txt"
    echo "Submitted cleanup job ${cleanup_job%%;*} for $job"
else
    echo "Cleanup submission failed; cancelling benchmark $job" >&2
    scancel "$job"
    exit 1
fi
