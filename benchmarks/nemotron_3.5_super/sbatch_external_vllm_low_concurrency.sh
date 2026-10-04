#!/bin/bash
#
# Run one benchmark's 50% split at production repeats under the low-concurrency recipe.
#
#   bash benchmarks/nemotron_3.5_super/sbatch_external_vllm_low_concurrency.sh swe
#   bash benchmarks/nemotron_3.5_super/sbatch_external_vllm_low_concurrency.sh tb
#
# Two *independent* vLLM engines, one per node, and one Gym collector per engine driving its
# own shard of the split against its own engine. This is a fork of sbatch_external_vllm.sh,
# not an edit of it: that script is hardwired to prefill/decode disaggregation behind a
# vllm-router, and this recipe needs plain independent engines. Leave the shared one alone.
#
# Concurrency is per replica, so the job runs 2x this many rollouts at once. Both default to
# 48 so the benchmarks share one operating point:
#   TB  48 — required; C=32 TIMED OUT twice.
#   SWE 48 — fits comfortably, but is measurably better at C=32 (see the note below).
#
# sandbox_timeout is 2 h, down from the 3 h default. It is a wall-clock cap on one rollout,
# so it sets the critical path: ~8% of TB rollouts run to it, and one starting after the 1 h
# mark cannot finish inside a 4 h wall. Costs 0.0028 reward on TB, nothing on SWE.
#
#
#   GYM              repo root                  derived from this script's location
#   MODEL            checkpoint to evaluate     the row90 step-28 candidate
#   MODEL_NAME       served model name          = MODEL
#   CONTAINER        vLLM container (.sqsh)     the validated nightly build
#   MOUNTS           container mounts           /lustre:/lustre,$GYM:/opt/Gym
#   VLLM_CONFIG      serving recipe             vllm_configs/..._low_concurrency.sh
#   CONCURRENCY      per replica                48
#   NUM_REPLICAS     engines = nodes            2   (shards exist for 2 only)
#   WALLTIME         sbatch --time              04:00:00
#   SANDBOX_TIMEOUT  harness cap, seconds       7200
#   EXPERIMENT_NAME  results/<this>/...         $USER-low-concurrency/<bench>_split50_c<C>_n<N>
#   SBATCH_ACCOUNT / SBATCH_PARTITION / SBATCH_GRES / SLURM_COMMENT   cluster-specific
#   DRY_RUN=1        print the generated scripts instead of submitting

set -euo pipefail

# Repo root, derived from this script's own location (<repo>/benchmarks/nemotron_3.5_super/),
# so it works whether you invoke it from the repo root, by absolute path, or via a symlink.
GYM="${GYM:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd -P)}"
[[ -d "$GYM/nemo_gym" && -d "$GYM/benchmarks" ]] || {
    echo "GYM=$GYM does not look like a NeMo Gym checkout (no nemo_gym/ + benchmarks/).
Run this from your Gym repo root, or set GYM=<path to your checkout>." >&2; exit 1; }

# Overridable so the recipe can be smoke-tested against a small hand-built shard without
# editing the committed 50% split.
SHARD_DIR="${SHARD_DIR:-benchmarks/nemotron_3.5_super/split50/shards}"

# C=48 on both, deliberately aligned so the two benchmarks share one operating point.
# Set CONCURRENCY=32 for SWE when latency fidelity matters more than matching TB.
BENCH="${1:-}"
case "$BENCH" in
    swe) DEFAULT_C=48
         inst=swebench_verified_opencode_sandboxed_agent
         agent=opencode_sandboxed_agent ;;
    tb)  DEFAULT_C=48
         inst=terminal_bench_2_1_terminus_2_sandboxed_agent
         agent=terminus_2_sandboxed_agent ;;
    *)   echo "usage: $0 {swe|tb}" >&2; exit 2 ;;
esac

CONCURRENCY="${CONCURRENCY:-$DEFAULT_C}"
NUM_REPLICAS="${NUM_REPLICAS:-2}"
WALLTIME="${WALLTIME:-04:00:00}"
SANDBOX_TIMEOUT="${SANDBOX_TIMEOUT:-7200}"

cd "$GYM"

bench_configs=""
for (( i = 0; i < NUM_REPLICAS; i++ )); do
    cfg="$SHARD_DIR/${BENCH}_r${i}.yaml"
    [[ -f "$cfg" ]] || { echo "missing shard config $cfg (shards exist for 2 replicas only)" >&2; exit 1; }
    bench_configs+="$cfg "
done

# Site/run-specific. The defaults are the checkpoint and container this recipe was validated
# on; override MODEL to evaluate a different checkpoint, CONTAINER for a different vLLM build.
MODEL="${MODEL:-/lustre/fsw/portfolios/nemotron/users/ygalron/super3dot5_ga_candidate/row90/joint-mopd-v1.sgt.2.v1-step28_boosted_mtp}"
MODEL_NAME="${MODEL_NAME:-$MODEL}"
CONTAINER="${CONTAINER:-/lustre/fs1/portfolios/nemotron/projects/nemotron_evals_dev/users/bxyu/Gym/results/vllm/vllm-openai:nightly-2a02f6efe319c885e3ccbcecde402e0028f9ec1e___with_gym.sqsh}"
# The checkout is mounted at /opt/Gym inside the container; /lustre carries model + container.
MOUNTS="${MOUNTS:-/lustre:/lustre,$GYM:/opt/Gym}"
for p in "$MODEL" "$CONTAINER"; do
    [[ -e "$p" ]] || echo "warning: $p does not exist on this filesystem" >&2
done
DEFAULT_VLLM_CONFIG=benchmarks/nemotron_3.5_super/vllm_configs/nemotron_3.5_super_low_concurrency.sh
VLLM_CONFIG="${VLLM_CONFIG:-$DEFAULT_VLLM_CONFIG}"
[[ -f "$VLLM_CONFIG" ]] || { echo "missing vllm config $VLLM_CONFIG" >&2; exit 1; }

# Every axis that changes what the run *is* goes in the name, so two runs can never land in the
# same directory: benchmark, concurrency, replica count, and the recipe when it is not the
# default one (the default keeps its historical unsuffixed name).
recipe_tag=""
if [[ "$VLLM_CONFIG" != "$DEFAULT_VLLM_CONFIG" ]]; then
    recipe_tag="_$(basename "$VLLM_CONFIG" .sh | sed 's/^nemotron_3.5_super_low_concurrency_//')"
fi
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$USER-low-concurrency/${BENCH}_split50_c${CONCURRENCY}_n${NUM_REPLICAS}${recipe_tag}}"
EXTRA_GYM_ARGS="++${inst}.responses_api_agents.${agent}.sandbox_timeout=${SANDBOX_TIMEOUT}"
# Slurm settings — all overridable, since account/partition are per-cluster and per-project.
# SLURM_COMMENT exempts the job from the idle-GPU reaper on oci-hsg; harmless elsewhere.
default_comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"benchmarking","description":"POR - https://nvaiinfa.aha.io/ideas/ideas/NMP-I-791"}}'
SLURM_COMMENT="${SLURM_COMMENT:-$default_comment}"
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-nemotron_n3_post}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:4}"

read -r -a bench_configs_arr <<< "${bench_configs% }"
WORKER_SERVER_PORT=8001

# max(8, 4 x concurrency) — see the recipe header for why.
MAX_NUM_SEQS=$(( CONCURRENCY * 4 ))
(( MAX_NUM_SEQS < 8 )) && MAX_NUM_SEQS=8

echo "bench=$BENCH C=$CONCURRENCY replicas=$NUM_REPLICAS wall=$WALLTIME sandbox_timeout=$SANDBOX_TIMEOUT"
echo "  max_num_seqs=$MAX_NUM_SEQS  experiment=$EXPERIMENT_NAME"

# ---------------------------------------------------------------- vLLM server, one per node
server_command=$(cat <<EOF
#!/bin/bash
set -euo pipefail

export MAX_NUM_SEQS=$MAX_NUM_SEQS

# Nemotron's three-read Mamba SSM state layout.
export VLLM_SSM_CONV_STATE_LAYOUT=DS
export VLLM_USE_FASTOKENS=1
# V2 model runner is the 0.29.0 default but carries a large speed regression.
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=180
# Rust frontend stays off: 1-2% SWE accuracy delta in this vLLM, and we have no accuracy budget.

source "$VLLM_CONFIG"

if [[ \$(ulimit -Hn) == "unlimited" ]] || [[ 65535 -lt \$(ulimit -Hn) ]]; then
  ulimit -Sn 65535
fi

vllm serve "$MODEL" --served-model-name "$MODEL_NAME" \
    "\${VLLM_COMMON_ARGS[@]}" "\${VLLM_SERVE_ARGS[@]}" \
    --host \$(hostname) \
    --port $WORKER_SERVER_PORT
EOF
)
export server_command

# ---------------------------------------------------------------- Gym collector, one per replica
for (( i = 0; i < NUM_REPLICAS; i++ )); do
    printf -v "eval_command_$i" '%s' "$(cat <<EOF
set -euo pipefail
source /opt/Gym_venv/bin/activate
cd /opt/Gym

export NEMO_GYM_RUN_ID="\$SLURM_JOB_ID"
export NEMO_GYM_USER="\${NEMO_GYM_USER:-\$SLURM_JOB_USER}"

source "$VLLM_CONFIG"

gym eval prepare \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config ${bench_configs_arr[i]} \
    +use_cached_prepared_benchmarks=true

experiment_name=$EXPERIMENT_NAME/slurm_job_id_\$SLURM_JOB_ID/replica_$i
rollouts_fpath=results/\$experiment_name/rollouts.jsonl

gym eval run \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config ${bench_configs_arr[i]} \
    --config benchmarks/nemotron_3.5_super/sandbox_utils.yaml \
    --config benchmarks/nemotron_3.5_super/policy_model_override.yaml \
    +wandb_project=$USER-gym-low-concurrency \
    +wandb_name=\$experiment_name \
    +uv_venv_dir=/opt/uv_venvs \
    +nemo_gym_log_dir=results/\$experiment_name/logs \
    +skip_venv_if_present=true \
    ++output_jsonl_fpath=\$rollouts_fpath \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++use_absolute_ip=true \
    ++reuse_existing_data_preparation=true \
    ++num_samples_in_parallel=$CONCURRENCY \
    ++policy_base_url=http://\$(getent hosts "\$REPLICA_NODE" | awk 'NR == 1 {print \$1}'):$WORKER_SERVER_PORT/v1 \
    ++policy_api_key=dummy_api_key \
    ++policy_model_name=$MODEL_NAME \
    ++upload_rollouts=false \
    ++global_aiohttp_connector_limit_per_host=16384 \
    ++port_range_low=63000 \
    ++port_range_high=64000 \
    ++observability_enabled=true \
    ++model_call_capture_dir=/opt/Gym/results/\$experiment_name/model_calls \
    $EXTRA_GYM_ARGS \
    "\${GYM_MODEL_PARAMS[@]}"
EOF
)"
    export "eval_command_$i"
done

# ---------------------------------------------------------------- batch driver
batch_command=$(cat <<EOF
set -euo pipefail

nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))

srun --nodes=$NUM_REPLICAS --ntasks=$NUM_REPLICAS --ntasks-per-node=1 --kill-on-bad-exit=1 \
    --container-image=$CONTAINER \
    --container-name=lowconc-server \
    --container-mounts=$MOUNTS \
    --container-workdir=\$SLURM_SUBMIT_DIR \
    --no-container-mount-home \
    bash -c '
        set -euo pipefail
        cd "\$SLURM_SUBMIT_DIR"
        exec "\$@"
    ' bash bash -c "\$server_command" &
server_step=\$!

cleanup_server() {
    job_status=\$?
    trap - EXIT INT TERM
    set +e
    kill "\$server_step" 2>/dev/null || true
    wait "\$server_step" 2>/dev/null || true
    exit "\$job_status"
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

eval_steps=()
for (( i = 0; i < $NUM_REPLICAS; i++ )); do
    var="eval_command_\$i"
    REPLICA_NODE="\${nodes[i]}" \
    eval_command="\${!var}" \
    srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=\$SLURM_CPUS_ON_NODE \
        --nodelist="\${nodes[i]}" --gpus=0 \
        --container-image=$CONTAINER \
        --container-name=lowconc-eval \
        --container-mounts=$MOUNTS \
        --container-workdir="\$SLURM_SUBMIT_DIR" \
        --no-container-mount-home \
        bash -c '
            set -euo pipefail
            cd "\$SLURM_SUBMIT_DIR"
            exec bash -c "\$eval_command"
        ' &
    eval_steps+=(\$!)
done

# Without this the collectors would spin for the whole allocation if the engines died --
# the failure mode that burned a 4 h allocation before.
watch_server() {
    while kill -0 "\$server_step" 2>/dev/null; do sleep 30; done
    echo "vLLM server step exited; stopping collectors" >&2
    kill "\${eval_steps[@]}" 2>/dev/null || true
}
watch_server &
watchdog=\$!

status=0
for pid in "\${eval_steps[@]}"; do
    wait "\$pid" || status=\$?
done
kill "\$watchdog" 2>/dev/null || true
exit "\$status"
EOF
)
export batch_command

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "================= server_command ================="
    echo "$server_command"
    for (( i = 0; i < NUM_REPLICAS; i++ )); do
        var="eval_command_$i"
        echo "================= eval_command_$i ================="
        echo "${!var}"
    done
    echo "================= batch_command ================="
    echo "$batch_command"
    exit 0
fi

submit_dir=$(pwd -P)
cleanup_user=${NEMO_GYM_USER:-$USER}
export NEMO_GYM_USER="$cleanup_user"

main_job_id=$(
    sbatch \
        --parsable \
        --nodes=$NUM_REPLICAS \
        --time=$WALLTIME \
        --job-name=lowconc-$BENCH-$USER \
        --output=slurm-logs/%j-%x.log \
        --ntasks-per-node=1 \
        --comment="$SLURM_COMMENT" \
        --exclusive \
        --segment=$NUM_REPLICAS \
        --wrap 'exec bash -c "$batch_command"'
)
main_job_id=${main_job_id%%;*}

unset SBATCH_RESERVATION
if ! cleanup_job_id=$(
    sbatch --parsable \
        --dependency=afterany:"$main_job_id" \
        --partition=cpu --qos=cpu-normal --gres=none --gpus-per-node=0 \
        --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=256M --time=00:30:00 \
        --job-name="lowconc-cleanup-$main_job_id" \
        --output="$submit_dir/slurm-logs/%j-lowconc-cleanup-$main_job_id.log" \
        "$submit_dir/nemo_gym/sandbox/providers/opensandbox/cleanup_sandboxes.py" \
        --connection-config "$submit_dir/env.yaml" \
        --run-id "$main_job_id" --user "$cleanup_user" --reap
); then
    echo "Submitted batch job $main_job_id"
    echo "Failed to submit the sandbox-cleanup job; its sandboxes will need reaping by hand" >&2
    exit 0
fi

echo "Submitted batch job $main_job_id"
echo "Submitted cleanup job ${cleanup_job_id%%;*} for batch job $main_job_id"
