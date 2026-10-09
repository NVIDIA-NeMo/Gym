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
#   EXPERIMENT_NAME  results/<this>/replica_<i>  $USER-low-concurrency/<bench>_split50_c<C>_n<N>
#                    The same name is the same run: launching it again continues it (Gym's
#                    ++resume_from_cache=true is always on and dispatches only the rollouts missing from
#                    each replica's rollouts file). A deliberate repeat needs a new name.
#   SHARD_DIR        per-replica shard configs  benchmarks/nemotron_3.5_super/split50/shards
#   LATENCY_METRICS  per-rollout latency timeline  0 (see LATENCY-METRICS.md); outputs go to
#                    results/<replica>/metrics/ and one report per job to results/<job>/metrics/
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
# LATENCY_METRICS=1 turns on the per-rollout latency timeline (docs: LATENCY-METRICS.md): model-call capture in
# metadata mode, the OpenCode header plugin, run_info.json, and the post-run build + check + analyze per replica
# and one benchmark report per job. The engine's per-request metrics and cached-token counts are already on in
# the vLLM recipe. Default off: byte-identical scripts.
LATENCY_METRICS="${LATENCY_METRICS:-0}"

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

# Every axis that changes what the run *is* goes in the name: benchmark, concurrency, replica count, and the
# recipe when it is not the default one (the default keeps its historical unsuffixed name). The same name is
# the same run; a second launch continues it (see the resume guard below).
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
export SBATCH_ACCOUNT="${SBATCH_ACCOUNT:-coreai_nvfm_llm}"
export SBATCH_PARTITION="${SBATCH_PARTITION:-batch}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:4}"

read -r -a bench_configs_arr <<< "${bench_configs% }"
WORKER_SERVER_PORT=8001

# max(8, 4 x concurrency) — see the recipe header for why.
MAX_NUM_SEQS=$(( CONCURRENCY * 4 ))
(( MAX_NUM_SEQS < 8 )) && MAX_NUM_SEQS=8

# ---------------------------------------------------------------- resume guard (login node)
# The same EXPERIMENT_NAME is the same run: Gym (++resume_from_cache=true, always on) dispatches only the
# rollouts missing from each replica's rollouts file. Before submitting into an existing experiment directory,
# refuse what Gym would mishandle silently: a replica without Gym's two files (Gym would start a fresh run in a
# directory that already holds one) or run facts that differ from the stored ones (Gym would mix two
# configurations into one run). A run made without LATENCY_METRICS has no run_info.json: only the files are
# checked.
exp_dir="results/$EXPERIMENT_NAME"
if compgen -G "$exp_dir/replica_*" > /dev/null; then
    command -v python3 > /dev/null || { echo "resume refused: python3 is needed on the login node to compare the stored run facts" >&2; exit 1; }
    for (( i = 0; i < NUM_REPLICAS; i++ )); do
        for f in rollouts.jsonl rollouts_materialized_inputs.jsonl; do
            [[ -f "$exp_dir/replica_$i/$f" ]] || { echo "resume refused: $exp_dir/replica_$i/$f is missing; Gym would start a fresh run in a directory that already holds one" >&2; exit 1; }
        done
        info="$exp_dir/replica_$i/metrics/run_info.json"
        [[ -f "$info" ]] || continue
        python3 - "$info" "$MODEL" "$MODEL_NAME" "$(basename "$CONTAINER")" "$(basename "$VLLM_CONFIG")" \
            "$CONCURRENCY" "$NUM_REPLICAS" "$MAX_NUM_SEQS" "$SANDBOX_TIMEOUT" "${bench_configs_arr[i]}" <<'PY' || exit 1
import json, os, sys
path, *now = sys.argv[1:]
keys = ("model", "model_name", "container", "vllm_config", "concurrency", "num_replicas", "max_num_seqs",
        "sandbox_timeout_s", "shard_config")
stored = json.load(open(path))
def same(key, old, new):
    if key == "shard_config":  # relative or absolute, with or without ".." segments: the same file is the same shard
        return os.path.realpath(str(old)) == os.path.realpath(new)
    return str(old) == new
bad = [(k, stored[k], v) for k, v in zip(keys, now) if k in stored and not same(k, stored[k], v)]
for key, old, new in bad:
    print(f"resume refused: {key} differs from {path}: stored {old!r}, now {new!r}", file=sys.stderr)
sys.exit(1 if bad else 0)
PY
    done
    echo "Resuming $exp_dir: Gym dispatches only the rollouts missing from each replica"
fi

# Latency-timeline fragments. All empty when LATENCY_METRICS is off, and each is glued onto an existing line
# below, so the rendered scripts are byte-identical to the default ones in that case.
latency_eval_pre=""
latency_gym_args=""
latency_eval_post=""
latency_batch_post=""
if [[ "$LATENCY_METRICS" == "1" ]]; then
    # run_info.json pins the run's knobs for the benchmark report; the commit is read here, on the login node,
    # because the container has no git.
    CODE_COMMIT=$(git -C "$GYM" rev-parse --short HEAD 2>/dev/null || echo unknown)
    latency_run_facts="\"benchmark\": \"$BENCH\", \"instance\": \"$inst\", \"agent\": \"$agent\", \"model\": \"$MODEL\", \"model_name\": \"$MODEL_NAME\", \"vllm_config\": \"$(basename "$VLLM_CONFIG")\", \"container\": \"$(basename "$CONTAINER")\", \"concurrency\": $CONCURRENCY, \"num_replicas\": $NUM_REPLICAS, \"sandbox_timeout_s\": $SANDBOX_TIMEOUT, \"walltime\": \"$WALLTIME\", \"max_num_seqs\": $MAX_NUM_SEQS, \"experiment_name\": \"$EXPERIMENT_NAME\", \"latency_metrics\": true"
    # Capture records carry metadata only (no prompts or completions). The OpenCode harness installs its header
    # plugin only under its own flag; Terminus records its spans regardless.
    latency_gym_args=" ++model_call_capture_payloads=metadata"
    [[ "$BENCH" == "swe" ]] && latency_gym_args+=" ++${inst}.responses_api_agents.${agent}.latency_metrics=true"
    # Glued onto the last line of `gym eval run`: capture the collector's exit, then run the build, the check and
    # the analysis whatever happened, and exit with the collector's code so the job status is still the
    # collection's.
    latency_eval_post=$(cat <<EOF
 || collector_status=\$?
python benchmarks/nemotron_3.5_super/analysis/timeline.py build --run-dir results/\$experiment_name \\
    --tokenize-url http://\$(getent hosts "\$REPLICA_NODE" | awk 'NR == 1 {print \$1}'):$WORKER_SERVER_PORT/v1 \\
    || echo "LATENCY TIMELINE BUILD FAILED" >&2
python benchmarks/nemotron_3.5_super/analysis/timeline.py check --run-dir results/\$experiment_name \\
    || echo "LATENCY CHECK FAILED (see results/\$experiment_name/metrics/timeline_check.json)" >&2
python benchmarks/nemotron_3.5_super/analysis/timeline.py analyze --run-dir results/\$experiment_name \\
    || echo "LATENCY ANALYZE FAILED" >&2
exit "\$collector_status"
EOF
)
    # One benchmark report per job, after every replica step has exited (whatever its status). A failure here
    # prints a line and leaves the job's exit code to the collectors.
    latency_batch_post=$(cat <<EOF

srun --overlap --nodes=1 --ntasks=1 --nodelist="\${nodes[0]}" --gpus=0 \\
    --container-image=$CONTAINER \\
    --container-name=lowconc-eval \\
    --container-mounts=$MOUNTS \\
    --container-workdir="\$SLURM_SUBMIT_DIR" \\
    --no-container-mount-home \\
    bash -c 'cd "\$SLURM_SUBMIT_DIR" && source /opt/Gym_venv/bin/activate && cd /opt/Gym && python benchmarks/nemotron_3.5_super/analysis/timeline.py report --experiment-dir results/$EXPERIMENT_NAME' \\
    || echo "LATENCY REPORT FAILED" >&2
EOF
)
fi

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
    latency_eval_pre=""
    if [[ "$LATENCY_METRICS" == "1" ]]; then
        latency_eval_pre=$(cat <<EOF
collector_status=0
mkdir -p results/\$experiment_name/metrics
[[ -f results/\$experiment_name/metrics/run_info.json ]] || cat > results/\$experiment_name/metrics/run_info.json <<RUNINFO
{"schema_version": 1, $latency_run_facts, "replica": $i, "shard_config": "${bench_configs_arr[i]}", "slurm_job_id": "\$SLURM_JOB_ID", "node": "\$REPLICA_NODE", "code_commit": "$CODE_COMMIT", "started_at": "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
RUNINFO
EOF
)
        latency_eval_pre+=$'\n\n'
    fi
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

experiment_name=$EXPERIMENT_NAME/replica_$i
rollouts_fpath=results/\$experiment_name/rollouts.jsonl

${latency_eval_pre}gym eval run \
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
    ++resume_from_cache=true \
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
    ++route_failures_to_sidecar=true \
    $EXTRA_GYM_ARGS${latency_gym_args} \
    "\${GYM_MODEL_PARAMS[@]}"${latency_eval_post}
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
done${latency_batch_post}
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
