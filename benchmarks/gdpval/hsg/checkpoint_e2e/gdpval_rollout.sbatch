#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Unified GDPVal rollout: produce deliverables, judge nothing.
#SBATCH --job-name=gdpval-rollout
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --switches=1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --open-mode=append
#SBATCH --comment={"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"240","reason":"benchmarking","description":"GDPVal rollouts"}}

set -euo pipefail
umask 077
export PYTHONUNBUFFERED=1

# Slurm executes a staged copy of this script from its spool directory, so
# BASH_SOURCE cannot locate package-owned helpers on compute nodes. The
# launcher exports the immutable package directory explicitly for initial and
# recovery submissions.
: "${ROLLOUT_PACKAGE_DIR:?set ROLLOUT_PACKAGE_DIR to the absolute checkpoint_e2e package directory}"
[[ $ROLLOUT_PACKAGE_DIR == /* ]] || {
    echo "FATAL: ROLLOUT_PACKAGE_DIR must be absolute: $ROLLOUT_PACKAGE_DIR" >&2
    exit 64
}
SCRIPT_DIR="$(cd -P -- "$ROLLOUT_PACKAGE_DIR" 2>/dev/null && pwd -P)" || {
    echo "FATAL: rollout package directory is inaccessible: $ROLLOUT_PACKAGE_DIR" >&2
    exit 64
}
ROLLOUT_LIFECYCLE_SH="$SCRIPT_DIR/rollout_lifecycle.sh"
ROLLOUT_SHARD_COVERAGE_PY="$SCRIPT_DIR/rollout_shard_coverage.py"

: "${RUN_DIR:?set RUN_DIR}"
: "${DATASET:?set DATASET}"
: "${PROFILE:?set PROFILE to a serving profile env file}"
[[ -r $ROLLOUT_LIFECYCLE_SH ]] || { echo "FATAL: rollout lifecycle helper is unreadable" >&2; exit 64; }
[[ -r $ROLLOUT_SHARD_COVERAGE_PY ]] || { echo "FATAL: shard coverage helper is unreadable" >&2; exit 64; }
# shellcheck disable=SC1090
source "$ROLLOUT_LIFECYCLE_SH"

CONCURRENCY=${CONCURRENCY:-8}
AGENT_MAX_TURNS=${AGENT_MAX_TURNS:-250}
NUM_REPEATS=${NUM_REPEATS:-1}
ROTATION=${SLURM_RESTART_COUNT:-0}

U=${U:-/lustre/fsw/portfolios/llmservice/users/spanev/gdpval_colo/unified}
TREE=${TREE:-/lustre/fsw/portfolios/llmservice/users/spanev/gdpval_integ}
AGENT_SIF=${AGENT_SIF:-/lustre/fsw/portfolios/llmservice/users/agronskiy/images/apptainer/python-3.13.gdpval.gym-32083a.sif}
APPTAINER_BIN=${APPTAINER_BIN:-/lustre/fsw/portfolios/llmservice/users/spanev/gdpval_colo/apptainer-1.5.1-bigsession/bin}

mkdir -p "$RUN_DIR"/{deliverables,tmp,logs,server_info}
[[ -n ${PERSIST_DELIVERABLES_DIR:-} ]] && mkdir -p "$PERSIST_DELIVERABLES_DIR"
OUTPUT_JSONL=$RUN_DIR/rollouts.jsonl
[[ ! -e $RUN_DIR/STOP ]] || { echo "STOP sentinel present"; exit 0; }

fail() { echo "FATAL: $*" >&2; exit 64; }
[[ $ROTATION =~ ^[0-9]+$ ]] || fail "SLURM_RESTART_COUNT must be a nonnegative integer"

# Source secrets from inside the job. Only fill values the submit environment
# did not already provide.
if [[ -n ${ENV_FILE:-} && -r ${ENV_FILE:-} ]]; then
    while IFS= read -r _line; do
        [[ $_line == export\ * ]] || continue
        _name=${_line#export }; _name=${_name%%=*}
        [[ -n ${!_name:-} ]] || eval "$_line"
    done < "$ENV_FILE"
fi

[[ -r $PROFILE ]] || fail "profile unreadable: $PROFILE"
set -a
# shellcheck disable=SC1090
source "$PROFILE"
set +a
: "${POLICY_SERVE_SCRIPT:?profile must set POLICY_SERVE_SCRIPT}"
: "${MODEL_NAME:?profile must set MODEL_NAME}"
: "${MODEL_PATH:?profile must set MODEL_PATH}"
NODES_PER_REPLICA=${NODES_PER_REPLICA:-1}

echo "=== assertions ==="
RUNNER_SHA=$(sha256sum "$TREE/benchmarks/gdpval/run_gdpval_rollouts.sh" | cut -c1-16)
echo "  runner sha256        $RUNNER_SHA"
if [[ -n ${EXPECTED_RUNNER_SHA:-} ]]; then
    [[ $RUNNER_SHA == "$EXPECTED_RUNNER_SHA" ]] \
        || fail "runner sha changed: $RUNNER_SHA != $EXPECTED_RUNNER_SHA"
    echo "  runner sha           pinned OK"
else
    echo "  runner sha           NOT pinned (set EXPECTED_RUNNER_SHA to enforce)"
fi
[[ -x $TREE/.venv/bin/gym ]] || fail "no gym CLI in $TREE"
[[ -f $DATASET ]] || fail "dataset missing: $DATASET"
rows=$(awk 'NF {n++} END {print n+0}' "$DATASET")
(( rows > 0 )) || fail "dataset has no rows: $DATASET"
if [[ ${REQUIRE_LOCAL_REFERENCES:-true} == true ]]; then
    http=$(grep -c '"https\?://' "$DATASET" || true)
    [[ $http == 0 ]] || fail "$http rows still carry remote reference urls"
fi
WORKING_DATASET="$RUN_DIR/input/dataset.jsonl"
gdpval_rollout_materialize_dataset "$DATASET" "$WORKING_DATASET" \
    || fail "could not materialize the immutable shard dataset into $RUN_DIR"
[[ -n ${TAVILY_API_KEY:-} ]] || fail "TAVILY_API_KEY unset"
[[ -f $AGENT_SIF ]] || fail "agent sandbox missing: $AGENT_SIF"
[[ -x $APPTAINER_BIN/apptainer ]] || fail "apptainer missing: $APPTAINER_BIN"
[[ -r $POLICY_SERVE_SCRIPT ]] || fail "POLICY_SERVE_SCRIPT unreadable: $POLICY_SERVE_SCRIPT"

# This tree's activation script can point at another checkout. Call the wrapper
# directly and prove it execs this tree's interpreter. Match on the tree name
# because fs1/fsw aliases can differ.
TREE_NAME=$(basename "$TREE")
grep -q "$TREE_NAME/.venv/bin/python" "$TREE/.venv/bin/gym" \
    || fail "gym wrapper does not exec $TREE_NAME's python: $(sed -n 2p "$TREE/.venv/bin/gym")"
echo "  gym binary           $TREE/.venv/bin/gym (verified)"

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
N_NODES=${#nodes[@]}
(( N_NODES % NODES_PER_REPLICA == 0 )) \
    || fail "$N_NODES nodes is not a multiple of NODES_PER_REPLICA=$NODES_PER_REPLICA"
N_REPLICAS=$((N_NODES / NODES_PER_REPLICA))
echo "  dataset              $rows rows"
echo "  profile              $(basename "$PROFILE") -> $MODEL_NAME"
echo "  topology             $N_NODES node(s) = $N_REPLICAS replica(s) x $NODES_PER_REPLICA, conc=$CONCURRENCY, turns=$AGENT_MAX_TURNS"

export PATH=$HOME/.local/bin:$PATH
export UV_CACHE_DIR=${UV_CACHE_DIR:-$U/uv-cache}
mkdir -p "$UV_CACHE_DIR"
command -v uv >/dev/null || fail "uv not on PATH (needed to start component servers)"
echo "=== assertions passed ==="

gym_pid=
serve_pids=()
replica_dirs=()
BASE_API_PORT=${BASE_API_PORT:-5000}
for ((r = 0; r < N_REPLICAS; r++)); do
    lo=$((r * NODES_PER_REPLICA))
    replica_nodes=$(IFS=,; echo "${nodes[*]:$lo:$NODES_PER_REPLICA}")
    replica_dir="$RUN_DIR/replica_$r/rotation_$ROTATION"
    [[ ! -e $replica_dir && ! -L $replica_dir ]] \
        || fail "refusing stale replica output directory: $replica_dir"
    mkdir -p "$replica_dir"
    replica_dirs+=("$replica_dir")
    REPLICA_NODES="$replica_nodes" \
        OUTPUT_DIR="$replica_dir" \
        API_PORT=$((BASE_API_PORT + r)) \
        LOG_DIR="$RUN_DIR/logs" \
        bash "$POLICY_SERVE_SCRIPT" &
    serve_pids+=("$!")
    echo "  replica $r -> $replica_nodes (port $((BASE_API_PORT + r)))"
done

# `#SBATCH --requeue` alone does not rotate a walltime-expired job. Cleanup can
# run here and again from EXIT, so it must be idempotent.
MAX_ROTATIONS=${MAX_ROTATIONS:-8}
on_walltime() {
    trap - TERM
    echo "=== walltime TERM at rotation ${ROTATION} ==="
    python3 "$ROLLOUT_SHARD_COVERAGE_PY" \
        --dataset "$DATASET" --deliverables "${PERSIST_DELIVERABLES_DIR:-$RUN_DIR/deliverables}" || true
    gdpval_rollout_cleanup
    if (( ROTATION + 1 >= MAX_ROTATIONS )); then
        echo "=== reached MAX_ROTATIONS=${MAX_ROTATIONS}; not requeueing ==="
        exit 0
    fi
    echo "=== requeueing for rotation $((ROTATION + 1)) ==="
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}
trap gdpval_rollout_on_exit EXIT
trap on_walltime TERM

echo "=== waiting for $N_REPLICAS replica(s) ==="
SERVE_WAIT=${SERVE_WAIT_SECONDS:-3600}
policy_urls=()
deadline=$((SECONDS + SERVE_WAIT))
while (( SECONDS < deadline )); do
    policy_urls=()
    for ((r = 0; r < N_REPLICAS; r++)); do
        envf=$(find "${replica_dirs[$r]}/server_info" -maxdepth 1 -type f -name '*.env' -print 2>/dev/null \
            | sort | head -1 || true)
        [[ -n $envf ]] || continue
        url=$(grep -m1 '^SERVER_URL=' "$envf" | cut -d= -f2-)
        [[ -n $url ]] && policy_urls+=("$url")
    done
    (( ${#policy_urls[@]} == N_REPLICAS )) && break
    alive=0
    for pid in "${serve_pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive=$((alive + 1))
    done
    (( alive > 0 )) || {
        echo "FATAL: every serve process exited before readiness" >&2
        tail -40 "$RUN_DIR"/logs/vllm-*.log 2>/dev/null
        exit 1
    }
    sleep 10
done
(( ${#policy_urls[@]} == N_REPLICAS )) || {
    echo "FATAL: readiness timeout after ${SERVE_WAIT}s (${#policy_urls[@]}/$N_REPLICAS)" >&2
    tail -40 "$RUN_DIR"/logs/vllm-*.log 2>/dev/null
    exit 1
}

served=$(curl -sf "${policy_urls[0]}/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
[[ $served == "$MODEL_NAME" ]] || {
    echo "FATAL: served model is $served, expected $MODEL_NAME" >&2
    exit 1
}
echo "=== policy READY ${#policy_urls[@]}/$N_REPLICAS, serving $served ==="

urls_omega="["
sep=
for url in "${policy_urls[@]}"; do
    urls_omega+="${sep}\"${url}\""
    sep=,
done
urls_omega+="]"

export PATH=$APPTAINER_BIN:$PATH
export GDPVAL_CONTAINER_PATH=$AGENT_SIF
# Residue runs use a private RUN_DIR for task-index caches while sharing the
# task-ID-keyed deliverables root with the initial shards.
export PERSIST_DELIVERABLES_DIR=${PERSIST_DELIVERABLES_DIR:-$RUN_DIR/deliverables}
export TMPDIR=$RUN_DIR/tmp
export RAY_TMPDIR=/tmp/gdpval_ray_${SLURM_JOB_ID}_${ROTATION}
mkdir -p "$RAY_TMPDIR"
export EXECUTE_ONLY=true JUDGE_ONLY=false RERUN_INCOMPLETE=true
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS=3 STIRRUP_PER_TASK_TIMEOUT_S=${STIRRUP_PER_TASK_TIMEOUT_S:-12600}
export NEMO_GYM_REASONING_FIELD=${NEMO_GYM_REASONING_FIELD:-both}

cd "$TREE"
GYM=$TREE/.venv/bin/gym

DATASET_OVERLAY=$RUN_DIR/dataset_overlay.yaml
cat > "$DATASET_OVERLAY" <<EOF
gdpval_stirrup_agent:
  responses_api_agents:
    stirrup_agent:
      datasets:
        - name: gdpval
          type: benchmark
          jsonl_fpath: $WORKING_DATASET
          prompt_config: null
          prepare_script: benchmarks/gdpval/prepare.py
          num_repeats: $NUM_REPEATS
EOF

declare -a policy_overrides=()
[[ -n ${USES_REASONING_PARSER:-} ]] \
    && policy_overrides+=("++policy_model.responses_api_models.vllm_model.uses_reasoning_parser=$USES_REASONING_PARSER")
[[ -n ${USES_INTERLEAVED_REASONING:-} ]] \
    && policy_overrides+=("++policy_model.responses_api_models.vllm_model.uses_interleaved_reasoning=$USES_INTERLEAVED_REASONING")

set +e
setsid "$GYM" eval run \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config benchmarks/gdpval/config.yaml \
    --config "$DATASET_OVERLAY" \
    --split benchmark --output "$OUTPUT_JSONL" \
    --model-url "${policy_urls[0]}" --model-api-key dummy --model "$MODEL_NAME" \
    --concurrency "$CONCURRENCY" --num-repeats "$NUM_REPEATS" \
    --max-output-tokens "${MAX_OUTPUT_TOKENS:-262144}" --resume \
    ++overwrite_metrics_conflicts=true \
    ++skip_venv_if_present=true \
    "++policy_model.responses_api_models.vllm_model.base_url=${urls_omega}" \
    ++policy_model.responses_api_models.vllm_model.model="$MODEL_NAME" \
    "${policy_overrides[@]}" \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.agent_max_turns="$AGENT_MAX_TURNS" \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.concurrency="$CONCURRENCY" \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.gdpval_container_path="$AGENT_SIF" \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.persist_deliverables_dir="$PERSIST_DELIVERABLES_DIR" \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.execute_only=true \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.rerun_incomplete=true \
    > >(tee -a "$RUN_DIR/logs/gym_${SLURM_JOB_ID}_${ROTATION}.log") 2>&1 &
gym_pid=$!
wait "$gym_pid"
gym_rc=$?
set -e

echo "=== gym rc=$gym_rc ==="
coverage_rc=0
python3 "$ROLLOUT_SHARD_COVERAGE_PY" \
    --dataset "$DATASET" --deliverables "$PERSIST_DELIVERABLES_DIR" || coverage_rc=$?
if (( gym_rc != 0 )); then
    exit "$gym_rc"
fi
exit "$coverage_rc"
