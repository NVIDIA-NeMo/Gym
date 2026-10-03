#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# PHASE 1: produce each task's deliverables once; judge later on CPU.
#SBATCH --job-name=aav2-rollout
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120

set -euo pipefail
umask 077
unset BASH_ENV
: "${AAV2_RUN_DIR:?set AAV2_RUN_DIR}"
AAV2_PACKAGE_DIR=$AAV2_RUN_DIR/package
SLURM_JOB_USER=${SLURM_JOB_USER:-${USER:-}}
[[ $SLURM_JOB_USER =~ ^[A-Za-z0-9._-]+$ && ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || exit 64
(cd "$AAV2_RUN_DIR" && sha256sum -c --status SHA256SUMS)
install -d -m 0700 "/raid/scratch/$SLURM_JOB_USER"
JOB_ROOT=$(mktemp -d "/raid/scratch/$SLURM_JOB_USER/aav2-$SLURM_JOB_ID.XXXXXX")
[[ $(readlink -f -- "$JOB_ROOT") == /raid/scratch/* && $(stat -f -c '%T' "$JOB_ROOT") != lustre ]] || exit 64
cp -- "$AAV2_PACKAGE_DIR/node_local.sh" "$JOB_ROOT/node_local.sh"
source "$JOB_ROOT/node_local.sh"
gdpval_prepare rollout
gdpval_prepare_serving
"$GYM_PYTHON" "$LOCAL_PACKAGE/rollout_runtime.py" verify \
    --gym-root "$GYM_ROOT" --component-venvs "$COMPONENT_VENVS" --module nemo_gym --module nemo_gym.cli.main
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ -n $nodes && $nodes != *$'\n'* ]] || gdpval_fail "rollout requires exactly one node"
serve_output=$RUN_DIR/serving-$SLURM_JOB_ID
[[ ! -e $serve_output ]] || gdpval_fail "serving output already exists"
mkdir "$serve_output"
REPLICA_NODES=$nodes OUTPUT_DIR=$serve_output API_PORT=${BASE_API_PORT:-5000} LOG_DIR=$RUN_DIR/logs \
    setsid bash "$JOB_ROOT/serve.sh" > "$RUN_DIR/logs/serve-$SLURM_JOB_ID.log" 2>&1 &
serve_pid=$!
JOB_PIDS+=("$serve_pid")
deadline=$((SECONDS + ${SERVE_WAIT_SECONDS:-3600}))
policy_url=
while (( SECONDS < deadline )); do
    kill -0 "$serve_pid" 2>/dev/null || gdpval_fail "serving exited before readiness"
    env_file=$(find "$serve_output/server_info" -maxdepth 1 -type f -name '*.env' -print 2>/dev/null | sort | head -1 || true)
    if [[ -n $env_file ]]; then
        policy_url=$(sed -n 's/^SERVER_URL=//p' "$env_file" | head -1)
        if [[ -n $policy_url ]] && curl -fsS --max-time 5 "$policy_url/models" > "$JOB_ROOT/models.json"; then
            "$GYM_PYTHON" - "$JOB_ROOT/models.json" "$MODEL_NAME" <<'PYTHON'
import json
import sys
with open(sys.argv[1]) as stream:
    assert sys.argv[2] in [model["id"] for model in json.load(stream)["data"]], "served model differs"
PYTHON
            break
        fi
    fi
    policy_url=
    sleep 2
done
[[ -n $policy_url ]] || gdpval_fail "serving readiness timed out"
export EXECUTE_ONLY=true JUDGE_ONLY=false RERUN_INCOMPLETE=true
export PERSIST_DELIVERABLES_DIR=$RUN_DIR/deliverables
policy_options=("++policy_model.responses_api_models.vllm_model.model=$MODEL_NAME")
[[ -z ${USES_REASONING_PARSER:-} ]] || policy_options+=("++policy_model.responses_api_models.vllm_model.uses_reasoning_parser=$USES_REASONING_PARSER")
[[ -z ${USES_INTERLEAVED_REASONING:-} ]] || policy_options+=("++policy_model.responses_api_models.vllm_model.uses_interleaved_reasoning=$USES_INTERLEAVED_REASONING")
# Do not give a task more time than remains after serving startup.
wall_left=$(squeue -h -j "$SLURM_JOB_ID" -o '%L' | awk -F'[-:]' '{
    if (NF==4) print (($1*24+$2)*60+$3)*60+$4;
    else if (NF==3) print ($1*60+$2)*60+$3;
    else if (NF==2) print $1*60+$2;
}')
[[ $wall_left =~ ^[0-9]+$ ]] && (( wall_left > 300 )) || gdpval_fail "insufficient or unknown remaining walltime"
usable=$((wall_left - 300))
export STIRRUP_PER_TASK_TIMEOUT_S=${STIRRUP_PER_TASK_TIMEOUT_S:-$((usable < 12600 ? usable : 12600))}
(( STIRRUP_PER_TASK_TIMEOUT_S > 0 && STIRRUP_PER_TASK_TIMEOUT_S <= usable )) || gdpval_fail "task timeout exceeds remaining walltime"
gdpval_run_gym "++dispatch_budget_s=$usable" --config "$JOB_ROOT/dataset.yaml" \
    --model-url "$policy_url" --model-api-key dummy --model "$MODEL_NAME" \
    --max-output-tokens "${MAX_OUTPUT_TOKENS:-262144}" "${policy_options[@]}" \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.execute_only=true \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.judge_only=false \
    "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.persist_deliverables_dir=$PERSIST_DELIVERABLES_DIR" \
    "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.agent_max_turns=$AGENT_MAX_TURNS" \
    "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.concurrency=$CONCURRENCY" \
    "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.gdpval_container_path=$GDPVAL_CONTAINER_PATH"
"$GYM_PYTHON" "$LOCAL_PACKAGE/completion.py" rollout --dataset "$DATASET" --deliverables "$RUN_DIR/deliverables"
