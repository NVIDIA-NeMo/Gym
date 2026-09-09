#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# PHASE 3: judge the prepared deliverables with the API panel; no policy GPU.
#SBATCH --job-name=aav2-judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=184G
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
AAV2_MODE=${AAV2_MODE:-full}
gdpval_prepare judge
"$GYM_PYTHON" "$LOCAL_PACKAGE/rollout_runtime.py" verify --without-sandbox \
    --gym-root "$GYM_ROOT" --component-venvs "$COMPONENT_VENVS" --module nemo_gym --module nemo_gym.cli.main
[[ -f $RUN_DIR/prepared/manifest.json ]] || gdpval_fail "run preconvert before judging"
"$GYM_PYTHON" "$LOCAL_PACKAGE/preconvert.py" "$RUN_DIR"
export EXECUTE_ONLY=false JUDGE_ONLY=true RERUN_INCOMPLETE=true
export PERSIST_DELIVERABLES_DIR=$RUN_DIR/prepared/candidate
gdpval_run_gym --config "$LOCAL_PACKAGE/true3_transport.yaml" \
    --config "$JOB_ROOT/judge.yaml" --config "$RUN_DIR/prepared/judge.yaml" --config "$JOB_ROOT/dataset.yaml" \
    --model-url http://127.0.0.1:9/v1 --model-api-key dummy --model judge-only \
    ++model_endpoint_readiness_timeout_seconds=0 \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.execute_only=false \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.judge_only=true \
    ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.gdpval_container_path=null \
    "++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.persist_deliverables_dir=$PERSIST_DELIVERABLES_DIR" \
    ++gdpval_resources_server.resources_servers.gdpval.reward_mode=comparison \
    ++gdpval_resources_server.resources_servers.gdpval.strict_comparison_trials=true \
    "++gdpval_resources_server.resources_servers.gdpval.num_comparison_trials=$NUM_COMPARISON_TRIALS" \
    ++gdpval_resources_server.resources_servers.gdpval.preconvert_office_to_pdf=false \
    ++multistage.enabled=true "++multistage.stages=$STAGES" "++multistage.dataset_path=$DATASET" ++multistage.seed=42
"$GYM_PYTHON" "$LOCAL_PACKAGE/completion.py" judge --output "$PHASE_DIR/rollouts.jsonl" --dataset "$DATASET" --mode "$AAV2_MODE"
