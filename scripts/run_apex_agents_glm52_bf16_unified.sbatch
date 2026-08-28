#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# GLM-5.2 BF16 Slurm entrypoint. The submit launcher overrides the 16-node
# defaults for the supported 4-node DP=4 and 8-node DP=8 topologies. Rollout sharding,
# node-local Gym workers, resume/retry, requeue, and merging are implemented
# by the shared unified harness used by the Kimi profile.

#SBATCH --job-name=apex-glm52-bf16
#SBATCH --account=nemotron_n4_post
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=16
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=144
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --switches=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"480","reason":"benchmarking","description":"Apex Agents GLM-5.2 BF16 rollouts"}}'
#SBATCH --output=results/slurm-logs/%x-%j.log

set -Eeuo pipefail

readonly DEFAULT_APEX_GYM_DIR="/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/artij/ApexGym"
export APEX_GYM_DIR=${APEX_GYM_DIR:-${DEFAULT_APEX_GYM_DIR}}
export PROFILE=${PROFILE:-${APEX_GYM_DIR}/scripts/profiles/apex-glm52-bf16.env}
export RUN_DIR=${RUN_DIR:-${APEX_GYM_DIR}/results/apex_glm52_bf16_unified/manual}
# Keep direct sbatch submissions on the same validated execution path as the
# submit launcher. An explicit environment override can re-enable compilation.
export GLM52_ENFORCE_EAGER=${GLM52_ENFORCE_EAGER:-true}

exec bash "${APEX_GYM_DIR}/scripts/run_apex_agents_k3_unified.sbatch" "$@"
