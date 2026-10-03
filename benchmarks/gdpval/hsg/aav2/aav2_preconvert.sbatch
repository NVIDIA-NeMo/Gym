#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# PHASE 2: prepare candidate copies; no provider requests.
#SBATCH --job-name=aav2-preconvert
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --signal=B:TERM@120

set -euo pipefail
umask 077
: "${AAV2_RUN_DIR:?set AAV2_RUN_DIR}"
AAV2_PACKAGE_DIR=$AAV2_RUN_DIR/package
SLURM_JOB_USER=${SLURM_JOB_USER:-${USER:-}}
[[ $SLURM_JOB_USER =~ ^[A-Za-z0-9._-]+$ && ${SLURM_JOB_ID:-} =~ ^[1-9][0-9]*$ ]] || exit 64
(cd "$AAV2_RUN_DIR" && sha256sum -c --status SHA256SUMS)
install -d -m 0700 "/raid/scratch/$SLURM_JOB_USER"
JOB_ROOT=$(mktemp -d "/raid/scratch/$SLURM_JOB_USER/aav2-preconvert-$SLURM_JOB_ID.XXXXXX")
cp -- "$AAV2_PACKAGE_DIR/node_local.sh" "$JOB_ROOT/node_local.sh"
source "$JOB_ROOT/node_local.sh"
gdpval_prepare preconvert
gdpval_prepare_sandbox "$JUDGE_SIF"
# The SIF needs only stdlib and its existing Office/media tools.
mkdir -p "$JOB_ROOT/container-home"
setsid "$APPTAINER_BIN/apptainer" exec --cleanenv --no-home \
    --bind /lustre:/lustre,/raid/scratch:/raid/scratch \
    --env "PYTHONPATH=$GYM_ROOT,TMPDIR=$TMPDIR,XDG_CACHE_HOME=$XDG_CACHE_HOME" \
    --pwd "$GYM_ROOT" "$GDPVAL_CONTAINER_PATH" env "HOME=$JOB_ROOT/container-home" \
    python3 "$LOCAL_PACKAGE/preconvert.py" "$RUN_DIR" \
    > "$PHASE_DIR/logs/preconvert-$SLURM_JOB_ID.log" 2>&1 &
JOB_PIDS+=("$!")
wait "${JOB_PIDS[${#JOB_PIDS[@]}-1]}"
echo "Prepared candidate: $RUN_DIR/prepared/candidate"
