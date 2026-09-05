#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=gdpval-e2e-preconvert
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --open-mode=append

set -euo pipefail
umask 077
export PYTHONUNBUFFERED=1
export PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

: "${RUN_DIR:?set RUN_DIR}"
: "${CHECKPOINT_E2E_EXECUTION_PACKAGE:?set CHECKPOINT_E2E_EXECUTION_PACKAGE}"

# Preserve the opt-in run-owned package before the campaign settings restore
# the package that originally prepared the run.
REQUESTED_ACTIVE_PACKAGE=${CHECKPOINT_E2E_ACTIVE_PACKAGE:-}

RUN_ID=$(sed -n 's/^RUN_ID=//p' "$RUN_DIR/settings.env")
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+-[0-9a-f]{16}$ ]] || {
    echo "CHECKPOINT_E2E_PRECONVERT_FAIL: invalid run id" >&2; exit 64;
}
JOB_USER=${SLURM_JOB_USER:-${USER:-}}
[[ $JOB_USER =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "CHECKPOINT_E2E_PRECONVERT_FAIL: invalid Slurm user" >&2; exit 64;
}
BOOTSTRAP_ROOT=/raid/scratch/${JOB_USER}/gdpval-e2e-bootstrap-${SLURM_JOB_ID}-preconvert
install -d -m 0700 "$BOOTSTRAP_ROOT"
[[ $(stat -f -c '%T' "$BOOTSTRAP_ROOT") != lustre ]] || {
    echo "CHECKPOINT_E2E_PRECONVERT_FAIL: bootstrap root resolves to Lustre" >&2; exit 64;
}
cp -- "$CHECKPOINT_E2E_EXECUTION_PACKAGE/mars_node_local.sh" "$BOOTSTRAP_ROOT/mars_node_local.sh"
chmod 0500 "$BOOTSTRAP_ROOT/mars_node_local.sh"
# shellcheck disable=SC1090
source "$BOOTSTRAP_ROOT/mars_node_local.sh"
mars_init "$RUN_ID" preconvert "$RUN_DIR"
mars_stage_package "$CHECKPOINT_E2E_EXECUTION_PACKAGE"
mars_stage_file "$RUN_DIR/settings.env" "$MARS_JOB_ROOT/settings.env"
# shellcheck disable=SC1090
source "$MARS_JOB_ROOT/settings.env"
CAMPAIGN_E2E_DIR=$E2E_DIR
if [[ -n $REQUESTED_ACTIVE_PACKAGE ]]; then
    [[ $REQUESTED_ACTIVE_PACKAGE == "$RUN_DIR"/* \
        && -d $REQUESTED_ACTIVE_PACKAGE && ! -L $REQUESTED_ACTIVE_PACKAGE \
        && $REQUESTED_ACTIVE_PACKAGE != "$CAMPAIGN_E2E_DIR" ]] || {
        echo "CHECKPOINT_E2E_PRECONVERT_FAIL: corrected package is invalid or aliases the campaign package" >&2
        exit 64
    }
fi
mars_stage_gym "$GYM_ROOT" "$GYM_REVISION"
mars_stage_container "$GDPVAL_SIF" "$GDPVAL_SIF_SIGNATURE"
E2E_DIR=$MARS_PACKAGE
GYM_ROOT=$MARS_GYM
GDPVAL_SIF=$MARS_GDPVAL_SIF

attempt=${PRECONVERT_ATTEMPT:-1}
receipt="$RUN_DIR/preconvert_attempt_${attempt}.json"
scratch="$MARS_JOB_ROOT/preconvert/attempt_${attempt}"
install -d -m 0700 "$scratch"

env -u SLURM_CPUS_PER_TASK -u SLURM_TRES_PER_TASK \
    srun --cpus-per-task=16 \
    --container-image="$GDPVAL_SIF" \
    --container-mounts=/lustre:/lustre,"$MARS_BASE":"$MARS_BASE" \
    --no-container-mount-home \
    python3 "$E2E_DIR/preconvert_closure.py" convert \
        --root "$DELIVERABLES" \
        --preconvert-py "$GYM_ROOT/resources_servers/gdpval/preconvert.py" \
        --scratch "$scratch" \
        --receipt "$receipt" \
        --workers 8
