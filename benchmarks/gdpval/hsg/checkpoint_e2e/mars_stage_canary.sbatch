#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provider-free node-local runtime staging canary.
#SBATCH --job-name=gdpval-mars-stage-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --no-requeue
#SBATCH --open-mode=append

set -euo pipefail
umask 077
export PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
: "${RUN_DIR:?set RUN_DIR}"
: "${CHECKPOINT_E2E_EXECUTION_PACKAGE:?set CHECKPOINT_E2E_EXECUTION_PACKAGE}"

RUN_ID=$(sed -n 's/^RUN_ID=//p' "$RUN_DIR/settings.env")
[[ $RUN_ID =~ ^[A-Za-z0-9._-]+-[0-9a-f]{16}$ ]]
JOB_USER=${SLURM_JOB_USER:-${USER:-}}
[[ $JOB_USER =~ ^[A-Za-z0-9._-]+$ ]]
BOOTSTRAP_ROOT=/raid/scratch/${JOB_USER}/gdpval-e2e-bootstrap-${SLURM_JOB_ID}-canary
install -d -m 0700 "$BOOTSTRAP_ROOT"
[[ $(stat -f -c '%T' "$BOOTSTRAP_ROOT") != lustre ]]
cp -- "$CHECKPOINT_E2E_EXECUTION_PACKAGE/mars_node_local.sh" "$BOOTSTRAP_ROOT/mars_node_local.sh"
chmod 0500 "$BOOTSTRAP_ROOT/mars_node_local.sh"
# shellcheck disable=SC1090
source "$BOOTSTRAP_ROOT/mars_node_local.sh"
mars_init "$RUN_ID" canary "$RUN_DIR"
mars_stage_package "$CHECKPOINT_E2E_EXECUTION_PACKAGE"
mars_stage_file "$RUN_DIR/settings.env" "$MARS_JOB_ROOT/settings.env"
# shellcheck disable=SC1090
source "$MARS_JOB_ROOT/settings.env"
mars_stage_gym "$GYM_ROOT" "$GYM_REVISION"
mars_stage_runtime
mars_stage_apptainer "$APPTAINER_BIN" "$APPTAINER_SIGNATURE"
mars_stage_container "$GDPVAL_SIF" "$GDPVAL_SIF_SIGNATURE"
mars_stage_uv

[[ $MARS_PACKAGE == /raid/scratch/* && $MARS_GYM == /raid/scratch/* \
    && $MARS_RUNTIME == /raid/scratch/* && $MARS_APPTAINER_BIN == /raid/scratch/* \
    && $MARS_GDPVAL_SIF == /raid/scratch/* && $TMPDIR == /raid/scratch/* \
    && $MARS_UV == /raid/scratch/* && $UV_CACHE_DIR == /raid/scratch/* \
    && $APPTAINER_CACHEDIR == /raid/scratch/* ]]
(( ${#RAY_TMPDIR} <= 48 ))
PYTHONPATH="$MARS_RUNTIME:$MARS_GYM" "$MARS_PYTHON" - <<'PY'
from pathlib import Path
import nemo_gym
import resources_servers.gdpval.comparison as comparison

for module in (nemo_gym, comparison):
    path = Path(module.__file__).resolve()
    if not str(path).startswith("/raid/scratch/"):
        raise SystemExit(f"module resolved outside node-local storage: {path}")
print("MARS_PYTHON_IMPORT_PASS")
PY
PYTHONPATH="$MARS_RUNTIME:$MARS_GYM" "$MARS_PYTHON" \
    "$MARS_PACKAGE/gym_entrypoint.py" --help >/dev/null
"$MARS_APPTAINER_BIN/apptainer" --version
[[ $(command -v "$MARS_UV") == "$MARS_UV" ]]
"$MARS_UV" --version
"$MARS_APPTAINER_BIN/apptainer" exec --containall --no-home \
    --bind "$MARS_BASE:$MARS_BASE:ro" "$MARS_GDPVAL_SIF" \
    env PYTHONPATH="$MARS_RUNTIME:$MARS_GYM" python3 -c \
    'from pathlib import Path; import nemo_gym; import resources_servers.gdpval.comparison as c; assert all(str(Path(m.__file__).resolve()).startswith("/raid/scratch/") for m in (nemo_gym,c))'
[[ $(stat -c '%s' "$MARS_GDPVAL_SIF") == "${GDPVAL_SIF_SIGNATURE%%:*}" ]]
printf 'MARS_STAGE_CANARY_PASS node=%s package=%s gym=%s runtime=%s sif=%s\n' \
    "${SLURMD_NODENAME:-unknown}" "$MARS_PACKAGE" "$MARS_GYM" "$MARS_RUNTIME" "$MARS_GDPVAL_SIF"
