#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --job-name=smoke-apex-sif
#SBATCH --account=nemotron_n3_post
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm-logs/%x-%j.log

set -Eeuo pipefail

readonly APEX_GYM_DIR=${APEX_GYM_DIR:-/lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/artij/ApexGym}
readonly APEX_IMAGE=${APEX_IMAGE:-${APEX_GYM_DIR}/responses_api_agents/apex_agent/images/archipelago-fixed.sif}
readonly GYM_CONTAINER=${GYM_CONTAINER:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/images/high_stripe/rl.48761755.sqsh}
readonly USER_NAME=${USER_NAME:-${USER:-$(whoami)}}

[[ -s "${APEX_IMAGE}" ]] || { echo "ERROR: image is missing or empty: ${APEX_IMAGE}" >&2; exit 66; }
[[ -r "${GYM_CONTAINER}" ]] || { echo "ERROR: outer Gym container is unreadable: ${GYM_CONTAINER}" >&2; exit 66; }

mkdir -p "${APEX_GYM_DIR}/results/slurm-logs"

export APEX_GYM_DIR APEX_IMAGE USER_NAME
export PATH="/home/${USER_NAME}/apptainer/bin:/home/${USER_NAME}/apptainer/usr/bin:${PATH}"

srun \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
    --kill-on-bad-exit=1 \
    --export=ALL \
    --no-container-mount-home \
    --container-image="${GYM_CONTAINER}" \
    --container-mounts="/lustre:/lustre,/home:/home" \
    --container-workdir="${APEX_GYM_DIR}" \
    --container-env="APEX_GYM_DIR,APEX_IMAGE,USER_NAME,PATH" \
    bash -s <<'INNER'
set -Eeuo pipefail

runtime=$(command -v apptainer || command -v singularity || true)
[[ -n "${runtime}" ]] || { echo "ERROR: Apptainer/Singularity is unavailable in the outer Pyxis container" >&2; exit 69; }

echo "Outer identity: $(id)"
echo "Host architecture: $(uname -m)"
echo "Runtime: ${runtime}"
"${runtime}" version
"${runtime}" inspect "${APEX_IMAGE}"

direct_exec_args=(
    --writable-tmpfs
    --cleanenv
    --pid
    --no-mount "home,tmp,bind-paths"
    --home /root
)

"${runtime}" exec "${direct_exec_args[@]}" "${APEX_IMAGE}" bash -s <<'IMAGE'
set -Eeuo pipefail

echo "Inner architecture: $(uname -m)"
case "$(uname -m)" in
    aarch64|arm64) ;;
    *) echo "ERROR: expected ARM64/aarch64 image execution" >&2; exit 1 ;;
esac

test -x /usr/bin/python3
test -x /app/.venv/bin/python
test -x /app/mcp_servers/code/.venv/bin/python3
test -x /app/.venv/bin/uvicorn
test -f /app/lib/sandbox_fs.so

/usr/bin/python3 - <<'PY'
import docx
import duckdb
import matplotlib
import numpy
import openpyxl
import pandas
import pdfplumber
import pptx
import pypdf
import scipy
import sklearn
import statsmodels
import xgboost

frame = pandas.DataFrame({"x": [1, 2, 3]})
assert int(frame["x"].sum()) == 6
print("system Python OK", numpy.__version__, pandas.__version__)
PY

/app/.venv/bin/python - <<'PY'
import numpy
import pandas

print("app venv Python OK", numpy.__version__, pandas.__version__)
PY

/app/mcp_servers/code/.venv/bin/python3 - <<'PY'
import fastmcp
import mcp
import numpy
import pandas

print("code MCP venv OK", numpy.__version__, pandas.__version__)
PY

cd /app/mcp_servers/code/mcp_servers/code_execution_server
/app/mcp_servers/code/.venv/bin/python3 -c 'import main; print("code MCP server import OK")'

mkdir -p /filesystem /.apps_data /app/apex-gym/output
touch /filesystem/.image-smoke-write
touch /.apps_data/.image-smoke-write
touch /app/apex-gym/output/.image-smoke-write
echo "writable sandbox paths OK"

cd /app
/app/.venv/bin/uvicorn runner.main:app --host 127.0.0.1 --port 18080 >/tmp/apex-gateway-smoke.log 2>&1 &
gateway_pid=$!
cleanup() {
    kill "${gateway_pid}" 2>/dev/null || true
    wait "${gateway_pid}" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:18080/health >/tmp/apex-gateway-health.json; then
        cat /tmp/apex-gateway-health.json
        echo
        echo "Archipelago gateway health OK"
        echo "APEX_IMAGE_SMOKE_PASS"
        exit 0
    fi
    if ! kill -0 "${gateway_pid}" 2>/dev/null; then
        echo "ERROR: Archipelago gateway exited early" >&2
        cat /tmp/apex-gateway-smoke.log >&2
        exit 1
    fi
    sleep 1
done

echo "ERROR: Archipelago gateway did not become healthy" >&2
cat /tmp/apex-gateway-smoke.log >&2
exit 1
IMAGE
INNER
