#!/bin/bash
# 03 - Run a sweep against any OpenAI-compatible endpoint. No Slurm, no sbatch, no GPUs.
#
# Takes a URL instead of allocating the policy: any reachable vLLM, gateway, or hosted API.
# Only the Gym servers and the collection driver run locally.
#
# USAGE
#   SWEEP_DIR=<out>/<nickname> POLICY_BASE_URL=http://host:8000/v1 \
#     POLICY_MODEL_NAME=<model> POLICY_API_KEY=<key> bash $R/scripts/03_run_endpoint.sh
#
# REQUIRED
#   SWEEP_DIR               the OUT_DIR/<nickname> directory 01 wrote
#   POLICY_BASE_URL         e.g. http://host:8000/v1
#   POLICY_MODEL_NAME       the served model
#
# OPTIONAL
#   POLICY_API_KEY          dummy_api_key
#   NUM_SAMPLES_IN_PARALLEL 128
#   SERVERS_READY_TIMEOUT_S 1800
#   ENV_PORT_RANGE_LOW/HIGH 20000 / 30000
#
# Needs `gym` on PATH: activate the venv (see README 00) or run inside the eval container.
# For the sandbox entries also set NEMO_SKILLS_SANDBOX_HOST/PORT at a reachable sandbox; without
# one they fail per rollout rather than at startup.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to the <out-dir>/<nickname> directory 01_materialize.sh wrote}
POLICY_BASE_URL=${POLICY_BASE_URL:?set POLICY_BASE_URL, e.g. http://host:8000/v1}
POLICY_MODEL_NAME=${POLICY_MODEL_NAME:?set POLICY_MODEL_NAME to the served model}
POLICY_API_KEY=${POLICY_API_KEY:-dummy_api_key}

# Settings the manifest declared, applied only where this environment leaves them unset, so
# precedence stays manifest -> env var -> command line as everywhere else.
while IFS='=' read -r _k _v; do
    [[ -n "$_k" ]] || continue
    [[ -n "${!_k:-}" ]] || export "$_k=$_v"
done < <(python - "$SWEEP_DIR" "srun,gym_eval_profile" <<'PY_MANIFEST'
import json, sys
from pathlib import Path
try:
    doc = json.loads((Path(sys.argv[1]) / "sweep_report.json").read_text())
except OSError:
    doc = {}
for block in sys.argv[2].split(","):
    for key, value in (doc.get(block) or {}).items():
        print(f"{key}={value}")
PY_MANIFEST
)

# The manifest's gym_eval_run keys, rendered as ++ overrides. Same treatment 03_run_single.sh
# gives them; this script used to hardcode the port ranges and the connector limit, which the
# manifest is supposed to own.
MANIFEST_OVERRIDES=$(python - "$SWEEP_DIR" <<'PY_RUN'
import json, os, sys
from pathlib import Path
try:
    doc = json.loads((Path(sys.argv[1]) / "sweep_report.json").read_text())
except OSError:
    doc = {}
out = []
for key, value in (doc.get("gym_eval_run") or {}).items():
    if key == "num_samples_in_parallel" or os.environ.get(key.upper()):
        continue
    out.append(f"++{key}={value}")
print(" ".join(out))
PY_RUN
)

_manifest_concurrency=$(python - "$SWEEP_DIR" <<'PY_C'
import json, sys
from pathlib import Path
try:
    print((json.loads((Path(sys.argv[1]) / "sweep_report.json").read_text()).get("gym_eval_run") or {}).get("num_samples_in_parallel", ""))
except OSError:
    print("")
PY_C
)
NUM_SAMPLES_IN_PARALLEL=${NUM_SAMPLES_IN_PARALLEL:-${_manifest_concurrency:-128}}
SERVERS_READY_TIMEOUT_S=${SERVERS_READY_TIMEOUT_S:-1800}
ENV_PORT_RANGE_LOW=${ENV_PORT_RANGE_LOW:-20000}
ENV_PORT_RANGE_HIGH=${ENV_PORT_RANGE_HIGH:-30000}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

if ! command -v gym >/dev/null 2>&1; then
    echo "ERROR: 'gym' is not on PATH." >&2
    echo "       source .venv/bin/activate  (see README 00 - Setup), or run in the eval container." >&2
    exit 2
fi

env_start_log=$SWEEP_DIR/env_start.log
: > "$env_start_log"

# No ++use_absolute_ip: it binds servers to the node IP, which puts the head server somewhere other
# than where gym eval run --no-serve looks for it.
gym env start --config "$SWEEP_DIR/sweep_config.yaml" \
    +uv_venv_dir=/opt/uv_venvs \
    +skip_venv_if_present=true \
    ++port_range_low="$ENV_PORT_RANGE_LOW" \
    ++port_range_high="$ENV_PORT_RANGE_HIGH" \
    ++policy_base_url="$POLICY_BASE_URL" \
    ++policy_api_key="$POLICY_API_KEY" \
    ++policy_model_name="$POLICY_MODEL_NAME" > "$env_start_log" 2>&1 &
gym_servers_pid=$!
trap 'kill $gym_servers_pid 2>/dev/null || true' EXIT

echo ">>> waiting for Gym servers (log: $env_start_log)"
for _i in $(seq 1 $((SERVERS_READY_TIMEOUT_S / 10))); do
    if grep -q "servers ready!" "$env_start_log" 2>/dev/null; then
        grep -m1 -oE "All [0-9]+ / [0-9]+ servers ready!" "$env_start_log"
        break
    fi
    if ! kill -0 "$gym_servers_pid" 2>/dev/null; then
        echo "ERROR: gym env start exited before the servers came up:" >&2
        tail -30 "$env_start_log" >&2 || true
        exit 1
    fi
    sleep 10
done
if ! grep -q "servers ready!" "$env_start_log" 2>/dev/null; then
    echo "ERROR: servers not ready within ${SERVERS_READY_TIMEOUT_S}s; see $env_start_log" >&2
    tail -30 "$env_start_log" >&2 || true
    exit 1
fi

# num_repeats=1 because 01_materialize.sh already wrote num_repeats copies of every row.
echo ">>> collecting"
gym eval run --no-serve --resume \
    --input "$SWEEP_DIR/rollouts_materialized_inputs.jsonl" \
    --output "$SWEEP_DIR/rollouts.jsonl" \
    ++num_repeats=1 \
    ++num_samples_in_parallel="$NUM_SAMPLES_IN_PARALLEL" \
    +nemo_gym_log_dir="$SWEEP_DIR/logs" \
    +uv_venv_dir=/opt/uv_venvs \
    +skip_venv_if_present=true \
    $MANIFEST_OVERRIDES \
    ++global_aiohttp_connector_limit_per_host=16384 \
    ++port_range_low=63000 \
    ++port_range_high=64000

echo ">>> profiling"
SWEEP_DIR="$SWEEP_DIR" bash "$(dirname "${BASH_SOURCE[0]}")/05_profile.sh"
