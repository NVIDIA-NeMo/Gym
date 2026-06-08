#!/usr/bin/env bash
#
# Run SWE-Bench-Ext golden-patch validation through swe_agents.
#
# This script is scheduler-neutral: it does not submit, monitor, or cancel jobs.
# Run it inside the execution environment prepared by your launcher.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_swe_agents_ext_golden_patch.sh \
    --input-jsonl PATH \
    --output-jsonl PATH \
    --task-image-root PATH

Required:
  --input-jsonl PATH              NeMo Gym JSONL containing SWE-Bench-Ext rows.
  --output-jsonl PATH             Rollout result JSONL to write.
  --task-image-root PATH          Directory containing {instance_id}.sif images.

Instead of --task-image-root, you may pass:
  --container-formatter VALUE     Raw swe_agents container_formatter override.

Common options:
  --gym-dir PATH                  Repo root. Defaults to the parent of scripts/.
  --agent-name NAME               Config key and rollout agent name. Default: swe_agents.
  --config-paths VALUE            Comma-separated Hydra config paths.
  --concurrency N                 Default: 1.
  --apptainer-memory-limit-mb N   Default: 65536.
  --test-timeout-seconds N        Default: 1200.
  --resume-from-cache true|false  Default: false.
  --force-agent-ref true|false    Rewrite input agent_ref to --agent-name. Default: true.
  --setup-venv true|false         Run uv venv + uv sync before launch. Default: true.
  --install-runtime-deps true|false
                                  Best-effort apt/apptainer setup inside this environment.
                                  Default: false.
  --policy-base-url URL           Default: http://127.0.0.1:9/v1.
  --policy-api-key VALUE          Default: dummy.
  --policy-model-name VALUE       Default: dummy.

This validates golden patches only. It sets verify_golden_patch=true, so no
real policy model is called.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GYM_DIR="${GYM_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

INPUT_JSONL_FPATH="${INPUT_JSONL_FPATH:-}"
OUTPUT_JSONL_FPATH="${OUTPUT_JSONL_FPATH:-}"
TASK_IMAGE_ROOT="${TASK_IMAGE_ROOT:-}"
CONTAINER_FORMATTER="${CONTAINER_FORMATTER:-}"
AGENT_NAME="${AGENT_NAME:-swe_agents}"
CONFIG_PATHS="${CONFIG_PATHS:-responses_api_agents/swe_agents/configs/swebench_openhands.yaml,responses_api_models/openai_model/configs/openai_model.yaml}"
CONCURRENCY="${CONCURRENCY:-1}"
APPTAINER_MEMORY_LIMIT_MB="${APPTAINER_MEMORY_LIMIT_MB:-65536}"
TEST_TIMEOUT_SECONDS="${TEST_TIMEOUT_SECONDS:-1200}"
RESUME_FROM_CACHE="${RESUME_FROM_CACHE:-false}"
FORCE_AGENT_REF="${FORCE_AGENT_REF:-true}"
SERVER_READY_TIMEOUT_SECONDS="${SERVER_READY_TIMEOUT_SECONDS:-1200}"
SERVER_READY_POLL_SECONDS="${SERVER_READY_POLL_SECONDS:-10}"
HEAD_SERVER_PORT="${HEAD_SERVER_PORT:-11000}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://127.0.0.1:9/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-dummy}"
POLICY_MODEL_NAME="${POLICY_MODEL_NAME:-dummy}"
SETUP_VENV="${SETUP_VENV:-true}"
VENV_DIR="${VENV_DIR:-}"
ISOLATE_RUNTIME_CACHE="${ISOLATE_RUNTIME_CACHE:-true}"
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-false}"
APPTAINER_VERSION="${APPTAINER_VERSION:-1.4.5}"
RUN_TAG="${RUN_TAG:-manual_$$}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --input-jsonl)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      INPUT_JSONL_FPATH="$2"; shift 2 ;;
    --input-jsonl=*)
      INPUT_JSONL_FPATH="${1#*=}"; shift ;;
    --output-jsonl)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      OUTPUT_JSONL_FPATH="$2"; shift 2 ;;
    --output-jsonl=*)
      OUTPUT_JSONL_FPATH="${1#*=}"; shift ;;
    --gym-dir)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      GYM_DIR="$2"; shift 2 ;;
    --gym-dir=*)
      GYM_DIR="${1#*=}"; shift ;;
    --task-image-root)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      TASK_IMAGE_ROOT="$2"; shift 2 ;;
    --task-image-root=*)
      TASK_IMAGE_ROOT="${1#*=}"; shift ;;
    --container-formatter)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      CONTAINER_FORMATTER="$2"; shift 2 ;;
    --container-formatter=*)
      CONTAINER_FORMATTER="${1#*=}"; shift ;;
    --agent-name)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      AGENT_NAME="$2"; shift 2 ;;
    --agent-name=*)
      AGENT_NAME="${1#*=}"; shift ;;
    --config-paths)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      CONFIG_PATHS="$2"; shift 2 ;;
    --config-paths=*)
      CONFIG_PATHS="${1#*=}"; shift ;;
    --concurrency)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      CONCURRENCY="$2"; shift 2 ;;
    --concurrency=*)
      CONCURRENCY="${1#*=}"; shift ;;
    --apptainer-memory-limit-mb)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      APPTAINER_MEMORY_LIMIT_MB="$2"; shift 2 ;;
    --apptainer-memory-limit-mb=*)
      APPTAINER_MEMORY_LIMIT_MB="${1#*=}"; shift ;;
    --test-timeout-seconds)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      TEST_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --test-timeout-seconds=*)
      TEST_TIMEOUT_SECONDS="${1#*=}"; shift ;;
    --resume-from-cache)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      RESUME_FROM_CACHE="$2"; shift 2 ;;
    --resume-from-cache=*)
      RESUME_FROM_CACHE="${1#*=}"; shift ;;
    --force-agent-ref)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      FORCE_AGENT_REF="$2"; shift 2 ;;
    --force-agent-ref=*)
      FORCE_AGENT_REF="${1#*=}"; shift ;;
    --server-ready-timeout-seconds)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      SERVER_READY_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --server-ready-timeout-seconds=*)
      SERVER_READY_TIMEOUT_SECONDS="${1#*=}"; shift ;;
    --server-ready-poll-seconds)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      SERVER_READY_POLL_SECONDS="$2"; shift 2 ;;
    --server-ready-poll-seconds=*)
      SERVER_READY_POLL_SECONDS="${1#*=}"; shift ;;
    --head-server-port)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      HEAD_SERVER_PORT="$2"; shift 2 ;;
    --head-server-port=*)
      HEAD_SERVER_PORT="${1#*=}"; shift ;;
    --policy-base-url)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      POLICY_BASE_URL="$2"; shift 2 ;;
    --policy-base-url=*)
      POLICY_BASE_URL="${1#*=}"; shift ;;
    --policy-api-key)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      POLICY_API_KEY="$2"; shift 2 ;;
    --policy-api-key=*)
      POLICY_API_KEY="${1#*=}"; shift ;;
    --policy-model-name)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      POLICY_MODEL_NAME="$2"; shift 2 ;;
    --policy-model-name=*)
      POLICY_MODEL_NAME="${1#*=}"; shift ;;
    --setup-venv)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      SETUP_VENV="$2"; shift 2 ;;
    --setup-venv=*)
      SETUP_VENV="${1#*=}"; shift ;;
    --venv-dir)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      VENV_DIR="$2"; shift 2 ;;
    --venv-dir=*)
      VENV_DIR="${1#*=}"; shift ;;
    --isolate-runtime-cache)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      ISOLATE_RUNTIME_CACHE="$2"; shift 2 ;;
    --isolate-runtime-cache=*)
      ISOLATE_RUNTIME_CACHE="${1#*=}"; shift ;;
    --install-runtime-deps)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      INSTALL_RUNTIME_DEPS="$2"; shift 2 ;;
    --install-runtime-deps=*)
      INSTALL_RUNTIME_DEPS="${1#*=}"; shift ;;
    --apptainer-version)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      APPTAINER_VERSION="$2"; shift 2 ;;
    --apptainer-version=*)
      APPTAINER_VERSION="${1#*=}"; shift ;;
    --run-tag)
      [ "$#" -ge 2 ] || { echo "Missing value for $1" >&2; exit 1; }
      RUN_TAG="$2"; shift 2 ;;
    --run-tag=*)
      RUN_TAG="${1#*=}"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1 ;;
  esac
done

require_bool() {
  case "$2" in
    true|false) ;;
    *) echo "$1 must be true or false. Got: $2" >&2; exit 1 ;;
  esac
}

require_positive_int() {
  if ! [[ "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "$1 must be a positive integer. Got: $2" >&2
    exit 1
  fi
}

[ -n "$INPUT_JSONL_FPATH" ] || { echo "--input-jsonl is required" >&2; exit 1; }
[ -n "$OUTPUT_JSONL_FPATH" ] || { echo "--output-jsonl is required" >&2; exit 1; }
if [ -z "$TASK_IMAGE_ROOT" ] && [ -z "$CONTAINER_FORMATTER" ]; then
  echo "Either --task-image-root or --container-formatter is required" >&2
  exit 1
fi

require_positive_int CONCURRENCY "$CONCURRENCY"
require_positive_int APPTAINER_MEMORY_LIMIT_MB "$APPTAINER_MEMORY_LIMIT_MB"
require_positive_int TEST_TIMEOUT_SECONDS "$TEST_TIMEOUT_SECONDS"
require_positive_int SERVER_READY_TIMEOUT_SECONDS "$SERVER_READY_TIMEOUT_SECONDS"
require_positive_int SERVER_READY_POLL_SECONDS "$SERVER_READY_POLL_SECONDS"
require_positive_int HEAD_SERVER_PORT "$HEAD_SERVER_PORT"
require_bool RESUME_FROM_CACHE "$RESUME_FROM_CACHE"
require_bool FORCE_AGENT_REF "$FORCE_AGENT_REF"
require_bool SETUP_VENV "$SETUP_VENV"
require_bool ISOLATE_RUNTIME_CACHE "$ISOLATE_RUNTIME_CACHE"
require_bool INSTALL_RUNTIME_DEPS "$INSTALL_RUNTIME_DEPS"

[ -d "$GYM_DIR" ] || { echo "GYM_DIR does not exist: $GYM_DIR" >&2; exit 1; }
GYM_DIR="$(cd "$GYM_DIR" && pwd)"
[ -f "$INPUT_JSONL_FPATH" ] || { echo "Input JSONL does not exist: $INPUT_JSONL_FPATH" >&2; exit 1; }
if [ -n "$TASK_IMAGE_ROOT" ]; then
  [ -d "$TASK_IMAGE_ROOT" ] || { echo "Task image root does not exist: $TASK_IMAGE_ROOT" >&2; exit 1; }
fi
mkdir -p "$(dirname "$OUTPUT_JSONL_FPATH")"

cleanup() {
  if [ -n "${NG_PID:-}" ]; then
    kill "$NG_PID" 2>/dev/null || true
    wait "$NG_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "$GYM_DIR"

if [ "$ISOLATE_RUNTIME_CACHE" = "true" ]; then
  export HOME="${RUNTIME_HOME:-/tmp/home_${RUN_TAG}}"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache_${RUN_TAG}}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/xdg_cache_${RUN_TAG}}"
  export XDG_DATA_HOME="${XDG_DATA_HOME:-/tmp/xdg_data_${RUN_TAG}}"
  export RAY_TMPDIR="${RAY_TMPDIR:-/tmp}"
  export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/tmp/apptainer_cache_${RUN_TAG}}"
  export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-/tmp/apptainer_tmp_${RUN_TAG}}"
  export TMPDIR="${TMPDIR:-/tmp}"
  mkdir -p "$HOME" "$UV_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" \
    "$RAY_TMPDIR" "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" "$TMPDIR"
fi

echo "================================================"
echo "Starting swe_agents golden patch validation"
echo "GYM_DIR: $GYM_DIR"
echo "INPUT_JSONL_FPATH: $INPUT_JSONL_FPATH"
echo "OUTPUT_JSONL_FPATH: $OUTPUT_JSONL_FPATH"
echo "TASK_IMAGE_ROOT: ${TASK_IMAGE_ROOT:-<using explicit container_formatter>}"
echo "AGENT_NAME: $AGENT_NAME"
echo "CONFIG_PATHS: $CONFIG_PATHS"
echo "CONCURRENCY: $CONCURRENCY"
echo "APPTAINER_MEMORY_LIMIT_MB: $APPTAINER_MEMORY_LIMIT_MB"
echo "TEST_TIMEOUT_SECONDS: $TEST_TIMEOUT_SECONDS"
echo "RESUME_FROM_CACHE: $RESUME_FROM_CACHE"
echo "FORCE_AGENT_REF: $FORCE_AGENT_REF"
echo "SETUP_VENV: $SETUP_VENV"
echo "INSTALL_RUNTIME_DEPS: $INSTALL_RUNTIME_DEPS"
echo "================================================"

if [ "$INSTALL_RUNTIME_DEPS" = "true" ]; then
  apt-get update
  apt-get install -y wget git build-essential fuse-overlayfs squashfuse slirp4netns curl
  if ! command -v apptainer >/dev/null 2>&1; then
    tmp_deb="/tmp/apptainer_${APPTAINER_VERSION}_amd64.deb"
    wget -q "https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}_amd64.deb" -O "$tmp_deb"
    apt install -y "$tmp_deb"
  fi
fi

if [ -f /etc/apptainer/apptainer.conf ] && [ -w /etc/apptainer/apptainer.conf ]; then
  sed -i 's/^[#[:space:]]*mount tmp.*/mount tmp = no/' /etc/apptainer/apptainer.conf
  grep -q '^[[:space:]]*mount tmp' /etc/apptainer/apptainer.conf || echo 'mount tmp = no' >> /etc/apptainer/apptainer.conf
fi

if [ "$SETUP_VENV" = "true" ]; then
  command -v uv >/dev/null 2>&1 || { echo "uv is required when --setup-venv=true" >&2; exit 1; }
  VENV_DIR="${VENV_DIR:-$GYM_DIR/.venv}"
  uv venv "$VENV_DIR" --python=3.12 --allow-existing
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  uv sync --active
elif [ -n "$VENV_DIR" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

command -v curl >/dev/null 2>&1 || { echo "curl is required" >&2; exit 1; }
command -v ng_run >/dev/null 2>&1 || { echo "ng_run is required on PATH" >&2; exit 1; }
command -v ng_collect_rollouts >/dev/null 2>&1 || { echo "ng_collect_rollouts is required on PATH" >&2; exit 1; }

COLLECT_INPUT_JSONL_FPATH="$INPUT_JSONL_FPATH"
if [ "$FORCE_AGENT_REF" = "true" ]; then
  input_base="$(basename "$INPUT_JSONL_FPATH")"
  COLLECT_INPUT_JSONL_FPATH="$(dirname "$OUTPUT_JSONL_FPATH")/${input_base%.jsonl}.${AGENT_NAME}.jsonl"
  python3 - "$INPUT_JSONL_FPATH" "$COLLECT_INPUT_JSONL_FPATH" "$AGENT_NAME" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
agent_name = sys.argv[3]
count = 0
missing_ids = 0

dst.parent.mkdir(parents=True, exist_ok=True)
with src.open() as f, dst.open("w") as out:
    for line_number, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception as exc:
            raise SystemExit(f"Invalid JSON in {src}:{line_number}: {exc}") from exc
        metadata = row.get("responses_create_params", {}).get("metadata", {})
        if not (row.get("instance_id") or metadata.get("instance_id")):
            missing_ids += 1
        row["agent_ref"] = {"type": "responses_api_agents", "name": agent_name}
        out.write(json.dumps(row) + "\n")
        count += 1

if count == 0:
    raise SystemExit(f"No task rows found in {src}")
if missing_ids:
    raise SystemExit(f"{missing_ids} task row(s) are missing instance_id in {src}")

print(f"Prepared {count} row(s) for agent_ref={agent_name}: {dst}")
PY
fi

if [ -z "$CONTAINER_FORMATTER" ]; then
  task_image_root_trimmed="${TASK_IMAGE_ROOT%/}"
  CONTAINER_FORMATTER="[\"${task_image_root_trimmed}/{instance_id}.sif\"]"
fi

ng_run "+config_paths=[$CONFIG_PATHS]" \
  "++policy_base_url=$POLICY_BASE_URL" \
  "++policy_api_key=$POLICY_API_KEY" \
  "++policy_model_name=$POLICY_MODEL_NAME" \
  "++${AGENT_NAME}.responses_api_agents.swe_agents.verify_golden_patch=true" \
  "++${AGENT_NAME}.responses_api_agents.swe_agents.dataset_path=$COLLECT_INPUT_JSONL_FPATH" \
  "++${AGENT_NAME}.responses_api_agents.swe_agents.container_formatter=$CONTAINER_FORMATTER" \
  "++${AGENT_NAME}.responses_api_agents.swe_agents.apptainer_memory_limit_mb=$APPTAINER_MEMORY_LIMIT_MB" \
  "++${AGENT_NAME}.responses_api_agents.swe_agents.swebench_tests_timeout=$TEST_TIMEOUT_SECONDS" \
  "++${AGENT_NAME}.responses_api_agents.swe_agents.concurrency=$CONCURRENCY" &

NG_PID=$!
echo "ng_run started with PID $NG_PID"

HEAD_SERVER_URL="http://127.0.0.1:${HEAD_SERVER_PORT}"
deadline=$((SECONDS + SERVER_READY_TIMEOUT_SECONDS))

while ! curl -sf "${HEAD_SERVER_URL}/global_config_dict_yaml" >/dev/null 2>&1; do
  if ! kill -0 "$NG_PID" 2>/dev/null; then
    echo "ERROR: ng_run exited before the NeMo Gym head server became ready" >&2
    wait "$NG_PID"
    exit 1
  fi
  if [ "$SECONDS" -ge "$deadline" ]; then
    echo "ERROR: NeMo Gym head server did not respond at ${HEAD_SERVER_URL}/global_config_dict_yaml within ${SERVER_READY_TIMEOUT_SECONDS}s" >&2
    exit 1
  fi
  echo "Waiting for NeMo Gym head server at ${HEAD_SERVER_URL}..."
  sleep "$SERVER_READY_POLL_SECONDS"
done
echo "NeMo Gym head server ready at ${HEAD_SERVER_URL}"

echo "Running swe_agents golden patch collection..."
start_time=$(date +%s)

ng_collect_rollouts \
  "+agent_name=$AGENT_NAME" \
  "+input_jsonl_fpath=$COLLECT_INPUT_JSONL_FPATH" \
  "+output_jsonl_fpath=$OUTPUT_JSONL_FPATH" \
  +num_repeats=1 \
  "+num_samples_in_parallel=$CONCURRENCY" \
  "+resume_from_cache=$RESUME_FROM_CACHE" \
  +upload_rollouts_to_wandb=false

end_time=$(date +%s)
echo "Time taken: $((end_time - start_time)) seconds"

python3 scripts/swe_agents_golden_patch_summary.py --output-jsonl "$OUTPUT_JSONL_FPATH"

echo "swe_agents golden patch validation complete"
