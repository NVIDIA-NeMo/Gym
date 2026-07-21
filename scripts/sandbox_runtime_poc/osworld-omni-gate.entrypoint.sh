#!/bin/bash
# Resume-aware entrypoint for the in-cluster OSWorld Omni-Nano-v3 domain gate.
# Runs `gym eval run` over a domain jsonl; if a prior pod already produced results,
# auto-adds --resume so an evicted/retried pod continues only the remaining tasks.
set -Eeuo pipefail
export PATH=$HOME/.local/bin:/usr/local/bin:$PATH

# The astral/uv bookworm image lacks the OpenGL/GLib system libs opencv-python needs;
# `import cv2` (via desktop_env evaluator metrics -> easyocr) otherwise crashes EVERY
# eval_task subprocess at import, silently failing all task setups.
if ! ldconfig -p 2>/dev/null | grep -q libGL.so.1; then
  echo "[entrypoint] installing libgl1/libglib2.0-0 (cv2 system deps)"
  apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 >/dev/null
fi

# Repo + runs locations come from the Job env (no site-specific edits needed here).
D="${OSWORLD_REPO_DIR:?set OSWORLD_REPO_DIR in the Job env (repo checkout on the shared volume)}"
cd "$D"

# Inputs (overridable via the Job env). NOTE: the dataset actually collected comes from the
# agent config's `datasets` list (gym's data preparation) — `--input` feeds a different code
# path — so OSWORLD_INPUT rewrites the config's jsonl_fpath in place before launching.
INPUT="${OSWORLD_INPUT:-resources_servers/osworld/data/os_domain.jsonl}"
RUN_NAME="${OSWORLD_RUN_NAME:-os-gate}"
AGENT_CFG=responses_api_agents/nemotron_osworld/configs/nemotron_osworld.yaml
sed -i -E "s#jsonl_fpath: resources_servers/osworld/data/.*\.jsonl#jsonl_fpath: ${INPUT}#" "$AGENT_CFG"
echo "[entrypoint] dataset -> $(grep jsonl_fpath "$AGENT_CFG" | head -1 | tr -d ' ')"
CONCURRENCY="${OSWORLD_CONCURRENCY:-8}"
OUT_DIR="${OSWORLD_RUNS_DIR:?set OSWORLD_RUNS_DIR in the Job env}/${RUN_NAME}"
OUT="${OUT_DIR}/results.jsonl"
mkdir -p "$OUT_DIR"

# First run on a fresh checkout: build the root venv (per-server venvs are
# created by gym itself; skip_venv_if_present makes retries fast).
if [ ! -x .venv/bin/gym ]; then
  echo "[entrypoint] bootstrapping root venv"
  uv venv .venv
  VIRTUAL_ENV="$PWD/.venv" uv sync --extra sandbox
fi

# A failed requirements install leaves a half-built venv that
# skip_venv_if_present would then trust forever — detect and remove those
# so gym's per-server setup rebuilds them.
venv_ok() {
  case "$1" in
    resources_servers/osworld)
      "$1/.venv/bin/python" -c "import fastapi, ray, cryptography; ray.__version__; from desktop_env.evaluators.getters import chrome" 2>/dev/null ;;
    responses_api_agents/nemotron_osworld)
      "$1/.venv/bin/python" -c "import fastapi, ray; ray.__version__; import mm_agents.nvidia.nemotron_agent" 2>/dev/null ;;
    *)
      "$1/.venv/bin/python" -c "import fastapi, ray; ray.__version__" 2>/dev/null ;;
  esac
}
for srv in resources_servers/osworld responses_api_agents/nemotron_osworld responses_api_models/vllm_model; do
  if [ -d "$srv/.venv" ] && ! venv_ok "$srv"; then
    echo "[entrypoint] $srv/.venv fails its import sentinel (half-installed) — removing for rebuild"
    rm -rf "$srv/.venv"
  fi
done

RESUME=""
if [ -s "$OUT" ]; then
  echo "[entrypoint] found existing results ($(wc -l <"$OUT") rows) — resuming"
  RESUME="--resume"
fi

echo "[entrypoint] input=$INPUT out=$OUT concurrency=$CONCURRENCY resume=${RESUME:-none}"
echo "[entrypoint] OPENSANDBOX_DOMAIN=$OPENSANDBOX_DOMAIN model=$POLICY_BASE_URL"

exec .venv/bin/gym eval run \
  --config responses_api_agents/nemotron_osworld/configs/nemotron_osworld.yaml \
  --config resources_servers/osworld/configs/osworld.yaml \
  --config resources_servers/osworld/configs/opensandbox_osworld.yaml \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml \
  --agent nemotron_osworld \
  --model "${POLICY_MODEL_NAME:-vllm_local}" \
  --model-url "$POLICY_BASE_URL" \
  --model-api-key "${POLICY_API_KEY:-EMPTY}" \
  --input "$INPUT" \
  --split validation \
  --concurrency "$CONCURRENCY" \
  --output "$OUT" \
  $RESUME \
  "+skip_venv_if_present=true"
