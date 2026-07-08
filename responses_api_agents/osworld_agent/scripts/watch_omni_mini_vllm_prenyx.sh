#!/usr/bin/env bash
# Wait for Prenyx assets/allocation, start vLLM, and run transport probes.
# Designed to run under nohup on login-prenyx.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 JOB_ID" >&2
  exit 2
fi

JOB_ID="$1"
ROOT="${PRENYX_OMNI_ROOT:-/lustre/fsw/general_sa/${USER}/omni-mini}"
MODEL="${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning}"
IMAGE="${VLLM_IMAGE:-${ROOT}/containers/vllm-openai-v0.20.0.sqsh}"
HF_CACHE="${ROOT}/cache/huggingface"
POLL_SECONDS="${POLL_SECONDS:-30}"

log() { printf '%s %s\n' "$(date -Is)" "$*"; }

while :; do
  state="$(squeue -h -j "${JOB_ID}" -o '%T')"
  case "${state}" in
    RUNNING) log "job ${JOB_ID} is RUNNING"; break ;;
    PENDING) log "job ${JOB_ID} pending" ;;
    "") log "job ${JOB_ID} no longer exists"; exit 1 ;;
    *) log "job ${JOB_ID} entered terminal/unexpected state ${state}"; exit 1 ;;
  esac
  sleep "${POLL_SECONDS}"
done

shard_count="$(find "${HF_CACHE}" -path '*/snapshots/*/model-*-of-*.safetensors' -type l 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${shard_count}" -ne 17 ]]; then
  pid="$(tr -dc '0-9' < "${ROOT}/logs/hf-download.pid" 2>/dev/null || true)"
  while [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; do
    log "waiting for Hugging Face download (${shard_count}/17 linked shards)"
    sleep "${POLL_SECONDS}"
    shard_count="$(find "${HF_CACHE}" -path '*/snapshots/*/model-*-of-*.safetensors' -type l 2>/dev/null | wc -l | tr -d ' ')"
  done
fi
if [[ "${shard_count}" -ne 17 ]]; then
  log "resuming model download from ${shard_count}/17 linked shards"
  env PATH="${HOME}/.local/bin:${PATH}" HF_HUB_DISABLE_XET=1 \
    hf download "${MODEL}" \
      --revision 24e67ea000b7c2837fc8f9488aa2008524fac8ba \
      --cache-dir "${HF_CACHE}" \
      --max-workers 2
  shard_count="$(find "${HF_CACHE}" -path '*/snapshots/*/model-*-of-*.safetensors' -type l 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "${shard_count}" -ne 17 ]]; then
    log "model download ended with only ${shard_count}/17 linked shards"
    exit 1
  fi
fi

log "assets ready; attaching vLLM"
"${ROOT}/src/start_omni_mini_vllm_prenyx.sh" "${JOB_ID}"
node="$(squeue -h -j "${JOB_ID}" -o '%N')"
base_url="http://${node}:8000/v1"

for attempt in $(seq 1 120); do
  if "${ROOT}/src/probe_omni_mini_vllm.py" \
      --base-url "${base_url}" --model "${SERVED_MODEL_NAME}" --models-only; then
    log "vLLM model endpoint ready after ${attempt} probes"
    "${ROOT}/src/probe_omni_mini_vllm.py" --base-url "${base_url}" --model "${SERVED_MODEL_NAME}"
    log "one-image completion probe passed; endpoint=${base_url}"
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done

log "vLLM did not become ready within $((120 * POLL_SECONDS)) seconds"
exit 1
