#!/usr/bin/env bash
# Start the official Omni Mini BF16 checkpoint with vLLM 0.20.0.
# Intended for one B200/H200/H100-80GB GPU. Extra command-line arguments are
# appended verbatim, for example: ./launch_omni_mini_vllm.sh --download-dir /lustre/cache

set -euo pipefail

MODEL="${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
MODEL_REVISION="${MODEL_REVISION:-24e67ea000b7c2837fc8f9488aa2008524fac8ba}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-64000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_BIN="${VLLM_BIN:-vllm}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
  "${VLLM_BIN}" serve "${MODEL}"
  --revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --trust-remote-code
  --reasoning-parser nemotron_v3
  --allowed-local-media-path /
)
cmd+=("$@")

printf ' %q' "${cmd[@]}"
echo

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

version="$(${VLLM_BIN} --version 2>&1 || true)"
if [[ "${version}" != *"0.20.0"* && "${ALLOW_UNSUPPORTED_VLLM:-0}" != "1" ]]; then
  echo "Omni Mini requires vLLM 0.20.0; found: ${version:-unavailable}" >&2
  echo "Set ALLOW_UNSUPPORTED_VLLM=1 only for an intentional compatibility probe." >&2
  exit 2
fi

exec "${cmd[@]}"
