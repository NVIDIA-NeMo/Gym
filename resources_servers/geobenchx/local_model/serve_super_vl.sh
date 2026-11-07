#!/usr/bin/env bash
# Serve one Nemotron 3.5 Super VL GA candidate with vLLM, following its model
# card. Candidates are evaluated one at a time: each takes the whole 8x B200
# node as data-parallel replicas of the card's tensor-parallel layout.
# OpenAI-compatible API on http://127.0.0.1:${PORT}/v1.
#
#   serve_super_vl.sh bf16|nvfp4|lilicorr|base
#
#   variant   GPUs  TP x DP  served name                   vLLM image
#   bf16      0-7   2 x 4    nvidia/super-vl-ga-bf16       v0.30.0   (training start)
#   nvfp4     0-7   2 x 4    nvidia/super-vl-ga-nvfp4      v0.31.0
#   lilicorr  0     1 x 1    nvidia/super-vl-ga-lilicorr   v0.31.0   (NVFP4 target + LiLiCorr draft)
#   base      0-7   2 x 4    nvidia/super-vl-base          v0.30.0   (pretrained only)
#
# LiLiCorr is tuned for low-concurrency serving, so it runs as one replica with
# few sequences; measure its throughput and latency with nothing else running.
#
#   MODELS_DIR    directory holding the downloaded checkpoints (default below)
#   GPUS, TP, PORT (default 8300), MAX_NUM_SEQS (per replica), MAX_MODEL_LEN
#                 override the per-variant defaults; DP is (number of GPUS) / TP
#   VLLM_ENGINE_READY_TIMEOUT_S  startup timeout (default 3600; checkpoints on NFS
#                 can take most of an hour to load)
#   FORCE=1       start even if the selected GPUs have memory in use
#   PYTHON        interpreter for the checkpoint check (default /usr/bin/python3)
#
# Stop with: docker rm -f super-vl-<variant>
set -euo pipefail

VARIANT="${1:?usage: $0 bf16|nvfp4|lilicorr|base}"
MODELS_DIR="${MODELS_DIR:-/datasets/pwilson/models}"
BF16=Nemotron-3.5-Super-VL-120B-A12B-GA-candidate-MTP-boosted-20261001
NVFP4=Nemotron-Super-3.5-GA-FINAL-row105-QAD-PreStitched-BoostedMTP
LILICORR=NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-LiLiCorr
BASE=NVIDIA-Nemotron-3.5-Super-midtrain-67B-vision-pretrained

MAMBA_ARGS=(
  --mamba-backend flashinfer
  --mamba-ssm-cache-dtype float16
  --enable-mamba-cache-stochastic-rounding
  --mamba-cache-philox-rounds 5
)
TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser qwen3_coder)
DRAFT_MOUNT=()

case "${VARIANT}" in
  bf16)
    CKPT="${BF16}"; IMAGE=vllm/vllm-openai:v0.30.0
    : "${GPUS:=0,1,2,3,4,5,6,7}" "${TP:=2}" "${MAX_NUM_SEQS:=32}"
    SERVED=nvidia/super-vl-ga-bf16
    ARGS=(
      --gpu-memory-utilization 0.85
      --async-scheduling
      --limit-mm-per-prompt '{"video": 1, "image": 1}'
      --video-pruning-rate 0.5
      --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
      --speculative_config.method mtp
      --speculative_config.num_speculative_tokens 3
      "${TOOL_ARGS[@]}" --reasoning-parser nemotron_v3
    )
    ;;
  nvfp4)
    CKPT="${NVFP4}"; IMAGE=vllm/vllm-openai:v0.31.0
    : "${GPUS:=0,1,2,3,4,5,6,7}" "${TP:=2}" "${MAX_NUM_SEQS:=128}"
    SERVED=nvidia/super-vl-ga-nvfp4
    ARGS=(
      --gpu-memory-utilization 0.89
      --max-num-batched-tokens 16384
      --block-size 64
      --prefix-match-unit 64
      --compilation_config.max_cudagraph_capture_size 256
      "${MAMBA_ARGS[@]}"
      --enable-mamba-shared-prefix-checkpoint
      --mamba-cache-mode align
      --speculative-config.method mtp
      --speculative-config.num_speculative_tokens 3
      --video-pruning-rate 0.5
      --media-io-kwargs '{"video": {"fps": 2, "num_frames": 256}}'
      "${TOOL_ARGS[@]}" --reasoning-parser nemotron_v3
    )
    ;;
  lilicorr)
    # LiLiCorr is a draft model; it is served behind the NVFP4 target.
    CKPT="${NVFP4}"; IMAGE=vllm/vllm-openai:v0.31.0
    : "${GPUS:=0}" "${TP:=1}" "${MAX_NUM_SEQS:=4}"
    SERVED=nvidia/super-vl-ga-lilicorr
    DRAFT_MOUNT=(-v "${MODELS_DIR}/${LILICORR}":/draft:ro)
    ARGS=(
      --gpu-memory-utilization 0.75
      --max-num-batched-tokens 8192
      --speculative_config.model /draft
      --speculative_config.method dflash
      --speculative_config.num_speculative_tokens 7
      "${MAMBA_ARGS[@]}"
      --enable-mamba-shared-prefix-checkpoint
      --prefix-match-unit 16
      "${TOOL_ARGS[@]}" --reasoning-parser nemotron_v3
    )
    ;;
  base)
    # Pretrained only and without an MTP head, but its chat template has the
    # same thinking and tool-call format as the GA candidates.
    CKPT="${BASE}"; IMAGE=vllm/vllm-openai:v0.30.0
    : "${GPUS:=0,1,2,3,4,5,6,7}" "${TP:=2}" "${MAX_NUM_SEQS:=32}"
    SERVED=nvidia/super-vl-base
    ARGS=(
      --gpu-memory-utilization 0.85
      --async-scheduling
      --limit-mm-per-prompt '{"video": 1, "image": 1}'
      "${TOOL_ARGS[@]}" --reasoning-parser nemotron_v3
    )
    ;;
  *)
    echo "Unknown variant ${VARIANT}; use bf16, nvfp4, lilicorr, or base" >&2
    exit 1
    ;;
esac

MODEL_CKPT="${MODELS_DIR}/${CKPT}"
NAME="super-vl-${VARIANT}"
PORT="${PORT:-8300}"
NUM_GPUS=$(awk -F, '{print NF}' <<<"${GPUS}")
if (( NUM_GPUS % TP != 0 )); then
  echo "GPUS=${GPUS} is not a multiple of TP=${TP}" >&2
  exit 1
fi
DP=$(( NUM_GPUS / TP ))

running=$(docker ps --format '{{.Names}}' | grep '^super-vl-' || true)
if [[ -n "${running}" && "${FORCE:-0}" != "1" ]]; then
  echo "Another candidate is still being served (${running}); evaluate one at a time." >&2
  echo "Stop it with: docker rm -f ${running}" >&2
  exit 1
fi

check_checkpoint() {
  "${PYTHON:-/usr/bin/python3}" - "$1" <<'EOF'
import json, os, sys
root = sys.argv[1]
index = os.path.join(root, "model.safetensors.index.json")
if os.path.isfile(index):
    shards = set(json.load(open(index))["weight_map"].values())
elif os.path.isfile(os.path.join(root, "model.safetensors")):
    shards = {"model.safetensors"}
else:
    sys.exit(f"No safetensors index or model.safetensors in {root}")
missing = sorted(s for s in shards if not os.path.isfile(os.path.join(root, s)))
if missing:
    sys.exit(f"Checkpoint is incomplete: {len(missing)} shard(s) missing in {root}")
EOF
}
check_checkpoint "${MODEL_CKPT}"
[[ "${VARIANT}" == lilicorr ]] && check_checkpoint "${MODELS_DIR}/${LILICORR}"

busy=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' -v gpus=",${GPUS}," 'index(gpus, "," $1 ",") && $2 > 1024')
if [[ -n "${busy}" && "${FORCE:-0}" != "1" ]]; then
  echo "GPUs ${GPUS} have memory in use (index, MiB); coordinate with the owners or set FORCE=1:" >&2
  echo "${busy}" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  echo "Container ${NAME} already exists; remove it with: docker rm -f ${NAME}" >&2
  exit 1
fi

docker run -d --name "${NAME}" \
  --gpus "\"device=${GPUS}\"" \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "${MODEL_CKPT}":/model:ro \
  "${DRAFT_MOUNT[@]}" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}" \
  -e SAFETENSORS_FAST_GPU=1 \
  -e VLLM_LOGGING_LEVEL=INFO \
  "${IMAGE}" \
  /model \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --served-model-name "${SERVED}" \
  --trust-remote-code \
  --tensor-parallel-size "${TP}" \
  --data-parallel-size "${DP}" \
  --max-model-len "${MAX_MODEL_LEN:-114688}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  "${ARGS[@]}"

echo "Started ${NAME} on GPUs ${GPUS} (TP=${TP} x DP=${DP}). Follow loading with: docker logs -f ${NAME}"
echo "Ready when this returns ${SERVED}: curl -s http://127.0.0.1:${PORT}/v1/models"
