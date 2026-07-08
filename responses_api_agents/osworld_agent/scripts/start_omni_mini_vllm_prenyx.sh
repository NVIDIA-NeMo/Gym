#!/usr/bin/env bash
# Attach Omni Mini vLLM to an already-running Prenyx Slurm allocation.
# Run on login-prenyx: ./start_omni_mini_vllm_prenyx.sh JOB_ID

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 JOB_ID" >&2
  exit 2
fi

JOB_ID="$1"
ROOT="${PRENYX_OMNI_ROOT:-/lustre/fsw/general_sa/${USER}/omni-mini}"
IMAGE="${VLLM_IMAGE:-${ROOT}/containers/vllm-openai-v0.20.0.sqsh}"
MODEL="${MODEL:-nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16}"
MODEL_REVISION="${MODEL_REVISION:-24e67ea000b7c2837fc8f9488aa2008524fac8ba}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning}"
PORT="${PORT:-8000}"
LOG="${VLLM_LOG:-${ROOT}/logs/vllm-${JOB_ID}.log}"
PID_FILE="${ROOT}/logs/vllm-${JOB_ID}.srun.pid"

state="$(squeue -h -j "${JOB_ID}" -o '%T')"
if [[ "${state}" != "RUNNING" ]]; then
  echo "job ${JOB_ID} is ${state:-not present}; wait for RUNNING before attaching vLLM" >&2
  exit 1
fi
mkdir -p "${ROOT}/logs"
node="$(squeue -h -j "${JOB_ID}" -o '%N')"

# Login-node /tmp is too small, while extracting directly on Lustre cannot set
# file capabilities. Reuse the downloaded layer cache but extract on the
# allocated compute node's local filesystem, then write the final squashfs to
# Lustre. This runs only once.
if [[ ! -f "${IMAGE}" ]]; then
  remote_tmp="/tmp/${USER}/omni-mini-enroot"
  srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 mkdir -p "${remote_tmp}"
  srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 \
    env TMPDIR="${remote_tmp}" \
        ENROOT_TEMP_PATH="${remote_tmp}" \
        ENROOT_CACHE_PATH="${ROOT}/cache/enroot" \
        enroot import -o "${IMAGE}" docker://vllm/vllm-openai:v0.20.0
fi

# --overlap permits this step to coexist with the allocation's lightweight
# keepalive. Use all eight B200s with TP8 to match internal-osworld-adapter-nano-omni's
# validated public-BF16 local-vLLM experiment.
nohup srun \
  --jobid="${JOB_ID}" \
  --overlap \
  --nodes=1 \
  --ntasks=1 \
  env HF_HOME="${ROOT}/cache/huggingface" \
      TRANSFORMERS_CACHE="${ROOT}/cache/huggingface" \
      enroot start --mount "${ROOT}:${ROOT}" "${IMAGE}" \
      vllm serve "${MODEL}" \
        --revision "${MODEL_REVISION}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --tensor-parallel-size 8 \
        --data-parallel-size 1 \
        --max-model-len "${MAX_MODEL_LEN:-64000}" \
        --max-num-seqs "${MAX_NUM_SEQS:-32}" \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
        --trust-remote-code \
        --reasoning-parser nemotron_v3 \
        --allowed-local-media-path / \
  > "${LOG}" 2>&1 < /dev/null &
echo "$!" > "${PID_FILE}"

echo "job=${JOB_ID} node=${node} srun_pid=$! log=${LOG}"
echo "endpoint=http://${node}:${PORT}/v1"
echo "served_model=${SERVED_MODEL_NAME}"
echo "model_revision=${MODEL_REVISION}"
