#!/bin/bash
#SBATCH --job-name=tau2_eval_qwen3.5_gpt_user
#SBATCH --output=logs/tau2_eval/slurm/%j.out
#SBATCH --error=logs/tau2_eval/slurm/%j.err
#SBATCH --time=02:00:00
#SBATCH --account=llmservice_fm_post
#SBATCH --partition=batch_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --export=ALL
#SBATCH --dependency=singleton

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
DOMAIN="${1:-retail}"
NUM_TRIALS="${2:-8}"
MODEL_ID="${3:-Qwen/Qwen3.5-397B-A17B-FP8}"
SERVED_MODEL_NAME="Qwen3.5-397B-A17B-FP8"
# NUM_TASKS="${5:-5}"

TENSOR_PARALLEL_SIZE=8
VLLM_PORT=8000
VLLM_HOST="0.0.0.0"
MAX_MODEL_LEN=131072
GPU_MEMORY_UTILIZATION=0.90

USER_LLM="gpt-5.2"
AGENT_TEMPERATURE=0.6
AGENT_TOP_P=0.95
AGENT_TOP_K=20
MAX_CONCURRENCY=4
SEED=42

VLLM_IMAGE="/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/users/zihanl/inform/sqshs/vllm-openai-nightly-3e802e8786e05ba94e8a585630f14c86f83b8ad4.sqsh"

SCRIPT_DIR="OFFICIAL_TAU2_BENCH_REPO_PATH"

OUTPUT_MAIN_DIR="${SCRIPT_DIR}/data/simulations"

SAVE_TO="${OUTPUT_MAIN_DIR}/${DOMAIN}_agent-${SERVED_MODEL_NAME}_user-gpt5.2_nt${NUM_TRIALS}_seed${SEED}/results"

LOG_DIR="${SCRIPT_DIR}/logs/tau2_eval/agent-${SERVED_MODEL_NAME}_user-gpt5.2"
mkdir -p "${LOG_DIR}"

echo "============================================"
echo "  Tau2 Evaluation"
echo "============================================"
echo "Model:           ${MODEL_ID}"
echo "Served name:     ${SERVED_MODEL_NAME}"
echo "Domain:          ${DOMAIN}"
echo "Num trials:      ${NUM_TRIALS}"
# echo "Num tasks:       ${NUM_TASKS}"
echo "TP size:         ${TENSOR_PARALLEL_SIZE}"
echo "Node:            $(hostname)"
echo "GPUs:            ${CUDA_VISIBLE_DEVICES:-all}"
echo "Job ID:          ${SLURM_JOB_ID:-local}"
echo "============================================"

# ─── Step 1: Launch vLLM server ──────────────────────────────────────────────
echo "[$(date)] Starting vLLM server..."

srun --nodes=1 --ntasks=1 --gpus-per-node=8 \
    --container-image="${VLLM_IMAGE}" \
    --container-mounts="/lustre" \
    vllm serve "${MODEL_ID}" \
        --served-model-name "${SERVED_MODEL_NAME}" \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --host ${VLLM_HOST} \
        --port ${VLLM_PORT} \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser qwen3_coder \
        --reasoning-parser qwen3 \
        --dtype auto \
    > "${LOG_DIR}/${DOMAIN}_vllm_server_${SLURM_JOB_ID:-local}.log" 2>&1 &

VLLM_PID=$!
echo "[$(date)] vLLM server started with PID ${VLLM_PID}"

# ─── Step 2: Wait for vLLM server to be ready ───────────────────────────────
VLLM_URL="http://localhost:${VLLM_PORT}"
MAX_WAIT=1800
WAIT_INTERVAL=10
ELAPSED=0

echo "[$(date)] Waiting for vLLM server at ${VLLM_URL} ..."
while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    if curl -s "${VLLM_URL}/health" > /dev/null 2>&1; then
        echo "[$(date)] vLLM server is ready! (waited ${ELAPSED}s)"
        break
    fi

    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM server process died. Check logs:"
        echo "  ${LOG_DIR}/vllm_server_${SLURM_JOB_ID:-local}.log"
        exit 1
    fi

    sleep ${WAIT_INTERVAL}
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    echo "[$(date)] Still waiting for vLLM server... (${ELAPSED}s / ${MAX_WAIT}s)"
done

if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    echo "[$(date)] ERROR: vLLM server did not start within ${MAX_WAIT}s"
    kill ${VLLM_PID} 2>/dev/null || true
    exit 1
fi

# ─── Step 3: Run Tau2 evaluation ─────────────────────────────────────────────
echo "[$(date)] Running tau2 evaluation..."

AGENT_LLM_ARGS="{\"api_base\": \"${VLLM_URL}/v1\", \"api_key\": \"EMPTY\", \"temperature\": ${AGENT_TEMPERATURE}, \"top_p\": ${AGENT_TOP_P}, \"top_k\": ${AGENT_TOP_K}}"

cd "${SCRIPT_DIR}"

tau2 run \
    --domain "${DOMAIN}" \
    --agent-llm "openai/${SERVED_MODEL_NAME}" \
    --agent-llm-args "${AGENT_LLM_ARGS}" \
    --user-llm "${USER_LLM}" \
    --num-trials ${NUM_TRIALS} \
    --max-concurrency ${MAX_CONCURRENCY} \
    --seed ${SEED} \
    --save-to "${SAVE_TO}" \
    --auto-resume

# --num-tasks ${NUM_TASKS} \


TAU2_EXIT_CODE=$?

# ─── Step 4: Cleanup ────────────────────────────────────────────────────────
echo "[$(date)] Shutting down vLLM server (PID ${VLLM_PID})..."
kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

if [ ${TAU2_EXIT_CODE} -eq 0 ]; then
    echo "[$(date)] Tau2 evaluation completed successfully."
else
    echo "[$(date)] Tau2 evaluation finished with exit code ${TAU2_EXIT_CODE}."
fi

exit ${TAU2_EXIT_CODE}


## commands
# cd SCRIPT_DIR
# source .venv/bin/activate
# sbatch --job-name="tau2_airline_qwen3.5_gpt_user" eval_tau2_gpt_user.sh airline 16
# sbatch --job-name="tau2_retail_qwen3.5_gpt_user" eval_tau2_gpt_user.sh retail 8
# sbatch --job-name="tau2_telecom_qwen3.5_gpt_user" eval_tau2_gpt_user.sh telecom 8


## check results
# tau2 evaluate-trajs data/simulations/airline_agent-Qwen3.5-397B-A17B-FP8_user-gpt5.2_nt16_seed42/results/
# tau2 evaluate-trajs data/simulations/retail_agent-Qwen3.5-397B-A17B-FP8_user-gpt5.2_nt8_seed42/results/
# tau2 evaluate-trajs data/simulations/telecom_agent-Qwen3.5-397B-A17B-FP8_user-gpt5.2_nt8_seed42/results/
