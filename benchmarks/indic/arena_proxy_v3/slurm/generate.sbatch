#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00

set +x
set -euo pipefail
: "${GYM_REPO:?Submit using slurm/submit.sh, or export GYM_REPO}"
source "$GYM_REPO/benchmarks/indic/arena_proxy_v3/slurm/common.sh"
set_run_paths
: "${MODEL_PATH:?Set MODEL_PATH to the checkpoint directory}"
: "${IMAGE:?Set IMAGE to the local vLLM .sqsh image}"
[[ -r "$MODEL_PATH/config.json" ]] || die "Checkpoint config.json is missing or unreadable."
[[ -r "$IMAGE" ]] || die "The local vLLM image is unreadable: $IMAGE"
check_generation_budget
# Generation never needs judge or Hub credentials.
unset ARENA_JUDGE_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN

VLLM_PORT="${VLLM_PORT:-8000}"
HEAD_PORT="${HEAD_PORT:-$((20000 + ${SLURM_JOB_ID:-0} % 30000))}"
vllm_args=(
    serve "$MODEL_PATH" --served-model-name "$MODEL_KEY" --host 127.0.0.1 --port "$VLLM_PORT"
    --tensor-parallel-size "${TP:-8}" --data-parallel-size "${DP:-1}"
    --max-model-len "${CONTEXT:-163840}" --gpu-memory-utilization "${GPU_UTIL:-0.85}"
    --max-num-seqs "${CONCURRENCY:-256}" --trust-remote-code
    --default-chat-template-kwargs '{"enable_thinking":true}'
)
[[ "${EP:-1}" != 1 ]] || vllm_args+=(--enable-expert-parallel)
[[ "${EAGER:-0}" != 1 ]] || vllm_args+=(--enforce-eager)
[[ "${PARSER:-nemotron_v3}" == none ]] || vllm_args+=(--reasoning-parser "${PARSER:-nemotron_v3}")
[[ "${TOOL_PARSER:-qwen3_coder}" == none ]] || vllm_args+=(--enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-qwen3_coder}")
[[ -z "${MAX_BATCHED_TOKENS:-}" ]] || vllm_args+=(--max-num-batched-tokens "$MAX_BATCHED_TOKENS")

srun --overlap --no-container-mount-home --container-image="$IMAGE" \
    --container-mounts="${CONTAINER_MOUNTS:-/lustre:/lustre}" \
    --output="$RUN_DIR/logs/vllm-${SLURM_JOB_ID:-local}.log" vllm "${vllm_args[@]}" &
vllm_pid=$!
cleanup() { kill "$vllm_pid" 2>/dev/null || true; wait "$vllm_pid" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ready=0
for ((attempt=0; attempt<${STARTUP_ATTEMPTS:-540}; attempt++)); do
    if curl --max-time 5 -fsS "http://127.0.0.1:$VLLM_PORT/health" >/dev/null 2>&1; then ready=1; break; fi
    kill -0 "$vllm_pid" 2>/dev/null || die "vLLM exited during startup; inspect $RUN_DIR/logs."
    sleep 5
done
((ready)) || die "vLLM did not become healthy before the startup deadline."

export ARENA_PROXY_INPUT_JSONL="$INPUT_JSONL"
"$GYM_BIN" eval run \
    --config "$BENCHMARK_DIR/config.yaml" --config "$BENCHMARK_DIR/slurm/dataset.yml" --model-type openai_model \
    --agent indic_arena_proxy_v3_agent --model "$MODEL_KEY" \
    --model-url "http://127.0.0.1:$VLLM_PORT/v1" --model-api-key EMPTY \
    --split benchmark --output "$GENERATIONS" --resume \
    --num-repeats 1 --concurrency "${CONCURRENCY:-256}" \
    --temperature "${TEMPERATURE:-1.0}" --top-p "${TOP_P:-0.95}" \
    --max-output-tokens "${MAX_OUTPUT_TOKENS:-120000}" \
    +skip_verification=true +require_complete=true \
    ++head_server.host=127.0.0.1 "++head_server.port=$HEAD_PORT" \
    "++policy_model.responses_api_models.openai_model.extra_body={top_k:${TOP_K:--1},seed:42,chat_template_kwargs:{enable_thinking:true}}"
