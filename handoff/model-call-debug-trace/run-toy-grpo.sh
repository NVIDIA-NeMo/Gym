#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Toy GRPO run (Qwen3-0.6B, workplace_assistant, one node, 2 GPUs) with Gym's
# model-call debug trace enabled, followed by trace validation.
# Run INSIDE the NeMo RL container, with the patched Gym mounted at
# /opt/nemo-rl/3rdparty/Gym-workspace/Gym and the patched nemo_rl/ at /opt/nemo-rl/nemo_rl.
#
# Usage: run-toy-grpo.sh <capture|nocapture>
#   capture   - token_capture.enabled=true (external staging through the vLLM worker)
#   nocapture - token capture disabled (legacy token echo on the served response)
#
# Required environment:
#   WORK_DIR    absolute, shared-filesystem directory for outputs
#   MODEL_PATH  local Qwen3-0.6B checkpoint directory (config.json, model.safetensors, tokenizer files)
#   DATA_PATH   JSONL of workplace_assistant rows that carry agent_ref (see README)
# Optional:
#   HANDOFF_DIR directory holding validate_trace.py (default: this script's directory)
#   RL_PYTHON   python for validation (default: /opt/nemo_rl_venv/bin/python)
set -euo pipefail

MODE="${1:?usage: run-toy-grpo.sh <capture|nocapture>}"
: "${WORK_DIR:?set WORK_DIR}" "${MODEL_PATH:?set MODEL_PATH}" "${DATA_PATH:?set DATA_PATH}"
HANDOFF_DIR="${HANDOFF_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
RL_PYTHON="${RL_PYTHON:-/opt/nemo_rl_venv/bin/python}"
RUN_ID="${SLURM_JOB_ID:-manual}-${MODE}"
OUT="${WORK_DIR}/runs/${RUN_ID}"
TRACE_DIR="${OUT}/model_call_debug"
CAPTURE_DIR="${OUT}/model_calls"
mkdir -p "${OUT}" "${TRACE_DIR}" "${CAPTURE_DIR}"

MODE_OVERRIDES=()
if [[ "${MODE}" == "capture" ]]; then
  MODE_OVERRIDES+=(++token_capture.enabled=true "++token_capture.capture_dir=/dev/shm/gym_token_capture/${RUN_ID}")
fi

# Separate HF_HOME per mode: parallel jobs would race on the HF->Megatron conversion cache.
export HF_HOME="${WORK_DIR}/hf_home_${MODE}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NEMO_GYM_VENV_DIR=/opt/gym_venvs RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export UV_CACHE_DIR="${WORK_DIR}/uv_cache" RAY_DEDUP_LOGS=1 PYTHONUNBUFFERED=1

cd /opt/nemo-rl
echo "[toy] mode=${MODE} out=${OUT} host=$(hostname) $(date)"
set +e
uv run --no-sync python examples/run_grpo_single_controller.py \
  --config examples/nemo_gym/grpo_qwen3_30ba3b_instruct.yaml \
  policy.model_name="${MODEL_PATH}" \
  policy.dtensor_cfg.enabled=false \
  policy.megatron_cfg.enabled=true \
  policy.megatron_cfg.tensor_model_parallel_size=1 \
  policy.megatron_cfg.pipeline_model_parallel_size=1 \
  policy.megatron_cfg.expert_model_parallel_size=1 \
  policy.megatron_cfg.context_parallel_size=1 \
  policy.megatron_cfg.sequence_parallel=false \
  policy.generation.vllm_cfg.tensor_parallel_size=1 \
  policy.generation.vllm_cfg.async_engine=true \
  policy.max_total_sequence_length=8192 \
  policy.generation.colocated.enabled=false \
  policy.generation.colocated.resources.num_nodes=1 \
  policy.generation.colocated.resources.gpus_per_node=1 \
  grpo.num_prompts_per_step=4 \
  grpo.num_generations_per_prompt=2 \
  grpo.max_num_steps=3 \
  grpo.val_period=-1 \
  grpo.val_at_start=false \
  grpo.async_grpo=null \
  policy.train_global_batch_size=8 \
  policy.train_micro_batch_size=1 \
  cluster.num_nodes=1 \
  cluster.gpus_per_node=2 \
  loss_fn.reference_policy_kl_penalty=0.01 \
  grpo.skip_reference_policy_logprobs_calculation=false \
  loss_fn.use_importance_sampling_correction=true \
  logger.tensorboard_enabled=true \
  logger.log_dir="${OUT}/logs" \
  logger.wandb_enabled=false \
  checkpointing.enabled=false \
  data.train.data_path="${DATA_PATH}" \
  data.validation.data_path="${DATA_PATH}" \
  ++data_plane.enabled=true \
  ++data_plane.impl=transfer_queue \
  ++data_plane.backend=simple \
  ++data_plane.simple.storage_capacity=1000000 \
  ++data_plane.simple.num_storage_units=2 \
  ++data_plane.claim_meta_poll_interval_s=0.5 \
  ++async_rl.sampler.name=in_order \
  ++async_rl.sampler.max_lookahead_versions=0 \
  ++async_rl.min_groups_for_streaming_train=4 \
  ++async_rl.max_inflight_prompts=4 \
  ++async_rl.max_buffered_rollouts=4 \
  ++env.nemo_gym.skip_venv_if_present=true \
  ++env.nemo_gym.nemo_gym_log_dir="${OUT}/nemo_gym_logs" \
  ++env.nemo_gym.observability_enabled=true \
  ++env.nemo_gym.model_call_capture_dir="${CAPTURE_DIR}" \
  ++env.nemo_gym.model_call_debug_trace.enabled=true \
  ++env.nemo_gym.model_call_debug_trace.dir="${TRACE_DIR}" \
  "${MODE_OVERRIDES[@]}" \
  2>&1 | tee "${OUT}/train.log"
TRAIN_RC=${PIPESTATUS[0]}
set -e
echo "[toy] training exit code ${TRAIN_RC}"

"${RL_PYTHON}" "${HANDOFF_DIR}/validate_trace.py" \
  --trace-dir "${TRACE_DIR}" \
  --capture-dir "${CAPTURE_DIR}" \
  --tokenizer "${MODEL_PATH}" \
  --mode "${MODE}" \
  --report "${OUT}/validation.json" 2>&1 | tee "${OUT}/validation.log"
VALIDATE_RC=${PIPESTATUS[0]}
echo "[toy] validation exit code ${VALIDATE_RC}"
exit $(( TRAIN_RC != 0 ? TRAIN_RC : VALIDATE_RC ))
