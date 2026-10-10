#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
umask 077
run_dir=${1:?}
model=${2:?}
image=${3:?}
job_id=${4:?}
model_name=${5:?}
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX
export PYTHONNOUSERSITE=1
export HF_HOME="$run_dir/hf-home"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mkdir -p "$HF_HOME"

args=(
  --model "$model"
  --trust-remote-code
  --served-model-name "$model_name"
  --host 0.0.0.0 --port 8000
  --tensor-parallel-size 4 --enable-expert-parallel
  --max-model-len 262144 --kv-cache-dtype fp8
  --skip-mm-profiling
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser nemotron_v3
  --max-num-batched-tokens 8192 --max-num-seqs 32
)
python3 "$run_dir/serving.py" record --run-dir "$run_dir" \
  --model "$model" --image "$image" --job-id "$job_id" --served-model "$model_name" -- "${args[@]}"
exec python3 -m vllm.entrypoints.openai.api_server "${args[@]}"
