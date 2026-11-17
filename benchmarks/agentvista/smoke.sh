#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run inside an existing allocation with the evaluation runtime available.
set -euo pipefail
umask 077
: "${AGENTVISTA_SMOKE_OUTPUT:?Set a new workspace output directory}"
: "${AGENTVISTA_SUPER_URL:?Set the existing local server base URL ending in /v1}"
: "${AGENTVISTA_ROOT:?Set the verified original source path}"
: "${AGENTVISTA_WORKER_PYTHON:?Set the worker interpreter}"
: "${AGENTVISTA_OPENCODE:?Set the pinned CLI path}"
: "${INFERENCE_API_KEY:?Load the hosted credential from a private environment file}"
: "${SERPAPI_KEY:?Load the original search credential from a private environment file}"
: "${VERIFIER_API_KEY:?Configure the fixed answer judge}"
: "${VERIFIER_END_POINT:?Configure the fixed answer judge base URL}"
: "${VERIFIER_MODEL_NAME:?Configure the fixed answer judge model}"
: "${AGENTVISTA_VISIT_API_KEY:?Configure the fixed webpage summarizer}"
: "${AGENTVISTA_VISIT_API_ENDPOINT:?Configure the fixed webpage summarizer URL}"
: "${AGENTVISTA_VISIT_MODEL_NAME:?Configure the fixed webpage summarizer model}"
: "${AGENTVISTA_SUPER_MODEL:?Set the local served-model name}"
: "${AGENTVISTA_HOSTED_URL:?Set the hosted policy base URL ending in /v1}"
: "${AGENTVISTA_HOSTED_MODEL:?Set the hosted policy model name}"
export PYTHONDONTWRITEBYTECODE=1 AGENTVISTA_TASKS=agentvista-000
mkdir "$AGENTVISTA_SMOKE_OUTPUT"
export XDG_CACHE_HOME="$AGENTVISTA_SMOKE_OUTPUT/storage/cache"
export XDG_CONFIG_HOME="$AGENTVISTA_SMOKE_OUTPUT/storage/config"
export XDG_DATA_HOME="$AGENTVISTA_SMOKE_OUTPUT/storage/data"
export XDG_STATE_HOME="$AGENTVISTA_SMOKE_OUTPUT/storage/state"
export UV_CACHE_DIR="$AGENTVISTA_SMOKE_OUTPUT/storage/uv"
export PIP_CACHE_DIR="$AGENTVISTA_SMOKE_OUTPUT/storage/pip"
export HF_HOME="$AGENTVISTA_SMOKE_OUTPUT/storage/huggingface"
export MPLCONFIGDIR="$AGENTVISTA_SMOKE_OUTPUT/storage/matplotlib"
export TMPDIR="$AGENTVISTA_SMOKE_OUTPUT/storage/tmp"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" \
  "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" "$TMPDIR"
for model in super kimi; do
  if [[ "$model" == super ]]; then
    endpoint="$AGENTVISTA_SUPER_URL"
    served_model="$AGENTVISTA_SUPER_MODEL"
    export AGENTVISTA_POLICY_API_KEY=EMPTY
  else
    endpoint="$AGENTVISTA_HOSTED_URL"
    served_model="$AGENTVISTA_HOSTED_MODEL"
    export AGENTVISTA_POLICY_API_KEY="$INFERENCE_API_KEY"
  fi
  for harness in default opencode; do
    out="$AGENTVISTA_SMOKE_OUTPUT/$model-$harness"
    mkdir "$out"
    common=("+agentvista_dataset_path=$out/input.jsonl"
      "+agentvista_output_root=$out/episodes" "+policy_base_url=$endpoint"
      "+policy_model_name=$served_model" '+policy_api_key=${oc.env:AGENTVISTA_POLICY_API_KEY}')
    gym eval prepare --benchmark "agentvista_$harness" "${common[@]}" \
      "+prepare_script_args.output_fpath=$out/input.jsonl" > "$out/prepare.log" 2>&1
    options=()
    if [[ "$harness" == opencode ]]; then
      options+=("++policy_model.responses_api_models.vllm_model.entrypoint=agentvista_opencode.py")
    fi
    gym eval run --benchmark "agentvista_$harness" --model-type vllm_model \
      --agent agentvista_agent --split benchmark --limit 1 --output "$out/rollouts.jsonl" \
      "${common[@]}" "${options[@]}" +observability_enabled=true +require_complete=true \
      "+model_call_capture_dir=$out/model_calls" +agentvista_max_steps=8 \
      +agentvista_default_max_turns=8 \
      '++agentvista_agent.responses_api_agents.agentvista_agent.agent_timeout=600' \
      '++agentvista_agent.responses_api_agents.agentvista_agent.episode_timeout=900' \
      > "$out/run.log" 2>&1
  done
done
