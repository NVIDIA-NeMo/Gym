#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# AA-LCR (long-context reasoning).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN, NVIDIA_API_KEY
# and JUDGE_API_KEY). Run from the Gym repo root — the benchmark's dataset and
# prepare script resolve relative to your working directory. Results land in
# ./results/aa-lcr.
#
#   ./aa-lcr.sh                          # full benchmark (100 tasks x 16)
#   LIMIT=3 ./aa-lcr.sh                  # quick smoke
#   OUT=<dir> PARALLEL=<n> ./aa-lcr.sh   # output dir, concurrency
#
# Note: top_p 1.0 overrides the 0.95 in env.yaml — intentional for this benchmark.

# Used judge: Qwen3-235B-A22B-Instruct-2507 (non-reasoning)
JUDGE_MODEL="${JUDGE_MODEL:?}"

gym eval prepare --benchmark aalcr

gym eval run \
  --benchmark aalcr \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/aa-lcr}/evaluator_rollouts.jsonl" \
  --top-p 1.0 \
  "++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.model=$JUDGE_MODEL" \
  "++policy_model.responses_api_models.vllm_model.sequential_reasoning_allowed=false" \
  "++policy_model.responses_api_models.vllm_model.chat_template_kwargs={enable_thinking: true}" \
  "++policy_model.responses_api_models.vllm_model.extra_body={skip_special_tokens: false}" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
