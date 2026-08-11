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
#   nemotron_recipes/lightning-3.5/instruct/gym/aa-lcr.sh                         # full benchmark (100 tasks x 16)
#   LIMIT=3 nemotron_recipes/lightning-3.5/instruct/gym/aa-lcr.sh                 # quick smoke
#   OUT=<dir> PARALLEL=<n> nemotron_recipes/lightning-3.5/instruct/gym/aa-lcr.sh  # output dir, concurrency
#
# Note: top_p 1.0 overrides the 0.95 in env.yaml — intentional for this benchmark.

# Used judge: Qwen3-235B-A22B-Instruct-2507 (non-reasoning)
AALCR_JUDGE_MODEL="${AALCR_JUDGE_MODEL:?}"

POLICY=policy_model.responses_api_models.vllm_model

# Pin Gym to the commit the tech report numbers were produced with. Set PIN_GYM=0 to
# run against your current checkout instead. `nemotron_recipes` is excluded, so this
# never touches the recipe that is running, and HEAD does not move. Undo the pin with
# `git restore .` from the repo root.
GYM_PIN="${GYM_PIN:-e446e4f415b9cde0e95bb813c85e9e3e23f5d893}"   # 0.5.0rc0
if [ "${PIN_GYM:-1}" != 0 ]; then
  git rev-parse --verify -q "$GYM_PIN^{commit}" >/dev/null 2>&1 || git fetch origin "$GYM_PIN"
  git restore --source="$GYM_PIN" -- . ':(exclude)nemotron_recipes' || exit 1
  # Files added after the pin would otherwise linger and mix two trees into one.
  comm -23 <(git ls-files -- . ':(exclude)nemotron_recipes' | LC_ALL=C sort) \
           <(git ls-tree -r --name-only "$GYM_PIN" | LC_ALL=C sort) | tr '\n' '\0' | xargs -0 -r rm -f
  echo "pinned Gym to $GYM_PIN (recipes untouched; PIN_GYM=0 to skip; git restore . to undo)"
fi

gym eval prepare --benchmark aalcr

gym eval run \
  --benchmark aalcr \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/aa-lcr}/evaluator_rollouts.jsonl" \
  --top-p 1.0 \
  "++Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model.model=$AALCR_JUDGE_MODEL" \
  "++$POLICY.sequential_reasoning_allowed=false" \
  "++$POLICY.chat_template_kwargs={enable_thinking: true}" \
  "++$POLICY.extra_body={skip_special_tokens: false}" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
