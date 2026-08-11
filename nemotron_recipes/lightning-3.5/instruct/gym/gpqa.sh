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
# GPQA Diamond (no tools).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN and
# NVIDIA_API_KEY). Run from the Gym repo root — the benchmark's dataset and
# prepare script resolve relative to your working directory. Results land in
# ./results/gpqa.
#
#   nemotron_recipes/lightning-3.5/instruct/gym/gpqa.sh                         # full benchmark (198 questions x 8)
#   LIMIT=3 nemotron_recipes/lightning-3.5/instruct/gym/gpqa.sh                 # quick smoke
#   OUT=<dir> PARALLEL=<n> nemotron_recipes/lightning-3.5/instruct/gym/gpqa.sh  # output dir, concurrency

# Pin Gym to the commit the tech report numbers were produced with. Set PIN_GYM=0 to
# run against your current checkout instead. `nemotron_recipes` is excluded, so this never
# touches the recipe that is running, and HEAD does not move.
GYM_PIN="${GYM_PIN:-e446e4f415b9cde0e95bb813c85e9e3e23f5d893}"   # v0.5.0
if [ "${PIN_GYM:-1}" != 0 ]; then
  git rev-parse --verify -q "$GYM_PIN^{commit}" >/dev/null 2>&1 || git fetch origin "$GYM_PIN"
  git checkout "$GYM_PIN" -- . ':(exclude)nemotron_recipes' || exit 1
  echo "pinned Gym to $GYM_PIN (recipes untouched; PIN_GYM=0 to skip)"
fi

gym eval prepare --benchmark gpqa

gym eval run \
  --benchmark gpqa \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/gpqa}/evaluator_rollouts.jsonl" \
  "++overwrite_metrics_conflicts=true" \
  "++policy_model.responses_api_models.vllm_model.chat_template_kwargs={enable_thinking: true}" \
  "++policy_model.responses_api_models.vllm_model.extra_body={skip_special_tokens: false}" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
