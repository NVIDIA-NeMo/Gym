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
# AA-Omniscience (knowledge & hallucination).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN, NVIDIA_API_KEY
# and JUDGE_API_KEY). Run from the Gym repo root — the benchmark's dataset and
# prepare script resolve relative to your working directory. Results land in
# ./results/omniscience.
#
#   nemotron_recipes/lightning-3.5/instruct/gym/omniscience.sh                         # full benchmark (600 questions x 10)
#   LIMIT=3 nemotron_recipes/lightning-3.5/instruct/gym/omniscience.sh                 # quick smoke
#   OUT=<dir> PARALLEL=<n> nemotron_recipes/lightning-3.5/instruct/gym/omniscience.sh  # output dir, concurrency
#
# Note: num_repeats 10 is set through the dataset override below — --num-repeats
# multiplies the config default (8) instead of replacing it.

# Used judge: Gemini 3 Flash (wired as genrm_model in env.yaml)

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

gym eval prepare --benchmark omniscience

gym eval run \
  --benchmark omniscience \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/omniscience}/evaluator_rollouts.jsonl" \
  "++omniscience_omniscience_simple_agent.responses_api_agents.simple_agent.datasets=[{name: omniscience, type: benchmark, jsonl_fpath: benchmarks/omniscience/data/omniscience_benchmark.jsonl, prompt_config: benchmarks/omniscience/prompts/generation.yaml, prepare_script: benchmarks/omniscience/prepare.py, num_repeats: 10}]" \
  "++omniscience_omniscience_resources_server.resources_servers.omniscience.judge_responses_create_params.max_output_tokens=32768" \
  "++genrm_model.responses_api_models.openai_model.max_concurrent_requests=16" \
  "++policy_model.responses_api_models.vllm_model.chat_template_kwargs={enable_thinking: true}" \
  "++policy_model.responses_api_models.vllm_model.extra_body={skip_special_tokens: false}" \
  "++policy_model.responses_api_models.vllm_model.sequential_reasoning_allowed=false" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
