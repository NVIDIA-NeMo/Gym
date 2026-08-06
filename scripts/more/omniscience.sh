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
#   ./omniscience.sh                          # full benchmark (600 questions x 10)
#   LIMIT=3 ./omniscience.sh                  # quick smoke
#   OUT=<dir> PARALLEL=<n> ./omniscience.sh   # output dir, concurrency
#
# Note: num_repeats 10 is set through the dataset override below — --num-repeats
# multiplies the config default (8) instead of replacing it.

# Used judge: Gemini 3 Flash (wired as genrm_model in env.yaml)

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
