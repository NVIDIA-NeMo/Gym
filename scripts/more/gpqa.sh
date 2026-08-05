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
#   ./gpqa.sh                          # full benchmark (198 questions x 8)
#   LIMIT=3 ./gpqa.sh                  # quick smoke
#   OUT=<dir> PARALLEL=<n> ./gpqa.sh   # output dir, concurrency

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
