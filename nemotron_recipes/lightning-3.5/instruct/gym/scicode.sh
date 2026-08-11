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
# SciCode (scientific coding).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN and
# NVIDIA_API_KEY). Also needs SciCode's ~1 GB test_data.h5, pointed at by
# TEST_DATA (see instruct/README.md). Run from the Gym repo root — the
# benchmark's dataset and prepare script resolve relative to your working
# directory. Results land in ./results/scicode.
#
#   scripts/more/instruct/gym/scicode.sh                         # full benchmark (65 problems x 8)
#   LIMIT=3 scripts/more/instruct/gym/scicode.sh                 # quick smoke
#   OUT=<dir> PARALLEL=<n> scripts/more/instruct/gym/scicode.sh  # output dir, concurrency
#
# Note: num_repeats 8 is set through the dataset override below — --num-repeats
# multiplies the config default (3) instead of replacing it.

TEST_DATA="${TEST_DATA:?export TEST_DATA (path to SciCode test_data.h5)}"
[ -r "$TEST_DATA" ] || { echo "test_data.h5 not readable at $TEST_DATA" >&2; exit 1; }

gym eval prepare --benchmark scicode

gym eval run \
  --benchmark scicode \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/scicode}/evaluator_rollouts.jsonl" \
  "++scicode_benchmark_agent.responses_api_agents.scicode_agent.datasets=[{name: scicode, type: benchmark, jsonl_fpath: benchmarks/scicode/data/scicode_benchmark.jsonl, prepare_script: benchmarks/scicode/prepare.py, prompt_config: null, num_repeats: 8}]" \
  "++scicode_benchmark_resources_server.resources_servers.scicode.test_data_fpath=$TEST_DATA" \
  "++policy_model.responses_api_models.vllm_model.chat_template_kwargs={enable_thinking: true}" \
  "++policy_model.responses_api_models.vllm_model.extra_body={skip_special_tokens: false}" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
