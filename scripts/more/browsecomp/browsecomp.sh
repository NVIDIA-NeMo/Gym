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
# BrowseComp (web-browsing agent).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN, NVIDIA_API_KEY,
# JUDGE_API_KEY and TAVILY_API_KEY). Run from the Gym repo root — the benchmark's
# dataset and prepare script resolve relative to your working directory. Results
# land in ./results/browsecomp.
#
#   ./browsecomp.sh                          # full benchmark (1266 tasks x 1)
#   LIMIT=3 ./browsecomp.sh                  # quick smoke
#   OUT=<dir> PARALLEL=<n> ./browsecomp.sh   # output dir, concurrency

# Runs all 1266 problems. Unset for prepare.py's default 400-problem subset.
export BROWSECOMP_RUN_FULL=1

# Used judge: GLM-5.1
BROWSECOMP_JUDGE_MODEL="${BROWSECOMP_JUDGE_MODEL:?}"
TAVILY_API_KEY="${TAVILY_API_KEY:?export TAVILY_API_KEY (one key, or [k1,k2] for several)}"

# The domain list search skips. Which domains are on it changes search coverage,
# so results shift if you swap in a different list.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXCLUDE_JSON="${EXCLUDE_JSON:-$HERE/exclude_domains.json}"
[ -r "$EXCLUDE_JSON" ] || { echo "exclude list not readable at $EXCLUDE_JSON" >&2; exit 1; }

QWEN=Qwen3-235B-A22B-Instruct-2507-FP8.responses_api_models.vllm_model
HARNESS=browsecomp_benchmark_resources_server.resources_servers.browsecomp_advanced_harness
AGENT=browsecomp_benchmark_agent.responses_api_agents.browsecomp_agent
# The agent runs on this derived node, not policy_model, so thinking is set here.
POLICY=policy_model_no_interleaved_reasoning.responses_api_models.vllm_model

# prepare has no --model-type flag, so vllm_model.yaml is composed via --config.
gym eval prepare --benchmark browsecomp \
  --config responses_api_models/vllm_model/configs/vllm_model.yaml

gym eval run \
  --benchmark browsecomp \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/browsecomp}/evaluator_rollouts.jsonl" \
  --max-output-tokens 32768 \
  "++$QWEN.model=$BROWSECOMP_JUDGE_MODEL" \
  "++$HARNESS.judge_model_server.name=Qwen3-235B-A22B-Instruct-2507-FP8" \
  "++$HARNESS.tavily_api_key=$TAVILY_API_KEY" \
  "++$HARNESS.exclude_domains_file_path=$EXCLUDE_JSON" \
  "++$AGENT.save_model_call_using_vllm_tokenize_endpoint=false" \
  "++$POLICY.chat_template_kwargs={enable_thinking: true}" \
  "++$POLICY.extra_body={skip_special_tokens: false}" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
