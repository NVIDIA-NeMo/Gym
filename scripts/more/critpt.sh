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
# CritPt (research-level physics problems).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN, NVIDIA_API_KEY and
# ARTIFICIAL_ANALYSIS_API_KEY). Run from the Gym repo root — the benchmark's dataset
# and prepare script resolve relative to your working directory. Results land in
# ./results/critpt.
#
#   ./critpt.sh                        # full benchmark (70 problems x 5)
#   LIMIT=3 ./critpt.sh                # rollouts only, cannot be scored (see below)
#   OUT=<dir> PARALLEL=<n> ./critpt.sh # output dir, concurrency
#
# Note: num_repeats 5 is set through the dataset override below — --num-repeats
# multiplies the config default instead of replacing it.
#
# Note: the grader scores 70 distinct problems per call, so a LIMIT below 70 cannot be
# scored, and concurrency stays at 350 (problems x repeats) — a rollout waiting to be
# scored holds its slot, so lower values starve the batches and the run wedges.

ARTIFICIAL_ANALYSIS_API_KEY="${ARTIFICIAL_ANALYSIS_API_KEY:?export ARTIFICIAL_ANALYSIS_API_KEY (one key, or [k1,k2] to fail over when a key is rate-limited)}"

AGENT=critpt_benchmark_agent.responses_api_agents.critpt_agent
CRITPT=critpt_resources_server.resources_servers.critpt
POLICY=policy_model.responses_api_models.vllm_model

# Persist every submission and grader response. Without it a run that exhausts the daily
# quota loses all of its inference; with it you can re-score once the quota resets:
#   python -m resources_servers.critpt.replay --cache-dir <dir>
export CRITPT_CACHE_DIR="${CRITPT_CACHE_DIR:-$(realpath -m "${OUT:-./results/critpt}")/critpt_cache}"

gym eval prepare --benchmark critpt

gym eval run \
  --benchmark critpt \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/critpt}/evaluator_rollouts.jsonl" \
  "++$AGENT.datasets=[{name: critpt, type: benchmark, jsonl_fpath: benchmarks/critpt/data/critpt_benchmark.jsonl, prompt_config: benchmarks/critpt/prompts/turn1.yaml, prepare_script: benchmarks/critpt/prepare.py, num_repeats: ${CRITPT_REPEATS:-5}}]" \
  "++artificial_analysis_api_key=$ARTIFICIAL_ANALYSIS_API_KEY" \
  "++$CRITPT.verify_timeout_seconds=21600" \
  "++$POLICY.chat_template_kwargs={enable_thinking: true}" \
  "++$POLICY.extra_body={skip_special_tokens: false}" \
  "++$POLICY.sequential_reasoning_allowed=false" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  --concurrency "${PARALLEL:-350}"
