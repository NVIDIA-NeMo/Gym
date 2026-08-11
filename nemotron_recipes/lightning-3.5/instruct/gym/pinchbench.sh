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
# PinchBench (the model as the brain of an OpenClaw agent).
#
# Needs an active Gym venv, .env loaded into your shell (this recipe uses
# PINCHBENCH_MODEL_BASE_URL, PINCHBENCH_MODEL_API_KEY, PINCHBENCH_MODEL_NAME,
# PINCHBENCH_JUDGE_MODEL, PINCHBENCH_JUDGE_BASE_URL, PINCHBENCH_JUDGE_API_KEY and
# PINCHBENCH_TAVILY_API_KEY) and the per-task sandbox image, pointed at by
# PINCHBENCH_SIF (see instruct/README.md). Run from the Gym repo root — the
# benchmark's dataset and prepare script resolve relative to your working
# directory. Results land in ./results/pinchbench.
#
#   scripts/more/instruct/gym/pinchbench.sh                         # full benchmark (147 tasks x 10)
#   LIMIT=3 scripts/more/instruct/gym/pinchbench.sh                 # quick smoke
#   OUT=<dir> PARALLEL=<n> scripts/more/instruct/gym/pinchbench.sh  # output dir, concurrency
#
# Note: OpenClaw calls the policy endpoint itself instead of going through a Gym
# model server, so the endpoint is given here rather than in env.yaml, and the
# sampling settings come from the agent's direct-exec wrapper rather than from
# this recipe.
#
# Note: PARALLEL also caps how many tasks the agent runs at once (16 by default, as
# in the reference run). Lower it if your endpoint slows under load — tasks time out
# rather than queue, so an overloaded endpoint costs score.
#
# Note: num_repeats is 10 below, where the benchmark's own config uses 3. 10 is what
# the published run used; both are deliberate, so don't reconcile them.

# Used judge: Claude Haiku 4.5
PINCHBENCH_JUDGE_MODEL="${PINCHBENCH_JUDGE_MODEL:?}"
PINCHBENCH_JUDGE_BASE_URL="${PINCHBENCH_JUDGE_BASE_URL:?}"
PINCHBENCH_JUDGE_API_KEY="${PINCHBENCH_JUDGE_API_KEY:?}"

PINCHBENCH_SIF="${PINCHBENCH_SIF:?export PINCHBENCH_SIF (image from setup_scripts/build_image.sh)}"
[ -r "$PINCHBENCH_SIF" ] || { echo "sandbox image not readable at $PINCHBENCH_SIF" >&2; exit 1; }

# Passed straight to OpenClaw, so a wrong name scores a different model. No default.
PINCHBENCH_MODEL_BASE_URL="${PINCHBENCH_MODEL_BASE_URL:?export PINCHBENCH_MODEL_BASE_URL (endpoint serving your model)}"
PINCHBENCH_MODEL_API_KEY="${PINCHBENCH_MODEL_API_KEY:?export PINCHBENCH_MODEL_API_KEY (key for that endpoint)}"
PINCHBENCH_MODEL_NAME="${PINCHBENCH_MODEL_NAME:?export PINCHBENCH_MODEL_NAME (name your endpoint serves the model under)}"

# One key only — this field is a plain string, so [k1,k2] is taken literally. Kept
# separate from TAVILY_API_KEY for that reason: the other recipes accept a key list.
PINCHBENCH_TAVILY_API_KEY="${PINCHBENCH_TAVILY_API_KEY:?export PINCHBENCH_TAVILY_API_KEY (one key)}"

# The published run raised Gym's rollout retry budget from its default of 3. An
# exhausted rollout scores 0, so on a flaky endpoint this moves the result.
export NEMO_GYM_MAX_ROLLOUT_ATTEMPTS="${NEMO_GYM_MAX_ROLLOUT_ATTEMPTS:-10}"

PB=pinchbench_agent.responses_api_agents.pinchbench
BENCH=pinchbench_benchmark_agent.responses_api_agents.pinchbench

# Absolute, and under the run's output dir: both default to /tmp/pinchbench_gym, which
# strands the per-task transcripts away from the results and collides between runs.
OUTDIR="$(realpath -m "${OUT:-./results/pinchbench}")"

gym eval prepare --benchmark pinchbench

gym eval run \
  --benchmark pinchbench \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/pinchbench}/evaluator_rollouts.jsonl" \
  "+sandbox_image=$PINCHBENCH_SIF" \
  "+model_base_url=$PINCHBENCH_MODEL_BASE_URL" \
  "+model_api_key=$PINCHBENCH_MODEL_API_KEY" \
  "+model_name=$PINCHBENCH_MODEL_NAME" \
  "+judge_model=$PINCHBENCH_JUDGE_MODEL" \
  "+judge_base_url=$PINCHBENCH_JUDGE_BASE_URL" \
  "+judge_api_key=$PINCHBENCH_JUDGE_API_KEY" \
  "+tavily_api_key=$PINCHBENCH_TAVILY_API_KEY" \
  "++$BENCH.datasets=[{name: pinchbench, type: benchmark, jsonl_fpath: benchmarks/pinchbench/data/pinchbench_benchmark.jsonl, prompt_config: null, prepare_script: benchmarks/pinchbench/prepare.py, num_repeats: 10, license: MIT}]" \
  "++$PB.web_search_provider=tavily" \
  "++$PB.sandbox_provider.apptainer.direct_exec=true" \
  "++$PB.context_window=262143" \
  "++$PB.max_tokens=65536" \
  "++$PB.max_concurrent=${PARALLEL:-16}" \
  "++$PB.work_root=$OUTDIR/pinchbench_work" \
  "++$PB.transcripts_dir=$OUTDIR/pinchbench_transcripts" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  --concurrency "${PARALLEL:-16}"
