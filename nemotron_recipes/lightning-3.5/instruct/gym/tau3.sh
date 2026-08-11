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
# Tau3-Banking (BM25 + grep knowledge tools).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN, NVIDIA_API_KEY
# and SIM_KEY). Run from the Gym repo root — the benchmark's dataset and prepare
# script resolve relative to your working directory. Results land in
# ./results/tau3.
#
#   nemotron_recipes/lightning-3.5/instruct/gym/tau3.sh                         # full benchmark (97 tasks x 5)
#   LIMIT=3 nemotron_recipes/lightning-3.5/instruct/gym/tau3.sh                 # quick smoke
#   OUT=<dir> PARALLEL=<n> nemotron_recipes/lightning-3.5/instruct/gym/tau3.sh  # output dir, concurrency
#
# Note: Gym installs tau2-bench from a branch, so what you get moves over time.
# This recipe pins the commit instead.

# Used simulator: gpt-5.4-mini plays the customer (wired in env.yaml)

export NEMO_GYM_TAU2_BENCH_DATA_REF="${NEMO_GYM_TAU2_BENCH_DATA_REF:-bxyu/nemo_gym_data}"

BENCH=tau2/configs/banking_bm25_grep_artificial_analysis
GYM_ROOT="${GYM_ROOT:-$PWD}"
TAU2_VENV="${TAU2_VENV:-$GYM_ROOT/responses_api_agents/tau2/.venv}"

# Gym builds the tau2 agent venv on first use, so pre-build it here to give the
# pin below somewhere to install.
[ -d "$TAU2_VENV" ] || gym env prefetch --benchmark "$BENCH" --model-type vllm_model
[ -d "$TAU2_VENV" ] || { echo "tau2 agent venv still missing at $TAU2_VENV" >&2; exit 1; }

uv pip install --python "$TAU2_VENV/bin/python" --force-reinstall --no-deps \
  "tau2[knowledge] @ git+https://github.com/bxyu-nvidia/tau2-bench@60c2a0dbf974ea7533456a4706f837c3a6d14afc"
uv pip install --python "$TAU2_VENV/bin/python" --quiet rank-bm25

SIM=gpt-5_4-mini-2026-03-17.responses_api_models.openai_model
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

gym eval prepare --benchmark "$BENCH"

gym eval run \
  --benchmark "$BENCH" \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/tau3}/evaluator_rollouts.jsonl" \
  "++$SIM.extra_body._delete_key=max_output_tokens" \
  "++$POLICY.sequential_reasoning_allowed=false" \
  "++$POLICY.chat_template_kwargs={enable_thinking: true}" \
  "++$POLICY.extra_body={skip_special_tokens: false}" \
  "++skip_venv_if_present=true" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
