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
# GDPval (real-world work products, judged).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN, NVIDIA_API_KEY,
# JUDGE_API_KEY and TAVILY_API_KEY). Also needs the Apptainer sandbox built by
# build-gdpval-sif.sh, pointed at by GDPVAL_CONTAINER_PATH (see
# instruct/README.md). Run from the Gym repo root — the benchmark's dataset and
# prepare script resolve relative to your working directory. Results land in
# ./results/gdpval.
#
#   nemotron_recipes/lightning-3.5/instruct/gym/gdpval/gdpval.sh                         # full benchmark (220 tasks x 1)
#   LIMIT=3 nemotron_recipes/lightning-3.5/instruct/gym/gdpval/gdpval.sh                 # quick smoke
#   OUT=<dir> PARALLEL=<n> nemotron_recipes/lightning-3.5/instruct/gym/gdpval/gdpval.sh  # output dir, concurrency
#
# Scores each deliverable against its rubric. Comparison mode instead scores against
# reference deliverables you generate yourself, one subdirectory per reference model;
# see instruct/README.md for the layout and how to produce them.
#
# Note: PARALLEL is applied twice on purpose — the agent caps its own concurrent runs
# at 32 regardless of --concurrency, so raising only one of them does nothing.

# Used judges: Gym's default panel — GPT-5.5, Gemini 3.1 Pro and Claude Opus 4.8,
# one sampled per call. All three route through gdpval_judge_model in env.yaml, so
# that endpoint has to serve every one of them.

# JUDGE_ONLY re-scores existing deliverables without running the agent, so it
# needs neither the sandbox nor a search key.
if [ "${JUDGE_ONLY:-false}" = "true" ]; then
  TAVILY_API_KEY=""
else
  GDPVAL_CONTAINER_PATH="${GDPVAL_CONTAINER_PATH:?export GDPVAL_CONTAINER_PATH (gdpval.sif from build-gdpval-sif.sh)}"
  [ -r "$GDPVAL_CONTAINER_PATH" ] || { echo "gdpval.sif not readable at $GDPVAL_CONTAINER_PATH" >&2; exit 1; }
  TAVILY_API_KEY="${TAVILY_API_KEY:?export TAVILY_API_KEY (one key, or [k1,k2] for several)}"
fi
export TAVILY_API_KEY

STIR=gdpval_stirrup_agent.responses_api_agents.stirrup_agent
GDR=gdpval_resources_server.resources_servers.gdpval
JUDGE=gdpval_judge_model.responses_api_models.openai_model
POLICY=policy_model.responses_api_models.vllm_model

# Absolute path required. A comparison run points GDPVAL_REFS at one of these.
DELIVERABLES="${PERSIST_DELIVERABLES_DIR:-$(realpath -m "${OUT:-./results/gdpval}/deliverables")}"

# Reference ELOs — the Artificial Analysis GDPval-AA v2 ratings the published figures
# were fitted against. A reference set is a subdirectory of GDPVAL_REFS named after one
# of these keys; supply as many as you have. Comparing against several opponents of
# known rating is what puts the result on the published scale — a single opponent only
# fixes an arbitrary offset.
#
# Pinned on purpose, and they no longer match the live board. These anchors define the
# scale, so refreshing them moves your score off the one the published figures sit on.
REF_ELOS="deepseek_v4_pro=1307 glm51_fp8=1257 kimi_k26=1191 nemotron3_ultra=1164
          qwen36_35b=1049 qwen35_397b=962 gptoss_120b=799 gemma4_26b=761
          qwen3_30b_thinking=308"

MODE="${GDPVAL_REWARD_MODE:-rubric}"
[ "$MODE" = rubric ] || [ "$MODE" = comparison ] ||
  { echo "GDPVAL_REWARD_MODE must be rubric or comparison (got '$MODE')" >&2; exit 1; }

MODE_OVR=()   # rubric is the config default and needs nothing added
if [ "$MODE" = comparison ]; then
  GDPVAL_REFS="${GDPVAL_REFS:?export GDPVAL_REFS (dir of reference deliverables)}"
  MODE_OVR=("++$GDR.reward_mode=comparison")
  found=0

  for kv in $REF_ELOS; do
    name="${kv%%=*}"
    [ -d "$GDPVAL_REFS/$name" ] || continue
    MODE_OVR+=("++$GDR.reference_models.$name.deliverables_dir=$GDPVAL_REFS/$name"
               "++$GDR.reference_models.$name.elo=${kv##*=}")
    found=$((found + 1))
  done

  if [ "$found" -eq 0 ]; then
    echo "no reference sets found in $GDPVAL_REFS. Expected subdirectories named:" >&2
    for kv in $REF_ELOS; do echo "  ${kv%%=*}" >&2; done
    exit 1
  fi

  # Two or more opponents: stage 1 places the model, stage 2 spends the full task
  # budget on the nearest of them. With one there is nothing to narrow to.
  if [ "$found" -ge 2 ]; then
    MODE_OVR+=("++multistage.enabled=true"
               "++multistage.stages=[{num_tasks: 45}, {num_models: $((found < 4 ? found : 4))}]")
  fi

  echo "gdpval: comparison against $found rated reference(s)" >&2
fi

# Pin Gym to the commit the tech report numbers were produced with. Set PIN_GYM=0 to
# run against your current checkout instead. `nemotron_recipes` is excluded, so this
# never touches the recipe that is running, and HEAD does not move. Undo the pin with
# `git restore .` from the repo root.
GYM_PIN="${GYM_PIN:-57c15a22f8b82d3d859b71468fe3329f4e2093b4}"
if [ "${PIN_GYM:-1}" != 0 ]; then
  if ! git diff --quiet -- . ':(exclude)nemotron_recipes' ||
    ! git diff --cached --quiet -- . ':(exclude)nemotron_recipes' ||
    [ -n "$(git ls-files --others --exclude-standard -- . ':(exclude)nemotron_recipes')" ]; then
    echo "refusing to pin over uncommitted changes outside nemotron_recipes; commit/stash them or set PIN_GYM=0" >&2
    exit 1
  fi
  git rev-parse --verify -q "$GYM_PIN^{commit}" >/dev/null 2>&1 || git fetch origin "$GYM_PIN"
  git restore --source="$GYM_PIN" -- . ':(exclude)nemotron_recipes' || exit 1
  echo "pinned Gym to $GYM_PIN (recipes untouched; PIN_GYM=0 to skip; git restore . to undo)"
fi

gym eval prepare --benchmark gdpval

gym eval run \
  --benchmark gdpval \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/gdpval}/evaluator_rollouts.jsonl" \
  "++$STIR.persist_deliverables_dir=$DELIVERABLES" \
  "++$GDR.judge_sampling_seed=${JUDGE_SAMPLING_SEED:-42}" \
  "++$GDR.persist_raw_judge_responses=true" \
  "++$GDR.preconvert_max_concurrent=30" \
  "++$JUDGE.max_concurrent_requests=10" \
  ${PARALLEL:+"++$STIR.concurrency=$PARALLEL"} \
  ${MODE_OVR[@]+"${MODE_OVR[@]}"} \
  "++$POLICY.chat_template_kwargs={enable_thinking: true}" \
  "++$POLICY.extra_body={skip_special_tokens: false}" \
  "++$POLICY.sequential_reasoning_allowed=false" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
