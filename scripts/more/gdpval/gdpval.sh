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
# reproducibility.md). Run from the Gym repo root — the benchmark's dataset and
# prepare script resolve relative to your working directory. Results land in
# ./results/gdpval.
#
#   ./gdpval.sh                          # full benchmark (220 tasks x 1)
#   LIMIT=3 ./gdpval.sh                  # quick smoke
#   OUT=<dir> PARALLEL=<n> ./gdpval.sh   # output dir, concurrency
#
# Scores each deliverable against its rubric. Comparison mode instead scores against
# reference deliverables you generate yourself, one subdirectory per reference model;
# see reproducibility.md for the layout and how to produce them.

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

STIR=gdpval_stirrup_agent.responses_api_agents.stirrup_agent
GDR=gdpval_resources_server.resources_servers.gdpval
POLICY=policy_model.responses_api_models.vllm_model

# Absolute path required. A comparison run points GDPVAL_REFS at one of these.
DELIVERABLES="${PERSIST_DELIVERABLES_DIR:-$(realpath -m "${OUT:-./results/gdpval}/deliverables")}"

# Reference ELOs — Artificial Analysis GDPval-AA v2 board, snapshot 2026-07-04. A
# reference set is a subdirectory of GDPVAL_REFS named after one of these keys; supply
# as many as you have. Comparing against several opponents of known rating is what puts
# the result on the published scale — a single opponent only fixes an arbitrary offset.
#
# Pinned on purpose. These anchors define the scale, so refreshing them to the current
# board moves your score off the one the published figures sit on.
REF_ELOS="deepseek_v4_pro=1307 glm51_fp8=1257 kimi_k26=1191 nemotron3_ultra=1164
          qwen36_35b=1049 qwen35_397b=962 gptoss_120b=799 gemma4_26b=761
          qwen3_30b_thinking=308"

MODE="${GDPVAL_REWARD_MODE:-rubric}"
case "$MODE" in
  rubric) MODE_OVR=() ;;   # the config default
  comparison)
    GDPVAL_REFS="${GDPVAL_REFS:?export GDPVAL_REFS (dir of reference deliverables)}"
    [ -d "$GDPVAL_REFS" ] || { echo "reference deliverables not found at $GDPVAL_REFS" >&2; exit 1; }
    MODE_OVR=("++$GDR.reward_mode=comparison")
    REFS_FOUND=0
    for kv in $REF_ELOS; do
      [ -d "$GDPVAL_REFS/${kv%%=*}" ] || continue
      MODE_OVR+=("++$GDR.reference_models.${kv%%=*}.deliverables_dir=$GDPVAL_REFS/${kv%%=*}"
                 "++$GDR.reference_models.${kv%%=*}.elo=${kv##*=}")
      REFS_FOUND=$((REFS_FOUND + 1))
    done
    if [ "$REFS_FOUND" -eq 0 ]; then
      # Nothing named — score against the directory as one unrated baseline.
      MODE_OVR+=("++$GDR.reference_models.baseline.deliverables_dir=$GDPVAL_REFS"
                 "++$GDR.reference_models.baseline.elo=${GDPVAL_REFERENCE_ELO:-1290}")
    elif [ "$REFS_FOUND" -ge 2 ]; then
      # Stage 1 places the model roughly, stage 2 spends the full task budget on the
      # nearest opponents. Needs at least two opponents to have anything to narrow to.
      MODE_OVR+=("++multistage.enabled=true"
                 "++multistage.stages=[{num_tasks: 45}, {num_models: $((REFS_FOUND < 4 ? REFS_FOUND : 4))}]")
    fi
    echo "gdpval: comparison against $REFS_FOUND rated reference(s)" >&2 ;;
  *) echo "GDPVAL_REWARD_MODE must be rubric or comparison (got '$MODE')" >&2; exit 1 ;;
esac

gym eval prepare --benchmark gdpval

gym eval run \
  --benchmark gdpval \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/gdpval}/evaluator_rollouts.jsonl" \
  "++$STIR.tavily_api_key=$TAVILY_API_KEY" \
  "++$STIR.persist_deliverables_dir=$DELIVERABLES" \
  "++$GDR.judge_sampling_seed=${JUDGE_SAMPLING_SEED:-42}" \
  "++$GDR.persist_raw_judge_responses=true" \
  "++$GDR.preconvert_max_concurrent=30" \
  ${MODE_OVR[@]+"${MODE_OVR[@]}"} \
  "++$POLICY.chat_template_kwargs={enable_thinking: true}" \
  "++$POLICY.extra_body={skip_special_tokens: false}" \
  "++$POLICY.sequential_reasoning_allowed=false" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
