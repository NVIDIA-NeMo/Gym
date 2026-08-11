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
# Humanity's Last Exam (no tools).
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded
# into your shell (copy .env.example; this recipe uses HF_TOKEN, NVIDIA_API_KEY
# and JUDGE_API_KEY). Run from the Gym repo root — the benchmark's dataset and
# prepare script resolve relative to your working directory. Results land in
# ./results/hle.
#
#   nemotron_recipes/lightning-3.5/instruct/gym/hle.sh                         # full benchmark (2158 questions x 1)
#   LIMIT=3 nemotron_recipes/lightning-3.5/instruct/gym/hle.sh                 # quick smoke
#   OUT=<dir> PARALLEL=<n> nemotron_recipes/lightning-3.5/instruct/gym/hle.sh  # output dir, concurrency
#
# Note: the judge overrides below are HLE's grading contract — labels, the
# answer-extraction regex and a strict JSON schema. Changing them changes scores.
# msg_extraction_failure is empty on purpose; Gym's default sentinel would
# otherwise reach the judge as the model's answer.

# Used judge: gpt-4o (wired as judge_model in env.yaml)

JUDGE=hle_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge
JPARAMS=equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params
POLICY=policy_model.responses_api_models.vllm_model

# Pin Gym to the commit the tech report numbers were produced with. Set PIN_GYM=0 to
# run against your current checkout instead. `nemotron_recipes` is excluded, so this
# never touches the recipe that is running, and HEAD does not move. Undo the pin with
# `git restore .` from the repo root.
GYM_PIN="${GYM_PIN:-e446e4f415b9cde0e95bb813c85e9e3e23f5d893}"   # 0.5.0rc0
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

gym eval prepare --benchmark hle

gym eval run \
  --benchmark hle \
  --model-type vllm_model \
  --resources-server labbench2_vlm/judge_model_openai \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/hle}/evaluator_rollouts.jsonl" \
  "++$JUDGE.judge_model_server.name=judge_model" \
  "++$JUDGE.judge_equal_label=HLE_JUDGE_CORRECT" \
  "++$JUDGE.judge_not_equal_label=HLE_JUDGE_INCORRECT" \
  "++$JUDGE.response_extract_regex='(?s)\A(?:(.{1,8192})\Z|(?=.{8193,}\Z))'" \
  "++$JUDGE.msg_extraction_failure=''" \
  "++$JPARAMS.temperature=0.0" \
  "++$JPARAMS.top_p=0.95" \
  "++$JPARAMS.text.format.type=json_schema" \
  "++$JPARAMS.text.format.name=hle_judge" \
  "++$JPARAMS.text.format.strict=true" \
  "++$JPARAMS.text.format.schema.type=object" \
  "++$JPARAMS.text.format.schema.properties.correct.type=string" \
  "++$JPARAMS.text.format.schema.properties.correct.enum=[HLE_JUDGE_CORRECT,HLE_JUDGE_INCORRECT]" \
  "++$JPARAMS.text.format.schema.properties.extracted_final_answer.type=string" \
  "++$JPARAMS.text.format.schema.properties.reasoning.type=string" \
  "++$JPARAMS.text.format.schema.properties.confidence.type=integer" \
  "++$JPARAMS.text.format.schema.required=[correct,extracted_final_answer,reasoning,confidence]" \
  "++$JPARAMS.text.format.schema.additionalProperties=false" \
  "++$POLICY.chat_template_kwargs={enable_thinking: true}" \
  "++$POLICY.extra_body={seed: 0, skip_special_tokens: false}" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
