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
#   ./hle.sh                          # full benchmark (2158 questions)
#   LIMIT=3 ./hle.sh                  # quick smoke
#   OUT=<dir> PARALLEL=<n> ./hle.sh   # output dir, concurrency
#
# Note: the judge overrides below are HLE's grading contract — labels, the
# answer-extraction regex and a strict JSON schema. Changing them changes scores.
# msg_extraction_failure is empty on purpose; Gym's default sentinel would
# otherwise reach the judge as the model's answer.

# Used judge: gpt-4o (wired as judge_model in env.yaml)

gym eval prepare --benchmark hle

gym eval run \
  --benchmark hle \
  --model-type vllm_model \
  --resources-server labbench2_vlm/judge_model_openai \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/hle}/evaluator_rollouts.jsonl" \
  "++hle_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge.judge_model_server.name=judge_model" \
  "++hle_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge.judge_equal_label=HLE_JUDGE_CORRECT" \
  "++hle_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge.judge_not_equal_label=HLE_JUDGE_INCORRECT" \
  "++hle_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge.response_extract_regex='(?s)\A(?:(.{1,8192})\Z|(?=.{8193,}\Z))'" \
  "++hle_equivalence_llm_judge_resources_server.resources_servers.equivalence_llm_judge.msg_extraction_failure=''" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.temperature=0.0" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.top_p=0.95" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.type=json_schema" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.name=hle_judge" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.strict=true" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.schema.type=object" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.schema.properties.correct.type=string" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.schema.properties.correct.enum=[HLE_JUDGE_CORRECT,HLE_JUDGE_INCORRECT]" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.schema.properties.extracted_final_answer.type=string" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.schema.properties.reasoning.type=string" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.schema.properties.confidence.type=integer" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.schema.required=[correct,extracted_final_answer,reasoning,confidence]" \
  "++equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_responses_create_params.text.format.schema.additionalProperties=false" \
  "++policy_model.responses_api_models.vllm_model.chat_template_kwargs={enable_thinking: true}" \
  "++policy_model.responses_api_models.vllm_model.extra_body={seed: 0, skip_special_tokens: false}" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
