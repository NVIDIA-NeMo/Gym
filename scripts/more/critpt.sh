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
#   PARALLEL=140 ./critpt.sh                  # full benchmark (70 problems x 5)
#   LIMIT=3 ./critpt.sh                       # rollouts only, no score (see below)
#   OUT=<dir> PARALLEL=<n> ./critpt.sh        # output dir, concurrency
#
# Note: scoring is atomic — the Artificial Analysis API grades all 70 problems in one
# call, so a LIMIT below 70 produces rollouts that cannot be scored. LIMIT counts
# problems, not rollouts.
#
# Note: keep concurrency comfortably above 70. A rollout waiting for its batch holds
# its slot, so 70 waiters fill 70 slots; at exactly 70 one failure wedges the run for
# good, because the freed slot goes to a repeat that cannot complete the batch. 140 is
# two full passes, so even a whole failed pass still leaves enough to finish.

ARTIFICIAL_ANALYSIS_API_KEY="${ARTIFICIAL_ANALYSIS_API_KEY:?export ARTIFICIAL_ANALYSIS_API_KEY (one key, or [k1,k2] to spread the daily quota)}"

AGENT=critpt_benchmark_agent.responses_api_agents.critpt_agent
CRITPT=critpt_resources_server.resources_servers.critpt
POLICY=policy_model.responses_api_models.vllm_model

# Persist every submission and AA response. Without it a run that exhausts the daily
# AA quota loses all of its inference; with it you can re-score once the quota resets:
#   python -m resources_servers.critpt.replay --cache-dir <dir>
export CRITPT_CACHE_DIR="${CRITPT_CACHE_DIR:-$(realpath -m "${OUT:-./results/critpt}")/critpt_cache}"

# Interleave the repeats — p1..p70 five times over, rather than Gym's grouped
# expansion (abc -> aabbcc). Scoring needs 70 DISTINCT problems in a batch, and under
# grouping the 70th distinct problem is row 346, which would force concurrency 346+.
# Gym keys a task on the row's content, so the five identical lines collapse into one
# task with rollout indices 0..4 — the same 70 tasks x 5 the grouped form produces.
CRITPT_SRC=benchmarks/critpt/data/critpt_benchmark.jsonl
CRITPT_JSONL="$(realpath -m "${OUT:-./results/critpt}")/critpt_interleaved.jsonl"

gym eval prepare --benchmark critpt

mkdir -p "$(dirname "$CRITPT_JSONL")"
python3 - "$CRITPT_SRC" "$CRITPT_JSONL" "${CRITPT_REPEATS:-5}" <<'PY'
import sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = [l if l.endswith("\n") else l + "\n" for l in open(src) if l.strip()]
with open(dst, "w") as f:
    for _ in range(n):          # one full pass over every problem, n times
        f.writelines(lines)
print(f"critpt: wrote {len(lines) * n} interleaved rows ({len(lines)} problems x {n})")
PY

gym eval run \
  --benchmark critpt \
  --model-type vllm_model \
  --split benchmark \
  ${RESUME:+--resume} \
  --output "${OUT:-./results/critpt}/evaluator_rollouts.jsonl" \
  "++$AGENT.datasets=[{name: critpt, type: benchmark, jsonl_fpath: $CRITPT_JSONL, prompt_config: benchmarks/critpt/prompts/turn1.yaml, prepare_script: benchmarks/critpt/prepare.py, num_repeats: 1}]" \
  "++artificial_analysis_api_key=$ARTIFICIAL_ANALYSIS_API_KEY" \
  "++$CRITPT.verify_timeout_seconds=21600" \
  "++$POLICY.chat_template_kwargs={enable_thinking: true}" \
  "++$POLICY.extra_body={skip_special_tokens: false}" \
  "++$POLICY.sequential_reasoning_allowed=false" \
  "++overwrite_metrics_conflicts=true" \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"}
