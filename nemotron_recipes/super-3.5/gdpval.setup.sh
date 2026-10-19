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
# Sourced by run.sh before `gym eval prepare`. Checks what GDPval needs and picks the scoring
# mode: comparison when GDPVAL_REFS is set, rubric otherwise (see gdpval.yaml).

if [ "${JUDGE_ONLY:-false}" != true ]; then
  [ -r "${GDPVAL_CONTAINER_PATH:-}" ] || {
    echo "gdpval: set GDPVAL_CONTAINER_PATH to the sandbox image (see gdpval.yaml)" >&2; exit 1; }
  [ -n "${TAVILY_API_KEY:-}" ] || { echo "gdpval: set TAVILY_API_KEY" >&2; exit 1; }
fi

# The agent caches each task's deliverable here (absolute path), so an interrupted run can
# resume without redoing finished tasks.
export PERSIST_DELIVERABLES_DIR="${PERSIST_DELIVERABLES_DIR:-$(realpath -m "${OUT:-./results/gdpval}/deliverables")}"

# The agent caps its own concurrent tasks besides Gym's --concurrency: set both.
if [ -n "${PARALLEL:-}" ]; then
  EXTRA_ARGS+=("++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.concurrency=$PARALLEL")
fi

# Reference models and the ratings the published ELO was fitted against, as in the published
# run. Pinned: newer ratings would move the score off the published scale.
GDPVAL_REF_ELOS="deepseek_v4_flash_0731=1468 deepseek_v4_pro=1223 inkling_small=1191 glm51_fp8=1181
  kimi_k26=1115 nemotron3_ultra=1091 qwen36_35b=992 qwen35_122b=925 qwen35_397b=905
  gptoss_120b=745 gemma4_26b=713 qwen3_30b_thinking=274"

if [ -n "${GDPVAL_REFS:-}" ]; then
  GDPVAL_SERVER=gdpval_resources_server.resources_servers.gdpval
  EXTRA_ARGS+=("++$GDPVAL_SERVER.reward_mode=comparison" "++multistage.enabled=true")
  GDPVAL_FOUND=0
  for kv in $GDPVAL_REF_ELOS; do
    name="${kv%%=*}"
    [ -d "$GDPVAL_REFS/$name" ] || continue
    EXTRA_ARGS+=("++$GDPVAL_SERVER.reference_models.$name.deliverables_dir=$(realpath "$GDPVAL_REFS/$name")"
                 "++$GDPVAL_SERVER.reference_models.$name.elo=${kv##*=}")
    GDPVAL_FOUND=$((GDPVAL_FOUND + 1))
  done
  [ "$GDPVAL_FOUND" -gt 0 ] || {
    echo "gdpval: no reference sets in $GDPVAL_REFS; expected subdirectories named:" >&2
    for kv in $GDPVAL_REF_ELOS; do echo "  ${kv%%=*}" >&2; done
    exit 1; }
  echo "gdpval: comparison mode against $GDPVAL_FOUND reference model(s)"
else
  echo "gdpval: rubric mode (set GDPVAL_REFS for comparison mode, the published metric)"
fi
