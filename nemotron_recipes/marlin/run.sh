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
# Runs one recipe from this folder: pins Gym to the commit the published number
# was produced with, prepares the benchmark data and collects rollouts.
#
# Needs an active Gym venv, ./env.yaml (copy env.yaml.example) and .env loaded into your
# shell (copy .env.example). Run from the Gym repo root.
#
#   nemotron_recipes/marlin/run.sh gpqa                          # full benchmark
#   LIMIT=3 nemotron_recipes/marlin/run.sh gpqa                  # quick smoke
#   nemotron_recipes/marlin/run.sh hle-vision                    # any <recipe>.yaml here
#   OUT=<dir> PARALLEL=<n> nemotron_recipes/marlin/run.sh gpqa   # output dir, concurrency
#   RESUME=1 nemotron_recipes/marlin/run.sh gpqa                 # continue an interrupted run
#
# Results land in ./results/<recipe>. Run without arguments to list the recipes.
#
# The pin overwrites and deletes tracked files (never the recipes), so commit or stash your
# work first; `git restore . && uv sync` puts your checkout back. Set PIN_GYM=0 to run
# against your current checkout instead, or GYM_PIN=<sha> to use a different commit.

set -euo pipefail

RECIPES=nemotron_recipes/marlin
[ -d "$RECIPES" ] || { echo "run this from the Gym repo root" >&2; exit 1; }

BENCH="${1:-}"
if [ -z "$BENCH" ]; then
  echo "usage: $RECIPES/run.sh <recipe>   (LIMIT, OUT, PARALLEL, RESUME optional)" >&2
  echo "recipes:" >&2
  (cd "$RECIPES" && ls -- *.yaml | grep -v '^common\.yaml$' | sed 's/\.yaml$//; s/^/  /') >&2
  exit 1
fi

# A recipe is <recipe>.yaml in this folder; common.yaml is the shared part, not a recipe.
CONFIG="$RECIPES/$BENCH.yaml"
case "$BENCH" in common|*/*) CONFIG="" ;; esac
[ -f "$CONFIG" ] || { echo "no recipe named '$BENCH' (run without arguments to list them)" >&2; exit 1; }

# Configs load in this order, each overriding the ones before it: the recipe's Gym benchmark
# config(s), the model config, common.yaml, the recipe. All are passed at the top level
# because Gym versions before 2026-08-16 merge included files after the file including them.
CONFIGS=()
while IFS= read -r bench_config; do
  if [ -n "$bench_config" ]; then CONFIGS+=(--config "$bench_config"); fi
done < <(python -c 'import sys, yaml
value = (yaml.safe_load(open(sys.argv[1])) or {}).get("benchmark_config") or []
print("\n".join(value if isinstance(value, list) else [value]))' "$CONFIG")
[ ${#CONFIGS[@]} -gt 0 ] || { echo "no benchmark_config in $CONFIG" >&2; exit 1; }
CONFIGS+=(
  --config responses_api_models/vllm_model/configs/vllm_model.yaml
  --config "$RECIPES/common.yaml"
  --config "$CONFIG"
)

# Pin Gym to the recipe's commit. `nemotron_recipes` is excluded, so the recipes are never
# touched, and HEAD does not move. The venv is synced to that commit's lockfile as well.
GYM_COMMIT="${GYM_PIN:-$(sed -n 's/^gym_commit:[[:space:]]*\([0-9a-f]\{40\}\).*/\1/p' "$CONFIG")}"
[ -n "$GYM_COMMIT" ] || { echo "no gym_commit in $CONFIG" >&2; exit 1; }
if [ "${PIN_GYM:-1}" != 0 ]; then
  git rev-parse --verify -q "$GYM_COMMIT^{commit}" >/dev/null || git fetch origin "$GYM_COMMIT"
  git restore --source="$GYM_COMMIT" -- . ':(exclude)nemotron_recipes'
  uv sync --frozen --quiet
  echo "pinned Gym to $GYM_COMMIT (PIN_GYM=0 to skip; git restore . && uv sync to undo)"
fi

# Benchmark-specific setup the config cannot express (installs, env vars, extra overrides).
# Lives next to the recipe as <recipe>.setup.sh and may append Gym overrides to EXTRA_ARGS.
EXTRA_ARGS=()
SETUP="$RECIPES/$BENCH.setup.sh"
if [ -f "$SETUP" ]; then
  # shellcheck source=/dev/null
  source "$SETUP"
fi

OUT="${OUT:-./results/$BENCH}"

gym eval prepare "${CONFIGS[@]}"

gym eval run \
  "${CONFIGS[@]}" \
  --output "$OUT/evaluator_rollouts.jsonl" \
  ${RESUME:+--resume} \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"} \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
