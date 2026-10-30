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
# Runs one Atom (NOOA) recipe: pins Gym to the recipe's commit, prepares the benchmark data and
# collects rollouts. The same command collects a one-row smoke and the full benchmark.
#
# Needs an active Gym venv, ./env.yaml (copy recipes/nooa-atom/env.yaml.example), .env loaded into
# your shell and NOOA_RUNTIME_WHEELHOUSE set. Run from the Gym repo root.
#
#   recipes/nooa-atom/run.sh terminal-bench-2.1                   # full benchmark
#   LIMIT=1 recipes/nooa-atom/run.sh terminal-bench-2.1           # one row
#   ONLY_TASKS='code-from-image,fp8-*' recipes/nooa-atom/run.sh terminal-bench-2.1   # named tasks (globs on task ids)
#   OUT=<dir> PARALLEL=<n> recipes/nooa-atom/run.sh terminal-bench-2.1
#   RESUME=1 recipes/nooa-atom/run.sh terminal-bench-2.1          # continue an interrupted run
#
# Results land in ./results/<recipe> (OUT overrides it) with the resolved config beside the rollouts.
set -euo pipefail

RECIPES=recipes/nooa-atom
[ -d "$RECIPES" ] || { echo "run this from the Gym repo root" >&2; exit 1; }
export RECIPES_DIR="$PWD/$RECIPES"

BENCH="${1:-}"
if [ -z "$BENCH" ]; then
  echo "usage: $RECIPES/run.sh <recipe>   (LIMIT, ONLY_TASKS, OUT, PARALLEL, RESUME optional)" >&2
  echo "recipes:" >&2
  (cd "$RECIPES" && ls -- *.yaml | sed 's/\.yaml$//; s/^/  /') >&2
  exit 1
fi
CONFIG="$RECIPES/$BENCH.yaml"
[ -f "$CONFIG" ] || { echo "no recipe named '$BENCH' (run without arguments to list them)" >&2; exit 1; }

# Config files the recipe lists, then the model server, then the recipe itself. env.yaml is loaded by Gym.
CONFIGS=()
while IFS= read -r bench_config; do
  [ -n "$bench_config" ] && CONFIGS+=(--config "$bench_config")
done < <(python -c 'import sys, yaml
value = (yaml.safe_load(open(sys.argv[1])) or {}).get("benchmark_config") or []
print("\n".join(value if isinstance(value, list) else [value]))' "$CONFIG")
[ ${#CONFIGS[@]} -gt 0 ] || { echo "no benchmark_config in $CONFIG" >&2; exit 1; }
CONFIGS+=(--config responses_api_models/vllm_model/configs/vllm_model.yaml --config "$CONFIG")

GYM_COMMIT="${GYM_PIN:-$(sed -n 's/^gym_commit:[[:space:]]*\([0-9a-f]\{40\}\).*/\1/p' "$CONFIG")}"
if [ "${PIN_GYM:-1}" != 0 ]; then
  [ -n "$GYM_COMMIT" ] || { echo "no 40-hex gym_commit in $CONFIG; set GYM_PIN=<sha> or PIN_GYM=0" >&2; exit 1; }
  [ "$(git rev-parse HEAD)" = "$GYM_COMMIT" ] || {
    echo "checkout is at $(git rev-parse --short HEAD), recipe pins $GYM_COMMIT; check it out (PIN_GYM=0 to run anyway)" >&2; exit 1; }
fi

OUT="${OUT:-./results/$BENCH}"
mkdir -p "$OUT"
# Hydra writes under the run directory, never into the checkout.
HYDRA=(hydra.run.dir="$OUT/hydra" hydra.output_subdir=null)
# REUSE_VENVS=1 (set by slurm/run.sbatch after preflight): reuse the per-server venvs preflight just built on this
# checkout instead of rebuilding their editable installs, whose setuptools file finder runs `git ls-files` against
# the whole repository and times out (40 s) on a busy Lustre .git.
[ "${REUSE_VENVS:-0}" = 1 ] && HYDRA+=(++skip_venv_if_present=true)
# Per-rollout NOOA artifacts (trace files) live beside the rollouts unless the caller chose a directory.
export NOOA_ARTIFACTS_DIR="${NOOA_ARTIFACTS_DIR:-$OUT/nooa-artifacts}"

gym eval prepare "${CONFIGS[@]}" "${HYDRA[@]}"
# The recipe's setup runs after preparation: it may pin a task repo, or write the rows the dataset entry names.
SETUP="$RECIPES/$BENCH.setup.sh"
[ -f "$SETUP" ] && source "$SETUP"
# ONLY_TASKS: `gym eval run --only-tasks` applies to Harbor task targets, not to a prepared dataset file. The recipe
# filters the prepared file to the matching rows instead and points the run at the copy (only_tasks.py).
if [ -n "${ONLY_TASKS:-}" ]; then
  gym env resolve "${CONFIGS[@]}" "${HYDRA[@]}" > "$OUT/only-tasks-resolved.yaml"
  ONLY_OVERLAY="$(python "$RECIPES/only_tasks.py" "$OUT/only-tasks-resolved.yaml" "$OUT/only-tasks" "$ONLY_TASKS")"
  CONFIGS+=(--config "$ONLY_OVERLAY")
  HYDRA+=(++use_cached_prepared_benchmarks=true)
fi
gym eval run \
  "${CONFIGS[@]}" \
  --output "$OUT/evaluator_rollouts.jsonl" \
  ${RESUME:+--resume} \
  ${LIMIT:+--limit "$LIMIT"} \
  ${PARALLEL:+--concurrency "$PARALLEL"} \
  "${HYDRA[@]}"
