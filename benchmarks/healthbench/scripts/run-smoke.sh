#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
cd "$(dirname "$0")/../../.."
benchmark=${1:-healthbench}
case "$benchmark" in
  healthbench|healthbench_professional) ;;
  *) echo 'Usage: run-smoke.sh [healthbench|healthbench_professional] [new output directory]' >&2; exit 2 ;;
esac
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY in your shell}"
gym_bin=${GYM_BIN:-.venv/bin/gym}
if [[ $# -ge 2 ]]; then
  mkdir "$2"
  run_dir=$(cd "$2" && pwd)
else
  mkdir -p results
  run_dir=$(mktemp -d "$PWD/results/$benchmark-smoke-XXXXXX")
fi
judge_model=azure/openai/gpt-4.1
if [[ "$benchmark" == healthbench_professional ]]; then
  judge_model=azure/openai/gpt-5.4
fi
configs=(--config "benchmarks/$benchmark/config.yaml" --config benchmarks/healthbench/configs/nemotron-lightning-smoke.yaml)
overrides=(
  "++smoke_judge_model_name=$judge_model"
  "++observability_enabled=true"
  "++model_call_capture_dir=$run_dir/model-calls"
  "++output_jsonl_fpath=$run_dir/rollouts.jsonl"
  "++split=benchmark"
  "++limit=8"
  "++num_repeats=1"
  "++num_samples_in_parallel=2"
)
"$gym_bin" env resolve "${configs[@]}" "${overrides[@]}" "${@:3}" > "$run_dir/resolved-config.yaml"
"$gym_bin" eval run "${configs[@]}" "${overrides[@]}" "${@:3}" > "$run_dir/engine.log" 2>&1
"$gym_bin" eval health-check "$run_dir"
printf 'Run artifacts: %s\n' "$run_dir"
"${PYTHON_BIN:-.venv/bin/python}" - "$run_dir" <<'PYTHON'
import json
import sys
from pathlib import Path
run = Path(sys.argv[1])
def count(path):
    return sum(1 for line in path.open() if line.strip()) if path.exists() else 0
scored = count(run / "rollouts.jsonl")
failed = count(run / "rollouts_failures.jsonl")
verdicts = json.loads((run / "quality_summary.json").read_text())["run"]["verdicts"]
print(f"Scored={scored}, failed/masked={failed}, health={verdicts}")
if scored != 8 or failed or verdicts.get("unhealthy") or verdicts.get("unobserved"):
    raise SystemExit("Smoke did not pass; inspect the retained artifacts")
PYTHON
