#!/bin/bash
# Validate a sweep manifest and expand it into Gym's materialized-inputs file.
#
# Run once per (manifest, checkpoint). The result is reusable: every later rollout job resumes
# from it and skips preprocessing, which is otherwise ~100 minutes single-threaded for a full
# sweep. Expansion is parallel across entries.
#
#   MANIFEST=benchmarks/nemotron_3.5_super/reward_profiling/manifests/no_judge_no_sandbox.yaml \
#   OUT_DIR=benchmarks/nemotron_3.5_super/reward_profiling/manifests_output \
#   bash benchmarks/nemotron_3.5_super/reward_profiling/scripts/prepare_sweep.sh
set -euo pipefail

MANIFEST=${MANIFEST:?set MANIFEST to a sweep manifest yaml}
OUT_DIR=${OUT_DIR:?set OUT_DIR to where artifacts should land}
JOBS=${JOBS:-}
LIMIT_PER_ENTRY=${LIMIT_PER_ENTRY:-}

args=(--out-dir "$OUT_DIR")
[[ -n "$JOBS" ]] && args+=(--jobs "$JOBS")
[[ -n "$LIMIT_PER_ENTRY" ]] && args+=(--limit-per-entry "$LIMIT_PER_ENTRY")
[[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)

echo ">>> validating $MANIFEST"
python -m nemo_gym.sweep validate "$MANIFEST"

echo ">>> materializing"
python -m nemo_gym.sweep materialize "$MANIFEST" "${args[@]}"
