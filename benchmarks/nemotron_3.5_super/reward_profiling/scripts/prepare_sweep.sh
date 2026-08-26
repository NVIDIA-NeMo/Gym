#!/bin/bash
# Validate a sweep manifest and expand it into Gym's materialized-inputs file.
#
# Run once per (manifest, checkpoint). The result is reusable: every later rollout job resumes
# from it and skips preprocessing, which is otherwise ~100 minutes single-threaded for a full
# sweep. Expansion is parallel across entries.
#
# Everything this benchmark writes goes under reward_profiling/outputs/, which is gitignored.
# OUT_DIR defaults to outputs/sweeps; the sweep dir is OUT_DIR/<nickname>, and that is what
# SWEEP_DIR must point at when running sbatch_reward_profiling.sh.
#
#   MANIFEST=benchmarks/nemotron_3.5_super/reward_profiling/manifests/nemotron_3_ultra.yaml \
#   bash benchmarks/nemotron_3.5_super/reward_profiling/scripts/prepare_sweep.sh
set -euo pipefail

MANIFEST=${MANIFEST:?set MANIFEST to a sweep manifest yaml}
RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR=${OUT_DIR:-$RP_DIR/outputs/sweeps}
mkdir -p "$OUT_DIR"
JOBS=${JOBS:-}
LIMIT_PER_ENTRY=${LIMIT_PER_ENTRY:-}

args=(--out-dir "$OUT_DIR")
[[ -n "$JOBS" ]] && args+=(--jobs "$JOBS")
[[ -n "$LIMIT_PER_ENTRY" ]] && args+=(--limit-per-entry "$LIMIT_PER_ENTRY")
[[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)

echo ">>> validating $MANIFEST"
python -m nemo_gym.sweep validate "$MANIFEST"

echo ">>> materializing into $OUT_DIR"
python -m nemo_gym.sweep materialize "$MANIFEST" "${args[@]}"
