#!/bin/bash
# Validate a sweep manifest and expand it into Gym's materialized-inputs file.
#
# Run once per (manifest, checkpoint). The result is reusable: every later rollout job resumes
# from it and skips preprocessing, which is otherwise ~100 minutes single-threaded for a full
# sweep. Expansion is parallel across entries.
#
# Everything this benchmark writes goes under reward_profiling/outputs/, which is gitignored.
# OUT_DIR defaults to outputs/sweeps; the sweep dir is OUT_DIR/<nickname>, and that is what
# SWEEP_DIR must point at when running 03_run.sh.
#
#   MANIFEST=benchmarks/nemotron_3.5_super/reward_profiling/manifests/nemotron_3_ultra.yaml \
#   bash benchmarks/nemotron_3.5_super/reward_profiling/scripts/01_prepare_sweep.sh
set -euo pipefail

MANIFEST=${MANIFEST:?set MANIFEST to a sweep manifest yaml}
RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR=${OUT_DIR:-$RP_DIR/outputs/sweeps}
mkdir -p "$OUT_DIR"
JOBS=${JOBS:-}
LIMIT_PER_ENTRY=${LIMIT_PER_ENTRY:-}

# Default to one row per task and let Gym expand num_repeats at collection time. It writes 8x
# fewer rows (34 GB rather than 291.5 GB for this manifest, 682s rather than 2087s), and keeps a
# task's repeats together so they share a vLLM prefix cache on an identical prompt -- which matters
# once sharded, since dealing expanded rows round-robin scatters repeats across jobs. Set EXPAND=1
# to pre-expand instead.
args=(--out-dir "$OUT_DIR")
[[ "${EXPAND:-0}" == "1" ]] || args+=(--no-expand)
[[ -n "$JOBS" ]] && args+=(--jobs "$JOBS")
[[ -n "$LIMIT_PER_ENTRY" ]] && args+=(--limit-per-entry "$LIMIT_PER_ENTRY")
[[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)

echo ">>> validating $MANIFEST"
python -m nemo_gym.sweep validate "$MANIFEST"

echo ">>> materializing into $OUT_DIR"
python -m nemo_gym.sweep materialize "$MANIFEST" "${args[@]}"
