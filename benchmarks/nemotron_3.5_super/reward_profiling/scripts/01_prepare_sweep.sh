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

# Pre-expand by default. NO_EXPAND=1 writes 8x fewer rows (34 GB rather than 291.5 GB here, 682s
# rather than 2087s) and keeps a task's repeats in one shard so they share a vLLM prefix cache --
# but it cannot be collected with --resume, which the launcher relies on:
#
#   rollout_collection.py:740  get_key = lambda r: (r[TASK_INDEX], r[ROLLOUT_INDEX])
#   KeyError: '_ng_rollout_index'          (job 6597590)
#
# _load_from_cache keys the materialized inputs on both indices, and unexpanded rows have only the
# task index. Losing --resume means re-preprocessing every restart and no recovery from a walltime
# kill, which costs more than the disk saves. Left available for a first run that does not need
# resume, or once Gym can key the cache on task index alone.
args=(--out-dir "$OUT_DIR")
[[ "${NO_EXPAND:-0}" == "1" ]] && args+=(--no-expand)
[[ -n "$JOBS" ]] && args+=(--jobs "$JOBS")
[[ -n "$LIMIT_PER_ENTRY" ]] && args+=(--limit-per-entry "$LIMIT_PER_ENTRY")
[[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# nemo_gym needs its dependencies importable. Automatic inside the eval container, not on a login
# node, where the failure is otherwise a bare ModuleNotFoundError from deep in the CLI.
if ! python -c "import orjson, nemo_gym" >/dev/null 2>&1; then
    if [[ -n "${GYM_SITE_PACKAGES:-}" ]]; then
        export PYTHONPATH="$REPO_ROOT:$GYM_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! python -c "import orjson, nemo_gym" >/dev/null 2>&1; then
        echo "ERROR: cannot import nemo_gym and its deps." >&2
        echo "       Run inside the eval container, activate the Gym venv, or set" >&2
        echo "       GYM_SITE_PACKAGES=<venv>/lib/python3.*/site-packages" >&2
        exit 2
    fi
fi

echo ">>> validating $MANIFEST"
python -m nemo_gym.sweep validate "$MANIFEST"

echo ">>> materializing into $OUT_DIR"
python -m nemo_gym.sweep materialize "$MANIFEST" "${args[@]}"
