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

# Repeats are always written out here rather than left to Gym. --resume keys the materialized
# inputs on (_ng_task_index, _ng_rollout_index) (rollout_collection.py:740), so a row without a
# rollout index dies with KeyError '_ng_rollout_index' about 90 seconds in (job 6597590). Every
# launcher relies on --resume, so there is no unexpanded path.
args=(--out-dir "$OUT_DIR")
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
