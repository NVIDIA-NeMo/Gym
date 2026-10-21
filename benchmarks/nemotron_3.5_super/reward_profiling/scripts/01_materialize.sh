#!/bin/bash
# 01 - Validate a manifest and materialize it into Gym's materialized-inputs file.
#
# Run once per (manifest, checkpoint). Every later rollout job resumes from the result and skips
# preprocessing, which is otherwise ~100 min single-threaded for the full sweep.
#
# USAGE
#   MANIFEST=$R/manifests/nemotron_3_5_super.yaml bash $R/scripts/01_materialize.sh
#
# REQUIRED
#   MANIFEST          path to a sweep manifest yaml
#
# OPTIONAL
#   OUT_DIR           where <nickname>/ is written        (default: <benchmark>/outputs/sweeps)
#                     OUT_DIR/<nickname> is what SWEEP_DIR points at in every later step
#   LIMIT_PER_ENTRY   cap every entry at N source rows, for smoke runs. 8 exercises every code
#                     path in minutes. Only lowers a manifest limit; for a committed subset use
#                     materialize.limit_per_entry or entry.limit. NOT `gym eval run --limit`,
#                     which takes the first N rows of the whole file -- only the first entry
#   JOBS              worker processes                    (default: one per CPU, capped at entries)
#   OVERWRITE=1       replace an existing materialized file
#   GYM_SITE_PACKAGES a venv's site-packages, if orjson/yaml/pydantic are not importable
#
# OUTPUT
#   OUT_DIR/<nickname>/rollouts_materialized_inputs.jsonl   expanded inputs
#   OUT_DIR/<nickname>/rollouts.jsonl                       empty; completes the --resume gate
#   OUT_DIR/<nickname>/sweep_config.yaml                    passed to `gym env start --config`
#   OUT_DIR/<nickname>/sweep_report.json                    per-entry counts and task_index_range
#
# Outputs are gitignored.
set -euo pipefail

MANIFEST=${MANIFEST:?set MANIFEST to a sweep manifest yaml}
RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR=${OUT_DIR:-$RP_DIR/outputs/sweeps}
mkdir -p "$OUT_DIR"
JOBS=${JOBS:-}
LIMIT_PER_ENTRY=${LIMIT_PER_ENTRY:-}

# Repeats are written out here rather than left to Gym, because every launcher resumes: --resume
# keys the materialized inputs on (_ng_task_index, _ng_rollout_index) (global_config.py:173-174),
# so a row without a rollout index dies with KeyError '_ng_rollout_index' ~90s in (job 6597590).
args=(--out-dir "$OUT_DIR")
[[ -n "$JOBS" ]] && args+=(--jobs "$JOBS")
[[ -n "$LIMIT_PER_ENTRY" ]] && args+=(--limit-per-entry "$LIMIT_PER_ENTRY")
[[ "${OVERWRITE:-0}" == "1" ]] && args+=(--overwrite)

# `infra` needs orjson/yaml/pydantic importable. It no longer imports nemo_gym -- the package is
# self-contained under reward_profiling/infra -- but the deps still have to be there, and on a login
# node the failure is otherwise a bare ModuleNotFoundError from deep in the CLI.
if ! PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -c "import orjson, infra" >/dev/null 2>&1; then
    if [[ -n "${GYM_SITE_PACKAGES:-}" ]]; then
        REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
        export PYTHONPATH="$REPO_ROOT:$GYM_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -c "import orjson, infra" >/dev/null 2>&1; then
        echo "ERROR: cannot import infra and its deps (orjson, yaml, pydantic)." >&2
        echo "       Run inside the eval container, activate the Gym venv, or set" >&2
        echo "       GYM_SITE_PACKAGES=<venv>/lib/python3.*/site-packages" >&2
        exit 2
    fi
fi

echo ">>> validating $MANIFEST"
PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m infra validate "$MANIFEST"

echo ">>> materializing into $OUT_DIR"
PYTHONPATH="$RP_DIR${PYTHONPATH:+:$PYTHONPATH}" python -m infra materialize "$MANIFEST" "${args[@]}"
