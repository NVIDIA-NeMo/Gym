#!/bin/bash
# 02 - Deal a materialized sweep into N sibling sweep directories, one per Slurm job.
#
# Only needed to go past one job's node ceiling (~16: --segment needs a topology-contiguous
# allocation and an NVL72 rack is 18 nodes). Each shard directory is a complete SWEEP_DIR, so
# 03_run_single.sh runs against one unmodified.
#
# USAGE
#   SWEEP_DIR=<sweep> NUM_SHARDS=16 bash $R/scripts/02_shard.sh
#
# REQUIRED
#   SWEEP_DIR     the OUT_DIR/<nickname> directory 01 wrote
#   (NUM_SHARDS is optional: the manifest's num_shards, else 16)
#
# OPTIONAL
#   SHARDS_DIR    where shard_NNN/ go                     (default: SWEEP_DIR/shards)
#
# Safe to re-run with a different NUM_SHARDS: collected rollouts are folded back into the parent,
# which is snapshotted to snapshots/<UTC>/ before any shard directory is touched, then carried into
# the new layout. Rows are dealt round-robin, so every shard carries every environment and none
# inherits a whole slow one.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to the <out-dir>/<nickname> directory 01_materialize.sh wrote}
# The manifest's num_shards when this environment does not set it, so running 02 standalone
# deals the count the manifest declares rather than silently defaulting to 16.
_manifest_shards=$(python - "$SWEEP_DIR" <<'PY_SHARDS' 2>/dev/null || true
import json, sys
from pathlib import Path
try:
    print(json.loads((Path(sys.argv[1]) / "sweep_report.json").read_text()).get("num_shards") or "")
except OSError:
    print("")
PY_SHARDS
)
NUM_SHARDS=${NUM_SHARDS:-${_manifest_shards:-16}}
SHARDS_DIR=${SHARDS_DIR:-$SWEEP_DIR/shards}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# The sweep package lives beside these scripts, not in nemo_gym.
RP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"

# `infra` needs orjson/yaml/pydantic importable. It no longer imports nemo_gym -- the package is
# self-contained under reward_profiling/infra -- but the deps still have to be there, and on a login
# node the failure is otherwise a bare ModuleNotFoundError from deep in the CLI.
if ! PYTHONPATH="$RP_DIR" python -c "import orjson, infra" >/dev/null 2>&1; then
    if [[ -n "${GYM_SITE_PACKAGES:-}" ]]; then
        export PYTHONPATH="$REPO_ROOT:$GYM_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
    fi
    if ! PYTHONPATH="$RP_DIR" python -c "import orjson, infra" >/dev/null 2>&1; then
        echo "ERROR: cannot import infra and its deps (orjson, yaml, pydantic)." >&2
        echo "       Run inside the eval container, activate the Gym venv, or set" >&2
        echo "       GYM_SITE_PACKAGES=<venv>/lib/python3.*/site-packages" >&2
        exit 2
    fi
fi

PYTHONPATH="$RP_DIR" python -m infra shard "$SWEEP_DIR" --num-shards "$NUM_SHARDS" --out-dir "$SHARDS_DIR"
echo
echo "Each shard directory is a valid SWEEP_DIR. Run them with:"
echo "  SWEEP_DIR=$SHARDS_DIR/shard_000 bash .../scripts/03_run_single.sh"
echo "or all of them, with resubmission of any that die:"
echo "  SWEEP_DIR=$SWEEP_DIR NUM_SHARDS=$NUM_SHARDS bash .../scripts/03_run_sharded.sh"
