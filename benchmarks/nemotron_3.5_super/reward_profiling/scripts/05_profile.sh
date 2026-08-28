#!/bin/bash
# 05 - Split the sweep by manifest entry, then reward-profile each entry and the whole sweep.
#
# No GPU and no Slurm: the profiler is CPU-only. Safe on a partial sweep -- partial groups are
# kept and the completion percentage is reported.
#
# USAGE
#   source .venv/bin/activate
#   SWEEP_DIR=<sweep> bash $R/scripts/05_profile.sh
#
# REQUIRED
#   SWEEP_DIR     a sweep directory with rollouts.jsonl
#
# OPTIONAL
#   PROFILE_JOBS  labels profiled concurrently            (default: 8)
#                 each label is a separate `gym` process and on Lustre interpreter start dominates
#                 (~45s vs ~5s in the container), so this is most of the wall time
#   VLLM_JOBID    borrow a container from this live job instead of using the local venv
#   CONTAINER     the sqsh to use with VLLM_JOBID
#   ENV_YAML      config whose ${oc.env:VAR} keys are checked  (default: <repo>/env.yaml)
#   GYM_SITE_PACKAGES a venv's site-packages, if nemo_gym is not already importable
#
# OUTPUT
#   SWEEP_DIR/by_label/<label>/profile.txt                per-entry
#   SWEEP_DIR/rollouts_reward_profiling.jsonl             one row per source task
#   SWEEP_DIR/rollouts_agent_metrics.json                 the same, aggregated per agent
#
# Checks up front that `gym eval profile` imports and that env.yaml's variables are exported,
# because either failing writes the same error into all 36 profile.txt files instead of once.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to the <out-dir>/<nickname> directory}
# Exported so the xargs children below see them.
export SWEEP_DIR VLLM_JOBID="${VLLM_JOBID:-}" CONTAINER="${CONTAINER:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

# nemo_gym needs its dependencies importable. Inside the eval container that is automatic; on a
# login node it is not, and the failure is otherwise a bare ModuleNotFoundError from deep inside
# the CLI. GYM_SITE_PACKAGES points PYTHONPATH at a venv's site-packages if you are not in one.
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

# The check above only proves nemo_gym loads; `gym eval profile` also pulls nemo_gym.cli.eval
# into openai_utils, so a stale openai pin passes there and then fails once per label. Fail here.
if [[ -z "${VLLM_JOBID:-}" ]]; then
    if ! profile_import_error=$(python -c "import nemo_gym.cli.eval" 2>&1); then
        echo "ERROR: 'gym eval profile' cannot run here, though nemo_gym itself imports:" >&2
        echo "$profile_import_error" | tail -3 >&2
        echo >&2
        echo "       Usually a stale venv: this branch pins openai==2.44.0 (pyproject.toml)." >&2
        echo "       Either sync one   -- uv venv && uv sync --extra dev, then re-run" >&2
        echo "       or use a container -- VLLM_JOBID=<live jobid> CONTAINER=<sqsh> $0" >&2
        exit 2
    fi
fi

# env.yaml interpolates ${oc.env:VAR} for judge keys. The profiler resolves the same config, so an
# unset one fails identically in all 36 labels -- each writing the same KeyError into its own
# profile.txt. Check once. A bare `VAR=value` in .bashrc is the usual cause: it is a shell variable,
# not an environment one, so a non-login shell never exports it.
ENV_YAML=${ENV_YAML:-$REPO_ROOT/env.yaml}
if [[ -z "${VLLM_JOBID:-}" && -f "$ENV_YAML" ]]; then
    _missing=$(python - "$ENV_YAML" <<'PY_ENVVARS'
import os, re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
names = sorted(set(re.findall(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)", text)))
print(" ".join(n for n in names if not os.environ.get(n)))
PY_ENVVARS
)
    if [[ -n "$_missing" ]]; then
        echo "ERROR: $ENV_YAML interpolates environment variables that are not exported:" >&2
        for _v in $_missing; do echo "         $_v" >&2; done
        echo "       'export VAR=...' in .bashrc only reaches a login shell, so either export" >&2
        echo "       them here, or re-run this under one:" >&2
        echo "         bash -lc \"SWEEP_DIR=$SWEEP_DIR bash $0\"" >&2
        exit 2
    fi
fi

# Container and profile settings the manifest declared, applied only where this environment
# leaves them unset -- so a standalone profile behaves like the one 03_run_single.sh runs.
while IFS='=' read -r _k _v; do
    [[ -n "$_k" ]] || continue
    [[ -n "${!_k:-}" ]] || export "$_k=$_v"
done < <(python - "$SWEEP_DIR" <<'PY_MANIFEST'
import json, sys
from pathlib import Path
try:
    doc = json.loads((Path(sys.argv[1]) / "sweep_report.json").read_text())
except OSError:
    doc = {}
for block in ("srun", "gym_eval_profile"):
    for key, value in (doc.get(block) or {}).items():
        print(f"{key}={value}")
PY_MANIFEST
)

python -m nemo_gym.sweep split "$SWEEP_DIR"

profile_cmd() {
    local inputs=$1 rollouts=$2 out=$3
    if [[ -n "${VLLM_JOBID:-}" ]]; then
        srun --overlap --jobid="$VLLM_JOBID" --nodes=1 --ntasks=1 --gpus=0 --cpus-per-task=8 \
            --container-image="${CONTAINER:?set CONTAINER when using VLLM_JOBID}" \
            --container-mounts=/lustre:/lustre --container-workdir=/opt/Gym --no-container-mount-home \
            bash -lc "source /opt/Gym_venv/bin/activate && gym eval profile \
                --inputs '$inputs' --rollouts '$rollouts' ++allow_partial_rollouts=True" > "$out" 2>&1
    else
        gym eval profile --inputs "$inputs" --rollouts "$rollouts" \
            ++allow_partial_rollouts=True > "$out" 2>&1
    fi
}

profile_one() {
    local label_dir=${1%/}
    local label; label="$(basename "$label_dir")"
    if [[ ! -s "$label_dir/rollouts.jsonl" ]]; then
        echo "  $label: no rollouts, skipping"
        return 0
    fi
    if profile_cmd "$label_dir/rollouts_materialized_inputs.jsonl" "$label_dir/rollouts.jsonl" \
                   "$label_dir/profile.txt"; then
        echo "  $label: $(grep -m1 'completion:' "$label_dir/profile.txt" 2>/dev/null || echo done)"
    else
        echo "  $label: profile FAILED, see $label_dir/profile.txt" >&2
        return 1
    fi
}
export -f profile_one profile_cmd

# Imports are page-cached after the first process, so concurrency recovers most of the
# per-label interpreter start the header describes.
PROFILE_JOBS=${PROFILE_JOBS:-8}
printf '%s\0' "$SWEEP_DIR"/by_label/*/ \
    | xargs -0 -P "$PROFILE_JOBS" -I{} bash -c 'profile_one "$@"' _ {} \
    || echo ">>> some labels failed; see the FAILED lines above" >&2

echo ">>> whole sweep"
profile_cmd "$SWEEP_DIR/rollouts_materialized_inputs.jsonl" "$SWEEP_DIR/rollouts.jsonl" \
            "$SWEEP_DIR/profile.txt" || true
tail -5 "$SWEEP_DIR/profile.txt" 2>/dev/null || true
