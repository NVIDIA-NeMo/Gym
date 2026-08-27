#!/bin/bash
# Split a sweep by manifest entry and reward-profile each, plus the sweep as a whole.
#
#   SWEEP_DIR=<outputs/sweeps>/<nickname> CONTAINER=<sqsh> VLLM_JOBID=<jobid> \
#   bash .../scripts/05_profile.sh
#
# VLLM_JOBID/CONTAINER are only needed to get a container with gym on PATH; the profiler itself
# needs no GPU, so --gpus=0 against any live allocation is enough. Omit them if gym is already on
# PATH (e.g. inside the container).
#
# Safe on a partial sweep: ++allow_partial_rollouts=True keeps groups that have some but not all
# repeats, and the profiler reports the completion percentage. 03_run.sh runs this at the end of a
# job; use it directly to profile mid-run or after merging shards.
set -euo pipefail

SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to the <out-dir>/<nickname> directory}
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

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

for label_dir in "$SWEEP_DIR"/by_label/*/; do
    [[ -d "$label_dir" ]] || continue
    label="$(basename "$label_dir")"
    if [[ ! -s "$label_dir/rollouts.jsonl" ]]; then
        echo "  $label: no rollouts, skipping"
        continue
    fi
    profile_cmd "$label_dir/rollouts_materialized_inputs.jsonl" "$label_dir/rollouts.jsonl" \
                "$label_dir/profile.txt" || echo "  $label: profile failed, see $label_dir/profile.txt" >&2
    echo "  $label: $(grep -m1 'completion:' "$label_dir/profile.txt" 2>/dev/null || echo done)"
done

echo ">>> whole sweep"
profile_cmd "$SWEEP_DIR/rollouts_materialized_inputs.jsonl" "$SWEEP_DIR/rollouts.jsonl" \
            "$SWEEP_DIR/profile.txt" || true
tail -5 "$SWEEP_DIR/profile.txt" 2>/dev/null || true
