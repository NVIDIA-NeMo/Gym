#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Driver for the AA-v2 multi-stage comparison ELO run.
#   ./run_aav2.sh rollout smoke|full          produce deliverables (GPU)
#   ./run_aav2.sh rollout resume RUN_DIR     resume unfinished tasks (GPU)
#   ./run_aav2.sh import SOURCE_RUN_DIR      copy existing evidence into a fresh run
#   ./run_aav2.sh preconvert RUN_DIR         prepare Office/media views (CPU)
#   ./run_aav2.sh judge RUN_DIR smoke|pilot|full (CPU, API panel)
# Each command submits one phase. Inspect its result before starting the next.
set -euo pipefail
umask 077
W=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
: "${AAV2_CONFIG:?set AAV2_CONFIG to the shared configuration shell file}"
set -a
source "$AAV2_CONFIG"
set +a
: "${ACCOUNT:?configure ACCOUNT}"
GYM_SOURCE=${GYM_SOURCE:-$(cd "$W/../../../.." && pwd -P)}
GYM_REVISION=${GYM_REVISION:-HEAD}
UV_SOURCE=${UV_SOURCE:-$HOME/.local/bin/uv}
RUNS_DIR=${RUNS_DIR:-$PWD/runs}
GPU_PARTITION=${GPU_PARTITION:-batch}
CPU_PARTITION=${CPU_PARTITION:-cpu}
gpu_args=(--partition="$GPU_PARTITION")
cpu_args=(--partition="$CPU_PARTITION")
[[ -z ${GPU_QOS:-} ]] || gpu_args+=(--qos="$GPU_QOS")
[[ -z ${CPU_QOS:-} ]] || cpu_args+=(--qos="$CPU_QOS")
CMD=${1:?usage: run_aav2.sh rollout|import|preconvert|judge ...}
shift

prepare_run() {
    local arguments=(prepare "$RUN_DIR" --source "$GYM_SOURCE" --revision "$GYM_REVISION"
        --dataset "$SELECTED_DATASET" --judge-config "$JUDGE_CONFIG" --env-file "$ENV_FILE"
        --uv-source "$UV_SOURCE" --agent-sif "$AGENT_SIF" --apptainer-bin "$APPTAINER_BIN"
        --concurrency "$CONC" --agent-max-turns "${AGENT_MAX_TURNS:-250}")
    [[ -z ${SMOKE_DATASET:-} ]] || arguments+=(--smoke-dataset "$SMOKE_DATASET")
    [[ -z ${JUDGE_SIF:-} ]] || arguments+=(--judge-sif "$JUDGE_SIF")
    [[ $CMD != rollout ]] || arguments+=(--profile "${PROFILE:?configure PROFILE}")
    python3 "$W/snapshot.py" "${arguments[@]}" "$@"
}

case "$CMD" in
rollout)
    SIZE=${1:-smoke}
    if [[ $SIZE == resume ]]; then
        RUN_DIR=${2:?usage: run_aav2.sh rollout resume RUN_DIR}
        python3 "$RUN_DIR/package/snapshot.py" verify "$RUN_DIR"
        existing=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("EXISTING_ROLLOUT", ""))' "$RUN_DIR/run.json")
        [[ -z $existing ]] || { echo "Imported evidence cannot resume rollout; run preconvert then judge." >&2; exit 64; }
    else
        case "$SIZE" in
            smoke) SELECTED_DATASET=${SMOKE_DATASET:?configure SMOKE_DATASET}; CONC=${CONCURRENCY:-4} ;;
            full) SELECTED_DATASET=${DATASET:?configure DATASET}; CONC=${CONCURRENCY:-40} ;;
            *) echo "rollout takes smoke|full|resume RUN_DIR" >&2; exit 64 ;;
        esac
        RUN_DIR=$RUNS_DIR/rollout_${SIZE}_$(date +%Y%m%d_%H%M%S)
        prepare_run
    fi
    RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
    mkdir -p "$RUN_DIR/logs"
    jid=$(sbatch --parsable --account="$ACCOUNT" "${gpu_args[@]}" --nodes=1 \
        --export=ALL,AAV2_RUN_DIR="$RUN_DIR" --output="$RUN_DIR/logs/rollout-%j.out" \
        "$RUN_DIR/package/aav2_rollout.sbatch")
    printf '%s\n' "$jid" > "$RUN_DIR/rollout.jobid"
    echo "SUBMITTED rollout/$SIZE job=$jid"
    echo "RUN_DIR=$RUN_DIR"
    ;;
import)
    SOURCE_RUN_DIR=${1:?usage: run_aav2.sh import SOURCE_RUN_DIR}
    SELECTED_DATASET=${DATASET:?configure DATASET}
    CONC=${CONCURRENCY:-40}
    RUN_DIR=$RUNS_DIR/import_$(date +%Y%m%d_%H%M%S)
    prepare_run --existing-rollout "$SOURCE_RUN_DIR"
    echo "IMPORTED: $RUN_DIR (run preconvert next)"
    ;;
preconvert)
    RUN_DIR=${1:?usage: run_aav2.sh preconvert RUN_DIR}
    RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
    python3 "$RUN_DIR/package/snapshot.py" verify "$RUN_DIR"
    mkdir -p "$RUN_DIR/logs"
    jid=$(sbatch --parsable --account="$ACCOUNT" "${cpu_args[@]}" \
        --export=ALL,AAV2_RUN_DIR="$RUN_DIR" --output="$RUN_DIR/logs/preconvert-%j.out" \
        "$RUN_DIR/package/aav2_preconvert.sbatch")
    printf '%s\n' "$jid" > "$RUN_DIR/preconvert.jobid"
    echo "SUBMITTED preconvert job=$jid (run_dir=$RUN_DIR)"
    ;;
judge)
    RUN_DIR=${1:?usage: run_aav2.sh judge RUN_DIR [smoke|pilot|full]}
    SIZE=${2:-full}
    case "$SIZE" in smoke|pilot|full) ;; *) echo "judge takes smoke|pilot|full" >&2; exit 64 ;; esac
    RUN_DIR=$(cd "$RUN_DIR" && pwd -P)
    python3 "$RUN_DIR/package/snapshot.py" verify "$RUN_DIR"
    mkdir -p "$RUN_DIR/logs"
    # Only the mode crosses --export: comma-containing stage specifications
    # are constructed by the frozen batch script and passed as one Gym argument.
    jid=$(sbatch --parsable --account="$ACCOUNT" "${cpu_args[@]}" \
        --export=ALL,AAV2_RUN_DIR="$RUN_DIR",AAV2_MODE="$SIZE" --output="$RUN_DIR/logs/judge-$SIZE-%j.out" \
        "$RUN_DIR/package/aav2_judge.sbatch")
    printf '%s\n' "$jid" > "$RUN_DIR/judge_$SIZE.jobid"
    echo "SUBMITTED judge/$SIZE job=$jid (run_dir=$RUN_DIR)"
    ;;
*) echo "unknown command: $CMD" >&2; exit 64 ;;
esac
