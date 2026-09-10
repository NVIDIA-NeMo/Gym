#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Common node-local setup for the three manual AA-v2 phases.

gdpval_fail() { echo "AAV2_JOB_FAIL: $*" >&2; exit 64; }
gdpval_local_path() {
    [[ $(readlink -f -- "$1") == /raid/scratch/* ]] || gdpval_fail "path is outside local scratch: $1"
    case $(stat -f -c '%T' "$1") in lustre*|nfs*) gdpval_fail "shared filesystem: $1" ;; esac
}

gdpval_prepare() {
    local phase=$1 env_file input python_version count rows calibration
    gdpval_local_path "$JOB_ROOT"
    cp -a -- "$AAV2_PACKAGE_DIR" "$JOB_ROOT/package"
    LOCAL_PACKAGE=$JOB_ROOT/package
    cp -- "$AAV2_RUN_DIR/run.env" "$JOB_ROOT/run.env"
    source "$JOB_ROOT/run.env"
    env_file=${ENV_FILE:-}
    if [[ -n $env_file ]]; then
        cp -- "$env_file" "$JOB_ROOT/credentials.env"
        chmod 0600 "$JOB_ROOT/credentials.env"
        set -a
        source "$JOB_ROOT/credentials.env"
        set +a
        source "$JOB_ROOT/run.env"
    fi
    [[ $RUN_DIR == "$AAV2_RUN_DIR" && -f $DATASET ]] || gdpval_fail "invalid run directory or dataset"
    exec 9>"$RUN_DIR/.phase.lock"
    flock -n 9 || gdpval_fail "another phase is already running in $RUN_DIR"
    if [[ $phase == rollout ]]; then
        [[ -z ${EXISTING_ROLLOUT:-} ]] || gdpval_fail "imported evidence cannot resume rollout"
        cp -- "$PROFILE" "$JOB_ROOT/serving.env"
        set -a
        source "$JOB_ROOT/serving.env"
        set +a
    else
        cp -- "$JUDGE_CONFIG" "$JOB_ROOT/judge.yaml"
    fi
    source "$JOB_ROOT/run.env"
    PHASE_DIR=$RUN_DIR
    [[ $phase != preconvert ]] || PHASE_DIR=$RUN_DIR/preconvert
    if [[ $phase == judge ]]; then
        case ${AAV2_MODE:-full} in
            smoke) DATASET=$SMOKE_DATASET; CONCURRENCY=4; NUM_COMPARISON_TRIALS=1; count=4 ;;
            pilot) CONCURRENCY=8; NUM_COMPARISON_TRIALS=2; count=12 ;;
            full) CONCURRENCY=16; NUM_COMPARISON_TRIALS=4; count=220 ;;
            *) gdpval_fail "judge takes smoke|pilot|full" ;;
        esac
        rows=$(awk 'NF {n++} END {print n+0}' "$DATASET")
        (( rows > 0 )) || gdpval_fail "dataset is empty"
        (( count <= rows )) || count=$rows
        STAGES="[{num_tasks: $count}]"
        [[ $AAV2_MODE != pilot ]] || STAGES="[{num_tasks: $count}, {num_tasks: $count, num_models: 4}]"
        if [[ $AAV2_MODE == full ]]; then
            calibration=$((rows < 45 ? rows : 45))
            STAGES="[{num_tasks: $calibration, partial_completion: {min_success_fraction: 0.97, min_per_reference_success_fraction: 0.88, min_successful_rows_per_reference: 1, tolerate_unresolved: true}}, {num_tasks: $rows, num_models: 4}]"
        fi
        PHASE_DIR=$RUN_DIR/judge_$AAV2_MODE
    fi
    mkdir -p "$PHASE_DIR/logs" "$RUN_DIR/deliverables"
    unset BASH_ENV PYTHONHOME PYTHONPATH NEMO_GYM_EXTRA_ROOTS NEMO_GYM_CONFIG_DICT NEMO_GYM_CONFIG_PATH
    unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT UV_NO_MANAGED_PYTHON NEMO_GYM_VENV_BIN
    unset UV_PYTHON UV_CONFIG_FILE UV_CONSTRAINT UV_OVERRIDE HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 UV_MANAGED_PYTHON=true UV_LINK_MODE=copy
    export PATH=/cm/local/apps/slurm/current/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
    export RAY_TMPDIR=/raid/scratch/$SLURM_JOB_USER/r/$SLURM_JOB_ID
    (( ${#RAY_TMPDIR} <= 48 )) || gdpval_fail "Ray scratch prefix is too long"
    export TMPDIR=$RAY_TMPDIR/tmp
    export UV_CACHE_DIR=/raid/scratch/$SLURM_JOB_USER/uv-cache
    export UV_PYTHON_INSTALL_DIR=$JOB_ROOT/python UV_PYTHON_BIN_DIR=$JOB_ROOT/bin
    export XDG_CACHE_HOME=$JOB_ROOT/cache/xdg PYTHONPYCACHEPREFIX=$JOB_ROOT/cache/pycache
    export HF_HOME=$JOB_ROOT/cache/huggingface HF_DATASETS_CACHE=$JOB_ROOT/cache/huggingface/datasets
    export PIP_CACHE_DIR=$JOB_ROOT/cache/pip GDPVAL_REF_FILES_DIR=$JOB_ROOT/reference_files
    export APPTAINER_TMPDIR=$JOB_ROOT/apptainer/tmp APPTAINER_CACHEDIR=$JOB_ROOT/apptainer/cache
    install -d -m 0700 "$UV_CACHE_DIR"
    gdpval_local_path "$UV_CACHE_DIR"
    mkdir -p "$TMPDIR" "$UV_PYTHON_INSTALL_DIR" "$UV_PYTHON_BIN_DIR" \
        "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR" \
        "$GDPVAL_REF_FILES_DIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
    gdpval_local_path "$RAY_TMPDIR"
    cp -- "$UV_SOURCE" "$JOB_ROOT/bin/uv"
    chmod 0500 "$JOB_ROOT/bin/uv"
    export MARS_UV=$JOB_ROOT/bin/uv PATH=$JOB_ROOT/bin:$PATH
    GYM_ROOT=$JOB_ROOT/gym
    [[ ! -e $GYM_ROOT ]] || gdpval_fail "refusing an existing source stage"
    mkdir "$GYM_ROOT"
    tar -xf "$RUN_DIR/source.tar" -C "$GYM_ROOT"
    [[ ! -e $GYM_ROOT/.venv && ! -L $GYM_ROOT/.venv ]] || gdpval_fail "source archive contains a virtualenv"
    for input in .python-version pyproject.toml uv.lock; do
        [[ -f $GYM_ROOT/$input && ! -L $GYM_ROOT/$input ]] || gdpval_fail "missing tracked dependency input: $input"
    done
    python_version=$(<"$GYM_ROOT/.python-version")
    [[ $python_version =~ ^3\.[0-9]+(\.[0-9]+)?$ ]] || gdpval_fail "invalid tracked Python version"
    mkdir -p "$GYM_ROOT/cache"
    (cd "$GYM_ROOT" && UV_PROJECT_ENVIRONMENT=$GYM_ROOT/.venv \
        "$MARS_UV" sync --frozen --no-dev --managed-python --python "$python_version")
    GYM_PYTHON=$GYM_ROOT/.venv/bin/python
    [[ -x $GYM_PYTHON && $(readlink -f -- "$GYM_PYTHON") == "$UV_PYTHON_INSTALL_DIR"/* ]] \
        || gdpval_fail "Gym Python is not the local managed interpreter"
    COMPONENT_VENVS=$JOB_ROOT/component_venvs
    mkdir "$COMPONENT_VENVS"
    export PATH=$GYM_ROOT/.venv/bin:$PATH PYTHONPATH=$GYM_ROOT
    cd "$GYM_ROOT"
    GYM_COMMAND=("$GYM_PYTHON" -c 'from nemo_gym.cli.main import main; main()')
    # Keep paths inside materialized rows and scientific config stable on resume.
    "$GYM_PYTHON" - "$DATASET" "$JOB_ROOT/dataset.yaml" <<'PYTHON'
import json
import sys
from pathlib import Path
dataset = {"name": "gdpval", "type": "benchmark", "jsonl_fpath": sys.argv[1],
           "prompt_config": None, "prepare_script": "benchmarks/gdpval/prepare.py", "num_repeats": 1}
config = {"gdpval_stirrup_agent": {"responses_api_agents": {"stirrup_agent": {"datasets": [dataset]}}}}
Path(sys.argv[2]).write_text(json.dumps(config))
PYTHON
    GYM_OPTIONS=(--config responses_api_models/vllm_model/configs/vllm_model.yaml
        --config benchmarks/gdpval/config.yaml --split benchmark --output "$PHASE_DIR/rollouts.jsonl"
        --concurrency "$CONCURRENCY" --num-repeats 1 --resume
        ++overwrite_metrics_conflicts=true ++skip_venv_if_present=false
        "++uv_venv_dir=$COMPONENT_VENVS" "++uv_cache_dir=$UV_CACHE_DIR" ++uv_pip_set_python=true
        ++gdpval_stirrup_agent.responses_api_agents.stirrup_agent.rerun_incomplete=true)
    JOB_PIDS=()
    trap gdpval_cleanup EXIT
    trap 'exit 143' TERM
    trap 'exit 130' INT
}

gdpval_prepare_sandbox() {
    local image=${1:-$AGENT_SIF}
    [[ -f $image && ! -L $image ]] || gdpval_fail "sandbox image must be a regular file"
    cp -- "$image" "$JOB_ROOT/agent.sif"
    [[ ${APPTAINER_BIN##*/} == bin && -d ${APPTAINER_BIN%/bin} && ! -L ${APPTAINER_BIN%/bin} \
        && -x $APPTAINER_BIN/apptainer ]] || gdpval_fail "invalid Apptainer installation"
    cp -a -- "${APPTAINER_BIN%/bin}" "$JOB_ROOT/apptainer-install"
    export APPTAINER_BIN=$JOB_ROOT/apptainer-install/bin GDPVAL_CONTAINER_PATH=$JOB_ROOT/agent.sif
    gdpval_local_path "$GDPVAL_CONTAINER_PATH"
    gdpval_local_path "$APPTAINER_BIN/apptainer"
    export PATH=$APPTAINER_BIN:$PATH
}

gdpval_prepare_serving() {
    : "${POLICY_SERVE_SCRIPT:?serving profile must set POLICY_SERVE_SCRIPT}"
    : "${MODEL_NAME:?serving profile must set MODEL_NAME}"
    [[ $MODEL_NAME =~ ^[A-Za-z0-9_./:-]+$ ]] || gdpval_fail "MODEL_NAME contains unsupported CLI characters"
    : "${MODEL_PATH:?serving profile must set MODEL_PATH}"
    : "${CONTAINER_IMAGE:?serving profile must set CONTAINER_IMAGE}"
    [[ ${NODES_PER_REPLICA:-1} == 1 ]] || gdpval_fail "serving requires one node per replica"
    gdpval_prepare_sandbox
    cp -- "$POLICY_SERVE_SCRIPT" "$JOB_ROOT/serve.sh"
    "$GYM_PYTHON" "$LOCAL_PACKAGE/rollout_serving.py" stage \
        --root "$JOB_ROOT/serving" --image "$CONTAINER_IMAGE" --model "$MODEL_PATH" \
        --runtime-root "$JOB_ROOT" --extra-mounts "${EXTRA_MOUNTS:-}"
    source "$JOB_ROOT/serving/environment.sh"
    cp -- "$JOB_ROOT/serving/manifest.json" "$RUN_DIR/logs/serving-$SLURM_JOB_ID.json"
}

gdpval_run_gym() {
    local head_port
    head_port=$("$GYM_PYTHON" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
    echo "Gym log: $PHASE_DIR/logs/gym-$SLURM_JOB_ID.log"
    setsid "${GYM_COMMAND[@]}" eval run "${GYM_OPTIONS[@]}" "$@" \
        ++head_server.host=127.0.0.1 "++head_server.port=$head_port" \
        > "$PHASE_DIR/logs/gym-$SLURM_JOB_ID.log" 2>&1 &
    JOB_PIDS+=("$!")
    wait "${JOB_PIDS[${#JOB_PIDS[@]}-1]}"
}

gdpval_cleanup() {
    local status=$? pid deadline=$((SECONDS + 10)) alive
    trap - EXIT TERM INT
    [[ -n ${JOB_PIDS[*]:-} ]] || exit "$status"
    for pid in "${JOB_PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
    while (( SECONDS < deadline )); do
        alive=false
        for pid in "${JOB_PIDS[@]}"; do kill -0 -- "-$pid" 2>/dev/null && alive=true; done
        [[ $alive == true ]] || break
        sleep 1
    done
    for pid in "${JOB_PIDS[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    exit "$status"
}
