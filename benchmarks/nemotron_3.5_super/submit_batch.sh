#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage() {
    echo 'Usage: bash benchmarks/nemotron_3.5_super/submit_batch.sh core|swe [--check] [--config PATH] [key=value ...]'
    echo 'Required environment: MODEL, CONTAINER, SBATCH_ACCOUNT, SBATCH_PARTITION.'
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
batch=${1:-}
case "$batch" in core|swe) ;; *) usage >&2; exit 2 ;; esac
shift

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)
cd "$repo_root"
gym_args=(--config "benchmarks/nemotron_3.5_super/batch_configs/$batch.yaml")
check_only=0
while (( $# )); do
    case "$1" in
        --check) check_only=1 ;;
        --config)
            [[ $# -ge 2 && -r "$2" ]] || { echo '--config requires a readable YAML file.' >&2; exit 2; }
            gym_args+=(--config "$2"); shift ;;
        --*) echo "Unexpected option: $1" >&2; exit 2 ;;
        *=*) gym_args+=("$1") ;;
        *) echo 'Expected --config PATH or a Hydra key=value override.' >&2; exit 2 ;;
    esac
    shift
done

: "${MODEL:?Set MODEL to the shared checkpoint directory.}"
: "${CONTAINER:?Set CONTAINER to a compatible Gym/vLLM .sqsh image.}"
: "${SBATCH_ACCOUNT:?Set SBATCH_ACCOUNT to your Slurm account.}"
: "${SBATCH_PARTITION:?Set SBATCH_PARTITION to a partition supporting the requested walltime.}"
export MODEL CONTAINER SBATCH_ACCOUNT SBATCH_PARTITION
export MODEL_NAME=${MODEL_NAME:-nemotron-3.5-super}
export SBATCH_TIME=${SBATCH_TIME:-04:00:00}
export SBATCH_QOS=${SBATCH_QOS:-normal}
export SBATCH_GRES=gpu:4
export NUM_PREFILL_NODES=${NUM_PREFILL_NODES:-2}
export NUM_DECODE_NODES=${NUM_DECODE_NODES:-2}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-nemotron-35-${batch}-$(date -u +%Y%m%dT%H%M%S)-$$}
export VLLM_CONFIG="$repo_root/benchmarks/nemotron_3.5_super/vllm_configs/batched.sh"
export EXPORT_TO_CSV=0 EXPORT_CSV_TO_MODEL_DIR=0 RAY_TMPDIR=/tmp

[[ $MODEL == /* && $CONTAINER == /* ]] || { echo 'MODEL and CONTAINER must be absolute paths.' >&2; exit 2; }
[[ $EXPERIMENT_NAME =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || {
    echo 'EXPERIMENT_NAME must be a single name containing only letters, digits, dots, underscores, or hyphens.' >&2
    exit 2
}
for nodes in "$NUM_PREFILL_NODES" "$NUM_DECODE_NODES"; do
    [[ $nodes =~ ^[1-9][0-9]*$ ]] || { echo 'Node counts must be positive integers.' >&2; exit 2; }
done
# The underlying launcher interpolates these values into shell commands and mount lists.
for value in "$repo_root" "$MODEL" "$CONTAINER" "$MODEL_NAME" "${ROLLOUTS_FPATH:-}"; do
    [[ ! $value =~ [^a-zA-Z0-9_./-] ]] || { echo 'Paths and MODEL_NAME must not contain spaces or shell metacharacters.' >&2; exit 2; }
done
for required in "$MODEL/config.json" "$MODEL/chat_template.jinja" "$MODEL/ultra_v3_reasoning_parser.py" "$CONTAINER"; do
    [[ -r $required ]] || { echo "Missing or unreadable: $required" >&2; exit 1; }
done
run_dir="$repo_root/results/$EXPERIMENT_NAME"
export MOUNTS="$repo_root:$repo_root,$repo_root:/opt/Gym,$MODEL:$MODEL:ro,$run_dir/uv_venvs:/opt/uv_venvs${MOUNTS:+,$MOUNTS}"
[[ ! $MOUNTS =~ [^a-zA-Z0-9_./,:=-] ]] || { echo 'MOUNTS must not contain spaces or shell metacharacters.' >&2; exit 2; }

# Use the checkout's installed Gym, without installing packages or contacting services.
"${GYM_PYTHON:-$repo_root/.venv/bin/python}" - "${gym_args[@]}" <<'PY'
import json
import sys
from omegaconf import OmegaConf
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionConfig

paths, overrides = [], []
args = iter(sys.argv[1:])
for arg in args:
    if arg == '--config':
        paths.append(next(args))
    else:
        overrides.append(arg)
paths += ['benchmarks/nemotron_3.5_super/sandbox_utils.yaml',
          'benchmarks/nemotron_3.5_super/policy_model_override.yaml']
sys.argv = ['gym', '+config_paths=' + json.dumps(paths), *overrides]
config = GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(
    initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, offline=True,
))
RolloutCollectionConfig.model_validate({
    **OmegaConf.to_container(config, resolve=True),
    'input_jsonl_fpath': '<preflight>', 'output_jsonl_fpath': '<preflight>',
})
credentials = ['sandbox.opensandbox.connection.domain', 'sandbox.opensandbox.connection.api_key']
if 'nv_inference_api_key' in config:
    credentials.append('nv_inference_api_key')
for path in credentials:
    if OmegaConf.select(config, path) in (None, '', 'dummy', '??', '???'):
        raise SystemExit(f'Missing connection setting: {path}')
print('PASS: Gym configuration resolves; model and service readiness are checked inside the job.')
PY

printf 'Batch: %s; nodes: %s+%s; walltime: %s; account: %s; partition: %s\n' \
    "$batch" "$NUM_PREFILL_NODES" "$NUM_DECODE_NODES" "$SBATCH_TIME" "$SBATCH_ACCOUNT" "$SBATCH_PARTITION"
if (( check_only )); then exit 0; fi

mkdir -p "$run_dir/uv_venvs" slurm-logs
# Preserve each argument through the launcher's generated shell and through prefetch.
# Only this printf-produced, shell-escaped string is decoded by batched.sh.
printf -v GYM_BATCH_ARGS '%q ' "${gym_args[@]}"
export GYM_BATCH_ARGS
printf 'Experiment: %s\n' "$EXPERIMENT_NAME"
printf 'Output: %s\n' "${ROLLOUTS_FPATH:-results/$EXPERIMENT_NAME/slurm_job_id_<job-id>/date_<timestamp>.jsonl}"
bash benchmarks/nemotron_3.5_super/sbatch_external_vllm.sh "$GYM_BATCH_ARGS"
