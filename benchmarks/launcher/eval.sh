#!/usr/bin/env bash
set -euo pipefail

launcher_dir=$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")
gym_root=$(realpath "$launcher_dir/../..")

if [[ -f "$launcher_dir/.env" ]]; then
    source "$launcher_dir/.env"
fi
export VLLM_CONTAINER=${VLLM_CONTAINER:-registry-1.docker.io#vllm/vllm-openai:v0.29.0-aarch64}

checkpoint=""
profile=""
smoke=false
usage() { echo "Usage: eval.sh --profile PATH --checkpoint PATH [--smoke]"; }
while (( $# )); do
    case "$1" in
        --profile)
            profile=${2:?--profile requires a script path}
            shift 2
            ;;
        --checkpoint=*) checkpoint=${1#*=}; shift ;;
        --checkpoint)
            checkpoint=${2:?--checkpoint requires a directory}
            shift 2
            ;;
        --smoke) smoke=true; shift ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

checkpoint=$(realpath "${checkpoint:?Pass --checkpoint PATH}")
profile=$(realpath "${profile:?Pass --profile PATH}")
source "$profile"
VLLM_CONFIG=$(realpath "$VLLM_CONFIG")
export VLLM_CONFIG BENCHMARK
source "$VLLM_CONFIG"
export GYM_CONTAINER=${GYM_CONTAINER:?Set GYM_CONTAINER to your Gym image}
export ROUTER_CONTAINER=${ROUTER_CONTAINER:?Set ROUTER_CONTAINER to your router image}

export CHECKPOINT="$checkpoint"
export EXPERIMENT_NAME="$(basename "$(dirname "$profile")")-$(basename "$profile" .sh)"
export WANDB_MODE=${WANDB_MODE:-disabled}
export RUNS_DIR=${RUNS_DIR:-$gym_root/../runs}

export GYM_INFERENCE_METRICS_ENABLED=${GYM_INFERENCE_METRICS_ENABLED:-false}

mkdir -p "$RUNS_DIR"
RUNS_DIR=$(realpath "$RUNS_DIR")
# Explicit mounts: vLLM profile, logs/results, and checkpoints.
export MOUNTS="$RUNS_DIR:$RUNS_DIR,$checkpoint:$checkpoint:ro"
MOUNTS+=",$VLLM_CONFIG:$VLLM_CONFIG:ro"

if [[ -n ${GYM_DEV_CHECKOUT:-} ]]; then
    MOUNTS+=",$(realpath "$GYM_DEV_CHECKOUT"):/mnt/gym-dev:ro"
fi

# Shared Gym run settings; prepare receives only the benchmark arguments below.
gym_run_args=(
    --config benchmarks/nemotron_3.5_super/sandbox_utils.yaml
    '++wandb_project=${oc.env:WANDB_PROJ,null}'
    '++wandb_api_key=${oc.env:WANDB_API_KEY,null}'
    +uv_venv_dir=/opt/uv_venvs
    ++split=benchmark
    ++use_absolute_ip=true
    ++reuse_existing_data_preparation=true
    ++policy_api_key=dummy_api_key
    ++policy_model_name=$MODEL_NAME
    ++upload_rollouts=false
)
printf -v GYM_RUN_ARGS '%q ' "${gym_run_args[@]}"
export GYM_RUN_ARGS

# Benchmark settings come from the selected evaluation profile.
config=$BENCHMARK_CONFIG
repeats=${BENCHMARK_REPEATS:-1}
concurrency=$BENCHMARK_CONCURRENCY
limit=${BENCHMARK_LIMIT:-null}

# Full benchmark by default; smoke uses one task and one rollout.
if [[ $smoke == true ]]; then
    limit=1
    repeats=1
    concurrency=1
fi

printf 'Profile: %s\nCheckpoint: %s\n' "$profile" "$checkpoint"
bash "$launcher_dir/sbatch.sh" \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config "$config" \
    ++limit="$limit" ++num_repeats="$repeats" \
    ++num_samples_in_parallel="$concurrency" \
    ++observability_enabled="${OBSERVABILITY_ENABLED:-true}"
