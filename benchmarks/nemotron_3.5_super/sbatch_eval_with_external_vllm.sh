#!/bin/bash

set -euo pipefail

# Input arguments and validation
EXPERIMENT_NAME=$EXPERIMENT_NAME
NUM_NODES=$NUM_NODES
CONTAINER=$CONTAINER
MOUNTS=$MOUNTS
VLLM_SCRIPT=$VLLM_SCRIPT

command=$(cat <<EOF
set -euo pipefail

host=\$(hostname -I | awk '{print \$1}')
source $VLLM_SCRIPT

# Activate environment in container and cd into Gym. The Gym path here may be mounted.
source /opt/Gym_venv/bin/activate
cd /opt/Gym

gym eval prepare $@ +use_cached_prepared_benchmarks=true

ip=http://\$host:8000/v1
until curl -s \$ip >/dev/null; do
    sleep 5
done

experiment_name=$EXPERIMENT_NAME-\$(date +%Y%m%d_%H%M%S)

poll_vllm_metrics() {
  while true; do
    printf "# SCRAPE %s\n" "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    curl -fsS "http://\$host:8000/metrics" || printf "# ERROR curl failed for %s\n" "\$host"
    printf "\n"
    sleep 60
  done
}
poll_vllm_metrics >> "results/\${experiment_name}_vllm_metrics.log" 2>&1 &

# +uv_venv_dir=/opt/uv_venvs is from the container.
# +skip_venv_if_present=true will reuse the venvs baked into the container if possible.
gym eval run \
    $@ \
    +wandb_project=$USER-gym-eval \
    +wandb_name=\$experiment_name \
    +uv_venv_dir=/opt/uv_venvs \
    +skip_venv_if_present=true \
    ++output_jsonl_fpath=results/\$experiment_name.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++use_absolute_ip=true \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=\$ip \
    ++policy_api_key=dummy_api_key \
    ++policy_model_name=$MODEL \
    ++global_aiohttp_connector_limit_per_host=16384

EOF
)

# --segment > 0 otherwise the engine will hang on the second or third engine step.
sbatch \
    --nodes=$NUM_NODES \
    --gres=gpu:4 \
    --time=04:00:00 \
    --job-name=gym-vllm-eval-$EXPERIMENT_NAME-$USER \
    --exclusive \
    --segment=$NUM_NODES \
    scripts/sbatch_base.sh bash -lc "$command"
