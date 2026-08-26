#!/bin/bash
# Collect reward-profiling rollouts against an already-running vLLM endpoint.
#
# Starts every Gym server for the sweep, then resumes from the materialized inputs written by
# prepare_sweep.sh. `--resume` is not optional: without it rollout collection clears the output
# and re-expands from scratch, and a sweep this size will not finish inside one allocation.
#
#   VLLM_JOBID=<jobid> \
#   SWEEP_DIR=<outputs/sweeps>/<nickname> \
#   POLICY_MODEL_NAME=<checkpoint-path> \
#   bash .../scripts/run_rollouts.sh
set -euo pipefail

VLLM_JOBID=${VLLM_JOBID:?set VLLM_JOBID to the running vLLM job}
SWEEP_DIR=${SWEEP_DIR:?set SWEEP_DIR to <out-dir>/<nickname> from prepare_sweep.sh}
POLICY_MODEL_NAME=${POLICY_MODEL_NAME:?set POLICY_MODEL_NAME to the served checkpoint path}
CONTAINER=${CONTAINER:?set CONTAINER to the reward-profiling sqsh}
GYM_CONFIG=${GYM_CONFIG:-sweep_config.yaml}
CONCURRENCY=${CONCURRENCY:-128}
ROUTER_PORT=${ROUTER_PORT:-8000}
MOUNTS=${MOUNTS:-/lustre:/lustre}
# Secrets reach Gym through env.yaml, which it auto-loads from its working directory. The judge
# lane needs it: the judge config_overlay interpolates ${nv_inference_api_key}, which env.yaml resolves
# from the shell. Without the mount the config fails to parse rather than failing at judge time.
ENV_YAML=${ENV_YAML:-$PWD/env.yaml}
if [[ -f "$ENV_YAML" ]]; then
    MOUNTS="$MOUNTS,$ENV_YAML:/opt/Gym/env.yaml"
else
    echo "WARNING: no env.yaml at $ENV_YAML; judge environments will fail to resolve their API key." >&2
fi

mapfile -t NODES < <(scontrol show hostnames "$(squeue -j "$VLLM_JOBID" -h -o '%N')")
ROUTER_IP=$(getent hosts "${NODES[0]}" | awk 'NR==1 {print $1}')
# nodes[0] runs prefill plus the router; keep the driver off it.
DRIVER_NODE=${NODES[1]:-${NODES[0]}}

echo "router : http://$ROUTER_IP:$ROUTER_PORT/v1"
echo "driver : $DRIVER_NODE"
echo "sweep  : $SWEEP_DIR"

srun --overlap --jobid="$VLLM_JOBID" --nodes=1 --ntasks=1 \
     --nodelist="$DRIVER_NODE" --gpus=0 --cpus-per-task="${CPUS:-64}" \
     --container-image="$CONTAINER" \
     --container-mounts="$MOUNTS" \
     --container-workdir=/opt/Gym --no-container-mount-home \
     bash -lc "
       set -euo pipefail
       source /opt/Gym_venv/bin/activate
       gym env start --config '$SWEEP_DIR/$GYM_CONFIG' \
           +uv_venv_dir=/opt/uv_venvs +skip_venv_if_present=true \
           ++policy_base_url=http://$ROUTER_IP:$ROUTER_PORT/v1 \
           ++policy_api_key=dummy_api_key \
           ++policy_model_name='$POLICY_MODEL_NAME' &
       gym_pid=\$!
       trap 'kill \$gym_pid 2>/dev/null || true' EXIT

       gym eval run --no-serve --resume \
           --input '$SWEEP_DIR/rollouts_materialized_inputs.jsonl' \
           --output '$SWEEP_DIR/rollouts.jsonl' \
           ++num_repeats=1 \
           ++num_samples_in_parallel=$CONCURRENCY \
           +nemo_gym_log_dir='$SWEEP_DIR/logs' \
           +uv_venv_dir=/opt/uv_venvs +skip_venv_if_present=true

       gym eval profile \
           --inputs '$SWEEP_DIR/rollouts_materialized_inputs.jsonl' \
           --rollouts '$SWEEP_DIR/rollouts.jsonl'
     "
