#!/bin/bash

command=$(cat <<EOF
set -euo pipefail

pip install ray==2.55.1

host=\$(hostname -I | awk '{print \$1}')
VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 \
vllm serve $MODEL \
    --gpu-memory-utilization 0.9 \
    --distributed-executor-backend ray \
    --data-parallel-backend ray \
    --data-parallel-size 4 \
    --data-parallel-size-local 1 \
    --tensor-parallel-size 4 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --api-server-count 1 \
    --kv-cache-dtype fp8 \
    -cc.pass_config.fuse_allreduce_rms=False \
    --mamba-ssm-cache-dtype float32 \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}' \
    --enable-expert-parallel \
    --max-num-batched-tokens 16384 \
    --host \$host \
    --port 8000 &

# Assume this is run from Gym repository root.
source .venv/bin/activate

gym eval prepare --config benchmarks/gpqa/config.yaml +use_cached_prepared_benchmarks=true

ip=http://\$host:8000/v1
until curl -s \$ip >/dev/null; do
    sleep 5
done

experiment_name=gpqa-\$(date +%Y%m%d_%H%M%S)
gym eval run \
    --config responses_api_models/vllm_model/configs/vllm_model.yaml \
    --config benchmarks/gpqa/config.yaml \
    +wandb_project=$USER-gym-eval \
    +wandb_name=\$experiment_name \
    ++output_jsonl_fpath=results/\$experiment_name.jsonl \
    ++overwrite_metrics_conflicts=true \
    ++split=benchmark \
    ++reuse_existing_data_preparation=true \
    ++policy_base_url=\$ip \
    ++policy_api_key=dummy_api_key \
    ++policy_model_name=$MODEL \
    ++global_aiohttp_connector_limit_per_host=512

EOF
)

CONTAINER=/lustre/fs1/portfolios/nemotron/projects/nemotron_evals_dev/users/bxyu/vllm/vllm-openai:v0.22.1___with_ray.sqsh \
MOUNTS=/lustre:/lustre \
sbatch \
    --nodes=4 \
    --account=nemotron_n4_post \
    --partition=batch \
    --gres=gpu:4 \
    --time=04:00:00 \
    --job-name=gym-vllm-eval-$USER \
    --exclusive \
    scripts/sbatch_base.sh bash -lc "$command"
