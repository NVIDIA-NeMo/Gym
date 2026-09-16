#!/bin/bash

# DS v4 Flash takes a super long time to start.
GYM_MODEL_PARAMS=(
    "++model_endpoint_readiness_timeout_seconds=1200"
)

export MOONCAKE_CONFIG_PATH=/etc/mooncake/mooncake_vllm_config.json

uv pip install --system 'mooncake-transfer-engine>=0.3.10'

if (( SLURM_PROCID == 0 )); then
cat > $MOONCAKE_CONFIG_PATH <<EOF
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "$(hostname):50051",
  "global_segment_size": "100GB",
  "local_buffer_size": "4GB",
  "protocol": "rdma",
  "device_name": "",
  "enable_offload": false
}
EOF

    mooncake_master \
        -rpc_port=50051 \
        -rpc_thread_num=4 \
        -default_kv_lease_ttl=30000 \
        -eviction_high_watermark_ratio=0.95 \
        -eviction_ratio=0.1 \
        -logtostderr
else
    until [ -f $MOONCAKE_CONFIG_PATH ]
    do
        sleep 1
    done
fi

VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser deepseek_v4
    --reasoning-parser deepseek_v4
    --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
    --tokenizer-mode deepseek_v4
    --enable-chunked-prefill
    --enable-prefix-caching
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --block-size 256
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --data-parallel-size 1
    --data-parallel-size-local 1
    --tensor-parallel-size 4
    --api-server-count 1
    --attention_config.use_fp4_indexer_cache True
    --moe-backend deep_gemm_mega_moe
    --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'
)
VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"lookup_async":true}}]}}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"},{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both","kv_connector_extra_config":{"load_async":true,"lookup_async":true}}]}}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
)
