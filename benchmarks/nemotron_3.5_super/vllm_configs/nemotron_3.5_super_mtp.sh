#!/bin/bash

GYM_MODEL_PARAMS=(
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.temperature=1.0"
    "++policy_model.responses_api_models.vllm_model.sampling_overrides.top_p=0.95"
)

# @bxyu-nvidia: `--skip-mm-profiling` Is needed to get Super VL checkpoint working, even with text benchmarks
VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.85
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --reasoning-parser nemotron_v3
    --enable-chunked-prefill
    --enable-prefix-caching
    --max-model-len 262144
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --block-size 128
    --mamba-cache-mode align
    --mamba-ssm-cache-dtype float32
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --skip-mm-profiling
    --data-parallel-size 1
    --api-server-count 1
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
    --enable-mamba-fine-grained-prefix-cache
    --prefix-match-unit 16
)
VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":180}}'
    --max-num-batched-tokens 33920
    --cudagraph-capture-sizes 1 2 4 8 16 16 24 32 32 40 48 56 64 64 72 80 88 96 104 112 120 128 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 256 272 288 304 320 336 352 368 384 400 416 432 448 464 480 496 512 512 544 576 608 640 672 704 736 768 800 832 864 896 928 960 992 1024 1024 1088 1152 1216 1280 1344 1408 1472 1536 1600 1664 1728 1792 1856 1920 1984 2048 2048 2176 2304 2432 2560 2688 2816 2944 3072 3200 3328 3456 3584 3712 3840 3968 4096 4096 4352 4608 4864 5120 5376 5632 5888 6144 6400 6656 6912 7168 7424 7680 7936 8192 8192 8704 9216 9728 10240 10752 11264 11776 12288 12800 13312 13824 14336 14848 15360 15872 16384 8192 8448 8704 8960 9216 9472 9728 9984 10240 10496 10752 11008 11264 11520 11776 12032 12288 12544 12800 13056 13312 13568 13824 14080 14336 14592 14848 15104 15360 15616 15872 16128 16384 16384 16896 17408 17920 18432 18944 19456 19968 20480 20992 21504 22016 22528 23040 23552 24064 24576 25088 25600 26112 26624 27136 27648 28160 28672 29184 29696 30208 30720 31232 31744 32256 32768 33920
    --cudagraph-metrics
    --max-num-seqs 1024
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)
VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail","kv_connector_extra_config":{"kv_lease_duration":180}}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)
