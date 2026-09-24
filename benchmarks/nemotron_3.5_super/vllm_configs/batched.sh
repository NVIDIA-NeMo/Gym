#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Serving flags from the batch pilots: use a compatible Super 3.5 vLLM image.
# Sourced in both containers; only the evaluation container checks Gym dependencies.
if [[ -n "${ROUTER_NODE:-}" ]]; then
    : "${GYM_BATCH_ARGS:?Submit with benchmarks/nemotron_3.5_super/submit_batch.sh.}"
    source /opt/Gym_venv/bin/activate

    # submit_batch.sh encodes each original argument with printf %q, including overrides.
    # Decode only that generated string so the check sees the evaluation's server list.
    eval "batch_args=($GYM_BATCH_ARGS)"
    if ! python - "${batch_args[@]}" <<'PY'
import json
import sys
from pathlib import Path
import nemo_gym.cli.eval
import nemo_gym.cli.env
import ray
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

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
parser = GlobalConfigDictParser()
config = parser.parse(GlobalConfigDictParserConfig(
    initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, offline=True,
))
venvs = {
    Path('/opt/uv_venvs', server.SERVER_TYPE, next(iter(getattr(server, server.SERVER_TYPE))), '.venv')
    for server in parser.filter_for_server_instance_configs(config)
}
missing = sorted(str(venv) for venv in venvs
                 if not all((venv / name).exists() for name in ('bin/python', 'bin/activate')))
if missing:
    raise SystemExit('Missing prebuilt server environments:\n' + '\n'.join(missing))
print(f'PASS: prebuilt Gym imports and {len(venvs)} server environment paths checked; no packages installed.')
PY
    then
        echo 'Prebuilt dependency check failed. Rebuild CONTAINER with benchmarks/nemotron_3.5_super/build_eval_container.sh for this checkout and batch config.' >&2
        return 1
    fi
fi

# Sampling is in batch_configs/*.yaml so CLI overrides remain effective.
GYM_MODEL_PARAMS=()
VLLM_COMMON_ARGS=(
    --trust-remote-code
    --disable-uvicorn-access-log
    --gpu-memory-utilization 0.9
    --distributed-executor-backend mp
    --data-parallel-backend mp
    --enable-auto-tool-choice
    --tool-call-parser qwen3_coder
    --chat-template "$MODEL/chat_template.jinja"
    --reasoning-parser-plugin "$MODEL/ultra_v3_reasoning_parser.py"
    --reasoning-parser ultra_v3
    --enable-chunked-prefill
    --enable-prefix-caching
    --max-model-len 262144
    --kv-cache-dtype fp8
    --no-disable-hybrid-kv-cache-manager
    --async-scheduling
    --block-size 128
    --mamba-cache-mode align
    --mamba-cache-dtype auto
    --mamba-ssm-cache-dtype float32
    --no-enable-mamba-cache-stochastic-rounding
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 96}'
    --enable-expert-parallel
    --skip-mm-profiling
    --data-parallel-size 1
    --api-server-count 1
)
VLLM_PREFILL_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
    --max-num-batched-tokens 135680
    --max-num-seqs 1024
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)
VLLM_DECODE_ARGS=(
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --max-num-batched-tokens 33920
    --max-num-seqs 1024
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)
