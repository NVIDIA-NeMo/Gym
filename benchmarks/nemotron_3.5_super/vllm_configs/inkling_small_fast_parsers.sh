#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Opt-in complete-response optimizations for vLLM 0.29.0.
# Select inkling_small.sh instead to use stock parsers with the same serving settings.
inkling_config_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
inkling_plugin_dir=$(cd "$inkling_config_dir/../vllm_plugins" && pwd)
source "$inkling_config_dir/inkling_small.sh"

for inkling_arg_index in "${!VLLM_COMMON_ARGS[@]}"; do
    case "${VLLM_COMMON_ARGS[$inkling_arg_index]}" in
        --tool-call-parser) VLLM_COMMON_ARGS[$((inkling_arg_index + 1))]=inkling_complete_fast ;;
        --reasoning-parser) VLLM_COMMON_ARGS[$((inkling_arg_index + 1))]=inkling_count_fast ;;
    esac
done
VLLM_COMMON_ARGS+=(
    --tool-parser-plugin "$inkling_plugin_dir/inkling_complete_tool_parser.py"
    --reasoning-parser-plugin "$inkling_plugin_dir/inkling_complete_reasoning_parser.py"
)
unset inkling_config_dir inkling_plugin_dir inkling_arg_index
