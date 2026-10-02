#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Nemotron 3.5 Super VL/omni checkpoints (NemotronH_Omni_Reasoning_V3) with image input, for the
# visual_agent benchmark where the policy reads screenshots. Same engine settings as
# nemotron_3.5_super.sh plus a per-prompt image cap (it bounds how many screenshots one OpenCode
# session can carry in its context). Meant for VLLM_MODE=aggregated: one TP4 replica per node
# behind vllm-router, so image preprocessing and KV stay on one engine (no P/D transfer of
# multimodal prompts). Chunks are sized like the decode tier so prefills do not stall decoding.

source "$(dirname "${BASH_SOURCE[0]}")/nemotron_3.5_super.sh"

VLLM_COMMON_ARGS+=(
    --limit-mm-per-prompt '{"image": 96, "video": 0}'
)
VLLM_AGGREGATED_ARGS=(
    --max-num-batched-tokens 33920
    --max-num-seqs 512
    --data-parallel-size-local 1
    --tensor-parallel-size 4
)
