#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Harbor invokes this without arguments; paths can be supplied for the same offline check.
answer_file="${1:-/app/answer.txt}"
reward_dir="${2:-/logs/verifier}"
mkdir -p "$reward_dir"
if [[ -f "$answer_file" ]] && cmp -s "$answer_file" <(printf '42\n'); then
    printf '1.0\n' > "$reward_dir/reward.txt"
else
    printf '0.0\n' > "$reward_dir/reward.txt"
fi
