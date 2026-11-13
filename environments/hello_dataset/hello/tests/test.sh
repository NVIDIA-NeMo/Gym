#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

mkdir -p /logs/verifier
# Accept the line with or without a final newline.
if printf 'Hello from NeMo Gym!' | cmp -s - /workspace/hello-gym.txt || printf 'Hello from NeMo Gym!\n' | cmp -s - /workspace/hello-gym.txt; then
  printf '1\n' > /logs/verifier/reward.txt
else
  printf '0\n' > /logs/verifier/reward.txt
fi
