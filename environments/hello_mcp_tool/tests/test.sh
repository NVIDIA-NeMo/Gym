#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -u

# environment/tools/server.py appends one JSON line per get_greeting call.
calls=/logs/agent/hello-tools-calls.jsonl
mkdir -p /logs/verifier
echo "get_greeting calls:"
cat "$calls" 2>/dev/null || echo "none"

reward=0
# The agent must have called the tool for "NeMo Gym" and saved its greeting, with or without a final newline.
if grep -qxF '{"name": "NeMo Gym"}' "$calls" 2>/dev/null; then
  if printf 'Hello, NeMo Gym!' | cmp -s - /workspace/tool-greeting.txt ||
    printf 'Hello, NeMo Gym!\n' | cmp -s - /workspace/tool-greeting.txt; then
    reward=1
  fi
fi
printf '%s\n' "$reward" > /logs/verifier/reward.txt
