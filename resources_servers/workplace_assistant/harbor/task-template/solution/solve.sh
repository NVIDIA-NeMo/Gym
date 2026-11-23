#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Reference solution: the row's ground-truth calls, sent in order to the episode's tool server.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
python3 "$here/replay.py" http://tools:8000/mcp "$here/calls.json"
