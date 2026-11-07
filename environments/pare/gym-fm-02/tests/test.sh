#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# Clear rewards from a previous invocation before Python starts, so an import
# or startup error cannot leave an old score for Harbor to read.
mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt /logs/verifier/reward.json
# Ask GPT-6.1 Terra to review the saved result and public execution records.
exec /opt/gym/.venv/bin/python /tests/verify.py
