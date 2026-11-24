#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs in a fresh verifier sandbox (task.toml selects separate mode). The tool server's journal arrives as an
# artifact copied out of the `tools` sidecar; the ground truth arrives with tests/. Exit 75 marks a verification
# that could not run (no journal, no dependencies), which Gym masks instead of scoring 0.
set -u
mkdir -p /logs/verifier
tests=$(cd "$(dirname "$0")" && pwd)
journal=/var/lib/wa-journal/calls.jsonl
if [ ! -f "$journal" ]; then
  echo "no tool-call journal at $journal"
  exit 75
fi
if ! uv pip install --system --quiet --require-hashes --no-build -r "$tests/requirements.txt"; then
  echo "could not install the grader's dependencies"
  exit 75
fi
python3 "$tests/grade.py" "$journal" "$tests/ground_truth.json" /logs/verifier/reward.json
