#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

mkdir -p /logs/verifier
cp /app/eval_answer.json /logs/verifier/eval_answer.json

python3 /tests/reference_grader.py \
    /tests/eval_config.json \
    /logs/verifier/eval_answer.json \
    > /logs/verifier/grading_result.json

python3 - <<'PY'
import json
from pathlib import Path

output_dir = Path("/logs/verifier")
result = json.loads((output_dir / "grading_result.json").read_text())
(output_dir / "reward.json").write_text(json.dumps({"reward": 1.0 if result["passed"] else 0.0}) + "\n")
PY
