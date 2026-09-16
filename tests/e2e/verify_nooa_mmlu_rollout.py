#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rollouts.read_text().splitlines() if line.strip()]

    assert len(rows) == 1, f"expected one rollout, found {len(rows)}"
    row = rows[0]
    assert row["agent_ref"] == {"name": "mmlu_nooa_agent"}
    assert row["expected_answer"] == "D"
    assert row["extracted_answer"] == "D"
    assert row["reward"] == 1.0
    assert row["ng_agent_observations"]["records"][0]["status"] == "completed"
    assert row["ng_trajectory"]["turns"]
    assert row["ng_trajectory"]["model_calls"][0]["response_metadata"]["response_id"] == "resp-nooa-mmlu-d"


if __name__ == "__main__":
    main()
