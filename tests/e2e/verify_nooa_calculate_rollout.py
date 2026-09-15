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

    assert len(rows) == 2, f"expected two rollouts, found {len(rows)}"
    assert {row["expected_result"] for row in rows} == {7, 56}
    for row in rows:
        expected = row["expected_result"]
        assert row["reward"] == 1.0
        assert row["actual_result"] == expected
        assert row["output_correct"] is True
        observations = row["ng_agent_observations"]
        invocations = [record for record in observations["records"] if record["kind"] == "agent_invocation"]
        assert invocations and invocations[0]["status"] == "completed"
        assert invocations[0]["model_calls"][0]["response_id"] == f"resp-nooa-{expected}"
        assert row["agent_ref"] == {"name": "nooa_calculate_capability"}


if __name__ == "__main__":
    main()
