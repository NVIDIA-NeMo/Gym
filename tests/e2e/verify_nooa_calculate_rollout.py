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
        trajectory = row["ng_trajectory"]
        assert len(trajectory["turns"]) == len(trajectory["model_calls"]) == 1
        assert trajectory["turns"][0]["invocation_id"] == invocations[0]["invocation_id"]
        captured = trajectory["model_calls"][0]
        assert captured["response_metadata"]["response_id"] == f"resp-nooa-{expected}"
        assert captured["request"] and captured["response"]
        usage = row["response"]["usage"]
        assert usage == captured["response"]["usage"]
        assert usage["input_tokens"] > 0
        assert usage["output_tokens"] > 0
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
        assert {gap["code"] for gap in trajectory["gaps"]} == {"non_trainable_terminal_output"}
        assert trajectory["tool_calls"]
        for tool in trajectory["tool_calls"]:
            owner = next(inv for inv in trajectory["invocations"] if inv["invocation_id"] == tool["invocation_id"])
            outputs = [
                item
                for item in owner["conversation"]
                if item["type"] == "function_call_output" and item["call_id"] == tool["tool_call_id"]
            ]
            assert len(outputs) == 1
            assert outputs[0]["output"] == tool["output"]
        assert all(item.get("id") != "nooa_fallback" for item in invocations[0]["conversation"])


if __name__ == "__main__":
    main()
