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
        assert row["ng_perf"]["token_observability_coverage"] == 1.0
        assert row["response"]["usage"]["input_tokens"] == row["ng_perf"]["prompt_tokens"]
        assert row["response"]["usage"]["output_tokens"] == row["ng_perf"]["completion_tokens"]
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
    health = json.loads(args.rollouts.with_name("quality_summary.json").read_text())
    assert health["run"]["verdicts"] == {"healthy": 2, "unhealthy": 0, "unobserved": 0}


if __name__ == "__main__":
    main()
