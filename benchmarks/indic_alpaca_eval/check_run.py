# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate a native rollout shard against its exact input tasks and judge settings."""

import argparse
import json
from pathlib import Path

from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.alpaca_pointwise_judge.app import PointwiseJudgment, final_answer_only


def check_run(inputs: Path, rollouts: Path, *, smoke: bool = False, max_output_tokens: int | None = None) -> dict:
    """Check coverage and scoring; an explicit budget overrides prepared defaults."""
    source = [json.loads(line) for line in inputs.read_text().splitlines() if line.strip()]
    rows = [json.loads(line) for line in rollouts.read_text().splitlines() if line.strip()]
    if not source:
        raise ValueError("Input tasks must not be empty")
    expected = {row["id"] for row in source}
    if len(expected) != len(source) or len(rows) != len(expected) or {row["id"] for row in rows} != expected:
        raise ValueError("Missing, unexpected, or duplicate rollout task IDs")
    tasks = {row["id"]: row for row in source}
    empty = 0
    for row in rows:
        if row.get("mask_sample") or row.get("_ng_failure_class"):
            raise ValueError(f"Unmeasured task: {row['id']}")
        if row["verifier_metadata"] != tasks[row["id"]]["verifier_metadata"]:
            raise ValueError(f"Task metadata changed: {row['id']}")
        judgment = PointwiseJudgment.model_validate(row["judgment"])
        if row["reward"] != (judgment.overall_score - 1) / 4:
            raise ValueError(f"Incorrect normalized reward: {row['id']}")
        response = NeMoGymResponse.model_validate(row["response"])
        answer = response.output_text
        if not answer:
            answer = "".join(
                content.refusal
                for item in response.output
                if item.type == "message"
                for content in item.content
                if content.type == "refusal"
            )
        if row["candidate_answer"] != final_answer_only(answer):
            raise ValueError(f"Final-answer extraction mismatch: {row['id']}")
        empty += not row["candidate_answer"].strip()
        expected_budget = max_output_tokens
        if expected_budget is None:
            expected_budget = tasks[row["id"]]["responses_create_params"].get("max_output_tokens", 120000)
        if row["responses_create_params"]["max_output_tokens"] != expected_budget:
            raise ValueError(f"Candidate output budget changed: {row['id']}")
    if smoke and empty == len(rows):
        raise ValueError("Smoke run must demonstrate at least one nonempty final answer")
    return {
        "complete": True,
        "expected": len(source),
        "judged": len(rows),
        "empty_final_answers": empty,
        "judge_models": sorted({row["judge_completion"]["model"] for row in rows}),
        "languages": sorted({row["verifier_metadata"]["language_code"] for row in rows}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-output-tokens", type=int, help="Expected run-level override of prepared token budgets")
    args = parser.parse_args()
    print(
        json.dumps(
            check_run(args.inputs, args.rollouts, smoke=args.smoke, max_output_tokens=args.max_output_tokens), indent=2
        )
    )


if __name__ == "__main__":
    main()
