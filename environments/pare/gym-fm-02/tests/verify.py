#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Coordinate the review of a GYM-FM-02 attempt and save Harbor's reward.

Load the reference data, gather saved Gym results and public execution
records, and ask the LLM judge whether the required workflow happened.
Save the report and write reward 1 or 0 only after a completed assessment.
Judge or verifier errors leave no reward file and exit unsuccessfully.
"""

import json
import os
from pathlib import Path

from hybrid import review, write_json
from read_execution import EvidenceFiles


def load_reference(hashes_path=Path("/opt/gym-fixture/bundled-output-hashes.json")):
    # Load the Carlos request and bundled sample fingerprints, then add GYM_AGENT
    # and GYM_MODEL so the judge knows which Gym agent should have run.
    reference = json.loads(Path(__file__).with_name("reference.json").read_text(encoding="utf-8"))
    baseline = json.loads(Path(hashes_path).read_text(encoding="utf-8"))
    hashes = baseline.get("bundled_output_hashes") if isinstance(baseline, dict) else None
    if not isinstance(hashes, list) or not hashes:
        raise ValueError("bundled_output_hashes_invalid")
    for value in hashes:
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("bundled_output_hashes_invalid")
    reference["bundled_output_hashes"] = hashes
    # Read the request Harbor supplied, rather than infer the requested harness
    # from an agent-written configuration or the saved response's model label.
    harness = os.environ["GYM_AGENT"]
    model = os.environ["GYM_MODEL"]
    if harness not in reference["gym_agent_harnesses"]:
        raise ValueError("gym_agent_harness_invalid")
    if not model.strip():
        raise ValueError("gym_agent_model_invalid")
    reference["gym_agent"] = {"harness": harness, "model": model}
    return reference


def main():
    # Save the review report and publish a reward only for a completed assessment,
    # so Harbor can distinguish a task failure from a verifier error.
    logs = Path(os.environ.get("VERIFIER_OUTPUT_DIR", "/logs/verifier"))
    logs.mkdir(parents=True, exist_ok=True)
    # Clear both reward formats on a direct rerun; Harbor accepts either one.
    for name in ("reward.txt", "reward.json"):
        (logs / name).unlink(missing_ok=True)
    result = {"status": "pending", "pare_success": False, "hybrid_reward": None, "gym_reward": None}
    write_json(logs / "report.json", result)
    try:
        reference = load_reference()
        evidence = EvidenceFiles(reference)
        result = review(evidence, report_dir=logs / "hybrid", artifacts=Path("/logs/artifacts/gym-fm-02-hybrid"))
        if result.get("status") == "completed":
            reward = result.get("hybrid_reward")
            if (
                type(reward) is not int
                or reward not in (0, 1)
                or type(result.get("pare_success")) is not bool
                or reward != int(result["pare_success"])
            ):
                raise ValueError("completed_judgment_invalid")
    except Exception as error:
        result.update(status="verifier_error", pare_success=False, hybrid_reward=None, error_type=type(error).__name__)
    write_json(logs / "report.json", result)
    print(json.dumps(result, indent=2))
    if result.get("status") != "completed":
        return 1
    # Publish the score last, using a temporary file so a failed write cannot
    # leave a partial reward for Harbor to read.
    temporary_reward = logs / "reward.tmp"
    temporary_reward.write_text(f"{result['hybrid_reward']}\n")
    temporary_reward.replace(logs / "reward.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
