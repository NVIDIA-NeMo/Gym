# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invoke the unchanged upstream answer judge outside the agent process."""

import json
import sys
from pathlib import Path

from responses_api_agents.agentvista_agent.runtime.data import inventory
from responses_api_agents.agentvista_agent.runtime.scoring import score
from responses_api_agents.agentvista_agent.runtime.source import verify_source


def run(config: dict) -> dict:
    result = dict(config["result"])
    try:
        verify_source(Path(config["source_root"]))
        sys.path.insert(0, str(Path(config["source_root"]).resolve(strict=True)))
        metadata = config["metadata"]
        task = inventory(config["dataset"], config["image_folder"], metadata["task_id"])[metadata["task_id"]]
        if task["source_sha256"] != metadata["source_sha256"]:
            raise ValueError("Prepared task/image hashes changed before verification")
        verdict = score(task["question"], task["record"].get("solution"), result["judge_responses"])
        result.pop("failure_kind", None)
        result.update(
            verify=verdict,
            reward=verdict["reward"],
            mask_sample=False,
            task_success=verdict["reward"] == 1,
            reward_kind="llm_judged",
            deterministically_verified=False,
        )
        if result.get("original_result"):
            result["original_result"].update(accuracy_score=verdict["reward"], trajectory_score=verdict["reward"],
                                             trajectory_analysis=verdict["analysis"])
        if result.get("execution_health"):
            result["execution_health"]["task_success"] = result["task_success"]
    except Exception as error:
        result.update(mask_sample=True, failure_kind="agentvista:verifier", error=f"{type(error).__name__}: {error}")
    result.pop("verification_pending", None)
    return result


if __name__ == "__main__":
    path = Path(sys.argv[1])
    temporary = path.parent / "result.json.tmp"
    temporary.write_text(json.dumps(run(json.loads(path.read_text())), indent=2))
    temporary.replace(path.parent / "result.json")
