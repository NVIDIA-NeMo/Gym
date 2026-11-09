# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Send the final answer to the existing host-side Gym VIALS verifier."""

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def grade(*, answer_path: Path, input_path: Path, output_dir: Path, verifier_url: str) -> dict:
    """Write Harbor rewards; transport/masked verifier failures remain trial errors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave an earlier successful reward behind.
    (output_dir / "reward.json").unlink(missing_ok=True)
    (output_dir / "info.json").unlink(missing_ok=True)
    text = answer_path.read_text() if answer_path.is_file() else ""
    if not text.strip():
        result = {"reward": 0.0, "correct": False, "judge_reasoning": "No answer file or empty answer."}
    else:
        payload = json.loads(input_path.read_text())
        payload["response"] = {
            "id": "harbor-response",
            "created_at": 0,
            "model": "harbor-policy",
            "object": "response",
            "output": [
                {
                    "id": "harbor-answer",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
        }
        request = Request(
            verifier_url,
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # Synchronous, dependency-free bridge. All model calls stay in Gym's verifier.
        with urlopen(request, timeout=240) as response:
            result = json.load(response)
        if result.get("mask_sample"):
            raise RuntimeError(f"Gym verifier failed: {result.get('failure_reason', 'masked sample')}")
        if result.get("reward") not in (0.0, 1.0):
            raise ValueError("Gym verifier did not return a binary reward")
    (output_dir / "info.json").write_text(json.dumps(result) + "\n")
    (output_dir / "reward.json").write_text(json.dumps({"reward": float(result["reward"])}) + "\n")
    return result


if __name__ == "__main__":
    grade(
        answer_path=Path("/task/answer.txt"),
        input_path=Path("/tests/verifier_input.json"),
        output_dir=Path("/logs/verifier"),
        verifier_url=os.environ["VIALS_VERIFIER_URL"],
    )
