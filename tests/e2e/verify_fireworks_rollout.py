# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def verify_rollout(rollout: dict[str, Any]) -> None:
    response = rollout["response"]
    assert response["status"] == "completed", response["status"]
    assert response.get("error") is None, response.get("error")
    assert response.get("incomplete_details") is None, response.get("incomplete_details")

    usage = response["usage"]
    assert usage["input_tokens"] > 0, usage
    assert usage["output_tokens"] > 0, usage

    output = response["output"]
    messages = [item for item in output if item.get("type") == "message" and item.get("role") == "assistant"]
    assert messages, output
    output_text = [
        part.get("text", "")
        for message in messages
        for part in message.get("content", [])
        if part.get("type") == "output_text"
    ]
    assert any(text.strip() for text in output_text), messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    args = parser.parse_args()

    rollouts = read_jsonl(args.rollouts)
    assert len(rollouts) == 1, f"expected one rollout, found {len(rollouts)}"
    verify_rollout(rollouts[0])


if __name__ == "__main__":
    main()
