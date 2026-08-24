# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def verify_rollout(rollout: dict[str, Any]) -> None:
    response = rollout["response"]
    assert response.get("error") is None, response.get("error")
    assert response.get("incomplete_details") is None, response.get("incomplete_details")
    assert response["status"] == "completed", response["status"]

    usage = response["usage"]
    assert usage["input_tokens"] > 0, usage
    assert usage["output_tokens"] > 0, usage

    output = response["output"]
    function_calls = [item for item in output if item.get("type") == "function_call"]
    assert len(function_calls) == 1, function_calls
    function_call = function_calls[0]
    assert function_call["name"] == "get_weather", function_call

    arguments = json.loads(function_call["arguments"])
    assert isinstance(arguments.get("city"), str) and arguments["city"].strip(), arguments

    function_outputs = [item for item in output if item.get("type") == "function_call_output"]
    matching_outputs = [item for item in function_outputs if item.get("call_id") == function_call["call_id"]]
    assert len(matching_outputs) == 1, function_outputs
    tool_output = json.loads(matching_outputs[0]["output"])
    assert tool_output.get("weather_description"), tool_output

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
