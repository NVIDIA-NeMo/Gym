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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--provider-events", type=Path, required=True)
    args = parser.parse_args()

    rollouts = read_jsonl(args.rollouts)
    assert len(rollouts) == 1, f"expected one rollout, found {len(rollouts)}"

    rollout = rollouts[0]
    assert rollout["reward"] == 1.0
    output = rollout["response"]["output"]
    output_types = [item["type"] for item in output]
    assert output_types == [
        "function_call",
        "function_call_output",
        "message",
    ], output_types

    function_call = output[0]
    assert function_call["name"] == "get_weather"
    assert json.loads(function_call["arguments"]) == {"city": "San Francisco"}

    tool_output = json.loads(output[1]["output"])
    assert tool_output == {
        "city": "San Francisco",
        "weather_description": "The weather in San Francisco is cold.",
    }

    final_text = output[2]["content"][0]["text"]
    assert final_text == "The weather in San Francisco is cold."

    provider_events = read_jsonl(args.provider_events)
    assert len(provider_events) == 2, f"expected two model requests, found {len(provider_events)}"
    tool_messages = [message for message in provider_events[1]["messages"] if message.get("role") == "tool"]
    assert len(tool_messages) == 1
    assert "weather_description" in tool_messages[0]["content"]


if __name__ == "__main__":
    main()
