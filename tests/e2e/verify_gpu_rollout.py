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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--expected-answer", required=True)
    args = parser.parse_args()

    with args.rollouts.open(encoding="utf-8") as rollouts_file:
        rollouts = [json.loads(line) for line in rollouts_file if line.strip()]

    assert len(rollouts) == 1, f"expected one rollout, found {len(rollouts)}"
    rollout = rollouts[0]
    response = rollout["response"]
    assert response["status"] == "completed"
    assert response["error"] is None
    assert response["model"] == args.expected_model
    assert response["usage"]["input_tokens"] > 0
    assert response["usage"]["output_tokens"] > 0

    messages = [item for item in response["output"] if item["type"] == "message"]
    assert len(messages) == 1
    output_text = [content["text"] for content in messages[0]["content"] if content["type"] == "output_text"]
    assert output_text and output_text[0].strip()
    assert rollout["reward"] == 1.0
    assert rollout["expected_answer"] == args.expected_answer
    assert isinstance(rollout["extracted_answer"], str)
    assert rollout["extracted_answer"].strip()
    assert rollout["agent_ref"] == {"name": "string_match_simple_agent"}


if __name__ == "__main__":
    main()
