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

import json
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"
SAMPLING_FIELDS = {
    "max_output_tokens",
    "temperature",
    "top_k",
    "top_p",
}
TRAJECTORY_RESULT_FIELDS = {
    "agent_verification_result",
    "continuation_start_index",
    "environment_verification_result",
    "generation_invalid_reason",
    "messages",
    "prefill_message_count",
    "terminal_error",
    "terminal_state",
    "user_verification_result",
}
SOURCE_NAME_BY_PROFILE = {
    "general": "conversational_tool_use_general",
    "proactive": "conversational_tool_use_proactive",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_example_inputs_have_profiles_and_delegate_sampling_to_model_servers() -> None:
    for filename in ("example.jsonl", "example_parallel_tool_calls.jsonl"):
        rows = _read_jsonl(DATA_DIR / filename)

        assert len(rows) == 5
        assert {row["profile"] for row in rows} <= {"general", "proactive"}
        for row in rows:
            expected_source_name = SOURCE_NAME_BY_PROFILE[row["profile"]]
            assert row["metadata"]["source_name"] == expected_source_name
            assert row["source_artifacts"]["source_name"] == expected_source_name
            assert row["id"].startswith(f"{row['metadata']['dataset_name']}_")
            assert not SAMPLING_FIELDS & row["responses_create_params"].keys()


def test_example_rollouts_match_current_conversation_result_contract() -> None:
    inputs = _read_jsonl(DATA_DIR / "example.jsonl")
    rollouts = _read_jsonl(DATA_DIR / "example_rollouts.jsonl")

    assert len(rollouts) == 5
    assert {row["id"] for row in rollouts} == {row["id"] for row in inputs}
    for row in rollouts:
        result = row["result"]
        trajectory = result["trajectory"]

        assert result["profile"] == row["profile"]
        assert result["source_artifacts"] == row["source_artifacts"]
        assert row["source_artifacts"]["source_name"] == SOURCE_NAME_BY_PROFILE[row["profile"]]
        assert row["id"].startswith(f"{row['metadata']['dataset_name']}_")
        assert TRAJECTORY_RESULT_FIELDS <= trajectory.keys()
        assert trajectory["terminal_state"] == row["terminal_state"]
        assert 0 <= trajectory["prefill_message_count"] <= len(trajectory["messages"])
        assert 0 <= trajectory["continuation_start_index"] <= len(trajectory["messages"])
        assert not [
            message
            for message in trajectory["messages"]
            if message["source"] == "agent" and message["type"] == "text" and not (message.get("text") or "").strip()
        ]
