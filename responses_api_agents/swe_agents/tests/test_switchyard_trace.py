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
import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseOutputMessageForTraining,
)
from nemo_gym.responses_converter import ResponsesConverter
from responses_api_agents.swe_agents.switchyard_trace import (
    SwitchyardTraceError,
    map_switchyard_tools,
    reconstruct_switchyard_rollout,
)


SESSION_ID = "abc123"

TOOL_SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}}
TOOLS = [{"id": "str_replace_editor", "description": "Edit files", "inputSchema": {"jsonSchema": TOOL_SCHEMA}}]
TOOL_CALL = {"id": "call_1", "type": "function", "function": {"name": "str_replace_editor", "arguments": "{}"}}

TRIPLE_0 = {"prompt_token_ids": [1, 2, 3], "generation_token_ids": [4, 5], "generation_log_probs": [-0.1, -0.2]}
TRIPLE_1 = {"prompt_token_ids": [1, 2, 3, 4, 5, 6], "generation_token_ids": [7], "generation_log_probs": [-0.3]}


def _record(i: int, messages: list, triple: dict, **overrides) -> dict:
    record = {
        "schema_version": 1,
        "uuid": f"uuid-{i}",
        "session_id": SESSION_ID,
        "captured_at": f"2026-07-14T00:00:0{i}+00:00",
        "request_id": f"req-{i}",
        "model": "Qwen/Qwen3-0.6B",
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "finish_reason": "stop",
        "is_valid": True,
        **triple,
    }
    record.update(overrides)
    return record


def _two_call_records() -> list:
    """A two-call tool trajectory, with the same logical messages appearing in
    both OpenHands' list-of-text-parts shape and Switchyard's captured string shape."""
    first = _record(
        0,
        [
            {"role": "system", "content": "You are a SWE agent."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "tool_calls": [TOOL_CALL]},
        ],
        TRIPLE_0,
    )
    second = _record(
        1,
        [
            {"role": "system", "content": [{"type": "text", "text": "You are a SWE agent."}]},
            {"role": "user", "content": [{"type": "text", "text": "Fix the bug."}]},
            {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL | {"index": 0}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "edited ok"},
            {"role": "assistant", "content": "All fixed."},
        ],
        TRIPLE_1,
    )
    return [first, second]


def _envelope(records: list, **overrides) -> dict:
    envelope = {"schema_version": 1, "session_id": SESSION_ID, "completions": records}
    envelope.update(overrides)
    return envelope


def _reconstruct(envelope: dict):
    converter = ResponsesConverter(return_token_id_information=True)
    return reconstruct_switchyard_rollout(envelope, SESSION_ID, converter)


class TestReconstruction:
    def test_two_call_tool_trajectory(self) -> None:
        trace = _reconstruct(_envelope(_two_call_records()))

        assert trace.record_uuids == ["uuid-0", "uuid-1"]
        assert trace.model == "Qwen/Qwen3-0.6B"

        assert [type(item) for item in trace.input_items] == [NeMoGymEasyInputMessage, NeMoGymEasyInputMessage]
        assert trace.input_items[0].role == "system"
        assert trace.input_items[1].content == "Fix the bug."

        assert [type(item) for item in trace.output_items] == [
            NeMoGymResponseFunctionToolCallForTraining,
            NeMoGymFunctionCallOutput,
            NeMoGymResponseOutputMessageForTraining,
        ]
        tool_call, tool_output, final_message = trace.output_items
        assert tool_call.call_id == "call_1"
        assert tool_call.prompt_token_ids == TRIPLE_0["prompt_token_ids"]
        assert tool_call.generation_token_ids == TRIPLE_0["generation_token_ids"]
        assert tool_call.generation_log_probs == TRIPLE_0["generation_log_probs"]
        assert tool_output.call_id == "call_1"
        assert tool_output.output == "edited ok"
        assert final_message.content[0].text == "All fixed."
        assert final_message.prompt_token_ids == TRIPLE_1["prompt_token_ids"]
        assert final_message.generation_token_ids == TRIPLE_1["generation_token_ids"]
        assert final_message.generation_log_probs == TRIPLE_1["generation_log_probs"]

        assert len(trace.tools) == 1
        assert trace.tools[0].name == "str_replace_editor"
        assert trace.tools[0].description == "Edit files"
        assert trace.tools[0].parameters == TOOL_SCHEMA
        assert trace.tools[0].type == "function"

    def test_single_record_session(self) -> None:
        records = [_two_call_records()[0]]
        trace = _reconstruct(_envelope(records))
        assert trace.record_uuids == ["uuid-0"]
        assert len(trace.input_items) == 2
        assert len(trace.output_items) == 1

    def test_divergent_history_rejected(self) -> None:
        records = _two_call_records()
        records[1]["messages"][1]["content"] = [{"type": "text", "text": "Fix a DIFFERENT bug."}]
        with pytest.raises(SwitchyardTraceError, match="does not extend"):
            _reconstruct(_envelope(records))

    def test_missing_environment_suffix_rejected(self) -> None:
        records = _two_call_records()
        # Second prompt replays the first prompt + assistant but adds no tool result.
        records[1]["messages"] = records[1]["messages"][:3] + [{"role": "assistant", "content": "All fixed."}]
        with pytest.raises(SwitchyardTraceError, match="adds no environment messages"):
            _reconstruct(_envelope(records))

    def test_assistant_in_suffix_rejected(self) -> None:
        records = _two_call_records()
        records[1]["messages"].insert(4, {"role": "assistant", "content": "uncaptured turn"})
        records[1]["messages"].insert(5, {"role": "user", "content": "go on"})
        with pytest.raises(SwitchyardTraceError, match="non-environment roles"):
            _reconstruct(_envelope(records))

    def test_reordered_records_rejected(self) -> None:
        records = list(reversed(_two_call_records()))
        with pytest.raises(SwitchyardTraceError, match="does not extend"):
            _reconstruct(_envelope(records))


class TestValidation:
    @pytest.mark.parametrize(
        "mutate, match",
        [
            (lambda e: e.update(schema_version=2), "schema_version"),
            (lambda e: e.update(session_id="other"), "session_id"),
            (lambda e: e.update(completions=[]), "no completions"),
            (lambda e: e.update(completions="nope"), "no completions"),
        ],
    )
    def test_bad_envelope(self, mutate, match) -> None:
        envelope = _envelope(_two_call_records())
        mutate(envelope)
        with pytest.raises(SwitchyardTraceError, match=match):
            _reconstruct(envelope)

    def test_non_object_envelope(self) -> None:
        with pytest.raises(SwitchyardTraceError, match="not an object"):
            _reconstruct([])

    @pytest.mark.parametrize(
        "field, value, match",
        [
            ("schema_version", 2, "schema_version"),
            ("session_id", "other", "session_id"),
            ("uuid", "", "no uuid"),
            ("uuid", "uuid-0", "duplicates uuid"),
            ("is_valid", False, "is_valid"),
            ("request_id", "", "empty request_id"),
            ("model", "", "empty model"),
            ("prompt_token_ids", [], "prompt_token_ids"),
            ("prompt_token_ids", [1, "2"], "prompt_token_ids"),
            ("generation_token_ids", None, "generation_token_ids"),
            ("generation_log_probs", [-0.1, float("nan")], "finite"),
            ("generation_log_probs", [-0.1, -0.2], "different lengths"),
            ("messages", [], "no messages"),
        ],
    )
    def test_bad_record_fields(self, field, value, match) -> None:
        records = _two_call_records()
        records[1][field] = value
        with pytest.raises(SwitchyardTraceError, match=match):
            _reconstruct(_envelope(records))

    def test_messages_must_end_with_assistant(self) -> None:
        records = _two_call_records()
        records[0]["messages"].append({"role": "user", "content": "trailing"})
        with pytest.raises(SwitchyardTraceError, match="end with an assistant"):
            _reconstruct(_envelope(records))

    @pytest.mark.parametrize(
        "field, value",
        [("model", "other-model"), ("tools", []), ("tool_choice", "required")],
    )
    def test_session_consistency(self, field, value) -> None:
        records = _two_call_records()
        records[1][field] = value
        with pytest.raises(SwitchyardTraceError):
            _reconstruct(_envelope(records))

    @pytest.mark.parametrize(
        "message, match",
        [
            ("nope", "non-object message"),
            ({"role": "narrator", "content": "hm"}, "unsupported role"),
            ({"role": "user", "content": {"nested": True}}, "unsupported content"),
            ({"role": "user", "content": [{"type": "image_url", "image_url": {}}]}, "content part"),
            ({"role": "tool", "content": "result"}, "without tool_call_id"),
        ],
    )
    def test_bad_messages(self, message, match) -> None:
        records = _two_call_records()
        records[1]["messages"].insert(3, message)
        with pytest.raises(SwitchyardTraceError, match=match):
            _reconstruct(_envelope(records))

    def test_malformed_tool_call(self) -> None:
        records = _two_call_records()
        records[0]["messages"][-1]["tool_calls"] = [{"function": {"name": "x", "arguments": "{}"}}]
        with pytest.raises(SwitchyardTraceError, match="malformed tool call"):
            _reconstruct(_envelope(records))


class TestToolMapping:
    def test_tool_without_schema(self) -> None:
        tools = map_switchyard_tools([{"id": "noop", "description": ""}])
        assert tools[0].name == "noop"
        assert tools[0].parameters is None
        assert tools[0].description is None

    def test_tool_without_id_rejected(self) -> None:
        with pytest.raises(SwitchyardTraceError, match="no usable id"):
            map_switchyard_tools([{"description": "no id"}])

    def test_non_list_rejected(self) -> None:
        with pytest.raises(SwitchyardTraceError, match="not a list"):
            map_switchyard_tools({"id": "x"})
