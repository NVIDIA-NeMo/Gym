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

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.trajectory import TRAJECTORY_SCHEMA_VERSION, validate_trajectory
from responses_api_agents.claude_code_agent.trajectory import build_trajectory, decode_jsonl


SESSION = "sess-1"


def _user_record(text: str, ts: str = "2026-07-09T00:00:00.000Z", **kwargs) -> dict:
    return {
        "type": "user",
        "uuid": "u-user",
        "timestamp": ts,
        "sessionId": SESSION,
        "message": {"role": "user", "content": text},
        **kwargs,
    }


def _assistant_record(
    uuid: str,
    message_id: str,
    blocks: list,
    usage: dict | None = None,
    ts: str = "2026-07-09T00:00:01.000Z",
    request_id: str = "req-1",
    stop_reason: str | None = None,
    **kwargs,
) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "requestId": request_id,
        "timestamp": ts,
        "sessionId": SESSION,
        "message": {
            "id": message_id,
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": stop_reason,
            "content": blocks,
            "usage": usage or {},
        },
        **kwargs,
    }


def _tool_result_record(
    call_id: str,
    content: str,
    ts: str,
    source_uuid: str = "a1",
    is_error: bool = False,
    tool_use_result=None,
) -> dict:
    return {
        "type": "user",
        "uuid": f"u-{call_id}",
        "timestamp": ts,
        "sessionId": SESSION,
        "sourceToolAssistantUUID": source_uuid,
        "toolUseResult": tool_use_result,
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error}],
        },
    }


USAGE = {"input_tokens": 100, "output_tokens": 20, "cache_read_input_tokens": 60, "cache_creation_input_tokens": 10}


class TestTranscriptSource:
    def _transcript(self) -> list[dict]:
        return [
            {"type": "queue-operation", "operation": "enqueue"},  # non-message noise is skipped
            _user_record("fix the bug"),
            # One API message written as two records (text block, then two parallel tool_use
            # blocks) sharing the same message id and usage.
            _assistant_record("a1", "msg_1", [{"type": "text", "text": "looking"}], usage=USAGE),
            _assistant_record(
                "a1",
                "msg_1",
                [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                    {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "/x"}},
                ],
                usage=USAGE,
                stop_reason="tool_use",
            ),
            _tool_result_record(
                "t1",
                "file.txt",
                "2026-07-09T00:00:01.500Z",
                tool_use_result={"interrupted": False, "stdout": "file.txt", "big": "x" * 5000, "nested": {"a": 1}},
            ),
            _tool_result_record("t2", "boom", "2026-07-09T00:00:03.000Z", is_error=True),
            _assistant_record(
                "a2",
                "msg_2",
                [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "fixed"}],
                usage={"input_tokens": 50, "output_tokens": 5},
                ts="2026-07-09T00:00:04.000Z",
                request_id="req-2",
                stop_reason="end_turn",
            ),
        ]

    def test_steps_and_native_items(self) -> None:
        trajectory = build_trajectory([], self._transcript())
        assert trajectory.source == "transcript"
        assert trajectory.session_id == SESSION
        assert trajectory.model == "claude-sonnet-4-6"
        assert [s.type for s in trajectory.steps] == ["user_message", "agent_step", "agent_step"]
        user, step1, step2 = trajectory.steps
        assert isinstance(user.items[0], NeMoGymEasyInputMessage)
        assert user.items[0].content == "fix the bug"
        # one agent step per API message: text + 2 tool calls + 2 tool outputs, all native items
        assert [type(i) for i in step1.items] == [
            NeMoGymResponseOutputMessage,
            NeMoGymResponseFunctionToolCall,
            NeMoGymResponseFunctionToolCall,
            NeMoGymFunctionCallOutput,
            NeMoGymFunctionCallOutput,
        ]
        assert step1.items[0].content[0].text == "looking"
        assert step1.items[1].name == "Bash"
        assert (step1.agent_step_no, step2.agent_step_no) == (1, 2)
        assert step1.stop_reason == "tool_use"
        assert isinstance(step2.items[0], NeMoGymResponseReasoningItem)
        assert step2.items[0].summary[0].text == "hmm"
        assert step2.items[1].content[0].text == "fixed"

    def test_generation_span_identity(self) -> None:
        trajectory = build_trajectory([], self._transcript())
        step1, step2 = trajectory.steps[1], trajectory.steps[2]
        gen1 = step1.spans[0]
        assert gen1.type == "generation"
        assert gen1.response_id == "msg_1"
        assert gen1.request_id == "req-1"
        assert step2.spans[0].request_id == "req-2"

    def test_usage_counted_once_per_message(self) -> None:
        trajectory = build_trajectory([], self._transcript())
        step1 = trajectory.steps[1]
        assert step1.usage.input_tokens == 100
        assert step1.usage.output_tokens == 20
        assert step1.usage.input_tokens_details.cached_tokens == 60
        # fields with no native slot survive verbatim on the generation span
        assert step1.spans[0].extra["provider_usage"]["cache_creation_input_tokens"] == 10
        assert trajectory.usage.input_tokens == 150
        assert trajectory.usage.output_tokens == 25
        assert trajectory.usage.input_tokens_details.cached_tokens == 60

    def test_parallel_function_spans_have_independent_timing(self) -> None:
        trajectory = build_trajectory([], self._transcript())
        spans = [s for s in trajectory.steps[1].spans if s.type == "function"]
        assert [s.call_id for s in spans] == ["t1", "t2"]
        assert spans[0].started_at == "2026-07-09T00:00:01.000Z"
        assert spans[0].ended_at == "2026-07-09T00:00:01.500Z"
        assert spans[0].duration_ms == 500.0
        assert spans[1].duration_ms == 2000.0

    def test_observation_error_and_curated_extra(self) -> None:
        trajectory = build_trajectory([], self._transcript())
        spans = [s for s in trajectory.steps[1].spans if s.type == "function"]
        assert spans[0].error is None
        # short scalars kept, oversized strings and nested structures dropped
        assert spans[0].extra == {"interrupted": False, "stdout": "file.txt"}
        assert spans[1].error is not None
        assert spans[1].extra is None
        # model-visible output is on the native function_call_output item
        outputs = {i.call_id: i.output for i in trajectory.steps[1].items if isinstance(i, NeMoGymFunctionCallOutput)}
        assert outputs == {"t1": "file.txt", "t2": "boom"}

    def test_sidechain_records_skipped_and_counted(self) -> None:
        records = self._transcript() + [
            _assistant_record("a3", "msg_3", [{"type": "text", "text": "sub"}], isSidechain=True)
        ]
        trajectory = build_trajectory([], records)
        assert trajectory.dropped_records == {"sidechain": 1}
        assert len([s for s in trajectory.steps if s.type == "agent_step"]) == 2

    def test_compact_summary_becomes_context_boundary(self) -> None:
        records = [
            _user_record("start"),
            _user_record("summary of history", ts="2026-07-09T00:01:00.000Z", isCompactSummary=True),
            _assistant_record("a1", "msg_1", [{"type": "text", "text": "go on"}]),
        ]
        trajectory = build_trajectory([], records)
        assert [s.type for s in trajectory.steps] == ["user_message", "context_boundary", "agent_step"]
        assert trajectory.steps[1].items[0].content == "summary of history"

    def test_unmatched_observation_attaches_to_last_agent_step(self) -> None:
        records = [
            _assistant_record("a1", "msg_1", [{"type": "text", "text": "hi"}]),
            _tool_result_record("orphan", "out", "2026-07-09T00:00:02.000Z", source_uuid="missing"),
        ]
        trajectory = build_trajectory([], records)
        assert trajectory.steps[0].spans[-1].call_id == "orphan"

    def test_orphan_observation_with_no_agent_step_counted_as_dropped(self) -> None:
        records = [_tool_result_record("orphan", "out", "2026-07-09T00:00:02.000Z")]
        trajectory = build_trajectory([], records)
        assert trajectory.steps == []
        assert trajectory.dropped_records == {"orphan_tool_results": 1}

    def test_meta_user_records_skipped(self) -> None:
        trajectory = build_trajectory([], [_user_record("injected", isMeta=True)])
        assert trajectory.steps == []

    def test_round_trip_validation(self) -> None:
        trajectory = build_trajectory([], self._transcript())
        dumped = trajectory.model_dump(mode="json")
        assert dumped["schema_version"] == TRAJECTORY_SCHEMA_VERSION
        assert validate_trajectory(dumped) == trajectory


class TestStreamJsonFallback:
    def _events(self) -> list[dict]:
        return [
            {"type": "system", "subtype": "init", "session_id": SESSION},
            {
                "type": "assistant",
                "message": {
                    "id": "msg_1",
                    "model": "claude-sonnet-4-6",
                    "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
            },
            {
                "type": "user",
                "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
            },
            {"type": "system", "subtype": "compact_boundary"},
            {
                "type": "result",
                "num_turns": 3,
                "duration_ms": 42.0,
                "total_cost_usd": 0.5,
                "usage": {"input_tokens": 10, "output_tokens": 2},
            },
        ]

    def test_fallback_used_when_no_transcript_messages(self) -> None:
        trajectory = build_trajectory(self._events(), transcript_records=[{"type": "queue-operation"}])
        assert trajectory.source == "stream_json"
        assert trajectory.session_id == SESSION
        assert [s.type for s in trajectory.steps] == ["agent_step", "context_boundary"]

    def test_no_timestamps_means_no_fabricated_timing(self) -> None:
        trajectory = build_trajectory(self._events(), [])
        (span,) = [s for s in trajectory.steps[0].spans if s.type == "function"]
        assert span.started_at is None
        assert span.ended_at is None
        assert span.duration_ms is None

    def test_result_event_totals(self) -> None:
        trajectory = build_trajectory(self._events(), [])
        assert trajectory.num_agent_steps == 3
        assert trajectory.duration_ms == 42.0
        assert trajectory.total_cost_usd == 0.5
        assert trajectory.provider_usage == {"input_tokens": 10, "output_tokens": 2}

    def test_empty_sources(self) -> None:
        trajectory = build_trajectory([], [])
        assert trajectory.source == "stream_json"
        assert trajectory.steps == []
        assert trajectory.usage.input_tokens == 0


class TestDecodeJsonl:
    def test_skips_blank_and_malformed_lines(self) -> None:
        text = '\n{"a": 1}\nnot-json\n[1, 2]\n{"b": 2}\n'
        assert decode_jsonl(text) == [{"a": 1}, {"b": 2}]
