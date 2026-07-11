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
    NeMoGymContextBoundaryMessage,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymFunctionCallOutputWithAgentTelemetry,
    NeMoGymResponseFunctionToolCallWithAgentTelemetry,
    NeMoGymResponseOutputMessageWithAgentTelemetry,
    NeMoGymResponseReasoningItemWithAgentTelemetry,
)
from nemo_gym.trajectory import reconstruct_model_input, summed_usage
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
            _user_record("fix the bug"),  # initial prompt: lives in create_params, not the content plane
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

    def test_output_is_native_lossless_and_tagged(self) -> None:
        output, generations, telemetry = build_trajectory([], self._transcript())
        assert telemetry.source == "transcript"
        assert telemetry.session_id == SESSION
        assert [type(i) for i in output] == [
            NeMoGymResponseOutputMessageWithAgentTelemetry,  # "looking"
            NeMoGymResponseFunctionToolCallWithAgentTelemetry,  # t1, issue order
            NeMoGymResponseFunctionToolCallWithAgentTelemetry,  # t2
            NeMoGymFunctionCallOutputWithAgentTelemetry,  # t1 result, arrival order
            NeMoGymFunctionCallOutputWithAgentTelemetry,  # t2 result
            NeMoGymResponseReasoningItemWithAgentTelemetry,  # native reasoning, not <think> text
            NeMoGymResponseOutputMessageWithAgentTelemetry,  # "fixed"
        ]
        assert [i.agent_step_no for i in output] == [1, 1, 1, 1, 1, 2, 2]
        assert output[1].name == "Bash"
        assert output[5].summary[0].text == "hmm"
        assert output[6].content[0].text == "fixed"

    def test_initial_prompt_not_duplicated_into_output(self) -> None:
        output, _, _ = build_trajectory([], self._transcript())
        assert not any(type(i) is NeMoGymEasyInputMessage for i in output)

    def test_mid_episode_user_message_is_recorded(self) -> None:
        records = self._transcript() + [
            _user_record("and now Berlin", ts="2026-07-09T00:01:00.000Z"),
            _assistant_record("a3", "msg_3", [{"type": "text", "text": "on it"}], ts="2026-07-09T00:01:01.000Z"),
        ]
        output, generations, _ = build_trajectory([], records)
        user_items = [i for i in output if type(i) is NeMoGymEasyInputMessage]
        assert [i.content for i in user_items] == ["and now Berlin"]
        assert output[output.index(user_items[0]) + 1].agent_step_no == generations[-1].agent_step_no

    def test_generation_identity_and_dedupe(self) -> None:
        _, generations, _ = build_trajectory([], self._transcript())
        g1, g2 = generations
        assert (g1.agent_step_no, g2.agent_step_no) == (1, 2)
        assert g1.stop_reason == "tool_use"
        assert g1.response_id == "msg_1"
        assert g1.request_id == "req-1"
        assert g2.request_id == "req-2"

    def test_usage_counted_once_per_message(self) -> None:
        _, generations, _ = build_trajectory([], self._transcript())
        g1 = generations[0]
        assert g1.usage.input_tokens == 100
        assert g1.usage.input_tokens_details.cached_tokens == 60
        assert g1.provider_usage["cache_creation_input_tokens"] == 10
        totals = summed_usage(generations)
        assert totals.input_tokens == 150
        assert totals.output_tokens == 25

    def test_parallel_tool_executions_have_independent_timing(self) -> None:
        output, _, _ = build_trajectory([], self._transcript())
        out1, out2 = output[3], output[4]
        assert (out1.call_id, out2.call_id) == ("t1", "t2")
        assert out1.execution.started_at == "2026-07-09T00:00:01.000Z"
        assert out1.execution.duration_ms == 500.0
        assert out2.execution.duration_ms == 2000.0

    def test_execution_error_and_curated_extra(self) -> None:
        output, _, _ = build_trajectory([], self._transcript())
        out1, out2 = output[3], output[4]
        assert out1.execution.error is None
        assert out1.execution.extra == {"interrupted": False, "stdout": "file.txt"}  # short scalars kept
        assert out2.execution.error is not None
        outputs = {i.call_id: i.output for i in output if isinstance(i, NeMoGymFunctionCallOutput)}
        assert outputs == {"t1": "file.txt", "t2": "boom"}

    def test_sidechain_records_skipped_and_counted(self) -> None:
        records = self._transcript() + [
            _assistant_record("a3", "msg_3", [{"type": "text", "text": "sub"}], isSidechain=True)
        ]
        _, generations, telemetry = build_trajectory([], records)
        assert telemetry.dropped_records == {"sidechain": 1}
        assert len(generations) == 2

    def test_compact_summary_becomes_boundary_with_summary_item(self) -> None:
        records = [
            _user_record("start"),
            _assistant_record("a1", "msg_1", [{"type": "text", "text": "working"}]),
            _user_record("summary of history", ts="2026-07-09T00:01:00.000Z", isCompactSummary=True),
            _assistant_record("a2", "msg_2", [{"type": "text", "text": "go on"}], ts="2026-07-09T00:01:01.000Z"),
        ]
        output, _, _ = build_trajectory([], records)
        boundaries = [i for i in output if isinstance(i, NeMoGymContextBoundaryMessage)]
        assert [b.content for b in boundaries] == ["summary of history"]
        # post-boundary reconstruction starts at the summary, not the base input
        items = reconstruct_model_input(
            output, agent_step_no=2, base_input=[NeMoGymEasyInputMessage(role="user", content="start")]
        )
        assert getattr(items[0], "content", None) == "summary of history"

    def test_orphan_observation_with_no_step_counted_as_dropped(self) -> None:
        records = [_tool_result_record("orphan", "out", "2026-07-09T00:00:02.000Z")]
        output, _, telemetry = build_trajectory([], records)
        assert output == []
        assert telemetry.dropped_records == {"orphan_tool_results": 1}

    def test_meta_user_records_skipped(self) -> None:
        output, _, _ = build_trajectory([], [_user_record("injected", isMeta=True)])
        assert output == []

    def test_telemetry_round_trips_through_the_contract(self) -> None:
        from time import time

        from nemo_gym.openai_utils import NeMoGymResponse

        output, generations, telemetry = build_trajectory([], self._transcript())
        response = NeMoGymResponse(
            id="r",
            created_at=int(time()),
            model="m",
            object="response",
            output=output,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
            generations=generations,
            agent_telemetry=telemetry,
        )
        revalidated = NeMoGymResponse.model_validate(response.model_dump(mode="json"))
        assert revalidated == response
        assert revalidated.output[3].execution.duration_ms == 500.0


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
        output, _, telemetry = build_trajectory(self._events(), transcript_records=[{"type": "queue-operation"}])
        assert telemetry.source == "stream_json"
        assert telemetry.session_id == SESSION
        assert [type(i).__name__ for i in output] == [
            "NeMoGymResponseFunctionToolCallWithAgentTelemetry",
            "NeMoGymFunctionCallOutputWithAgentTelemetry",
            "NeMoGymContextBoundaryMessage",
        ]

    def test_no_timestamps_means_no_fabricated_timing(self) -> None:
        output, _, _ = build_trajectory(self._events(), [])
        execution = output[1].execution
        assert execution.started_at is None
        assert execution.duration_ms is None

    def test_result_event_totals(self) -> None:
        _, _, telemetry = build_trajectory(self._events(), [])
        assert telemetry.num_agent_steps == 3  # Claude Code's num_turns, normalized
        assert telemetry.duration_ms == 42.0
        assert telemetry.total_cost_usd == 0.5
        assert telemetry.provider_usage == {"input_tokens": 10, "output_tokens": 2}

    def test_empty_sources(self) -> None:
        output, generations, telemetry = build_trajectory([], [])
        assert output == []
        assert generations == []
        assert telemetry.source == "stream_json"


class TestDecodeJsonl:
    def test_skips_blank_and_malformed_lines(self) -> None:
        text = '\n{"a": 1}\nnot-json\n[1, 2]\n{"b": 2}\n'
        assert decode_jsonl(text) == [{"a": 1}, {"b": 2}]
