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
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryBuilder,
    TrajectorySpanError,
    reconstruct_model_input,
    to_response_create_params,
    usage_from_provider,
    validate_trajectory,
)


ANTHROPIC_USAGE = {
    "input_tokens": 100,
    "output_tokens": 20,
    "cache_read_input_tokens": 60,
    "cache_creation_input_tokens": 10,
}
OPENAI_USAGE = {
    "input_tokens": 100,
    "output_tokens": 20,
    "total_tokens": 120,
    "input_tokens_details": {"cached_tokens": 60},
    "output_tokens_details": {"reasoning_tokens": 5},
}


class TestUsageFromProvider:
    def test_anthropic_dialect(self) -> None:
        usage = usage_from_provider(ANTHROPIC_USAGE)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 20
        assert usage.total_tokens == 120
        assert usage.input_tokens_details.cached_tokens == 60
        assert usage.output_tokens_details.reasoning_tokens == 0

    def test_openai_dialect(self) -> None:
        usage = usage_from_provider(OPENAI_USAGE)
        assert usage.input_tokens_details.cached_tokens == 60
        assert usage.output_tokens_details.reasoning_tokens == 5
        assert usage.total_tokens == 120

    def test_empty(self) -> None:
        usage = usage_from_provider({})
        assert usage.input_tokens == 0
        assert usage.total_tokens == 0


class TestBuilderSteps:
    def _build(self) -> TrajectoryBuilder:
        builder = TrajectoryBuilder(agent="test_agent", source="unit_test")
        builder.set_session_id("sess-1")
        builder.add_user_message("fix the bug", timestamp="2026-07-09T00:00:00.000Z")
        builder.start_agent_turn(
            response_id="msg_1",
            request_id="req-1",
            model="test-model",
            timestamp="2026-07-09T00:00:01.000Z",
            stop_reason="tool_use",
            provider_usage=ANTHROPIC_USAGE,
        )
        builder.add_output_text("looking")
        builder.add_tool_call("t1", "Bash", '{"command": "ls"}')
        builder.add_tool_call("t2", "Read", '{"file_path": "/x"}')
        builder.add_tool_result(
            "t1",
            "file.txt",
            completed_at="2026-07-09T00:00:01.500Z",
            extra={"interrupted": False},
        )
        builder.add_tool_result(
            "t2",
            "boom",
            completed_at="2026-07-09T00:00:03.000Z",
            error="tool_result flagged is_error",
        )
        builder.start_agent_turn(
            response_id="msg_2",
            model="test-model",
            timestamp="2026-07-09T00:00:04.000Z",
            stop_reason="end_turn",
            provider_usage={"input_tokens": 50, "output_tokens": 5},
        )
        builder.add_reasoning("hmm")
        builder.add_output_text("fixed")
        return builder

    def test_step_structure_uses_native_items(self) -> None:
        trajectory = self._build().build()
        assert [s.type for s in trajectory.steps] == ["user_message", "agent_turn", "agent_turn"]
        user, turn1, turn2 = trajectory.steps
        assert isinstance(user.items[0], NeMoGymEasyInputMessage)
        assert user.items[0].content == "fix the bug"
        assert [type(i) for i in turn1.items] == [
            NeMoGymResponseOutputMessage,
            NeMoGymResponseFunctionToolCall,
            NeMoGymResponseFunctionToolCall,
            NeMoGymFunctionCallOutput,
            NeMoGymFunctionCallOutput,
        ]
        assert turn1.items[0].content[0].text == "looking"
        assert turn1.items[3].output == "file.txt"
        assert isinstance(turn2.items[0], NeMoGymResponseReasoningItem)
        assert turn2.items[0].summary[0].text == "hmm"
        assert (turn1.turn_no, turn2.turn_no) == (1, 2)
        assert turn1.stop_reason == "tool_use"

    def test_generation_span_identity_and_raw_usage(self) -> None:
        trajectory = self._build().build()
        span = trajectory.steps[1].spans[0]
        assert span.type == "generation"
        assert span.response_id == "msg_1"
        assert span.request_id == "req-1"
        assert span.ended_at == "2026-07-09T00:00:01.000Z"
        # raw provider usage preserved verbatim (incl. fields with no native slot)
        assert span.extra["provider_usage"]["cache_creation_input_tokens"] == 10

    def test_function_spans_have_independent_timing(self) -> None:
        trajectory = self._build().build()
        spans = [s for s in trajectory.steps[1].spans if s.type == "function"]
        assert [s.call_id for s in spans] == ["t1", "t2"]
        assert spans[0].started_at == "2026-07-09T00:00:01.000Z"
        assert spans[0].duration_ms == 500.0
        assert spans[0].error is None
        assert spans[0].extra == {"interrupted": False}
        assert spans[1].duration_ms == 2000.0
        assert spans[1].error.message == "tool_result flagged is_error"

    def test_explicit_started_at_overrides_registration(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_turn(response_id="m1", timestamp="2026-07-09T00:00:00.000Z")
        builder.add_tool_call("t1", "Bash", "{}")
        builder.add_tool_result(
            "t1", "ok", started_at="2026-07-09T00:00:02.000Z", completed_at="2026-07-09T00:00:03.000Z"
        )
        (span,) = [s for s in builder.steps[0].spans if s.type == "function"]
        assert span.duration_ms == 1000.0

    def test_no_timestamps_means_no_fabricated_timing(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_turn(response_id="m1")
        builder.add_tool_call("t1", "Bash", "{}")
        builder.add_tool_result("t1", "ok")
        (span,) = [s for s in builder.steps[0].spans if s.type == "function"]
        assert span.started_at is None
        assert span.ended_at is None
        assert span.duration_ms is None

    def test_usage_native_per_turn_and_totals(self) -> None:
        trajectory = self._build().build()
        turn1 = trajectory.steps[1]
        assert turn1.usage.input_tokens == 100
        assert turn1.usage.input_tokens_details.cached_tokens == 60
        assert trajectory.usage.input_tokens == 150
        assert trajectory.usage.output_tokens == 25
        assert trajectory.usage.total_tokens == 175
        assert trajectory.usage.input_tokens_details.cached_tokens == 60

    def test_same_response_id_continues_turn_without_double_count(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_turn(response_id="m1", provider_usage={"input_tokens": 10, "output_tokens": 5})
        builder.add_output_text("part 1")
        step = builder.start_agent_turn(
            response_id="m1", provider_usage={"input_tokens": 10, "output_tokens": 5}, stop_reason="end_turn"
        )
        builder.add_output_text("part 2")
        trajectory = builder.build()
        assert len(trajectory.steps) == 1
        assert step.turn_no == 1
        assert step.stop_reason == "end_turn"
        assert trajectory.usage.input_tokens == 10
        # text accumulated into one native output message
        assert [b.text for b in step.items[0].content] == ["part 1", "part 2"]

    def test_different_response_id_starts_new_turn(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_turn(response_id="m1")
        builder.start_agent_turn(response_id="m2")
        assert [s.turn_no for s in builder.steps] == [1, 2]

    def test_output_without_turn_raises(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        with pytest.raises(ValueError):
            builder.add_output_text("no turn")

    def test_unmatched_result_attaches_to_last_turn(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_turn(response_id="m1")
        builder.add_tool_result("orphan", "out")
        step = builder.steps[0]
        assert step.items[-1].call_id == "orphan"
        assert step.spans[-1].call_id == "orphan"

    def test_orphan_result_with_no_turn_counted_as_dropped(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.add_tool_result("orphan", "out")
        trajectory = builder.build()
        assert trajectory.steps == []
        assert trajectory.dropped_records == {"orphan_tool_results": 1}

    def test_dropped_and_session_and_totals(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.set_session_id("sess-1")
        builder.set_session_id("sess-2")  # first one wins
        builder.count_dropped("sidechain")
        builder.count_dropped("sidechain")
        builder.set_run_totals(num_turns=3, duration_ms=42.0, total_cost_usd=0.5, provider_usage={"input_tokens": 1})
        trajectory = builder.build()
        assert trajectory.session_id == "sess-1"
        assert trajectory.dropped_records == {"sidechain": 2}
        assert trajectory.num_turns == 3
        assert trajectory.duration_ms == 42.0
        assert trajectory.total_cost_usd == 0.5
        assert trajectory.provider_usage == {"input_tokens": 1}

    def test_error_can_be_span_error(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_turn(response_id="m1")
        builder.add_tool_call("t1", "Bash", "{}")
        builder.add_tool_result("t1", "boom", error=TrajectorySpanError(message="timeout", data={"signal": 9}))
        (span,) = [s for s in builder.steps[0].spans if s.type == "function"]
        assert span.error.data == {"signal": 9}


class TestReconstruction:
    def _trajectory(self):
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.add_user_message("q1")
        builder.start_agent_turn(response_id="m1", model="test-model")
        builder.add_output_text("a1")
        builder.add_context_boundary(summary="summary of q1/a1")
        builder.add_user_message("q2")
        builder.start_agent_turn(response_id="m2", model="test-model")
        builder.add_output_text("a2")
        return builder.build()

    def test_full_reconstruction_starts_at_boundary(self) -> None:
        items = reconstruct_model_input(self._trajectory())
        assert [getattr(i, "content", None) for i in items][:2] == ["summary of q1/a1", "q2"]
        assert len(items) == 3  # boundary summary, q2, a2

    def test_before_step_id_excludes_later_steps(self) -> None:
        trajectory = self._trajectory()
        # input visible to the model call at step 1 (turn 1): just q1
        items = reconstruct_model_input(trajectory, before_step_id=1)
        assert len(items) == 1
        assert items[0].content == "q1"
        # input visible to the model call at step 4 (turn 2): boundary summary + q2
        items = reconstruct_model_input(trajectory, before_step_id=4)
        assert [i.content for i in items] == ["summary of q1/a1", "q2"]

    def test_to_response_create_params_is_native(self) -> None:
        params = to_response_create_params(self._trajectory())
        assert params.model == "test-model"
        assert len(params.input) == 3
        # round-trips through the strict native model
        assert type(params).model_validate(params.model_dump(mode="json")).model == "test-model"


class TestValidation:
    def test_round_trip(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.add_user_message("hi", timestamp="2026-07-09T00:00:00.000Z")
        builder.start_agent_turn(response_id="m1", provider_usage=ANTHROPIC_USAGE)
        builder.add_output_text("hello")
        builder.add_reasoning("think")
        builder.add_tool_call("t1", "Bash", "{}")
        builder.add_tool_result("t1", "ok", extra={"code": 0})
        trajectory = builder.build()

        dumped = trajectory.model_dump(mode="json")
        assert dumped["schema_version"] == TRAJECTORY_SCHEMA_VERSION
        revalidated = validate_trajectory(dumped)
        assert revalidated == trajectory
        # native item types survive the round trip through the union
        items = revalidated.steps[1].items
        assert [type(i) for i in items] == [
            NeMoGymResponseOutputMessage,
            NeMoGymResponseReasoningItem,
            NeMoGymResponseFunctionToolCall,
            NeMoGymFunctionCallOutput,
        ]
