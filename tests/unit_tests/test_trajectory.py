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

from time import time

import pytest

from nemo_gym.openai_utils import (
    NeMoGymContextBoundaryMessage,
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymFunctionCallOutputWithAgentTelemetry,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseFunctionToolCallWithAgentTelemetry,
    NeMoGymResponseOutputMessageWithAgentTelemetry,
    NeMoGymResponseReasoningItemWithAgentTelemetry,
    NeMoGymToolExecutionError,
)
from nemo_gym.trajectory import (
    TrajectoryBuilder,
    agent_step_slices,
    reconstruct_model_input,
    summed_usage,
    to_response_create_params,
    usage_from_provider,
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


def _two_step_builder() -> TrajectoryBuilder:
    """Step 1: text + two parallel tool calls; step 2: reasoning + final answer."""
    builder = TrajectoryBuilder(agent="test_agent", source="unit_test")
    builder.set_session_id("sess-1")
    builder.start_agent_step(
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
    builder.add_tool_result("t1", "file.txt", completed_at="2026-07-09T00:00:01.500Z", extra={"interrupted": False})
    builder.add_tool_result(
        "t2", "boom", completed_at="2026-07-09T00:00:03.000Z", error="tool_result flagged is_error"
    )
    builder.start_agent_step(
        response_id="msg_2",
        model="test-model",
        timestamp="2026-07-09T00:00:04.000Z",
        stop_reason="end_turn",
        provider_usage={"input_tokens": 50, "output_tokens": 5},
    )
    builder.add_reasoning("hmm")
    builder.add_output_text("fixed")
    return builder


class TestBuilder:
    def test_output_is_tagged_native_items_in_execution_order(self) -> None:
        output, _, _ = _two_step_builder().build()
        assert [type(i) for i in output] == [
            NeMoGymResponseOutputMessageWithAgentTelemetry,
            NeMoGymResponseFunctionToolCallWithAgentTelemetry,  # issue order
            NeMoGymResponseFunctionToolCallWithAgentTelemetry,
            NeMoGymFunctionCallOutputWithAgentTelemetry,  # arrival order
            NeMoGymFunctionCallOutputWithAgentTelemetry,
            NeMoGymResponseReasoningItemWithAgentTelemetry,
            NeMoGymResponseOutputMessageWithAgentTelemetry,
        ]
        # every item carries its agent step tag — grouping without any copy
        assert [i.agent_step_no for i in output] == [1, 1, 1, 1, 1, 2, 2]
        assert output[0].content[0].text == "looking"
        assert output[3].output == "file.txt"
        assert output[5].summary[0].text == "hmm"

    def test_tool_execution_rides_on_the_output_item(self) -> None:
        output, _, _ = _two_step_builder().build()
        out_t1, out_t2 = output[3], output[4]
        assert out_t1.execution.started_at == "2026-07-09T00:00:01.000Z"
        assert out_t1.execution.duration_ms == 500.0
        assert out_t1.execution.error is None
        assert out_t1.execution.extra == {"interrupted": False}
        assert out_t2.execution.duration_ms == 2000.0
        assert out_t2.execution.error.message == "tool_result flagged is_error"

    def test_generations_carry_per_call_identity_and_usage(self) -> None:
        _, generations, _ = _two_step_builder().build()
        g1, g2 = generations
        assert (g1.agent_step_no, g2.agent_step_no) == (1, 2)
        assert g1.response_id == "msg_1"
        assert g1.request_id == "req-1"
        assert g1.stop_reason == "tool_use"
        assert g2.stop_reason == "end_turn"
        assert g1.usage.input_tokens == 100
        assert g1.usage.input_tokens_details.cached_tokens == 60
        # raw provider usage preserved verbatim (incl. fields with no native slot)
        assert g1.provider_usage["cache_creation_input_tokens"] == 10

    def test_same_response_id_continues_step_without_double_count(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_step(response_id="m1", provider_usage={"input_tokens": 10, "output_tokens": 5})
        builder.add_output_text("part 1")
        builder.start_agent_step(
            response_id="m1", provider_usage={"input_tokens": 10, "output_tokens": 5}, stop_reason="end_turn"
        )
        builder.add_output_text("part 2")
        output, generations, _ = builder.build()
        assert len(generations) == 1
        assert generations[0].stop_reason == "end_turn"
        assert summed_usage(generations).input_tokens == 10
        # text accumulated into one message item under the same tag
        assert [b.text for b in output[0].content] == ["part 1", "part 2"]

    def test_output_without_step_raises(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        with pytest.raises(ValueError):
            builder.add_output_text("no step")

    def test_mid_episode_user_message_is_plain_and_closes_the_step(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_step(response_id="m1")
        builder.add_output_text("answer 1")
        builder.add_user_message("follow-up")
        builder.start_agent_step(response_id="m1")  # same provider id, but step was closed by the user turn
        builder.add_output_text("answer 2")
        output, generations, _ = builder.build()
        assert type(output[1]) is NeMoGymEasyInputMessage  # untagged: not produced by a step
        assert output[1].content == "follow-up"
        assert [g.agent_step_no for g in generations] == [1, 2]
        assert output[2].agent_step_no == 2

    def test_no_timestamps_means_no_fabricated_timing(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_step(response_id="m1")
        builder.add_tool_call("t1", "Bash", "{}")
        builder.add_tool_result("t1", "ok")
        output, _, _ = builder.build()
        execution = output[-1].execution
        assert execution.started_at is None
        assert execution.completed_at is None
        assert execution.duration_ms is None

    def test_explicit_started_at_overrides_registration(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_step(response_id="m1", timestamp="2026-07-09T00:00:00.000Z")
        builder.add_tool_call("t1", "Bash", "{}")
        builder.add_tool_result(
            "t1", "ok", started_at="2026-07-09T00:00:02.000Z", completed_at="2026-07-09T00:00:03.000Z"
        )
        output, _, _ = builder.build()
        assert output[-1].execution.duration_ms == 1000.0

    def test_orphan_result_with_no_step_counted_as_dropped(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.add_tool_result("orphan", "out")
        output, _, telemetry = builder.build()
        assert output == []
        assert telemetry.dropped_records == {"orphan_tool_results": 1}

    def test_unmatched_result_attaches_to_last_step(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_step(response_id="m1")
        builder.add_tool_call("known", "Bash", "{}")
        builder.add_tool_result("known", "ok")
        builder.add_tool_result("orphan", "late")
        output, _, telemetry = builder.build()
        assert output[-1].call_id == "orphan"
        assert output[-1].agent_step_no == 1
        assert telemetry.dropped_records == {}

    def test_error_can_be_structured(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_step(response_id="m1")
        builder.add_tool_call("t1", "Bash", "{}")
        builder.add_tool_result("t1", "boom", error=NeMoGymToolExecutionError(message="timeout", data={"signal": 9}))
        output, _, _ = builder.build()
        assert output[-1].execution.error.data == {"signal": 9}

    def test_run_totals_and_session(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.set_session_id("sess-1")
        builder.set_session_id("sess-2")  # first one wins
        builder.count_dropped("sidechain")
        builder.set_run_totals(
            num_agent_steps=3, duration_ms=42.0, total_cost_usd=0.5, provider_usage={"input_tokens": 1}
        )
        _, _, telemetry = builder.build()
        assert telemetry.agent == "a"
        assert telemetry.source == "s"
        assert telemetry.session_id == "sess-1"
        assert telemetry.dropped_records == {"sidechain": 1}
        assert telemetry.num_agent_steps == 3
        assert telemetry.duration_ms == 42.0
        assert telemetry.total_cost_usd == 0.5
        assert telemetry.provider_usage == {"input_tokens": 1}

    def test_summed_usage(self) -> None:
        _, generations, _ = _two_step_builder().build()
        totals = summed_usage(generations)
        assert totals.input_tokens == 150
        assert totals.output_tokens == 25
        assert totals.total_tokens == 175
        assert totals.input_tokens_details.cached_tokens == 60

    def test_agent_step_slices(self) -> None:
        output, _, _ = _two_step_builder().build()
        slices = dict(agent_step_slices(output))
        assert [len(items) for items in slices.values()] == [5, 2]
        assert slices[2][0].summary[0].text == "hmm"


class TestReconstruction:
    def _episode(self):
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_step(response_id="m1", model="test-model")
        builder.add_output_text("a1")
        builder.add_context_boundary(summary="summary of q1/a1")
        builder.add_user_message("q2")
        builder.start_agent_step(response_id="m2", model="test-model")
        builder.add_output_text("a2")
        output, generations, telemetry = builder.build()
        base_input = [NeMoGymEasyInputMessage(role="user", content="q1")]
        return output, base_input

    def test_step_1_sees_base_input_only(self) -> None:
        output, base = self._episode()
        items = reconstruct_model_input(output, agent_step_no=1, base_input=base)
        assert [i.content for i in items] == ["q1"]

    def test_post_boundary_step_sees_summary_not_base(self) -> None:
        output, base = self._episode()
        items = reconstruct_model_input(output, agent_step_no=2, base_input=base)
        assert [getattr(i, "content", None) for i in items] == ["summary of q1/a1", "q2"]
        assert isinstance(items[0], NeMoGymContextBoundaryMessage)

    def test_full_reconstruction(self) -> None:
        output, base = self._episode()
        items = reconstruct_model_input(output, base_input=base)
        assert len(items) == 3  # summary, q2, a2 — the boundary replaced q1 + a1
        assert items[-1].content[0].text == "a2"

    def test_no_boundary_prepends_base_input(self) -> None:
        builder = TrajectoryBuilder(agent="a", source="s")
        builder.start_agent_step(response_id="m1")
        builder.add_output_text("a1")
        builder.add_user_message("q2")
        builder.start_agent_step(response_id="m2")
        output, _, _ = builder.build()
        base = [NeMoGymEasyInputMessage(role="user", content="q1")]
        items = reconstruct_model_input(output, agent_step_no=2, base_input=base)
        assert [getattr(i, "content", None) for i in items][0] == "q1"
        assert len(items) == 3  # q1 + a1 + q2

    def test_to_response_create_params_is_native(self) -> None:
        output, base = self._episode()
        params = to_response_create_params(output, base_input=base, model="test-model")
        assert params.model == "test-model"
        assert type(params).model_validate(params.model_dump(mode="json")).model == "test-model"


class TestContractRoundTrip:
    """Telemetry must survive NeMoGymResponse validation — the whole point of putting it
    on the contract: every server hop revalidates, and nothing may be stripped."""

    def _response(self) -> NeMoGymResponse:
        output, generations, telemetry = _two_step_builder().build()
        return NeMoGymResponse(
            id="resp_x",
            created_at=int(time()),
            model="test-model",
            object="response",
            output=output,
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
            generations=generations,
            agent_telemetry=telemetry,
        )

    def test_items_keep_telemetry_through_revalidation(self) -> None:
        response = self._response()
        revalidated = NeMoGymResponse.model_validate(response.model_dump(mode="json"))
        assert [getattr(i, "agent_step_no", None) for i in revalidated.output] == [1, 1, 1, 1, 1, 2, 2]
        assert revalidated.output[3].execution.duration_ms == 500.0
        assert revalidated == response

    def test_generations_and_telemetry_survive(self) -> None:
        revalidated = NeMoGymResponse.model_validate(self._response().model_dump(mode="json"))
        assert revalidated.generations[0].provider_usage["cache_creation_input_tokens"] == 10
        assert revalidated.agent_telemetry.source == "unit_test"
        assert revalidated.agent_telemetry.schema_version == "1.0"

    def test_plain_payloads_stay_plain(self) -> None:
        # A model-server response (no telemetry) must validate to the plain classes and
        # keep generations/agent_telemetry as None.
        plain = {
            "id": "resp_y",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "tool_choice": "auto",
            "tools": [],
            "parallel_tool_calls": True,
            "output": [
                {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c", "output": "ok"},
            ],
        }
        response = NeMoGymResponse.model_validate(plain)
        assert type(response.output[0]) is NeMoGymResponseFunctionToolCall
        assert type(response.output[1]) is NeMoGymFunctionCallOutput
        assert response.generations is None
        assert response.agent_telemetry is None

    def test_context_boundary_survives_and_plain_user_message_stays_plain(self) -> None:
        boundary = NeMoGymContextBoundaryMessage(role="user", content="summary", context_boundary=True)
        plain_user = {"role": "user", "content": "hi", "type": "message"}
        response = self._response()
        response.output = [boundary] + response.output
        revalidated = NeMoGymResponse.model_validate(response.model_dump(mode="json"))
        assert isinstance(revalidated.output[0], NeMoGymContextBoundaryMessage)
        from pydantic import TypeAdapter

        from nemo_gym.openai_utils import NeMoGymResponseInputItem

        assert type(TypeAdapter(NeMoGymResponseInputItem).validate_python(plain_user)) is NeMoGymEasyInputMessage
