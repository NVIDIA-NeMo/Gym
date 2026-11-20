# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError
from stirrup.core.models import AssistantMessage, Reasoning, TokenUsage, ToolCall, ToolMessage, UserMessage

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY, AgentOutcome, OutcomeAction, agent_outcome_policy
from nemo_gym.cli.env import validate_server_configs
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    AGENT_INVALID_OUTPUT,
    MODEL_REQUEST_ERROR,
    EpisodeId,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_runtime import DirectHTTPToolAccess
from nemo_gym.trajectory import project_bundle_v1
from nemo_gym.utils import ClockContext, ToolCallRecorder
from responses_api_agents.stirrup_agent.app import StirrupAgentWrapper, StirrupAgentWrapperConfig, _observations
from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent, NeMoUserMessage
from responses_api_agents.stirrup_agent.nemo_client import DynamicMaxTokensChatCompletionsClient
from responses_api_agents.stirrup_agent.stirrup_utils import convert_stirrup_history_to_output_items


def _make_config(**fields) -> StirrupAgentWrapperConfig:
    return StirrupAgentWrapperConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="stirrup_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        **fields,
    )


@pytest.mark.parametrize(
    "option",
    [
        "judge_only",
        "execute_only",
        "rerun_incomplete",
        "persist_deliverables_dir",
        "tavily_api_key",
        "tavily_max_sweeps",
        "gdpval_container_path",
        "system_prompt_template",
        "user_prompt_template",
        "model_id",
        "task",
    ],
)
def test_removed_options_are_rejected(option):
    with pytest.raises(ValueError, match=rf"removed options that are no longer supported: `{option}`"):
        _make_config(**{option: "x"})


def test_removed_options_are_each_located_at_their_own_key():
    with pytest.raises(ValidationError) as error:
        _make_config(judge_only=False, task="t")
    assert [(detail["loc"], detail["type"]) for detail in error.value.errors()] == [
        (("judge_only",), "removed_option"),
        (("task",), "removed_option"),
    ]


def test_validate_server_configs_reports_a_removed_option_at_its_full_path():
    block = {
        "entrypoint": "app.py",
        "host": "0.0.0.0",
        "port": 8080,
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
        "judge_only": False,
    }
    config = OmegaConf.create({"gdpval_agent": {"responses_api_agents": {"stirrup_agent": block}}})
    [problem] = validate_server_configs(config, ["responses_api_agents"])
    assert (problem.path, problem.config_class, problem.value) == (
        "gdpval_agent.responses_api_agents.stirrup_agent.judge_only",
        "StirrupAgentWrapperConfig",
        False,
    )
    assert "removed options that are no longer supported: `judge_only`" in problem.message


def _report(**fields) -> dict:
    return {
        "input_items": [],
        "output_items": [],
        "elapsed_seconds": 0,
        "usages": [],
        "stop_reason": None,
        "resources_cookies": {},
        "observations": {"status": "completed", "model_response_ids": ["chatcmpl-1"]},
        **fields,
    }


async def _run_episode(config, body, report=None, *, execute_error=None, artifacts=None):
    """Run one episode against a state whose sandbox runner returns ``report`` (or raises); return response, payload, state."""
    sent = []

    async def execute(payload, **kwargs):
        sent.append(payload)
        state.execute_kwargs = kwargs
        if execute_error:
            raise execute_error
        return report or _report()

    wrapper = StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))
    state = SimpleNamespace(
        request=SimpleNamespace(episode_id=EpisodeId(rollout_id="rollout", attempt=0)),
        session=SimpleNamespace(workdir="/root", stop_request_path="/tmp/stop-requested", artifacts=artifacts),
        tool_access=DirectHTTPToolAccess(name="tools", required=True, base_url="http://resources:8000"),
        resources_cookies={},
        execute=execute,
    )
    with patch.object(StirrupAgentWrapper, "resolve_model_base_url", return_value="http://policy.invalid/v1"):
        response = await wrapper._run_sandbox_episode(body, state)
    return response, sent[0], state


async def _runner_payload(config: StirrupAgentWrapperConfig, body: NeMoGymResponseCreateParamsNonStreaming) -> dict:
    """The payload the agent server hands the sandbox runner for one episode."""
    return (await _run_episode(config, body))[1]


class TestApp:
    def test_sanity(self) -> None:
        """Config instantiation + wrapper construction should not raise."""
        config = _make_config()
        StirrupAgentWrapper(config=config, server_client=MagicMock(spec=ServerClient))

        # Generic Stirrup users retain the historical behavior. GDPVal opts in
        # to its larger floor and matching template semantics in its benchmark
        # config below.
        assert config.min_completion_tokens == 1024
        assert config.context_window_tokens == 262144
        assert config.prompt_estimator_truncate_history_thinking is None
        assert config.min_compaction_summary_words == 1
        assert config.truncation_recovery is False

    @pytest.mark.asyncio
    async def test_request_output_cap_does_not_replace_model_context_window(self) -> None:
        config = _make_config(context_window_tokens=262144, max_completion_tokens_cap=64000)
        body = NeMoGymResponseCreateParamsNonStreaming(input="ignored", model="policy", max_output_tokens=8192)

        payload = await _runner_payload(config, body)

        assert payload["client"]["max_tokens"] == 262144
        assert payload["client"]["max_completion_tokens_cap"] == 8192

    @pytest.mark.parametrize("truncation_recovery", [None, False, True])
    @pytest.mark.asyncio
    async def test_truncation_recovery_is_forwarded_to_the_rollout(self, truncation_recovery) -> None:
        fields = {} if truncation_recovery is None else {"truncation_recovery": truncation_recovery}
        body = NeMoGymResponseCreateParamsNonStreaming(input="ignored", model="policy")

        payload = await _runner_payload(_make_config(**fields), body)

        # Unset means off: only configs that opt in (GDPVal's benchmark config) get recovery nudges.
        assert payload["client"]["truncation_recovery"] is bool(truncation_recovery)

    @pytest.mark.parametrize("recovery", [None, False, True])
    @pytest.mark.asyncio
    async def test_context_overflow_recovery_is_forwarded_to_the_rollout(self, recovery) -> None:
        fields = {} if recovery is None else {"context_overflow_recovery": recovery}
        body = NeMoGymResponseCreateParamsNonStreaming(input="ignored", model="policy")

        payload = await _runner_payload(_make_config(**fields), body)

        # Unset means off: a rejected request ends the episode unless the benchmark's config opts in.
        assert payload["client"]["context_overflow_recovery"] is bool(recovery)

    @pytest.mark.parametrize("code_exec", [None, False, True])
    @pytest.mark.asyncio
    async def test_code_exec_is_forwarded_to_the_rollout(self, code_exec) -> None:
        fields = {} if code_exec is None else {"code_exec": code_exec}
        body = NeMoGymResponseCreateParamsNonStreaming(input="ignored", model="policy")

        payload = await _runner_payload(_make_config(**fields), body)

        # Unset keeps the shell, as before the option existed.
        assert payload["code_exec"] is (code_exec is not False)

    @pytest.mark.asyncio
    async def test_runtime_knobs_shape_the_model_requests_the_runner_makes(self) -> None:
        config = _make_config(
            context_window_tokens=131072,
            min_completion_tokens=8192,
            prompt_estimator_truncate_history_thinking=True,
            truncation_recovery=True,
            min_compaction_summary_words=50,
        )
        payload = await _runner_payload(config, NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"))
        client = DynamicMaxTokensChatCompletionsClient(api_key="gym", **payload["client"])
        NeMoAgent(
            client=client,
            name="stirrup_agent",
            max_turns=payload["max_turns"],
            min_compaction_summary_words=payload["min_compaction_summary_words"],
        )
        overrun = MagicMock(**{"choices": [MagicMock(finish_reason="length")], "usage": None})
        overrun.choices[0].message = MagicMock(
            content="<think>x", tool_calls=[], reasoning_content=None, reasoning=None
        )
        ok = MagicMock(**{"choices": [MagicMock(finish_reason="stop")], "usage": None})
        ok.choices[0].message = MagicMock(content="ok", tool_calls=[], reasoning_content=None, reasoning=None)
        create = AsyncMock(side_effect=[ok, overrun, ok])
        client._client = MagicMock()
        client._client.chat.completions.create = create
        old_thinking = f"<think>{'r' * 300_000}</think>done"
        history = [
            UserMessage(content="task"),
            AssistantMessage(content=old_thinking, token_usage=TokenUsage(input=1, answer=1, reasoning=0)),
            UserMessage(content="continue"),
        ]

        await client.generate(history, tools={})
        await client.generate([UserMessage(content="x" * 500_000)], tools={})
        await client.generate([UserMessage(content="x")], tools={})

        first, filled, recovery = (call.kwargs for call in create.await_args_list)
        # The old reasoning block is not counted, so the cap rather than the context bounds the budget.
        assert first["max_completion_tokens"] == 64000
        assert filled["max_completion_tokens"] == 8192
        assert recovery["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    @pytest.mark.asyncio
    async def test_the_response_is_built_from_the_runner_report(self) -> None:
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 4},
            "completion_tokens_details": {"reasoning_tokens": 2},
        }
        report = _report(
            input_items=[{"role": "user", "content": "hi", "type": "message"}],
            output_items=[
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "yo", "annotations": []}],
                    "type": "message",
                    "status": "completed",
                    "id": "m1",
                }
            ],
            usages=[usage, usage],
            elapsed_seconds=1.5,
            observations={"status": "incomplete", "model_response_ids": ["chatcmpl-1"]},
            stop_reason="cancelled",
        )

        response, _, _ = await _run_episode(
            _make_config(), NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"), report
        )

        assert response.status == "incomplete"
        assert response.metadata == {
            "elapsed_seconds": "1.5",
            "agent_timeout_sec": "12600.0",
            "stop_reason": "cancelled",
        }
        assert [item.type for item in response.output] == ["message", "message"]
        assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (20, 10, 30)
        assert response.usage.input_tokens_details.cached_tokens == 8
        assert response.usage.output_tokens_details.reasoning_tokens == 4

    @pytest.mark.parametrize(
        "metadata,budget",
        [
            (None, 900.0),
            ({"agent_timeout_sec": "600"}, 600.0),
            ({"agent_timeout_sec": "5000"}, 900.0),
            ({"agent_timeout_sec": "not-a-number"}, 900.0),
        ],
    )
    async def test_task_timeout_shortens_but_never_extends_the_configured_budget(
        self, metadata: dict | None, budget: float
    ) -> None:
        """A Harbor task row carries `metadata.agent_timeout_sec`; the runner gets min(config, task) seconds."""
        response, _, state = await _run_episode(
            _make_config(sandbox_runner_timeout_seconds=900),
            NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy", metadata=metadata),
        )

        assert state.execute_kwargs["timeout"] == budget
        assert response.metadata["agent_timeout_sec"] == str(budget)

    @pytest.mark.asyncio
    async def test_a_run_whose_model_calls_reported_no_usage_has_none(self) -> None:
        response, _, _ = await _run_episode(
            _make_config(), NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy")
        )

        assert (response.status, response.usage) == ("completed", None)
        assert "stop_reason" not in response.metadata
        assert AGENT_OUTCOME_METADATA_KEY not in response.metadata

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 413, 422])
    async def test_a_run_ended_by_a_rejected_request_is_a_failed_response_that_used_no_tokens(self, status) -> None:
        report = _report(
            stop_reason="model_error",
            model_error="rejected",
            model_error_status=status,
            observations={"status": "failed", "model_response_ids": []},
        )

        response, _, _ = await _run_episode(
            _make_config(), NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"), report
        )

        assert response.status == "failed"
        assert (response.error.code, response.error.message) == ("server_error", "rejected")
        assert response.metadata["stop_reason"] == "model_error"
        assert AgentOutcome.model_validate_json(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == AgentOutcome(
            failure_kind=MODEL_REQUEST_ERROR, status_code=status
        )
        assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (0, 0, 0)
        assert response.usage.input_tokens_details.cached_tokens is None
        assert response.usage.output_tokens_details.reasoning_tokens is None

    @pytest.mark.asyncio
    async def test_a_run_ended_by_unusable_compaction_summaries_is_the_models_invalid_output(self) -> None:
        report = _report(
            stop_reason="compaction_failed",
            invalid_output="Compaction summarizer produced no usable text after 3 attempts",
            observations={"status": "failed", "model_response_ids": ["chatcmpl-1"]},
        )

        response, _, _ = await _run_episode(
            _make_config(), NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"), report
        )

        assert response.status == "failed"
        assert response.error is None
        assert response.metadata["stop_reason"] == "compaction_failed"
        assert AgentOutcome.model_validate_json(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == AgentOutcome(
            failure_kind=AGENT_INVALID_OUTPUT
        )
        assert agent_outcome_policy(response) == OutcomeAction.VERIFY

    @pytest.mark.asyncio
    async def test_a_run_killed_before_any_model_response_has_unknown_usage_and_no_error(self) -> None:
        report = _report(observations={"status": "incomplete", "model_response_ids": []}, stop_reason="wall_time")

        response, _, _ = await _run_episode(
            _make_config(), NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"), report
        )

        assert response.error is None
        assert response.usage is None

    @pytest.mark.asyncio
    async def test_a_rejected_request_after_responses_that_reported_no_usage_has_unknown_usage(self) -> None:
        report = _report(
            stop_reason="model_error",
            model_error="Error code: 400 - rejected",
            observations={"status": "failed", "model_response_ids": ["chatcmpl-1"]},
        )

        response, _, _ = await _run_episode(
            _make_config(), NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"), report
        )

        assert (response.status, response.usage) == ("failed", None)
        # A report without a structured status must not guess from the error message.
        assert (
            AgentOutcome.model_validate_json(response.metadata[AGENT_OUTCOME_METADATA_KEY]).failure_kind == "unknown"
        )

    @pytest.mark.asyncio
    async def test_usage_without_detail_fields_has_no_cached_or_reasoning_counts(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        response, _, _ = await _run_episode(
            _make_config(), NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"), _report(usages=[usage])
        )

        assert response.usage.input_tokens_details.cached_tokens is None
        assert response.usage.output_tokens_details.reasoning_tokens is None

    @pytest.mark.asyncio
    async def test_a_supervisor_timeout_with_the_runners_wall_time_report_is_an_incomplete_response(self) -> None:
        partial = _report(
            output_items=[{"type": "function_call", "call_id": "c1", "name": "code_exec", "arguments": "{}"}],
            stop_reason="wall_time",
            observations={"status": "incomplete", "model_response_ids": ["chatcmpl-1"]},
            resources_cookies={"session": "late"},
        )

        response, _, state = await _run_episode(
            _make_config(),
            NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"),
            execute_error=TimeoutError(),
            artifacts=partial,
        )

        assert (response.status, response.metadata["stop_reason"]) == ("incomplete", "wall_time")
        assert [item.type for item in response.output] == ["function_call"]
        assert state.resources_cookies == {"session": "late"}

    @pytest.mark.parametrize(
        "artifacts", [None, _report(stop_reason="cancelled"), _report(stop_reason=None), "not a report"]
    )
    @pytest.mark.asyncio
    async def test_a_supervisor_timeout_without_a_wall_time_report_is_raised(self, artifacts) -> None:
        with pytest.raises(TimeoutError):
            await _run_episode(
                _make_config(),
                NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"),
                execute_error=TimeoutError(),
                artifacts=artifacts,
            )

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            ({"temperature": 0, "top_p": 0}, (0, 0)),
            ({"temperature": 0.2, "top_p": 0.5}, (0.2, 0.5)),
            ({}, (0.6, 0.95)),
        ],
    )
    @pytest.mark.asyncio
    async def test_the_request_sampling_overrides_the_config_even_when_it_is_zero(self, requested, expected) -> None:
        config = _make_config(temperature=0.6, top_p=0.95)
        body = NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy", **requested)

        client = (await _runner_payload(config, body))["client"]

        assert (client["temperature"], client["top_p"]) == expected

    def test_output_history_preserves_nemo_user_tool_results(self) -> None:
        """Run-history export should keep NeMo user-role tool results as tool outputs."""
        history = [
            [
                AssistantMessage(
                    content="",
                    tool_calls=[ToolCall(tool_call_id="call_1", name="code_exec", arguments='{"cmd":"true"}')],
                    token_usage=TokenUsage(input=1, answer=1, reasoning=0),
                ),
                NeMoUserMessage(content="ok", name="code_exec", success=True, tool_call_id="call_1"),
            ]
        ]

        input_items, output_items = convert_stirrup_history_to_output_items(history)

        assert input_items == []
        assert len(output_items) == 2
        assert output_items[0]["type"] == "function_call"
        assert output_items[0]["call_id"] == "call_1"
        assert output_items[1]["type"] == "function_call_output"
        assert output_items[1]["call_id"] == "call_1"
        assert output_items[1]["output"] == "ok"

    def test_output_history_keeps_reasoning_returned_in_its_own_field(self) -> None:
        call = ToolCall(tool_call_id="call_1", name="code_exec", arguments='{"cmd":"ls"}')
        history = [
            [
                AssistantMessage(
                    content="Listing the inputs.", tool_calls=[call], reasoning=Reasoning(content="Look first.")
                ),
                ToolMessage(content="a.xlsx", name="code_exec", tool_call_id="call_1"),
                AssistantMessage(content="Done.", reasoning=Reasoning(content="")),
            ]
        ]

        _, output_items = convert_stirrup_history_to_output_items(history)

        assert [item["type"] for item in output_items] == [
            "reasoning",
            "message",
            "function_call",
            "function_call_output",
            "message",
        ]
        assert output_items[0]["summary"] == [{"type": "summary_text", "text": "Look first."}]
        response = NeMoGymResponse(
            id="r",
            created_at=0,
            model="m",
            object="response",
            output=output_items,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        assert response.output[0].type == "reasoning"

    def test_output_history_without_reasoning_is_unchanged(self) -> None:
        call = ToolCall(tool_call_id="call_1", name="code_exec", arguments='{"cmd":"ls"}')
        inline = AssistantMessage(content="<think>Look first.</think>Listing.", tool_calls=[call])
        history = [[inline, ToolMessage(content="a.xlsx", name="code_exec", tool_call_id="call_1")]]

        _, output_items = convert_stirrup_history_to_output_items(history)

        assert [item["type"] for item in output_items] == ["message", "function_call", "function_call_output"]
        assert output_items[0]["content"][0]["text"] == "<think>Look first.</think>Listing."


class TestObservations:
    MODEL = ModelServerRef(type="responses_api_models", name="policy_model")

    @staticmethod
    def _recorder(path: Path) -> ToolCallRecorder:
        return ToolCallRecorder(
            path=path,
            producer_id="stirrup:test",
            invocation_id="root",
            agent_instance=None,
            clock=ClockContext(location="sandbox", owner="test"),
            boundary="lifecycle",
            timing_source="stirrup.run_tool",
        )

    def _events(self, tmp_path: Path, *outcomes: str) -> tuple[str, list[str]]:
        """Tool events as the runner records them: one call per outcome, all with native ID ``code_exec:0``."""
        path = tmp_path / "tool_events.jsonl"
        recorder, spans = self._recorder(path), []
        for outcome in outcomes:
            with recorder.call(
                native_call_id="code_exec:0",
                tool_call_id="code_exec:0",
                tool_name="code_exec",
                model_call_ref={"model_ref": self.MODEL.model_dump(mode="json"), "response_id": "r1"},
            ) as observed:
                observed.set_outcome(outcome)
            spans.append(observed.span_id)
        return path.read_text(), spans

    def _output(self, **observations):
        raw = {
            "invocation_id": "root",
            "status": "completed",
            "model_response_ids": ["r1"],
            "tool_call_ids": {},
            "tool_capture_failed_spans": [],
        }
        raw |= observations
        return {"input_items": [{"role": "user", "content": "hi", "type": "message"}], "observations": raw}

    def test_recorded_tool_calls_take_the_conversations_call_ids(self, tmp_path) -> None:
        events, spans = self._events(tmp_path, "completed", "failed")
        call_ids = dict(zip(spans, ["code_exec:0-1", "code_exec:0-2"]))

        bundle = _observations(self._output(tool_call_ids=call_ids), self.MODEL, events)

        assert bundle.schema_version == "2.0"
        tools = sorted((r for r in bundle.records if r.kind == "tool_call"), key=lambda t: t.timings[0].started_at)
        assert [(t.tool_call_id, t.native_tool_call_id, t.status) for t in tools] == [
            ("code_exec:0-1", "code_exec:0", "completed"),
            ("code_exec:0-2", "code_exec:0", "failed"),
        ]
        assert bundle.gaps == []

    def test_the_v1_view_keeps_one_harness_timing_per_call(self, tmp_path) -> None:
        events, spans = self._events(tmp_path, "completed", "timeout")
        bundle = _observations(self._output(tool_call_ids=dict(zip(spans, ["c1", "c2"]))), self.MODEL, events)

        v1 = project_bundle_v1(bundle.model_dump(mode="json"))

        tools = [record for record in v1["records"] if record["kind"] == "tool_call"]
        assert len(tools) == 2
        for tool in tools:
            assert tool["timing_source"] == "harness"
            assert tool["completed_at"] >= tool["started_at"] > 1e9 and tool["duration_ms"] >= 0

    def test_a_runner_without_output_keeps_the_calls_it_started(self, tmp_path) -> None:
        path = tmp_path / "tool_events.jsonl"
        self._recorder(path).start(native_call_id="c1", tool_call_id="c1", tool_name="code_exec")

        bundle = _observations(None, self.MODEL, path.read_text())

        (tool,) = bundle.records
        assert (tool.tool_call_id, tool.status) == ("c1", "incomplete")
        assert tool.timings[0].started_at is not None and tool.timings[0].completed_at is None
        assert {"observation_capture_failed", "tool_completion_unavailable"} <= {gap.code for gap in bundle.gaps}

    @pytest.mark.parametrize("ran_tools", [False, True])
    def test_unreadable_events_are_a_gap_only_when_tools_ran(self, ran_tools) -> None:
        output = self._output(tool_call_ids={"span-1": "c1"} if ran_tools else {})

        bundle = _observations(output, self.MODEL, None)

        assert [r.kind for r in bundle.records] == ["agent_invocation"]
        assert [gap.code for gap in bundle.gaps] == (["tool_capture_unavailable"] if ran_tools else [])

    def test_a_call_the_recorder_could_not_write_is_a_gap_on_its_span(self, tmp_path) -> None:
        events, _ = self._events(tmp_path, "completed")

        bundle = _observations(self._output(tool_capture_failed_spans=["lost"]), self.MODEL, events)

        assert [(gap.code, gap.span_id) for gap in bundle.gaps] == [("tool_capture_unavailable", "lost")]

    def test_an_invalid_conversation_keeps_the_model_call_references(self) -> None:
        output = self._output()
        output["input_items"] = [{"type": "no_such_item"}]

        bundle = _observations(output, self.MODEL)

        invocation = bundle.records[0]
        assert [ref.response_id for ref in invocation.model_calls] == ["r1"]
        assert invocation.conversation == []
        assert [gap.detail for gap in bundle.gaps] == ["conversation: ValidationError"]

    def test_no_runner_output_is_one_gap(self) -> None:
        bundle = _observations(None, self.MODEL)

        assert bundle.records == []
        assert [gap.code for gap in bundle.gaps] == ["observation_capture_failed"]
