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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from stirrup.core.models import AssistantMessage, TokenUsage, ToolCall, UserMessage

from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.tool_access import DirectHTTPToolAccess
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

    async def execute(payload, **_):
        sent.append(payload)
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
        assert response.metadata == {"elapsed_seconds": "1.5", "stop_reason": "cancelled"}
        assert [item.type for item in response.output] == ["message", "message"]
        assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (20, 10, 30)
        assert response.usage.input_tokens_details.cached_tokens == 8
        assert response.usage.output_tokens_details.reasoning_tokens == 4

    @pytest.mark.asyncio
    async def test_a_run_whose_model_calls_reported_no_usage_has_none(self) -> None:
        response, _, _ = await _run_episode(
            _make_config(), NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy")
        )

        assert (response.status, response.usage) == ("completed", None)
        assert "stop_reason" not in response.metadata

    @pytest.mark.asyncio
    async def test_a_run_ended_by_a_rejected_request_is_a_failed_response_that_used_no_tokens(self) -> None:
        report = _report(
            stop_reason="model_error",
            model_error="Error code: 400 - rejected",
            observations={"status": "failed", "model_response_ids": []},
        )

        response, _, _ = await _run_episode(
            _make_config(), NeMoGymResponseCreateParamsNonStreaming(input="x", model="policy"), report
        )

        assert response.status == "failed"
        assert (response.error.code, response.error.message) == ("server_error", "Error code: 400 - rejected")
        assert response.metadata["stop_reason"] == "model_error"
        assert (response.usage.input_tokens, response.usage.output_tokens, response.usage.total_tokens) == (0, 0, 0)
        assert response.usage.input_tokens_details.cached_tokens is None
        assert response.usage.output_tokens_details.reasoning_tokens is None

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


class TestObservations:
    MODEL = ModelServerRef(type="responses_api_models", name="policy_model")
    TOOL = {"kind": "tool_call", "invocation_id": "root", "tool_call_id": "c1", "status": "completed"}

    def _output(self, **observations):
        raw = {"invocation_id": "root", "status": "completed", "model_response_ids": ["r1"], "tool_calls": [self.TOOL]}
        raw |= observations
        return {"input_items": [{"role": "user", "content": "hi", "type": "message"}], "observations": raw}

    def test_a_part_that_fails_validation_becomes_a_gap_and_the_rest_is_kept(self) -> None:
        bundle = _observations(self._output(tool_calls=[self.TOOL, {**self.TOOL, "status": "exploded"}]), self.MODEL)

        assert [record.kind for record in bundle.records] == ["agent_invocation", "tool_call"]
        assert [(gap.code, gap.detail) for gap in bundle.gaps] == [
            ("observation_capture_failed", "tool_calls[1]: ValidationError")
        ]

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
