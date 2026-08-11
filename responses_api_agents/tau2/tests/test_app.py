# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_responses_api_agent import AggregateMetricsRequest
from nemo_gym.rollout_collection import NG_FAILURE_CLASS_KEY, NG_NO_PERSIST_KEY, NG_TERMINAL_KEY
from nemo_gym.server_utils import ServerClient


# Import the adapter before Tau2 so it can set TAU2_DATA_DIR before Tau2
# initializes its module-level data paths.
# isort: off
from responses_api_agents.tau2.app import (
    ModelServerRef,
    TAU2_AGENT_FAILURE_CLASS,
    TAU2_MALFORMED_TOOL_CALL_FAILURE_CLASS,
    Tau2Agent,
    Tau2Config,
    Tau2FailureResponse,
    Tau2MalformedToolCallError,
    Tau2RunRequest,
    Tau2ToolValidatingAsyncOpenAI,
)
from tau2.data_model.message import AssistantMessage, UserMessage
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.utils import llm_utils as tau2_llm_utils


def _drop_nulls(value):
    """Recursively remove keys whose value is None."""
    if isinstance(value, dict):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


# isort: on


class TestApp:
    def _dummy_server(
        self,
        max_agent_steps: Optional[int] = None,
        turns_remaining_interval: int = 1,
        malformed_tool_call_max_retries: int = 0,
    ) -> Tuple[Tau2Config, Tau2Agent]:
        config = Tau2Config(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
            user_model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
            max_steps=4,
            max_agent_steps=max_agent_steps,
            turns_remaining_interval=turns_remaining_interval,
            malformed_tool_call_max_retries=malformed_tool_call_max_retries,
        )
        server = Tau2Agent(config=config, server_client=MagicMock(spec=ServerClient))

        return config, server

    def _example_run_request(self) -> Tau2RunRequest:
        example_jsonl = Path(__file__).parent.parent / "data" / "example.jsonl"
        with example_jsonl.open() as f:
            data = list(map(json.loads, f))
        return Tau2RunRequest.model_validate(data[0])

    def _fake_simulation_run(
        self,
        *,
        messages=None,
        agent_messages=None,
        agent_steps: Optional[int] = 2,
        max_agent_steps: Optional[int] = 3,
        termination_reason: TerminationReason = TerminationReason.USER_STOP,
    ) -> SimulationRun:
        if messages is None:
            messages = [
                AssistantMessage.text("hello"),
                UserMessage.text("hi"),
                AssistantMessage.text("done"),
            ]

        return SimulationRun(
            id="fake-simulation",
            task_id="fake-task",
            start_time="2026-05-05T00:00:00",
            end_time="2026-05-05T00:00:01",
            duration=1.0,
            num_steps=len(messages),
            agent_steps=agent_steps,
            max_agent_steps=max_agent_steps,
            termination_reason=termination_reason,
            reward_info=RewardInfo(reward=1.0),
            messages=messages,
            agent_messages=agent_messages,
        )

    def _chat_completion(self, arguments: Optional[str] = None) -> dict:
        tool_calls = []
        finish_reason = "stop"
        if arguments is not None:
            finish_reason = "tool_calls"
            tool_calls = [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": arguments},
                }
            ]
        return {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {
                        "content": "done" if arguments is None else None,
                        "role": "assistant",
                        "tool_calls": tool_calls,
                    },
                }
            ],
            "created": 0,
            "model": "dummy_model",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    def test_sanity(self) -> None:
        config, _ = self._dummy_server()

        assert config.max_agent_steps is None
        assert config.turns_remaining_interval == 1
        assert config.malformed_tool_call_max_retries == 0

    def test_setup_webserver_installs_tool_validating_client(self) -> None:
        _, server = self._dummy_server(malformed_tool_call_max_retries=2)

        with (
            patch("responses_api_agents.tau2.app.ensure_tau2_data_dir"),
            patch.object(tau2_llm_utils, "NeMoGymAsyncOpenAI"),
        ):
            server.setup_webserver()
            client = tau2_llm_utils.NeMoGymAsyncOpenAI(
                base_url="http://model/v1",
                api_key="dummy",  # pragma: allowlist secret
            )

        assert isinstance(client, Tau2ToolValidatingAsyncOpenAI)
        assert client.malformed_tool_call_max_retries == 2

    async def test_tool_argument_validation_does_not_retry_malformed_response(self) -> None:
        malformed = self._chat_completion('{"account_id": "123"')
        messages = [{"role": "user", "content": "look it up"}]
        original_messages = deepcopy(messages)
        client = Tau2ToolValidatingAsyncOpenAI(base_url="http://model/v1", api_key="dummy")  # pragma: allowlist secret

        with patch(
            "nemo_gym.openai_utils.NeMoGymAsyncOpenAI.create_chat_completion",
            AsyncMock(return_value=malformed),
        ) as create_chat_completion:
            with pytest.raises(Tau2MalformedToolCallError, match="1 consecutive attempts"):
                await client.create_chat_completion(model="model", messages=messages)

        assert create_chat_completion.await_count == 1
        assert all(call.kwargs["messages"] == original_messages for call in create_chat_completion.await_args_list)
        assert messages == original_messages

    async def test_tool_argument_validation_stops_after_first_malformed_response(self) -> None:
        malformed = self._chat_completion('{"account_id": "123"')
        client = Tau2ToolValidatingAsyncOpenAI(base_url="http://model/v1", api_key="dummy")  # pragma: allowlist secret

        with patch(
            "nemo_gym.openai_utils.NeMoGymAsyncOpenAI.create_chat_completion",
            AsyncMock(return_value=malformed),
        ) as create_chat_completion:
            with pytest.raises(Tau2MalformedToolCallError, match="1 consecutive attempts"):
                await client.create_chat_completion(model="model", messages=[])

        assert create_chat_completion.await_count == 1

    async def test_tool_argument_validation_honors_retry_override(self) -> None:
        malformed = self._chat_completion('{"account_id": "123"')
        valid = self._chat_completion('{"account_id": "123"}')
        client = Tau2ToolValidatingAsyncOpenAI(
            base_url="http://model/v1",
            api_key="dummy",  # pragma: allowlist secret
            malformed_tool_call_max_retries=1,
        )

        with patch(
            "nemo_gym.openai_utils.NeMoGymAsyncOpenAI.create_chat_completion",
            AsyncMock(side_effect=[malformed, valid]),
        ) as create_chat_completion:
            response = await client.create_chat_completion(model="model", messages=[])

        assert response == valid
        assert create_chat_completion.await_count == 2

    @pytest.mark.parametrize("arguments", ['["not", "an", "object"]', '"not an object"', "null"])
    async def test_tool_argument_validation_rejects_valid_json_that_is_not_an_object(self, arguments: str) -> None:
        response = self._chat_completion(arguments)
        client = Tau2ToolValidatingAsyncOpenAI(base_url="http://model/v1", api_key="dummy")  # pragma: allowlist secret

        with patch(
            "nemo_gym.openai_utils.NeMoGymAsyncOpenAI.create_chat_completion",
            AsyncMock(return_value=response),
        ) as create_chat_completion:
            with pytest.raises(Tau2MalformedToolCallError, match="must decode to an object"):
                await client.create_chat_completion(model="model", messages=[])

        assert create_chat_completion.await_count == 1

    @pytest.mark.parametrize("arguments", ['{"account_id": "123"}', None])
    async def test_tool_argument_validation_passes_valid_and_no_tool_responses_without_retry(
        self, arguments: Optional[str]
    ) -> None:
        expected = self._chat_completion(arguments)
        client = Tau2ToolValidatingAsyncOpenAI(base_url="http://model/v1", api_key="dummy")  # pragma: allowlist secret

        with patch(
            "nemo_gym.openai_utils.NeMoGymAsyncOpenAI.create_chat_completion",
            AsyncMock(return_value=expected),
        ) as create_chat_completion:
            response = await client.create_chat_completion(model="model", messages=[])

        assert response == expected
        create_chat_completion.assert_awaited_once()

    @pytest.mark.parametrize(
        ("error", "expected_failure_class"),
        [
            (
                ExceptionGroup("task group failed", [Tau2MalformedToolCallError("invalid arguments")]),
                TAU2_MALFORMED_TOOL_CALL_FAILURE_CLASS,
            ),
            (RuntimeError("unexpected failure"), TAU2_AGENT_FAILURE_CLASS),
        ],
    )
    def test_run_returns_http_200_failure_record_and_sanitizes_stale_routing_fields(
        self, error: Exception, expected_failure_class: str
    ) -> None:
        data = self._example_run_request().model_dump(mode="json")
        data |= {
            "reward": 1.0,
            "response": {"stale": True},
            "error": "stale",
            NG_FAILURE_CLASS_KEY: "stale",
            NG_NO_PERSIST_KEY: True,
            NG_TERMINAL_KEY: True,
        }
        _, server = self._dummy_server()
        with patch("responses_api_agents.tau2.app.ensure_tau2_data_dir"):
            app = server.setup_webserver()
        client = TestClient(app)

        with (
            patch("responses_api_agents.tau2.app.get_server_url", return_value="http://model"),
            patch("responses_api_agents.tau2.app.run_single_task", AsyncMock(side_effect=error)),
        ):
            response = client.post("/run", json=data)

        assert response.status_code == 200
        result = response.json()
        validated = Tau2FailureResponse.model_validate(result)
        assert validated.reward == 0.0
        assert result[NG_FAILURE_CLASS_KEY] == expected_failure_class
        assert result["error"] != "stale"
        assert result["response"]["object"] == "response"
        assert result["response"]["output"][0]["content"][0]["text"] == ""
        assert NG_NO_PERSIST_KEY not in result
        assert NG_TERMINAL_KEY not in result

    def test_sanity_query_input(self) -> None:
        example_jsonl = Path(__file__).parent.parent / "data" / "example.jsonl"
        with example_jsonl.open() as f:
            data = list(map(json.loads, f))

        _, server = self._dummy_server()

        app = server.setup_webserver()
        client = TestClient(app)

        async_openai_mock = MagicMock()
        async_openai_mock.create_chat_completion = AsyncMock(
            return_value={
                "id": "chtcmpl-123",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "message": {"content": "<think>thinking</think>hello", "role": "assistant", "tool_calls": []},
                    }
                ],
                "created": 0,
                "model": "dummy_model",
                "object": "chat.completion",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

        with (
            patch("responses_api_agents.tau2.app.get_server_url", return_value="dummy base url"),
            patch("tau2.utils.llm_utils.NeMoGymAsyncOpenAI", return_value=async_openai_mock),
        ):
            response = client.post("/run", json=data[0])

        actual_response_dict = response.json()
        expected_response_dict = json.loads((Path(__file__).parent / "test_data.json").read_text())
        # with open("temp.json", "w") as f:
        #     json.dump(actual_response_dict, f, indent=4)

        def _clean(d):
            d.pop("agent_steps", None)
            d.pop("max_agent_steps", None)
            d["config"].pop("max_agent_steps", None)
            d["config"].pop("turns_remaining_interval", None)
            d["config"].pop("review_model", None)
            d["result"].pop("agent_messages", None)
            d["result"].pop("agent_steps", None)
            d["result"].pop("max_agent_steps", None)
            d["result"].pop("num_steps", None)
            d["result"].pop("duration")
            d["result"].pop("end_time")
            d["result"].pop("id")
            d["result"].pop("start_time")
            d["result"].pop("timestamp")
            for m in d["result"]["messages"]:
                m.pop("timestamp")
                m.pop("generation_time_seconds", None)

            d["response"].pop("created_at")

            for o in d["response"]["output"]:
                o.pop("id", None)

            for o in d["responses_create_params"]["input"]:
                o.pop("id", None)

            d["duration"] = 0.0

            # SDK releases can add optional response fields.
            # Ignore these fields when their values are `None`.
            # Expected non-null values remain part of the comparison.
            return _drop_nulls(d)

        assert _clean(expected_response_dict) == _clean(actual_response_dict)

    async def test_run_passes_agent_step_budget_config(self) -> None:
        _, server = self._dummy_server(max_agent_steps=3, turns_remaining_interval=2)
        body = self._example_run_request()
        captured_kwargs = {}

        async def fake_run_single_task(**kwargs):
            captured_kwargs.update(kwargs)
            return self._fake_simulation_run(
                agent_steps=3,
                max_agent_steps=3,
                termination_reason=TerminationReason.MAX_AGENT_STEPS,
            )

        with (
            patch("responses_api_agents.tau2.app.get_server_url", return_value="dummy base url"),
            patch("responses_api_agents.tau2.app.run_single_task", side_effect=fake_run_single_task),
        ):
            response = await server.run(body)

        assert captured_kwargs["config"].max_agent_steps == 3
        assert captured_kwargs["config"].turns_remaining_interval == 2
        assert response.agent_steps == 3
        assert response.max_agent_steps == 3
        assert response.result.agent_steps == 3
        assert response.result.max_agent_steps == 3
        assert response.result.termination_reason == TerminationReason.MAX_AGENT_STEPS

    async def test_run_uses_agent_visible_messages_for_responses_conversion(self) -> None:
        _, server = self._dummy_server(max_agent_steps=3)
        body = self._example_run_request()
        reminder = "ENVIRONMENT REMINDER: You have 2 agent steps remaining."
        clean_messages = [
            AssistantMessage.text("hello"),
            UserMessage.text("hi"),
            AssistantMessage.text("done"),
        ]
        agent_messages = [
            AssistantMessage.text("hello"),
            UserMessage.text(f"hi\n\n{reminder}"),
            AssistantMessage.text("done"),
        ]

        with (
            patch("responses_api_agents.tau2.app.get_server_url", return_value="dummy base url"),
            patch(
                "responses_api_agents.tau2.app.run_single_task",
                AsyncMock(
                    return_value=self._fake_simulation_run(
                        messages=clean_messages,
                        agent_messages=agent_messages,
                    )
                ),
            ),
        ):
            response = await server.run(body)

        response_dict = response.model_dump(mode="json")
        assert reminder not in json.dumps(response_dict["result"]["messages"])
        assert reminder in json.dumps(response_dict["result"]["agent_messages"])
        assert reminder in json.dumps(response_dict["responses_create_params"]["input"])

    async def test_compute_metrics(self) -> None:
        example_rollouts_fpath = Path(__file__).parent.parent / "data" / "example_rollouts.jsonl"
        with example_rollouts_fpath.open() as f:
            rollouts = list(map(json.loads, f))

        _, server = self._dummy_server()

        actual_metrics = server.compute_metrics([rollouts])
        expected_metrics = {
            "macro_average": 1.0,
            "telecom/num_samples_unique": 1,
            "retail/num_samples_total": 1,
            "telecom/num_samples_total": 3,
            "airline/num_samples_total": 1,
            "retail/reward": 1.0,
            "telecom/reward": 1.0,
            "airline/reward": 1.0,
            "telecom/service_issue/reward": 1.0,
            "retail/trajectory_termination_reason/user_stop/count": 1,
            "telecom/trajectory_termination_reason/user_stop/count": 3,
            "airline/trajectory_termination_reason/user_stop/count": 1,
            "trajectory_termination_reason/user_stop/count": 5,
            "trajectory_termination_reason/user_stop/pct": 1.0,
            "message_finish_reason/tool_calls/count": 20,
            "message_finish_reason/stop/count": 5,
            "message_finish_reason/tool_calls/pct": 0.8,
            "message_finish_reason/stop/pct": 0.2,
            "trajectory_transfer_to_human_agents/count": 4,
            "trajectory_transfer_to_human_agents/pct": 0.8,
            "tool_call_hallucination/count/total": 0,
            "trajectory_missing_tool_call/count": 0,
            "trajectory_missing_tool_call/pct": 0.0,
            "messages_with_incomplete_reasoning/count": 0,
            "messages_with_incomplete_reasoning/pct": 0.0,
        }
        assert expected_metrics == actual_metrics

        actual_aggregate_metrics = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=rollouts))
        expected_key_aggregate_metrics = {
            "mean/reward": 1.0,
            "macro_average": 1.0,
            "airline/num_samples_unique": 1,
            "telecom/num_samples_unique": 3,
            "retail/num_samples_unique": 1,
            "airline/num_samples_total": 1,
            "telecom/num_samples_total": 3,
            "retail/num_samples_total": 1,
            "airline/reward": 1.0,
            "telecom/reward": 1.0,
            "retail/reward": 1.0,
            "telecom/service_issue/reward": 1.0,
            "airline/trajectory_termination_reason/user_stop/count": 1,
            "telecom/trajectory_termination_reason/user_stop/count": 3,
            "retail/trajectory_termination_reason/user_stop/count": 1,
            "trajectory_termination_reason/user_stop/count": 5,
            "trajectory_termination_reason/user_stop/pct": 1.0,
            "message_finish_reason/tool_calls/count": 20,
            "message_finish_reason/stop/count": 5,
            "message_finish_reason/tool_calls/pct": 0.8,
            "message_finish_reason/stop/pct": 0.2,
            "trajectory_transfer_to_human_agents/count": 4,
            "trajectory_transfer_to_human_agents/pct": 0.8,
            "tool_call_hallucination/count/total": 0,
            "trajectory_missing_tool_call/count": 0,
            "trajectory_missing_tool_call/pct": 0.0,
            "messages_with_incomplete_reasoning/count": 0,
            "messages_with_incomplete_reasoning/pct": 0.0,
        }
        assert expected_key_aggregate_metrics == actual_aggregate_metrics.key_metrics

    async def test_compute_metrics_counts_max_agent_steps_termination(self) -> None:
        example_rollouts_fpath = Path(__file__).parent.parent / "data" / "example_rollouts.jsonl"
        with example_rollouts_fpath.open() as f:
            rollouts = list(map(json.loads, f))

        max_agent_steps_rollout = deepcopy(rollouts[0])
        max_agent_steps_rollout["result"]["termination_reason"] = "max_agent_steps"
        max_agent_steps_rollout["reward"] = 0.0

        _, server = self._dummy_server()

        actual_metrics = server.compute_metrics([[max_agent_steps_rollout]])

        assert actual_metrics["trajectory_termination_reason/max_agent_steps/count"] == 1
        assert actual_metrics["trajectory_termination_reason/max_agent_steps/pct"] == 1.0
