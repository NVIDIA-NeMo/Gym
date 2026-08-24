# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import uuid
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.gymv_agent.app import (
    BOXED_PATTERN,
    GymVAgent,
    GymVAgentConfig,
    GymVAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
)


def _make_config(**overrides) -> GymVAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        model_server=ModelServerRef(type="responses_api_models", name="my model name"),
        resources_server=ResourcesServerRef(type="resources_servers", name="my resources name"),
    )
    base.update(overrides)
    return GymVAgentConfig(**base)


def _make_agent(**overrides) -> GymVAgent:
    config = _make_config(**overrides)
    return GymVAgent(config=config, server_client=MagicMock(spec=ServerClient))


def _assistant_message(text: str, msg_id: str = "msg_1") -> dict:
    return {
        "id": msg_id,
        "content": [
            {"annotations": [], "text": text, "type": "output_text"},
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _model_response(text: str, resp_id: str = "resp_1") -> dict:
    return {
        "id": resp_id,
        "created_at": 1753983920.0,
        "model": "dummy_model",
        "object": "response",
        "output": [_assistant_message(text)],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _seed_session(env_id: str | None = None, content: str = "Initial obs") -> dict:
    return {
        "env_id": env_id or str(uuid.uuid4()),
        "obs": [{"role": "user", "content": content, "env_info": None}],
        "tools": [],
    }


def _step(obs_text: str = "Next obs", reward: float = 0.0, done: bool = False) -> dict:
    return {
        "obs": [{"role": "user", "content": obs_text, "env_info": None}],
        "reward": reward,
        "done": done,
    }


class TestExtractBoxed:
    def test_finds_last(self) -> None:
        text = "Let me think. The example would be \\boxed{example} but the real answer is \\boxed{[up]}."
        assert GymVAgent._extract_boxed(text) == "[up]"

    def test_returns_none_if_missing(self) -> None:
        assert GymVAgent._extract_boxed("nothing here") is None

    def test_handles_whitespace(self) -> None:
        assert GymVAgent._extract_boxed("foo \\boxed{   [up]   } bar") == "[up]"

    def test_empty_box_returns_none(self) -> None:
        assert GymVAgent._extract_boxed("\\boxed{}") is None
        assert GymVAgent._extract_boxed("\\boxed{   }") is None

    def test_multiline_capture(self) -> None:
        text = "before \\boxed{0 0 1\n1 0 0\n0 1 0} after"
        assert GymVAgent._extract_boxed(text) == "0 0 1\n1 0 0\n0 1 0"

    def test_pattern_constant(self) -> None:
        assert BOXED_PATTERN.pattern == r"\\boxed\{\s*(.*?)\s*\}"


class TestExtractAssistantText:
    def test_concatenates_content_parts(self) -> None:
        from nemo_gym.openai_utils import NeMoGymResponseOutputMessage

        msg = NeMoGymResponseOutputMessage.model_validate(
            {
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "content": [
                    {"annotations": [], "text": "first", "type": "output_text"},
                    {"annotations": [], "text": "second", "type": "output_text"},
                ],
            }
        )
        assert GymVAgent._extract_assistant_text([msg]) == "first\nsecond"

    def test_skips_non_assistant_messages(self) -> None:
        from nemo_gym.openai_utils import (
            NeMoGymEasyInputMessage,
            NeMoGymResponseOutputMessage,
        )

        user_msg = NeMoGymEasyInputMessage(role="user", content="hello")
        assistant_msg = NeMoGymResponseOutputMessage.model_validate(
            {
                "id": "m1",
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "content": [{"annotations": [], "text": "world", "type": "output_text"}],
            }
        )
        assert GymVAgent._extract_assistant_text([user_msg, assistant_msg]) == "world"


class TestLifecycleHappyPath:
    async def test_lifecycle_happy_path(self) -> None:
        agent = _make_agent(max_steps=1)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        response = _model_response("Reasoning... \\boxed{step}")
        step_data = _step(done=True)
        close_data = {"message": "ok", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [seed, response, step_data, close_data]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = GymVAgentRunRequest(
            task_idx=0,
            env_id="Games/FrozenLake-v0",
            env_kwargs={"size": 4},
            seed=1234,
            task_id="frozenlake_4x4_h3_seed1234_train",
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        result = await agent.responses(request)

        assert result.env_id == env_id
        assert result.group_id == "0"
        assert result.contains_transitions is False

        # seed_obs mirror carries the initial observation.
        assert result.seed_obs is not None
        assert len(result.seed_obs) == 1
        assert result.seed_obs[0].content == "Initial obs"

        # output begins with the seed observation, then the assistant turn.
        # The terminal /step observation is omitted because the model never saw it.
        assert len(result.output) == 2  # seed_obs + assistant
        assert result.output[0].role == "user"
        assert result.output[0].content == "Initial obs"
        assert result.output[1].role == "assistant"

        calls = agent.server_client.post.await_args_list
        assert len(calls) == 4
        assert calls[0][1]["server_name"] == "my resources name"
        assert calls[0][1]["url_path"] == "/seed_session"
        assert calls[0][1]["json"]["task_idx"] == 0
        assert calls[0][1]["json"]["task_row"]["env_id"] == "Games/FrozenLake-v0"
        assert calls[0][1]["json"]["task_row"]["seed"] == 1234
        assert calls[0][1]["json"]["task_row"]["responses_create_params"]["input"] == []
        assert calls[1][1]["server_name"] == "my model name"
        assert calls[1][1]["url_path"] == "/v1/responses"
        assert calls[2][1]["server_name"] == "my resources name"
        assert calls[2][1]["url_path"] == "/step"
        assert calls[2][1]["json"] == {"env_id": env_id, "action_string": "step"}
        assert calls[3] == call(server_name="my resources name", url_path="/close", json={"env_id": env_id})


class TestNoBoxedRecovery:
    async def test_no_boxed_recovery_does_not_call_step(self) -> None:
        agent = _make_agent(max_steps=2)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        bad_response = _model_response("I do not know what to do.", resp_id="resp_1")
        good_response = _model_response("\\boxed{step}", resp_id="resp_2")
        step_data = _step(done=True)
        close_data = {"message": "ok", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            seed,
            bad_response,
            good_response,
            step_data,
            close_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = GymVAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        await agent.responses(request)

        calls = agent.server_client.post.await_args_list
        urls = [c[1]["url_path"] for c in calls]
        assert urls == [
            "/seed_session",
            "/v1/responses",
            "/v1/responses",
            "/step",
            "/close",
        ]

        second_model_call = calls[2]
        second_input = second_model_call[1]["json"].input
        recovery_present = any(
            isinstance(m, NeMoGymEasyInputMessage)
            and m.role == "user"
            and "did not find a \\boxed{...}" in str(m.content)
            for m in second_input
        )
        assert recovery_present, "Expected the no-boxed-answer recovery message to be in agent state"


class TestDoneIfNoBoxedAnswer:
    async def test_done_if_no_boxed_answer_true(self) -> None:
        agent = _make_agent(max_steps=5, done_if_no_boxed_answer=True)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        bad_response = _model_response("nothing useful")
        close_data = {"message": "ok", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [seed, bad_response, close_data]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = GymVAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        await agent.responses(request)

        urls = [c[1]["url_path"] for c in agent.server_client.post.await_args_list]
        assert urls == ["/seed_session", "/v1/responses", "/close"]


class TestReEmitRules:
    async def test_re_emit_rules_each_turn_prepends_summary(self) -> None:
        agent = _make_agent(
            max_steps=2,
            re_emit_rules_each_turn=True,
            rules_summary_template="REMINDER!",
        )
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id, content="Seed obs")
        first_response = _model_response("\\boxed{a}", resp_id="r1")
        first_step = _step(obs_text="Env obs after step 1", done=False)
        second_response = _model_response("\\boxed{b}", resp_id="r2")
        second_step = _step(obs_text="Env obs after step 2", done=True)
        close_data = {"message": "ok", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            seed,
            first_response,
            first_step,
            second_response,
            second_step,
            close_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = GymVAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        result = await agent.responses(request)

        assert result.seed_obs is not None
        seed_obs_contents = [
            s.get("content") if isinstance(s, dict) else getattr(s, "content", None) for s in result.seed_obs
        ]
        assert "Seed obs" in seed_obs_contents
        assert "REMINDER!" in seed_obs_contents

        output_contents = [getattr(m, "content", None) for m in result.output]
        assert "Seed obs" in output_contents
        assert "REMINDER!" in output_contents
        # seed_obs entries must come before the first assistant message.
        assistant_idx = next(i for i, m in enumerate(result.output) if getattr(m, "role", None) == "assistant")
        assert output_contents.index("Seed obs") < assistant_idx
        assert output_contents.index("REMINDER!") < assistant_idx

        calls = agent.server_client.post.await_args_list
        second_model_call_input = calls[3][1]["json"].input
        contents = [getattr(m, "content", None) for m in second_model_call_input]
        assert "REMINDER!" in contents
        assert "Env obs after step 1" in contents
        assert contents.index("REMINDER!") < contents.index("Env obs after step 1")

    async def test_re_emit_rules_off_by_default(self) -> None:
        agent = _make_agent(max_steps=2)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id, content="Seed obs")
        first_response = _model_response("\\boxed{a}", resp_id="r1")
        first_step = _step(obs_text="Env obs", done=True)
        close_data = {"message": "ok", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            seed,
            first_response,
            first_step,
            close_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = GymVAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        result = await agent.responses(request)

        contents = [getattr(m, "content", None) for m in result.output]
        assert "Env obs" in contents
        assert agent.config.rules_summary_template not in contents


class TestCloseLifecycle:
    async def test_close_called_on_normal_termination(self) -> None:
        agent = _make_agent(max_steps=1)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        response = _model_response("\\boxed{x}")
        step_data = _step(done=True)
        close_data = {"message": "ok", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [seed, response, step_data, close_data]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = GymVAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        await agent.responses(request)

        urls = [c[1]["url_path"] for c in agent.server_client.post.await_args_list]
        assert urls[-1] == "/close"

    async def test_close_called_on_exception(self) -> None:
        import aiohttp

        agent = _make_agent(max_steps=5)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        close_data = {"message": "ok", "success": True}

        seed_mock = AsyncMock()
        seed_mock.json = AsyncMock(return_value=seed)
        seed_mock.raise_for_status = MagicMock()
        seed_mock.cookies = None

        model_err_mock = AsyncMock()
        model_err_mock.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(request_info=MagicMock(), history=(), status=500, message="boom")
        )
        model_err_mock.text = "boom"

        close_mock = AsyncMock()
        close_mock.json = AsyncMock(return_value=close_data)
        close_mock.raise_for_status = MagicMock()
        close_mock.cookies = None

        agent.server_client.post = AsyncMock(side_effect=[seed_mock, model_err_mock, close_mock])

        request = GymVAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        with pytest.raises(AssertionError):
            await agent.responses(request)

        urls = [c[1]["url_path"] for c in agent.server_client.post.await_args_list]
        assert urls == ["/seed_session", "/v1/responses", "/close"]


class TestRunWorkflow:
    async def test_run_workflow_calls_verify(self) -> None:
        agent = _make_agent(max_steps=1)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        response = _model_response("\\boxed{step}")
        step_data = _step(done=True, reward=1.0)
        verify_data = {
            "reward": 1.0,
            "response": {
                "id": "resp_1",
                "created_at": 1753983920.0,
                "model": "dummy_model",
                "object": "response",
                "env_id": env_id,
                "group_id": "0",
                "contains_transitions": False,
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            seed,
            response,
            step_data,
            verify_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = GymVAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        verify_response = await agent.run(request)

        assert verify_response.reward == 1.0

        urls = [c[1]["url_path"] for c in agent.server_client.post.await_args_list]
        assert urls == [
            "/seed_session",
            "/v1/responses",
            "/step",
            "/close",
            "/verify",
        ]


class TestSanity:
    def test_construct_with_minimal_config(self) -> None:
        agent = _make_agent()
        assert agent.config.done_if_no_boxed_answer is False
        assert agent.config.re_emit_rules_each_turn is False
        assert agent.config.return_transitions is False
