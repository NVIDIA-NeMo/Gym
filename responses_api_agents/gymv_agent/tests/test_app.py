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
import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, call, patch

import anyio
import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymEasyInputMessage,
    NeMoGymResponseCreateParamsNonStreaming,
)
from resources_servers.gym_v.schemas import GymVNeMoGymResponse
from responses_api_agents.gymv_agent import app as gymv_agent_app
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
        from nemo_gym.core import NeMoGymResponseOutputMessage

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
        from nemo_gym.core import (
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


class TestTaskIdentity:
    def test_accepts_canonical_collated_index_without_legacy_task_idx(self) -> None:
        request = GymVAgentRunRequest.model_validate(
            {
                "_ng_task_index": 137,
                "env_id": "Games/FrozenLake-v0",
                "seed": 1234,
                "responses_create_params": {"input": []},
            }
        )

        assert request.task_idx is None
        assert request.group_task_index == 137
        assert GymVAgent._task_row_from_request(request) is not None

    def test_canonical_index_wins_over_legacy_local_index_for_grouping(self) -> None:
        request = GymVAgentRunRequest.model_validate(
            {
                "task_idx": 0,
                "_ng_task_index": 137,
                "responses_create_params": {"input": []},
            }
        )

        assert request.task_idx == 0
        assert request.group_task_index == 137

    def test_requires_canonical_or_legacy_index(self) -> None:
        with pytest.raises(ValueError, match="_ng_task_index"):
            GymVAgentRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]))


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

        # seed_obs carries the initial observation separately from output.
        assert result.seed_obs is not None
        assert len(result.seed_obs) == 1
        assert result.seed_obs[0].content == "Initial obs"

        # output contains the assistant turn but not the separately returned
        # seed observation. The terminal /step observation is also omitted.
        assert len(result.output) == 1
        assert result.output[0].role == "assistant"

        calls = agent.server_client.post.await_args_list
        assert len(calls) == 4
        assert calls[0][1]["server_name"] == "my resources name"
        assert calls[0][1]["url_path"] == "/seed_session"
        assert "task_idx" not in calls[0][1]["json"]
        assert calls[0][1]["json"]["task_row"]["env_id"] == "Games/FrozenLake-v0"
        assert calls[0][1]["json"]["task_row"]["seed"] == 1234
        assert calls[0][1]["json"]["task_row"]["responses_create_params"]["input"] == []
        assert calls[1][1]["server_name"] == "my model name"
        assert calls[1][1]["url_path"] == "/v1/responses"
        assert calls[2][1]["server_name"] == "my resources name"
        assert calls[2][1]["url_path"] == "/step"
        assert calls[2][1]["json"] == {"env_id": env_id, "action_string": "step"}
        assert calls[3] == call(
            server_name="my resources name",
            url_path="/close",
            json={"env_id": env_id},
            _max_num_tries=3,
        )
        assert calls[0][1]["_max_num_tries"] == 1
        assert calls[2][1]["_max_num_tries"] == 1

    async def test_canonical_only_row_routes_inline_and_sets_group_id(self) -> None:
        agent = _make_agent(max_steps=1)
        env_id = str(uuid.uuid4())
        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            _seed_session(env_id=env_id),
            _model_response("\\boxed{step}"),
            _step(done=True),
            {"message": "ok", "success": True},
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = GymVAgentRunRequest.model_validate(
            {
                "_ng_task_index": 137,
                "env_id": "Games/FrozenLake-v0",
                "seed": 1234,
                "responses_create_params": {"input": []},
            }
        )
        result = await agent.responses(request)

        assert result.group_id == "137"
        seed_payload = agent.server_client.post.await_args_list[0][1]["json"]
        assert "task_idx" not in seed_payload
        assert seed_payload["task_row"]["env_id"] == "Games/FrozenLake-v0"


class TestMultiTurnTokenReplay:
    async def test_second_turn_replays_complete_token_metadata(self) -> None:
        agent = _make_agent(max_steps=2)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        first_response = _model_response("\\boxed{a}", resp_id="r1")
        first_response["output"][0] |= {
            "prompt_token_ids": [1, 2],
            "generation_token_ids": [3],
            "generation_log_probs": [-0.1],
        }
        first_step = _step(obs_text="Next obs", done=False)
        second_response = _model_response("\\boxed{b}", resp_id="r2")
        second_response["output"][0] |= {
            "prompt_token_ids": [1, 2, 3, 4],
            "generation_token_ids": [5],
            "generation_log_probs": [-0.2],
        }
        second_step = _step(done=True)
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

        second_request = agent.server_client.post.await_args_list[3][1]["json"]
        second_input = second_request.input
        replayed = next(item for item in second_input if getattr(item, "role", None) == "assistant")
        assert replayed.prompt_token_ids == [1, 2]
        assert replayed.generation_token_ids == [3]
        assert replayed.generation_log_probs == [-0.1]
        assert result.output[0].prompt_token_ids == [1, 2]

        serialized_input = second_request.model_dump(exclude_none=True)["input"]
        serialized_replay = next(item for item in serialized_input if item.get("role") == "assistant")
        assert serialized_replay["prompt_token_ids"] == [1, 2]
        assert serialized_replay["generation_token_ids"] == [3]
        assert serialized_replay["generation_log_probs"] == [-0.1]


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
        result = await agent.responses(request)

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

        # The second successful model call covered the recovery text in its
        # prompt, so the flat token-replay output must retain it between the
        # two assistant turns.
        output_recovery_present = any(
            getattr(m, "role", None) == "user" and "did not find a \\boxed{...}" in str(getattr(m, "content", ""))
            for m in result.output
        )
        assert output_recovery_present


class TestTokenCoveredOutputBoundary:
    async def test_max_steps_omits_unseen_trailing_observation(self) -> None:
        agent = _make_agent(max_steps=1)
        env_id = str(uuid.uuid4())

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            _seed_session(env_id=env_id),
            _model_response("\\boxed{step}"),
            _step(obs_text="Unseen trailing obs", done=False),
            {"message": "ok", "success": True},
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        result = await agent.responses(
            GymVAgentRunRequest(
                task_idx=0,
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            )
        )

        assert [item.role for item in result.output] == ["assistant"]
        assert all(getattr(item, "content", None) != "Unseen trailing obs" for item in result.output)

    async def test_failed_next_model_call_omits_pending_observation(self) -> None:
        import aiohttp

        agent = _make_agent(max_steps=2)
        env_id = str(uuid.uuid4())

        seed_mock = AsyncMock()
        seed_mock.json = AsyncMock(return_value=_seed_session(env_id=env_id))
        seed_mock.raise_for_status = MagicMock()

        first_model_mock = AsyncMock()
        first_model_mock.json = AsyncMock(return_value=_model_response("\\boxed{step}"))
        first_model_mock.raise_for_status = MagicMock()
        first_model_mock.cookies = None

        step_mock = AsyncMock()
        step_mock.json = AsyncMock(return_value=_step(obs_text="Uncovered obs", done=False))

        failed_model_mock = AsyncMock()
        failed_model_mock.raise_for_status = MagicMock(
            side_effect=aiohttp.ClientResponseError(
                request_info=MagicMock(), history=(), status=500, message="context full"
            )
        )
        failed_model_mock.text = "context full"

        close_mock = AsyncMock()
        close_mock.json = AsyncMock(return_value={"message": "ok", "success": True})
        agent.server_client.post = AsyncMock(
            side_effect=[seed_mock, first_model_mock, step_mock, failed_model_mock, close_mock]
        )

        result = await agent.responses(
            GymVAgentRunRequest(
                task_idx=0,
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            )
        )

        assert [item.role for item in result.output] == ["assistant"]
        assert all(getattr(item, "content", None) != "Uncovered obs" for item in result.output)


class TestDoneIfNoBoxedAnswer:
    async def test_task_horizon_caps_no_boxed_recovery_turns(self) -> None:
        agent = _make_agent(max_steps=64)
        env_id = str(uuid.uuid4())

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            _seed_session(env_id=env_id),
            _model_response("missing action", resp_id="r1"),
            _model_response("still missing action", resp_id="r2"),
            {"message": "ok", "success": True},
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        result = await agent.responses(
            GymVAgentRunRequest(
                task_idx=0,
                env_id="Games/FrozenLake-v0",
                env_kwargs={"size": 4},
                seed=1234,
                horizon_cap=2,
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            )
        )

        urls = [c[1]["url_path"] for c in agent.server_client.post.await_args_list]
        assert urls == [
            "/seed_session",
            "/v1/responses",
            "/v1/responses",
            "/close",
        ]
        assert [item.role for item in result.output] == [
            "assistant",
            "user",
            "assistant",
        ]

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
        assert "Seed obs" not in output_contents
        # The seed reminder is returned via seed_obs; the reminder following
        # the first non-terminal step remains in the flat model-visible log.
        assert output_contents.count("REMINDER!") == 1

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
        # Terminal observations are omitted because no subsequent model call
        # saw them.
        assert "Env obs" not in contents
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


class TestCloseSemantics:
    async def test_retries_false_and_malformed_responses_until_success(self) -> None:
        agent = _make_agent()
        false_response = AsyncMock(ok=True)
        false_response.json.return_value = {"success": False, "message": "native close failed"}
        malformed_response = AsyncMock(ok=True)
        malformed_response.json.return_value = {"message": "missing success"}
        success_response = AsyncMock(ok=True)
        success_response.json.return_value = {"success": True, "message": "ok"}
        agent.server_client.post = AsyncMock(side_effect=[false_response, malformed_response, success_response])

        await agent._close_session("retry-session", discard_reward=True)

        assert agent.server_client.post.await_count == 3
        assert all(item.kwargs["_max_num_tries"] == 3 for item in agent.server_client.post.await_args_list)

    async def test_persistent_close_failure_is_bounded(self) -> None:
        agent = _make_agent()
        false_response = AsyncMock(ok=True)
        false_response.json.return_value = {"success": False, "message": "still open"}
        agent.server_client.post = AsyncMock(return_value=false_response)

        await agent._close_session("bounded-session", discard_reward=False)

        assert agent.server_client.post.await_count == 3

    async def test_raw_asyncio_cancellation_waits_for_cleanup_request(self) -> None:
        agent = _make_agent()
        post_started = asyncio.Event()
        release_post = asyncio.Event()
        response = AsyncMock(ok=True)
        response.json.return_value = {"success": True, "message": "ok"}

        async def blocked_post(**_kwargs: object) -> AsyncMock:
            post_started.set()
            await release_post.wait()
            return response

        agent.server_client.post = AsyncMock(side_effect=blocked_post)
        close_task = asyncio.create_task(agent._close_session("cancel-session", discard_reward=True))
        await post_started.wait()

        close_task.cancel()
        await asyncio.sleep(0)
        assert close_task.done() is False

        release_post.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task
        assert agent.server_client.post.await_count == 1

    async def test_anyio_cancel_scope_waits_for_cleanup_request(self) -> None:
        agent = _make_agent()
        post_started = asyncio.Event()
        release_post = asyncio.Event()
        cleanup_finished = asyncio.Event()
        response = AsyncMock(ok=True)
        response.json.return_value = {"success": True, "message": "ok"}

        async def blocked_post(**_kwargs: object) -> AsyncMock:
            post_started.set()
            await release_post.wait()
            cleanup_finished.set()
            return response

        agent.server_client.post = AsyncMock(side_effect=blocked_post)
        cancel_scope = anyio.CancelScope()

        async def cancel_then_release() -> None:
            await post_started.wait()
            cancel_scope.cancel()
            await asyncio.sleep(0)
            release_post.set()

        driver = asyncio.create_task(cancel_then_release())
        with cancel_scope:
            await agent._close_session("cancel-scope-session", discard_reward=True)
        await driver

        assert cancel_scope.cancel_called is True
        assert cleanup_finished.is_set()
        assert agent.server_client.post.await_count == 1


class TestRunWorkflow:
    @pytest.mark.parametrize("masked", [False, True])
    async def test_run_workflow_calls_verify(self, masked: bool) -> None:
        agent = _make_agent(max_steps=1)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        response = _model_response("\\boxed{step}")
        step_data = _step(done=True, reward=1.0)
        verify_data = {
            "reward": 0.0 if masked else 1.0,
            "mask_sample": masked,
            "failure_kind": "gym_v:simulator_failed" if masked else None,
            "failure_reason": "provider failed" if masked else None,
            "responses_create_params": {"input": []},
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
            {"message": "ok", "success": True},
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

        assert verify_response.reward == (0.0 if masked else 1.0)
        assert verify_response.mask_sample is masked
        assert verify_response.failure_kind == verify_data["failure_kind"]

        urls = [c[1]["url_path"] for c in agent.server_client.post.await_args_list]
        assert urls == [
            "/seed_session",
            "/v1/responses",
            "/step",
            "/close",
            "/verify",
        ]
        assert agent.server_client.post.await_args_list[-1][1]["json"]["responses_create_params"]["input"] == []
        assert agent.server_client.post.await_args_list[-1][1]["_max_num_tries"] == 1

    async def test_verify_failure_discards_preserved_reward(self) -> None:
        agent = _make_agent(max_steps=1)
        env_id = str(uuid.uuid4())

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            _seed_session(env_id=env_id),
            _model_response("\\boxed{step}"),
            _step(done=True, reward=1.0),
            {"message": "ok", "success": True},
            ValueError("invalid verify response"),
            {"message": "already closed", "success": True},
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        with pytest.raises(ValueError, match="invalid verify response"):
            await agent.run(
                GymVAgentRunRequest(
                    task_idx=0,
                    responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                )
            )

        close_calls = [c for c in agent.server_client.post.await_args_list if c[1]["url_path"] == "/close"]
        assert close_calls == [
            call(
                server_name="my resources name",
                url_path="/close",
                json={"env_id": env_id},
                _max_num_tries=3,
            ),
            call(
                server_name="my resources name",
                url_path="/close",
                json={"env_id": env_id, "discard_reward": True},
                _max_num_tries=3,
            ),
        ]

    async def test_cancellation_during_verify_aborts_preserved_reward(self) -> None:
        agent = _make_agent(max_steps=1)
        env_id = str(uuid.uuid4())
        episode_response = GymVNeMoGymResponse.model_validate(
            {
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
            }
        )
        verify_started = asyncio.Event()
        close_response = AsyncMock(ok=True)
        close_response.json.return_value = {"success": True, "message": "ok"}

        async def post(**kwargs: object) -> AsyncMock:
            if kwargs["url_path"] == "/verify":
                verify_started.set()
                await asyncio.Future()
            return close_response

        agent.server_client.post = AsyncMock(side_effect=post)
        with patch.object(GymVAgent, "responses", AsyncMock(return_value=episode_response)):
            run_task = asyncio.create_task(
                agent.run(
                    GymVAgentRunRequest(
                        task_idx=0,
                        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
                    )
                )
            )
            await verify_started.wait()
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task

        close_call = agent.server_client.post.await_args_list[-1]
        assert close_call.kwargs["url_path"] == "/close"
        assert close_call.kwargs["json"] == {"env_id": env_id, "discard_reward": True}


class TestSeedMutationDoesNotRetry:
    async def test_unusable_seed_response_is_closed_without_retrying(self) -> None:
        agent = _make_agent(max_steps=1)
        requested_env_id = str(uuid.uuid4())

        seed_response = AsyncMock()
        seed_response.json.return_value = {"env_id": requested_env_id, "obs": [], "tools": []}
        seed_response.ok = True
        close_response = AsyncMock()
        close_response.json.return_value = {"message": "ok", "success": True}
        close_response.ok = True
        agent.server_client.post = AsyncMock(side_effect=[seed_response, close_response])

        with (
            patch.object(gymv_agent_app.uuid, "uuid4", return_value=uuid.UUID(requested_env_id)),
            pytest.raises(ValueError, match="No observations"),
        ):
            await agent._seed_session(task_idx=0, task_row=None)

        calls = agent.server_client.post.await_args_list
        assert [item[1]["url_path"] for item in calls] == ["/seed_session", "/close"]
        assert calls[0][1]["json"] == {"env_id": requested_env_id, "task_idx": 0}
        assert calls[0][1]["_max_num_tries"] == 1
        assert calls[1][1]["json"] == {"env_id": requested_env_id, "discard_reward": True}
        assert calls[1][1]["_max_num_tries"] == 3

    async def test_seed_collision_does_not_close_preexisting_session(self) -> None:
        import aiohttp

        agent = _make_agent(max_steps=1)
        collision = aiohttp.ClientResponseError(
            request_info=MagicMock(),
            history=(),
            status=409,
            message="env_id already in use",
        )
        agent.server_client.post = AsyncMock(return_value=AsyncMock())

        with (
            patch.object(gymv_agent_app, "raise_for_status", AsyncMock(side_effect=collision)),
            pytest.raises(aiohttp.ClientResponseError) as exc_info,
        ):
            await agent._seed_session(task_idx=0, task_row=None)

        assert exc_info.value.status == 409
        assert [item[1]["url_path"] for item in agent.server_client.post.await_args_list] == ["/seed_session"]


class TestSanity:
    def test_construct_with_minimal_config(self) -> None:
        agent = _make_agent()
        assert agent.config.done_if_no_boxed_answer is False
        assert agent.config.re_emit_rules_each_turn is False
        assert agent.config.return_transitions is False

    def test_return_transitions_true_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_config(return_transitions=True)

    def test_nonpositive_max_steps_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            _make_config(max_steps=0)


class TestModelRequestSerialization:
    """The model request must keep the token metadata that makes replay on-policy."""

    async def test_observations_do_not_strip_token_ids_from_the_wire_body(self) -> None:
        # GymVEnvStateEasyInputMessage is not a member of NeMo-Gym's request
        # input union. Passing one through unconverted makes Pydantic serialize
        # every item in the list against the base union members, which drops
        # prompt_token_ids/generation_token_ids from the assistant turns without
        # raising. The model server would then have no exact prefix to replay
        # and turn 2 onwards would silently go off-policy.
        agent = _make_agent(max_steps=2)
        env_id = str(uuid.uuid4())

        first_response = _model_response("\\boxed{step}", resp_id="resp_1")
        first_response["output"][0].update(
            {
                "prompt_token_ids": [10, 11],
                "generation_token_ids": [20, 21],
                "generation_log_probs": [-0.1, -0.2],
            }
        )
        second_response = _model_response("\\boxed{finish}", resp_id="resp_2")

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            _seed_session(env_id=env_id),
            first_response,
            _step(done=False),
            second_response,
            _step(done=True),
            {"message": "ok", "success": True},
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        await agent.responses(
            GymVAgentRunRequest(
                task_idx=0,
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            )
        )

        # ServerClient posts the body as model_dump(exclude_unset=True), so
        # assert on exactly that shape rather than on the in-memory objects.
        second_request = agent.server_client.post.await_args_list[3][1]["json"]
        wire_input = second_request.model_dump(exclude_unset=True)["input"]
        assistant_items = [item for item in wire_input if "prompt_token_ids" in item]
        assert len(assistant_items) == 1
        assert assistant_items[0]["prompt_token_ids"] == [10, 11]
        assert assistant_items[0]["generation_token_ids"] == [20, 21]
        assert assistant_items[0]["generation_log_probs"] == [-0.1, -0.2]
