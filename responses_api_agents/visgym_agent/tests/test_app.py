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
from responses_api_agents.visgym_agent.app import (
    BOXED_PATTERN,
    ModelServerRef,
    ResourcesServerRef,
    TextActionAgent,
    TextActionAgentConfig,
    TextActionAgentRunRequest,
)


def _make_config(**overrides) -> TextActionAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        model_server=ModelServerRef(type="responses_api_models", name="my model name"),
        resources_server=ResourcesServerRef(type="resources_servers", name="my resources name"),
    )
    base.update(overrides)
    return TextActionAgentConfig(**base)


def _make_agent(**overrides) -> TextActionAgent:
    config = _make_config(**overrides)
    return TextActionAgent(config=config, server_client=MagicMock(spec=ServerClient))


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


def _model_response(
    text: str,
    resp_id: str = "resp_1",
    incomplete_reason: str | None = None,
) -> dict:
    return {
        "id": resp_id,
        "created_at": 1753983920.0,
        "model": "dummy_model",
        "object": "response",
        "output": [_assistant_message(text)],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "incomplete_details": ({"reason": incomplete_reason} if incomplete_reason is not None else None),
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
        assert TextActionAgent._extract_boxed(text) == "[up]"

    def test_returns_none_if_missing(self) -> None:
        assert TextActionAgent._extract_boxed("nothing here") is None

    def test_handles_whitespace(self) -> None:
        assert TextActionAgent._extract_boxed("foo \\boxed{   [up]   } bar") == "[up]"

    def test_empty_box_returns_none(self) -> None:
        assert TextActionAgent._extract_boxed("\\boxed{}") is None
        assert TextActionAgent._extract_boxed("\\boxed{   }") is None

    def test_multiline_capture(self) -> None:
        text = "before \\boxed{0 0 1\n1 0 0\n0 1 0} after"
        assert TextActionAgent._extract_boxed(text) == "0 0 1\n1 0 0\n0 1 0"

    def test_pattern_constant_matches_string_match(self) -> None:
        # Doc 2 R7: BOXED_PATTERN must stay byte-equal with string_match's copy.
        # If string_match's pattern changes, this test must be updated alongside it.
        assert BOXED_PATTERN.pattern == r"\\boxed\{\s*(.*?)\s*\}"


class TestExtractAction:
    maze_action_regex = r"^\('(?:move|stop)',\s*(?:[0-3]|'stop')\)$"

    def test_boxed_action_still_takes_precedence(self) -> None:
        assert (
            TextActionAgent._extract_action("reasoning \\boxed{('move', 3)}", self.maze_action_regex) == "('move', 3)"
        )

    def test_accepts_exact_unboxed_action_when_configured(self) -> None:
        assert TextActionAgent._extract_action("  ('move', 0)\n", self.maze_action_regex) == "('move', 0)"

    def test_accepts_observed_unboxed_action_with_terminal_chat_token(self) -> None:
        assert TextActionAgent._extract_action(" ('move', 3)<|im_end|>\n", self.maze_action_regex) == "('move', 3)"

    def test_rejects_unboxed_action_by_default(self) -> None:
        assert TextActionAgent._extract_action("('move', 0)") is None

    def test_rejects_prose_around_unboxed_action(self) -> None:
        assert TextActionAgent._extract_action("I choose ('move', 0)", self.maze_action_regex) is None

    def test_rejects_action_outside_grammar(self) -> None:
        assert TextActionAgent._extract_action("('move', 4)", self.maze_action_regex) is None


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
        assert TextActionAgent._extract_assistant_text([msg]) == "first\nsecond"

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
        assert TextActionAgent._extract_assistant_text([user_msg, assistant_msg]) == "world"


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

        request = TextActionAgentRunRequest(
            task_idx=0,
            env_id="maze_2d/easy",
            env_kwargs={"size": 4},
            seed=1234,
            task_id="maze_seed1234_train",
            init_state={"agent": [1, 1]},
            seed_key="level_seed",
            prompt_kwargs={"include_action_space": True},
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        result = await agent.responses(request)

        assert result.env_id == env_id
        assert result.group_id == "0"
        assert result.contains_transitions is False

        # seed_obs carries the initial observation separately from output
        # (Option B per seed-obs-persistence-problem.md).
        assert result.seed_obs is not None
        assert len(result.seed_obs) == 1
        assert result.seed_obs[0].content == "Initial obs"

        # output must NOT start with the raw seed observation — that would
        # crash NeMo-RL's tokenized message-log flattening (KeyError: 'token_ids').
        assert len(result.output) == 2  # assistant + step_obs
        assert result.output[0].role == "assistant"
        assert result.output[1].role == "user"
        assert result.output[1].content == "Next obs"

        calls = agent.server_client.post.await_args_list
        assert len(calls) == 4
        assert calls[0][1]["server_name"] == "my resources name"
        assert calls[0][1]["url_path"] == "/seed_session"
        assert calls[0][1]["json"]["task_idx"] == 0
        assert calls[0][1]["json"]["task_row"]["env_id"] == "maze_2d/easy"
        assert calls[0][1]["json"]["task_row"]["seed"] == 1234
        assert calls[0][1]["json"]["task_row"]["init_state"] == {"agent": [1, 1]}
        assert calls[0][1]["json"]["task_row"]["seed_key"] == "level_seed"
        assert calls[0][1]["json"]["task_row"]["prompt_kwargs"] == {"include_action_space": True}
        assert calls[0][1]["json"]["task_row"]["responses_create_params"]["input"] == []
        assert calls[1][1]["server_name"] == "my model name"
        assert calls[1][1]["url_path"] == "/v1/responses"
        # Step body MUST contain action_string with the extracted boxed answer
        # and MUST NOT carry tool_calls (Path B contract).
        assert calls[2][1]["server_name"] == "my resources name"
        assert calls[2][1]["url_path"] == "/step"
        assert calls[2][1]["json"] == {"env_id": env_id, "action_string": "step"}
        assert calls[3] == call(server_name="my resources name", url_path="/close", json={"env_id": env_id})

    async def test_two_turn_history_preserves_training_token_ids(self) -> None:
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
        second_response["output"][0].update(
            {
                "prompt_token_ids": [10, 11, 20, 21, 30],
                "generation_token_ids": [40],
                "generation_log_probs": [-0.3],
            }
        )

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

        result = await agent.responses(
            TextActionAgentRunRequest(
                task_idx=0,
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            )
        )

        calls = agent.server_client.post.await_args_list
        second_model_input = calls[3][1]["json"].input
        prior_assistant = next(item for item in second_model_input if hasattr(item, "prompt_token_ids"))
        assert prior_assistant.prompt_token_ids == [10, 11]
        assert prior_assistant.generation_token_ids == [20, 21]
        second_model_wire = calls[3][1]["json"].model_dump(exclude_unset=True, mode="json")
        prior_assistant_wire = next(item for item in second_model_wire["input"] if "prompt_token_ids" in item)
        assert prior_assistant_wire["prompt_token_ids"] == [10, 11]
        assert prior_assistant_wire["generation_token_ids"] == [20, 21]
        assert [item.role for item in result.output] == [
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert result.output[0].prompt_token_ids == [10, 11]
        assert result.output[2].prompt_token_ids == [10, 11, 20, 21, 30]


class TestNoBoxedRecovery:
    async def test_no_boxed_recovery_does_not_call_step(self) -> None:
        agent = _make_agent(max_steps=2, max_no_boxed_truncation_retries=1)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        bad_response = _model_response(
            "I do not know what to do.",
            resp_id="resp_1",
            incomplete_reason="max_output_tokens",
        )
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

        request = TextActionAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        await agent.responses(request)

        calls = agent.server_client.post.await_args_list
        urls = [c[1]["url_path"] for c in calls]
        # /step must be called exactly ONCE (after the second, valid response).
        assert urls == [
            "/seed_session",
            "/v1/responses",
            "/v1/responses",
            "/step",
            "/close",
        ]

        # The recovery user message must be present in the model state on the
        # second /v1/responses call (so the model can see it).
        second_model_call = calls[2]
        second_input = second_model_call[1]["json"].input
        recovery_present = any(
            isinstance(m, NeMoGymEasyInputMessage)
            and m.role == "user"
            and "produced no \\boxed{...} action" in str(m.content)
            for m in second_input
        )
        assert recovery_present, "Expected the no-boxed-answer recovery message to be in agent state"
        assert calls[1][1]["json"].max_output_tokens is None
        assert calls[2][1]["json"].max_output_tokens is None

    async def test_truncation_retry_cap_terminates_same_state(self) -> None:
        agent = _make_agent(max_steps=4, max_no_boxed_truncation_retries=2)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        miss_1 = _model_response(
            "still thinking",
            resp_id="resp_1",
            incomplete_reason="max_output_tokens",
        )
        miss_2 = _model_response(
            "still thinking more",
            resp_id="resp_2",
            incomplete_reason="max_output_tokens",
        )
        miss_3 = _model_response(
            "still no box",
            resp_id="resp_3",
            incomplete_reason="max_output_tokens",
        )
        close_data = {"message": "ok", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [seed, miss_1, miss_2, miss_3, close_data]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = TextActionAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=100),
        )
        result = await agent.responses(request)

        urls = [c[1]["url_path"] for c in agent.server_client.post.await_args_list]
        assert urls == [
            "/seed_session",
            "/v1/responses",
            "/v1/responses",
            "/v1/responses",
            "/close",
        ]
        model_calls = [c for c in agent.server_client.post.await_args_list if c[1]["url_path"] == "/v1/responses"]
        assert [c[1]["json"].max_output_tokens for c in model_calls] == [100, 200, 400]
        assert result.metadata["termination_reason"] == "no_boxed_retry_cap"
        assert result.metadata["no_boxed_truncation_retries"] == "2"

    async def test_real_env_transition_resets_truncation_budget(self) -> None:
        agent = _make_agent(max_steps=4, max_no_boxed_truncation_retries=2)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        truncated_miss = _model_response(
            "long answer",
            resp_id="resp_1",
            incomplete_reason="max_output_tokens",
        )
        good_response = _model_response("\\boxed{step}", resp_id="resp_2")
        good_step = _step(obs_text="Next obs", done=False)
        final_response = _model_response("\\boxed{finish}", resp_id="resp_3")
        final_step = _step(obs_text="Done", done=True)
        close_data = {"message": "ok", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            seed,
            truncated_miss,
            good_response,
            good_step,
            final_response,
            final_step,
            close_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = TextActionAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=100),
        )
        await agent.responses(request)

        model_calls = [c for c in agent.server_client.post.await_args_list if c[1]["url_path"] == "/v1/responses"]
        assert [c[1]["json"].max_output_tokens for c in model_calls] == [100, 200, 100]

    async def test_invalid_env_action_does_not_reset_truncation_budget(self) -> None:
        agent = _make_agent(max_steps=4, max_no_boxed_truncation_retries=2)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        truncated_miss = _model_response(
            "long answer",
            resp_id="resp_1",
            incomplete_reason="max_output_tokens",
        )
        boxed_invalid = _model_response("\\boxed{up}", resp_id="resp_2")
        invalid_step = _step(obs_text="Invalid action 'up': try again", done=False)
        boxed_valid = _model_response("\\boxed{down}", resp_id="resp_3")
        valid_step = _step(obs_text="Moved", done=True)
        close_data = {"message": "ok", "success": True}

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            seed,
            truncated_miss,
            boxed_invalid,
            invalid_step,
            boxed_valid,
            valid_step,
            close_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = TextActionAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=100),
        )
        await agent.responses(request)

        model_calls = [c for c in agent.server_client.post.await_args_list if c[1]["url_path"] == "/v1/responses"]
        assert [c[1]["json"].max_output_tokens for c in model_calls] == [100, 200, 200]


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

        request = TextActionAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        await agent.responses(request)

        urls = [c[1]["url_path"] for c in agent.server_client.post.await_args_list]
        # No /step call; agent went straight to /close after one model response.
        assert urls == ["/seed_session", "/v1/responses", "/close"]

    def test_done_if_no_boxed_answer_true_rejects_retry_cap(self) -> None:
        with pytest.raises(ValueError, match="incompatible"):
            _make_agent(
                done_if_no_boxed_answer=True,
                max_no_boxed_truncation_retries=1,
            )


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

        request = TextActionAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        result = await agent.responses(request)

        # seed_obs carries the initial observation + rules reminder (post-injection).
        assert result.seed_obs is not None
        seed_obs_contents = [
            s.get("content") if isinstance(s, dict) else getattr(s, "content", None) for s in result.seed_obs
        ]
        assert "Seed obs" in seed_obs_contents
        assert "REMINDER!" in seed_obs_contents

        # output must NOT contain the raw seed observation.
        output_contents = [getattr(m, "content", None) for m in result.output]
        assert "Seed obs" not in output_contents

        # The second /v1/responses call (calls[3]) must include a "REMINDER!"
        # user message (the rules summary) prepended ahead of the env obs.
        calls = agent.server_client.post.await_args_list
        second_model_call_input = calls[3][1]["json"].input
        contents = [getattr(m, "content", None) for m in second_model_call_input]
        # The reminder must appear BEFORE the env obs from step 1.
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

        request = TextActionAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        result = await agent.responses(request)

        contents = [getattr(m, "content", None) for m in result.output]
        assert "Env obs" in contents
        # Default template must NOT be injected when re_emit_rules_each_turn=False.
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

        request = TextActionAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        await agent.responses(request)

        urls = [c[1]["url_path"] for c in agent.server_client.post.await_args_list]
        assert urls[-1] == "/close"

    async def test_close_called_on_exception(self) -> None:
        # Simulate the model server raising mid-rollout; /close MUST still fire.
        import aiohttp

        agent = _make_agent(max_steps=5)
        env_id = str(uuid.uuid4())

        seed = _seed_session(env_id=env_id)
        close_data = {"message": "ok", "success": True}

        # First post -> seed_session (success); second -> model server (raises);
        # third -> close (success).
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

        request = TextActionAgentRunRequest(
            task_idx=0,
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )
        # The agent catches ClientResponseError internally and breaks; the run
        # should NOT raise, but the rollout should still call /close. Since the
        # rollout has no successful first transition we expect the trailing
        # assert to fire: confirm this is the path.
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
        # Note: /close result.json() is never read by the agent (Aviary parity),
        # so close_data is intentionally omitted from the side_effect list.
        verify_data = {
            "reward": 1.0,
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
            verify_data,
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        request = TextActionAgentRunRequest(
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


class TestModelRequestSerialization:
    """The model request must keep the token metadata that makes replay on-policy."""

    async def test_observations_do_not_strip_token_ids_from_the_wire_body(self) -> None:
        # VisGymEnvStateEasyInputMessage is not a member of NeMo-Gym's request
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
            TextActionAgentRunRequest(
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


class TestNeMoRLResultContract:
    """The /run result has to carry what NeMo-RL reads off a rollout."""

    async def test_run_result_carries_the_fields_nemo_rl_reads(self) -> None:
        # NeMo-RL's postprocessing indexes these directly:
        #   nemo_gym_result["responses_create_params"]["input"] (rebuilds the
        #   initial prompt), nemo_gym_result["response"]["output"] (the trained
        #   turns) and full_result["reward"] (the scalar that becomes
        #   total_reward). A missing key is not caught until postprocessing, by
        #   which point the episode has already been played on real GPUs.
        agent = _make_agent(max_steps=1)
        env_id = str(uuid.uuid4())

        dotjson_mock = AsyncMock()
        dotjson_mock.json.side_effect = [
            _seed_session(env_id=env_id),
            _model_response("\\boxed{step}"),
            _step(done=True, reward=1.0),
            {
                "reward": 1.0,
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
            },
        ]
        dotjson_mock.raise_for_status = MagicMock()
        dotjson_mock.cookies = None
        agent.server_client.post = AsyncMock(return_value=dotjson_mock)

        result = await agent.run(
            TextActionAgentRunRequest(
                task_idx=0,
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            )
        )

        wire = result.model_dump(mode="json")
        assert {"reward", "responses_create_params", "response"} <= set(wire)
        assert "input" in wire["responses_create_params"]
        assert "output" in wire["response"]
