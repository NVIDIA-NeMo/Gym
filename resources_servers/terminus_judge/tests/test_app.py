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
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pytest import approx, fixture

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.terminus_judge.app import (
    TerminusJudgeResourcesServer,
    TerminusJudgeResourcesServerConfig,
    TerminusJudgeVerifyRequest,
)


class TestApp:
    @fixture
    def config(self) -> TerminusJudgeResourcesServerConfig:
        cfg = TerminusJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template_fpath="prompt_templates/terminus_prompt.txt",
        )
        cfg.judge_equal_label = "[[A=B]]"
        cfg.judge_not_equal_label = "[[A!=B]]"
        return cfg

    def _create_response(self, id: str, output_item: NeMoGymResponseOutputItem) -> dict[str, Any]:
        return NeMoGymResponse(
            id=id,
            created_at=123.0,
            model="judge_model",
            object="response",
            output=[output_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()

    def _msg(self, text: str) -> NeMoGymResponseOutputMessage:
        return NeMoGymResponseOutputMessage(
            id="msg_id",
            content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
            role="assistant",
            status="completed",
            type="message",
        )

    async def test_verify_equal_simple(self, config: TerminusJudgeResourcesServerConfig) -> None:
        """Test basic equal verdict without swap check."""
        server_mock = MagicMock(spec=ServerClient)
        rs = TerminusJudgeResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        # Use return_value instead of side_effect for single response
        post_mock.json = AsyncMock(
            return_value=self._create_response("first", self._msg("some text [[A=B]] trailing"))
        )
        server_mock.post = AsyncMock(return_value=post_mock)

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q: 1+1?"}])
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("It is 2.")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        req = TerminusJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="2",
        )
        res = await rs.verify(req)
        assert res.reward == approx(1.0)
        assert res.expected_answer == "2"
        assert len(res.judge_evaluations) == 1
        assert res.judge_evaluations[0].verdict_label == "[[A=B]]"

    async def test_verify_equal_with_swap_check_both_equal(self, config: TerminusJudgeResourcesServerConfig) -> None:
        """Test swap check when both evaluations return equal."""
        config_twice = config.model_copy(deep=True)
        config_twice.check_twice_swap = True

        server_mock = MagicMock(spec=ServerClient)
        rs = TerminusJudgeResourcesServer(config=config_twice, server_client=server_mock)

        # Create two separate response mocks for two separate POST calls
        post_mock_1 = MagicMock()
        post_mock_1.json = AsyncMock(return_value=self._create_response("first", self._msg("[[A=B]]")))

        post_mock_2 = MagicMock()
        post_mock_2.json = AsyncMock(return_value=self._create_response("second", self._msg("[[A=B]]")))

        # Use side_effect on server_mock.post to return different response mocks
        server_mock.post = AsyncMock(side_effect=[post_mock_1, post_mock_2])

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q: 1+1?"}])
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("It is 2.")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        req = TerminusJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="2",
        )
        res = await rs.verify(req)
        assert res.reward == approx(1.0)
        assert len(res.judge_evaluations) == 2
        assert res.judge_evaluations[0].verdict_label == "[[A=B]]"
        assert res.judge_evaluations[1].verdict_label == "[[A=B]]"

    async def test_verify_not_equal_first(self, config: TerminusJudgeResourcesServerConfig) -> None:
        """Test when first evaluation returns not equal."""
        server_mock = MagicMock(spec=ServerClient)
        rs = TerminusJudgeResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_response("f", self._msg("[[A!=B]]")))
        server_mock.post = AsyncMock(return_value=post_mock)

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q: 1+1?"}])
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("It is 3.")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        req = TerminusJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="2",
        )
        res = await rs.verify(req)
        assert res.reward == approx(0.0)
        assert len(res.judge_evaluations) == 1
        assert res.judge_evaluations[0].verdict_label == "[[A!=B]]"

    async def test_unexpected_judge_output_defaults_to_not_equal(
        self, config: TerminusJudgeResourcesServerConfig
    ) -> None:
        """Test that missing verdict labels default to not equal."""
        server_mock = MagicMock(spec=ServerClient)
        rs = TerminusJudgeResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_response("f", self._msg("no label present")))
        server_mock.post = AsyncMock(return_value=post_mock)

        req = TerminusJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=NeMoGymResponse(
                id="r",
                created_at=0.0,
                model="m",
                object="response",
                output=[self._msg("text")],
                parallel_tool_calls=False,
                tool_choice="none",
                tools=[],
            ),
            expected_answer="x",
        )
        res = await rs.verify(req)
        assert res.reward == approx(0.0)
        assert res.judge_evaluations[0].verdict_label is None

    async def test_swap_fails_uses_configured_reward(self, config: TerminusJudgeResourcesServerConfig) -> None:
        """Test that swap failure uses configured reward_if_swap_fails."""
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.check_twice_swap = True
        cfg.reward_if_swap_fails = -1.0
        rs = TerminusJudgeResourcesServer(config=cfg, server_client=server_mock)

        # Create two separate response mocks for two separate POST calls
        post_mock_1 = MagicMock()
        post_mock_1.json = AsyncMock(return_value=self._create_response("first", self._msg("[[A=B]]")))

        post_mock_2 = MagicMock()
        post_mock_2.json = AsyncMock(return_value=self._create_response("second", self._msg("[[A!=B]]")))

        # Use side_effect on server_mock.post to return different response mocks
        server_mock.post = AsyncMock(side_effect=[post_mock_1, post_mock_2])

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q?"}])
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("A")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        req = TerminusJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="B",
        )
        res = await rs.verify(req)
        assert res.reward == approx(-1.0)
        assert len(res.judge_evaluations) == 2

    async def test_equal_label_appears_first(self, config: TerminusJudgeResourcesServerConfig) -> None:
        """Test that when both labels appear, the first one wins."""
        server_mock = MagicMock(spec=ServerClient)
        rs = TerminusJudgeResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_response("f", self._msg("[[A=B]] some text [[A!=B]]")))
        server_mock.post = AsyncMock(return_value=post_mock)

        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("answer")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        req = TerminusJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            expected_answer="expected",
        )
        res = await rs.verify(req)
        assert res.reward == approx(1.0)
        assert res.judge_evaluations[0].verdict_label == "[[A=B]]"

    async def test_not_equal_label_appears_first(self, config: TerminusJudgeResourcesServerConfig) -> None:
        """Test that when not-equal appears first, it wins."""
        server_mock = MagicMock(spec=ServerClient)
        rs = TerminusJudgeResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_response("f", self._msg("[[A!=B]] some text [[A=B]]")))
        server_mock.post = AsyncMock(return_value=post_mock)

        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("answer")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        req = TerminusJudgeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=model_response,
            expected_answer="expected",
        )
        res = await rs.verify(req)
        assert res.reward == approx(0.0)
        assert res.judge_evaluations[0].verdict_label == "[[A!=B]]"
