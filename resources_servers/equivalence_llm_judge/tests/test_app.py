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
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from omegaconf import OmegaConf
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
from resources_servers.equivalence_llm_judge.app import (
    LLMJudgeResourcesServer,
    LLMJudgeResourcesServerConfig,
    LLMJudgeVerifyRequest,
)


class TestApp:
    @fixture
    def config(self) -> LLMJudgeResourcesServerConfig:
        # Load judge template from YAML so tests mirror runtime config
        yaml_path = Path(__file__).resolve().parents[1] / "configs" / "equivalence_llm_judge.yaml"
        yaml_cfg = OmegaConf.load(str(yaml_path))
        judge_template: str = (
            yaml_cfg.equivalence_llm_judge.resources_servers.equivalence_llm_judge.judge_prompt_template
        )

        cfg = LLMJudgeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            judge_model_server=ModelServerRef(type="responses_api_models", name="judge"),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            judge_prompt_template=judge_template,
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

    async def test_verify_equal_then_confirm(self, config: LLMJudgeResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        # Default: check_twice_swap = False
        rs = LLMJudgeResourcesServer(config=config, server_client=server_mock)

        # First: judge says equal; Second: judge says equal => reward 1
        post_mock = MagicMock()
        post_mock.json = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)

        # Only the first call is used when check_twice_swap is False
        post_mock.json.side_effect = [
            self._create_response("first", self._msg("some text [[A=B]] trailing")),
        ]

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

        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="2",
        )
        res = await rs.verify(req)
        assert res.reward == approx(1.0)
        assert res.expected_answer == "2"
        assert len(res.judge_evaluations) == 1

        # Now enable double-check and ensure two evaluations are returned
        config_twice = config.model_copy(deep=True)
        config_twice.check_twice_swap = True
        rs_twice = LLMJudgeResourcesServer(config=config_twice, server_client=server_mock)

        post_mock2 = MagicMock()
        post_mock2.json = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock2)
        post_mock2.json.side_effect = [
            self._create_response("first", self._msg("[[A=B]]")),
            self._create_response("second", self._msg("[[A=B]]")),
        ]
        req2 = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="2",
        )
        res2 = await rs_twice.verify(req2)
        assert res2.reward == approx(1.0)
        assert len(res2.judge_evaluations) == 2

    async def test_verify_not_equal_first(self, config: LLMJudgeResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        rs = LLMJudgeResourcesServer(config=config, server_client=server_mock)

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

        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="2",
        )
        res = await rs.verify(req)
        assert res.reward == approx(0.0)
        assert len(res.judge_evaluations) == 1

    async def test_unexpected_judge_output_defaults_to_not_equal(self, config: LLMJudgeResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        rs = LLMJudgeResourcesServer(config=config, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_response("f", self._msg("no label present")))
        server_mock.post = AsyncMock(return_value=post_mock)

        req = LLMJudgeVerifyRequest(
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

    async def test_swap_fails_uses_configured_reward(self, config: LLMJudgeResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.check_twice_swap = True
        cfg.reward_if_swap_fails = -1.0
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)
        # First pass equal, second pass not equal -> use configured -1.0
        post_mock.json.side_effect = [
            self._create_response("first", self._msg("[[A=B]]")),
            self._create_response("second", self._msg("[[A!=B]]")),
        ]

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
        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="B",
        )
        res = await rs.verify(req)
        assert res.reward == approx(-1.0)
        assert len(res.judge_evaluations) == 2

    async def test_per_record_regex_extraction(self, config: LLMJudgeResourcesServerConfig) -> None:
        """Test that template_metadata.output_regex extracts answer correctly."""
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.use_per_record_regex = True
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_response("first", self._msg("[[A=B]]")))
        server_mock.post = AsyncMock(return_value=post_mock)

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What is 2+2?"}]
        )
        # Model generates answer wrapped in \boxed{}
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("Let me explain: The answer is \\boxed{4} because 2+2=4.")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        # Request with per-record regex in template_metadata
        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="4",
            template_metadata={"output_regex": r"\\boxed\{(.*?)\}"},
        )

        res = await rs.verify(req)
        assert res.reward == approx(1.0)
        assert len(res.judge_evaluations) == 1
        # Verify the regex extraction worked by checking judge was called once
        assert server_mock.post.call_count == 1

    async def test_full_generation_rescue_on_extraction_failure(self, config: LLMJudgeResourcesServerConfig) -> None:
        """When regex-extracted answer fails, retry with full generation for partial credit."""
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.use_per_record_regex = True
        cfg.check_full_generation_on_fail = True
        cfg.reward_if_full_generation_succeeds = 0.5
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock()
        server_mock.post = AsyncMock(return_value=post_mock)
        # First call (extracted answer) fails, second call (full generation) succeeds
        post_mock.json.side_effect = [
            self._create_response("first", self._msg("[[A!=B]]")),
            self._create_response("second", self._msg("[[A=B]]")),
        ]

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "What is 2+2?"}]
        )
        # Model output: correct answer is in full text but regex won't match properly
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg("The final answer is clearly 4")],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        # Regex pattern that won't match the model output
        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer="4",
            template_metadata={"output_regex": r"ANSWER:\s*(.+)"},  # Won't match
        )

        res = await rs.verify(req)
        assert res.reward == approx(0.5)  # Partial credit
        assert len(res.judge_evaluations) == 2  # Both passes recorded
        assert server_mock.post.call_count == 2

    async def test_extraction_length_threshold_skips_regex(self, config: LLMJudgeResourcesServerConfig) -> None:
        """Long expected answers skip regex extraction and use full generation."""
        server_mock = MagicMock(spec=ServerClient)
        cfg = config.model_copy(deep=True)
        cfg.use_per_record_regex = True
        cfg.extraction_length_threshold = 50  # 50 characters
        rs = LLMJudgeResourcesServer(config=cfg, server_client=server_mock)

        post_mock = MagicMock()
        post_mock.json = AsyncMock(return_value=self._create_response("first", self._msg("[[A=B]]")))
        server_mock.post = AsyncMock(return_value=post_mock)

        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": "Explain photosynthesis."}]
        )
        # Long answer that exceeds threshold
        long_answer = "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose."
        model_response = NeMoGymResponse(
            id="resp",
            created_at=0.0,
            model="m",
            object="response",
            output=[self._msg(long_answer)],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

        # Even though regex is provided, it should be ignored due to length threshold
        req = LLMJudgeVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            expected_answer=long_answer,  # >50 chars, will skip regex
            template_metadata={"output_regex": r"\\boxed\{(.*?)\}"},  # Should be ignored
        )

        res = await rs.verify(req)
        assert res.reward == approx(1.0)
        assert len(res.judge_evaluations) == 1
        # Verify only one judge call (no second pass due to length threshold)
        assert server_mock.post.call_count == 1
