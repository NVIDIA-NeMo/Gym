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

from math_verify.errors import TimeoutException
from pydantic import ValidationError
from pytest import approx, fixture, raises

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputItem,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputRefusal,
    NeMoGymResponseOutputText,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.math_with_judge.app import (
    JudgeEvaluation,
    LibraryJudgeMathResourcesServer,
    LibraryJudgeMathResourcesServerConfig,
    LibraryJudgeMathVerifyRequest,
)


class TestApp:
    @fixture
    def config(self) -> LibraryJudgeMathResourcesServerConfig:
        return LibraryJudgeMathResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            judge_model_server=ModelServerRef(
                type="responses_api_models",
                name="math_judge",
            ),
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        )

    def _create_response(self, id: str, output_item: NeMoGymResponseOutputItem) -> dict[str, Any]:
        return NeMoGymResponse(
            id=id,
            created_at=1234.5,
            model="response_model",
            object="response",
            output=[output_item],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump()

    def _check_judge_evaluation(
        self,
        judge_evaluation: JudgeEvaluation,
        question: str,
        first_answer: str,
        second_answer: str,
        expected_create_params: dict[str, Any],
        expected_response_id: str,
        expected_output_item: NeMoGymResponseOutputItem,
    ) -> None:
        expected_prompt = LibraryJudgeMathResourcesServer.JUDGE_PROMPT_TEMPLATE.format(
            question=question, first_answer=first_answer, second_answer=second_answer
        )
        assert judge_evaluation.responses_create_params == NeMoGymResponseCreateParamsNonStreaming(
            **expected_create_params,
            input=[
                {
                    "role": "system",
                    "content": LibraryJudgeMathResourcesServer.JUDGE_SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": expected_prompt,
                },
            ],
        )

        actual_response = judge_evaluation.response
        assert actual_response.output == [expected_output_item]
        response_map = actual_response.model_dump(exclude_none=True)
        assert response_map.pop("id") == expected_response_id
        assert response_map.pop("created_at") == approx(1234.5)
        assert response_map.pop("model") == "response_model"
        assert response_map.pop("object") == "response"
        assert response_map.pop("parallel_tool_calls") is False
        assert response_map.pop("tool_choice") == "none"
        assert response_map.pop("tools") == []
        assert list(response_map) == ["output"]

    def _create_response_output_message(self, message_text: str) -> NeMoGymResponseOutputMessage:
        return NeMoGymResponseOutputMessage(
            id=f"ID for {message_text}",
            content=[NeMoGymResponseOutputText(annotations=[], text=message_text, type="output_text")],
            role="assistant",
            status="in_progress",
            type="message",
        )

    async def test_verify(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.json = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)
        not_equal_item = self._create_response_output_message(
            f"{LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL} done"
        )
        response_mock.return_value = self._create_response("verify_not_equal_id", not_equal_item)

        question = "Simplify the expression x + 7 - 6"
        expected_answer = "x + 1"
        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "content": question,
                }
            ]
        )
        first_part = "$1"
        first_part_item = self._create_response_output_message(first_part)
        first_model_response = NeMoGymResponse(**self._create_response("first_part_id", first_part_item))
        not_equal_verify_request = LibraryJudgeMathVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=first_model_response.model_copy(deep=True),
            question=question,
            expected_answer=expected_answer,
        )
        not_equal_verify_response = await resources_server.verify(not_equal_verify_request)
        assert not_equal_verify_response.responses_create_params == model_create_params
        assert not_equal_verify_response.response == first_model_response
        assert not_equal_verify_response.reward == approx(0.0)
        assert not_equal_verify_response.expected_answer == expected_answer
        assert not_equal_verify_response.extracted_answer == "1"
        assert not_equal_verify_response.library_reward == approx(0.0)
        judge_evaluations = not_equal_verify_response.judge_evaluations
        assert len(judge_evaluations) == 1
        self._check_judge_evaluation(
            judge_evaluations[0],
            question,
            expected_answer,
            first_part,
            {},
            "verify_not_equal_id",
            not_equal_item,
        )
        assert sorted(list(not_equal_verify_response.model_dump())) == [
            "expected_answer",
            "extracted_answer",
            "judge_evaluations",
            "library_reward",
            "response",
            "responses_create_params",
            "reward",
        ]

        second_model_response = first_model_response.model_copy(deep=True)
        second_model_response.output = second_model_response.output + [
            NeMoGymResponseReasoningItem(id="extra_reasoning", summary=[], type="reasoning"),
            self._create_response_output_message(" + x$"),
            NeMoGymResponseOutputMessage(
                id="refusal_finish",
                content=[
                    NeMoGymResponseOutputRefusal(refusal="no response", type="refusal"),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ]
        equal_verify_request = LibraryJudgeMathVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=second_model_response.model_copy(deep=True),
            question=question,
            expected_answer=expected_answer,
        )
        equal_verify_response = await resources_server.verify(equal_verify_request)
        assert equal_verify_response.responses_create_params == model_create_params
        assert equal_verify_response.response == second_model_response
        assert equal_verify_response.reward == approx(1.0)
        assert equal_verify_response.expected_answer == expected_answer
        assert equal_verify_response.extracted_answer == "x + 1"
        assert equal_verify_response.library_reward == approx(1.0)
        assert equal_verify_response.judge_evaluations is None
        assert sorted(list(equal_verify_response.model_dump())) == [
            "expected_answer",
            "extracted_answer",
            "judge_evaluations",
            "library_reward",
            "response",
            "responses_create_params",
            "reward",
        ]

    async def test_verify_answer(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.json = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        (
            equal_reward,
            equal_extracted_answer,
            equal_library_reward,
            equal_judge_evaluations,
        ) = await resources_server._verify_answer("What is 3 plus 5?", "8", "3 + 5 = \\boxed{8}")
        assert equal_reward == approx(1.0)
        assert equal_extracted_answer == "8"
        assert equal_library_reward == approx(1.0)
        assert equal_judge_evaluations is None

        not_equal_item = self._create_response_output_message(
            f"Conclusion: {LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL}"
        )
        response_mock.side_effect = [self._create_response("verify_answer_not_equal_id", not_equal_item)]
        not_equal_question = "What is 1 + 1?"
        not_equal_expected_answer = "2"
        not_equal_generated_answer = "3"
        (
            not_equal_reward,
            not_equal_extracted_answer,
            not_equal_library_reward,
            not_equal_judge_evaluations,
        ) = await resources_server._verify_answer(
            not_equal_question,
            not_equal_expected_answer,
            not_equal_generated_answer,
        )
        assert not_equal_reward == approx(0.0)
        assert not_equal_extracted_answer == "3"
        assert not_equal_library_reward == approx(0.0)
        assert len(not_equal_judge_evaluations) == 1
        self._check_judge_evaluation(
            not_equal_judge_evaluations[0],
            not_equal_question,
            not_equal_expected_answer,
            not_equal_generated_answer,
            {},
            "verify_answer_not_equal_id",
            not_equal_item,
        )

        first_judge_equal_item = self._create_response_output_message(
            f"I say {LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL} as the verdict"
        )
        second_judge_equal_item = self._create_response_output_message(
            LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL
        )
        response_mock.side_effect = [
            self._create_response("verify_answer_first_judge_equal_id", first_judge_equal_item),
            self._create_response("verify_answer_second_judge_equal_id", second_judge_equal_item),
        ]
        judge_equal_question = "What is 14 divided by 10?"
        judge_equal_expected_answer = "1.4"
        judge_equal_generated_answer = "Final answer: {7 / 5}"
        (
            judge_equal_reward,
            judge_equal_extracted_answer,
            judge_equal_library_reward,
            judge_equal_judge_evaluations,
        ) = await resources_server._verify_answer(
            judge_equal_question,
            judge_equal_expected_answer,
            judge_equal_generated_answer,
        )
        assert judge_equal_reward == approx(1.0)
        assert judge_equal_extracted_answer == "5"
        assert judge_equal_library_reward == approx(0.0)
        assert len(judge_equal_judge_evaluations) == 2
        self._check_judge_evaluation(
            judge_equal_judge_evaluations[0],
            judge_equal_question,
            judge_equal_expected_answer,
            judge_equal_generated_answer,
            {},
            "verify_answer_first_judge_equal_id",
            first_judge_equal_item,
        )
        self._check_judge_evaluation(
            judge_equal_judge_evaluations[1],
            judge_equal_question,
            judge_equal_generated_answer,
            judge_equal_expected_answer,
            {},
            "verify_answer_second_judge_equal_id",
            second_judge_equal_item,
        )

    def test_verify_answer_with_library(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        assert resources_server._verify_answer_with_library("4", "2 + 2 = \\boxed{4}") == (approx(1.0), "4")
        assert resources_server._verify_answer_with_library("\\boxed{12}", "3 * 4 = \\boxed{12}") == (
            approx(1.0),
            "12",
        )
        assert resources_server._verify_answer_with_library("\\boxed{5}", "10 - 5 = \\boxed{5}") == (approx(1.0), "5")
        assert resources_server._verify_answer_with_library("4.0", "2 + 2 = \\boxed{\\frac{8}{2}}") == (
            approx(1.0),
            "4",
        )

        assert resources_server._verify_answer_with_library("\\boxed{12}", "3 * 4 = 13") == (approx(0.0), "13")
        assert resources_server._verify_answer_with_library("17.001", "17") == (
            approx(0.0),
            "17",
        )

        assert resources_server._verify_answer_with_library("", "") == (
            approx(0.0),
            None,
        )

        assert resources_server._verify_answer_with_library("3", "3") == (
            approx(1.0),
            "3",
        )
        timeout_mock = MagicMock(side_effect=TimeoutException())
        resources_server._library_verifier = timeout_mock
        assert resources_server._verify_answer_with_library("3", "3") == (
            approx(0.0),
            None,
        )

    async def test_verify_answer_with_judge(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = LibraryJudgeMathResourcesServer(config=config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.json = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        first_not_equal_item = self._create_response_output_message(
            f"{LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL} is the evaluation"
        )
        response_mock.side_effect = [self._create_response("first_not_equal_id", first_not_equal_item)]
        first_not_equal_question = "What is 2 + 2?"
        first_not_equal_expected_answer = "4"
        first_not_equal_generated_answer = "5"
        (
            first_not_equal_reward,
            first_not_equal_evaluations,
        ) = await resources_server._verify_answer_with_judge(
            first_not_equal_question,
            first_not_equal_expected_answer,
            first_not_equal_generated_answer,
        )
        assert first_not_equal_reward == approx(0.0)
        assert len(first_not_equal_evaluations) == 1
        self._check_judge_evaluation(
            first_not_equal_evaluations[0],
            first_not_equal_question,
            first_not_equal_expected_answer,
            first_not_equal_generated_answer,
            {},
            "first_not_equal_id",
            first_not_equal_item,
        )

        first_equal_item = self._create_response_output_message(LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL)
        second_equal_item = self._create_response_output_message(
            f"I conclude that {LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL}"
        )
        response_mock.side_effect = [
            self._create_response("second_equal_first_id", first_equal_item),
            self._create_response("second_equal_second_id", second_equal_item),
        ]
        second_equal_question = "What is 3 divided by 6?"
        second_equal_expected_answer = "1/2"
        second_equal_generated_answer = "0.5"
        (
            second_equal_reward,
            second_equal_evaluations,
        ) = await resources_server._verify_answer_with_judge(
            second_equal_question,
            second_equal_expected_answer,
            second_equal_generated_answer,
        )
        assert second_equal_reward == approx(1.0)
        assert len(second_equal_evaluations) == 2
        self._check_judge_evaluation(
            second_equal_evaluations[0],
            second_equal_question,
            second_equal_expected_answer,
            second_equal_generated_answer,
            {},
            "second_equal_first_id",
            first_equal_item,
        )
        self._check_judge_evaluation(
            second_equal_evaluations[1],
            second_equal_question,
            second_equal_generated_answer,
            second_equal_expected_answer,
            {},
            "second_equal_second_id",
            second_equal_item,
        )

        second_not_equal_item = self._create_response_output_message(
            LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL
        )
        response_mock.side_effect = [
            self._create_response("second_not_equal_first_id", second_equal_item),
            self._create_response("second_not_equal_second_id", second_not_equal_item),
        ]
        second_not_equal_question = "What is 4 times 5?"
        second_not_equal_expected_answer = "20"
        second_not_equal_generated_answer = "20.0"
        (
            second_not_equal_reward,
            second_not_equal_evaluations,
        ) = await resources_server._verify_answer_with_judge(
            second_not_equal_question,
            second_not_equal_expected_answer,
            second_not_equal_generated_answer,
        )
        assert second_not_equal_reward == approx(0.0)
        assert len(second_not_equal_evaluations) == 2
        self._check_judge_evaluation(
            second_not_equal_evaluations[0],
            second_not_equal_question,
            second_not_equal_expected_answer,
            second_not_equal_generated_answer,
            {},
            "second_not_equal_first_id",
            second_equal_item,
        )
        self._check_judge_evaluation(
            second_not_equal_evaluations[1],
            second_not_equal_question,
            second_not_equal_generated_answer,
            second_not_equal_expected_answer,
            {},
            "second_not_equal_second_id",
            second_not_equal_item,
        )

    async def _generate_and_check_judge_evaluation(
        self,
        resources_server: LibraryJudgeMathResourcesServer,
        question: str,
        expected_answers_equal: bool,
        expected_response_id: str,
        expected_output_item: NeMoGymResponseOutputItem,
    ) -> None:
        first_answer = f"{question}_1"
        second_answer = f"{question}_2"
        (
            actual_answers_equal,
            judge_evaluation,
        ) = await resources_server._generate_judge_evaluation(question, first_answer, second_answer)
        assert actual_answers_equal == expected_answers_equal
        self._check_judge_evaluation(
            judge_evaluation,
            question,
            first_answer,
            second_answer,
            {"max_output_tokens": 1024},
            expected_response_id,
            expected_output_item,
        )

    async def test_generate_judge_evaluation(self, config: LibraryJudgeMathResourcesServerConfig) -> None:
        judge_config = config.model_copy(deep=True)
        judge_config.judge_responses_create_params.max_output_tokens = 1024
        server_mock = MagicMock(spec=ServerClient)
        resources_server = LibraryJudgeMathResourcesServer(config=judge_config, server_client=server_mock)
        response_mock = AsyncMock()
        post_mock = MagicMock()
        post_mock.json = response_mock
        server_mock.post = AsyncMock(return_value=post_mock)

        response_mock.return_value = {}
        with raises(ValidationError, match="Field required"):
            await resources_server._generate_judge_evaluation("invalid_response", "invalid_1", "invalid_2")

        reasoning_item = NeMoGymResponseReasoningItem(id="reasoning_item", summary=[], type="reasoning")
        response_mock.return_value = self._create_response("reasoning_id", reasoning_item)
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "reasoning_question",
            False,
            "reasoning_id",
            reasoning_item,
        )

        refusal_item = NeMoGymResponseOutputMessage(
            id="refusal_item",
            content=[
                NeMoGymResponseOutputRefusal(refusal="I refuse", type="refusal"),
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        response_mock.return_value = self._create_response("refusal_id", refusal_item)
        await self._generate_and_check_judge_evaluation(
            resources_server, "refusal_question", False, "refusal_id", refusal_item
        )

        no_evaluation_item = self._create_response_output_message("no evaluation")
        response_mock.return_value = self._create_response("no_evaluation_id", no_evaluation_item)
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "no_evaluation_question",
            False,
            "no_evaluation_id",
            no_evaluation_item,
        )

        not_equal_item = self._create_response_output_message(
            f"Evaluation: {LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL}"
        )
        response_mock.return_value = self._create_response("not_equal_id", not_equal_item)
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "not_equal_question",
            False,
            "not_equal_id",
            not_equal_item,
        )

        equal_item = self._create_response_output_message(
            f"The evaluation is {LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL}"
        )
        response_mock.return_value = self._create_response("equal_id", equal_item)
        await self._generate_and_check_judge_evaluation(
            resources_server, "equal_question", True, "equal_id", equal_item
        )

        equal_first_item = self._create_response_output_message(
            f"First {LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL}, "
            f"then {LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL}"
        )
        response_mock.return_value = self._create_response("equal_first_id", equal_first_item)
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "equal_first_question",
            True,
            "equal_first_id",
            equal_first_item,
        )

        not_equal_first_item = self._create_response_output_message(
            f"{LibraryJudgeMathResourcesServer.JUDGE_NOT_EQUAL_LABEL} "
            f"{LibraryJudgeMathResourcesServer.JUDGE_EQUAL_LABEL}"
        )
        response_mock.return_value = self._create_response("not_equal_first_id", not_equal_first_item)
        await self._generate_and_check_judge_evaluation(
            resources_server,
            "not_equal_first_question",
            False,
            "not_equal_first_id",
            not_equal_first_item,
        )
