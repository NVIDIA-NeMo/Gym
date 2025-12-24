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


from copy import deepcopy
from typing import Any
from unittest.mock import MagicMock

from pytest import approx, fixture

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.translation_bleu.app import (
    TranslationBleuResourcesServer,
    TranslationBleuResourcesServerConfig,
    TranslationBleuVerifyRequest,
)


class TestApp:
    @fixture
    def config(self) -> TranslationBleuResourcesServerConfig:
        return TranslationBleuResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    def _create_response(self, id: str, model_response_text: str) -> dict[str, Any]:
        return NeMoGymResponse(
            id=id,
            created_at=1234.5,
            model="response_model",
            object="response",
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
            output=[
                NeMoGymResponseOutputMessage(
                    id=f"ID for {model_response_text}",
                    role="assistant",
                    status="in_progress",
                    type="message",
                    content=[NeMoGymResponseOutputText(annotations=[], text=model_response_text, type="output_text")],
                )
            ],
        ).model_dump()

    async def test_verify_identical(self, config: TranslationBleuResourcesServerConfig) -> None:
        server_mock = MagicMock(spec=ServerClient)
        resources_server = TranslationBleuResourcesServer(config=config, server_client=server_mock)

        source_text = "What is the name of your cat?"
        ground_truth = "Was ist der Name deiner Katze?"
        target_lang = "de"
        target_lang_name = "German"
        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "content": f'Translate this into {target_lang_name}: "{source_text}"',
                }
            ]
        )
        model_response = NeMoGymResponse(**self._create_response("model_response_id", ground_truth))
        identical_verify_request = TranslationBleuVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            trg_txt=ground_truth,
            trg_lang=target_lang,
        )
        identical_verify_response = await resources_server.verify(identical_verify_request)
        assert identical_verify_response.responses_create_params == model_create_params
        assert identical_verify_response.response == model_response
        assert identical_verify_response.trg_txt == ground_truth
        assert identical_verify_response.trg_lang == target_lang
        assert identical_verify_response.reward == approx(1.0)
        assert identical_verify_response.extracted_answer == ground_truth

        assert sorted(list(identical_verify_response.model_dump())) == [
            "extracted_answer",
            "response",
            "responses_create_params",
            "reward",
            "trg_lang",
            "trg_txt",
        ]

    def test_verify_answer_identical(self, config: TranslationBleuResourcesServerConfig) -> None:
        resources_server = TranslationBleuResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # source_text = "What is the name of your cat?"
        ground_truth = "Was ist der Name deiner Katze?"
        target_lang = "de"
        model_response_text = ground_truth

        assert resources_server._verify_answer(ground_truth, target_lang, model_response_text) == (
            approx(1.0),
            ground_truth,
        )

    def test_verify_answer_think_tags(self, config: TranslationBleuResourcesServerConfig) -> None:
        resources_server = TranslationBleuResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # source_text = "What is the name of your cat?"
        ground_truth = "Was ist der Name deiner Katze?"
        target_lang = "de"
        model_response_text = f"<think></think>\n\n{ground_truth}"

        assert resources_server._verify_answer(ground_truth, target_lang, model_response_text) == (
            approx(1.0),
            ground_truth,
        )

    def test_verify_answer_no_match(self, config: TranslationBleuResourcesServerConfig) -> None:
        resources_server = TranslationBleuResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        # source_text = "What is the name of your cat?"
        ground_truth = "Was ist der Name deiner Katze?"
        target_lang = "de"
        model_response_text = "Incorrect translation."

        assert resources_server._verify_answer(ground_truth, target_lang, model_response_text) == (
            approx(0.0),
            model_response_text,
        )
