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
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from pytest import approx, fixture

from nemo_gym import CACHE_DIR
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.translation_metricx.app import (
    TranslationMetricxResourcesServer,
    TranslationMetricxResourcesServerConfig,
    TranslationMetricxVerifyRequest,
)


logger = logging.getLogger(__name__)

os.environ["HF_HOME"] = str(Path(CACHE_DIR) / "hf_cache")


class TestApp:
    @fixture(scope="class")
    def resources_server(self) -> TranslationMetricxResourcesServer:
        """We only want to spin up the server once since it has to load the model."""
        logger.info("Spinning up server with MetricX model...")

        server = TranslationMetricxResourcesServer(
            config=TranslationMetricxResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                use_reference=True,
                # 1.2B parameter model runs fine on CPU, though tests will take a couple of minutes
                metricx_model_name="google/metricx-24-hybrid-large-v2p6-bfloat16",
                tokenizer_name="google/mt5-large",
                device_map="cpu",
                max_input_length=1536,
                output_dir=str(Path(CACHE_DIR) / "metricx_output"),
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        logger.info("Model loaded and server started successfully")
        return server

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

    async def test_verify_identical(self, resources_server: TranslationMetricxResourcesServer) -> None:
        source_text = "What is the name of your cat?"
        target_text = "Was ist der Name deiner Katze?"
        target_lang_name = "German"
        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "content": f'Translate this into {target_lang_name}: "{source_text}"',
                }
            ]
        )
        model_response = NeMoGymResponse(**self._create_response("model_response_id", target_text))
        identical_verify_request = TranslationMetricxVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            src_txt=source_text,
            trg_txt=target_text,
        )
        identical_verify_response = await resources_server.verify(identical_verify_request)
        assert identical_verify_response.responses_create_params == model_create_params
        assert identical_verify_response.response == model_response
        assert identical_verify_response.src_txt == source_text
        assert identical_verify_response.trg_txt == target_text
        assert identical_verify_response.reward == approx(1.0, abs=0.1)
        assert identical_verify_response.extracted_answer == target_text

        assert sorted(list(identical_verify_response.model_dump())) == [
            "extracted_answer",
            "response",
            "responses_create_params",
            "reward",
            "src_txt",
            "trg_txt",
        ]

    async def test_verify_identical_without_reference(
        self, resources_server: TranslationMetricxResourcesServer
    ) -> None:
        source_text = "two three"
        target_text = "zwei drei"
        target_lang_name = "German"
        model_create_params = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "content": f'Translate this into {target_lang_name}: "{source_text}"',
                }
            ]
        )
        model_response = NeMoGymResponse(**self._create_response("model_response_id", target_text))
        identical_verify_request = TranslationMetricxVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            src_txt=source_text,
            trg_txt=None,  # Technically the model config is set up to use a reference but this triggers the same behavior
        )
        identical_verify_response = await resources_server.verify(identical_verify_request)
        assert identical_verify_response.responses_create_params == model_create_params
        assert identical_verify_response.response == model_response
        assert identical_verify_response.src_txt == source_text
        assert identical_verify_response.trg_txt is None
        assert identical_verify_response.reward == approx(1.0, abs=0.1)
        assert identical_verify_response.extracted_answer == target_text

        assert sorted(list(identical_verify_response.model_dump())) == [
            "extracted_answer",
            "response",
            "responses_create_params",
            "reward",
            "src_txt",
            "trg_txt",
        ]

    def test_verify_answer_identical(self, resources_server: TranslationMetricxResourcesServer) -> None:
        source_text = "two three"
        target_text = "zwei drei"
        model_response_text = target_text

        assert resources_server._verify_answer(model_response_text, source_text, target_text) == (
            approx(1.0, abs=0.1),  # It's a model output so it won't be exact
            target_text,
        )

    def test_verify_answer_think_tags(self, resources_server: TranslationMetricxResourcesServer) -> None:
        source_text = "What is the name of your cat?"
        target_text = "Was ist der Name deiner Katze?"
        model_response_text = f"<think></think>\n\n{target_text}"

        assert resources_server._verify_answer(model_response_text, source_text, target_text) == (
            approx(1.0, abs=0.1),  # It's a model output so it won't be exact
            target_text,
        )

    def test_verify_answer_no_match(self, resources_server: TranslationMetricxResourcesServer) -> None:
        source_text = "What is the name of your cat?"
        target_text = "Was ist der Name deiner Katze?"
        model_response_text = "Incorrect translation."

        reward, extracted_answer = resources_server._verify_answer(model_response_text, source_text, target_text)
        assert reward <= 0.6  # Raw score is around 10 for this example, where 25 is worst
        assert extracted_answer == model_response_text

    def test_verify_answer_without_reference(self, resources_server: TranslationMetricxResourcesServer) -> None:
        source_text = "two three"
        model_response_text = "zwei drei"

        assert resources_server._verify_answer(model_response_text, source_text) == (
            approx(1.0, abs=0.1),  # It's a model output so it won't be exact
            model_response_text,
        )
