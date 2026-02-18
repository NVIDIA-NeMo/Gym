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
from copy import deepcopy
from os.path import dirname, join
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
from resources_servers.translation_comet.app import (
    TranslationCometResourcesServer,
    TranslationCometResourcesServerConfig,
    TranslationCometVerifyRequest,
)


logger = logging.getLogger(__name__)


class TestApp:
    @fixture(scope="class")
    def resources_server(self) -> TranslationCometResourcesServer:
        """We only want to spin up the server once since it has to load the comet model.
        Although, the slowest part is actually `import comet` :)
        """
        logger.info("Spinning up server with COMET model...")

        server = TranslationCometResourcesServer(
            config=TranslationCometResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                use_reference=True,
                comet_model_name="Unbabel/wmt22-comet-da",  # 0.5B parameter model runs fine on CPU
                # Need to use the actual model as the cometinho model does not return values in [0,1]
                comet_gpu_count=0,  # CPU
                comet_gpu_devices="auto",  # CPU
                model_cache_dir=join(CACHE_DIR, "ptl_cache"),
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        logger.info("Model loaded and server started successfully")
        return server

    def reference_free_resources_server(self) -> TranslationCometResourcesServer:
        logger.info("Spinning up server with reference-free COMET model...")

        server = TranslationCometResourcesServer(
            config=TranslationCometResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                use_reference=False,
                comet_model_name="Unbabel/wmt22-cometkiwi-da",  # reference-free COMET
                comet_gpu_count=0,  # CPU
                comet_gpu_devices="auto",  # CPU
                model_cache_dir=join(dirname(__file__), "..", "..", "..", "cache", "ptl_cache"),
            ),
            server_client=MagicMock(spec=ServerClient),
        )

        logger.info("Reference-free COMET model loaded and server started successfully")
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

    async def test_verify_identical(self, resources_server: TranslationCometResourcesServer) -> None:
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
        identical_verify_request = TranslationCometVerifyRequest(
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
        assert identical_verify_response.reward == approx(1.0, abs=0.05)
        assert identical_verify_response.extracted_answer == target_text

        assert sorted(list(identical_verify_response.model_dump())) == [
            "extracted_answer",
            "response",
            "responses_create_params",
            "reward",
            "src_txt",
            "trg_txt",
        ]

    def test_verify_answer_identical(self, resources_server: TranslationCometResourcesServer) -> None:
        source_text = "What is the name of your cat?"
        target_text = "Was ist der Name deiner Katze?"
        model_response_text = target_text

        assert resources_server._verify_answer(source_text, target_text, model_response_text) == (
            approx(1.0, abs=0.05),  # It's a model output so it won't be exact
            target_text,
        )

    def test_verify_answer_think_tags(self, resources_server: TranslationCometResourcesServer) -> None:
        source_text = "What is the name of your cat?"
        target_text = "Was ist der Name deiner Katze?"
        model_response_text = f"<think></think>\n\n{target_text}"

        assert resources_server._verify_answer(source_text, target_text, model_response_text) == (
            approx(1.0, abs=0.05),  # It's a model output so it won't be exact
            target_text,
        )

    def test_verify_answer_no_match(self, resources_server: TranslationCometResourcesServer) -> None:
        source_text = "What is the name of your cat?"
        target_text = "Was ist der Name deiner Katze?"
        model_response_text = "Incorrect translation."

        assert resources_server._verify_answer(source_text, target_text, model_response_text) == (
            approx(0.0, abs=0.5),  # This returns about 0.3 in practice but it's fine as long as it's low
            model_response_text,
        )

    async def test_verify_identical_reference_free(self, resources_server: TranslationCometResourcesServer) -> None:
        reference_free_resources_server = self.reference_free_resources_server()

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
        identical_verify_request = TranslationCometVerifyRequest(
            responses_create_params=deepcopy(model_create_params),
            response=model_response.model_copy(deep=True),
            src_txt=source_text,
        )
        identical_verify_response = await reference_free_resources_server.verify(identical_verify_request)
        assert identical_verify_response.responses_create_params == model_create_params
        assert identical_verify_response.response == model_response
        assert identical_verify_response.src_txt == source_text
        assert identical_verify_response.trg_txt is None
        assert identical_verify_response.reward == approx(
            1.0, abs=0.25
        )  # It's hard to get a score near 1.0 with the reference-free model
        assert identical_verify_response.extracted_answer == target_text

        assert sorted(list(identical_verify_response.model_dump())) == [
            "extracted_answer",
            "response",
            "responses_create_params",
            "reward",
            "src_txt",
            "trg_txt",  # Should be present but None
        ]
