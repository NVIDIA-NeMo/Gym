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

from unittest.mock import MagicMock

import pytest
import reasoning_gym

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.reasoning_gym.app import (
    ReasoningGymResourcesServer,
    ReasoningGymResourcesServerConfig,
    ReasoningGymVerifyRequest,
)


class TestApp:
    @pytest.fixture
    def config(self) -> ReasoningGymResourcesServerConfig:
        return ReasoningGymResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )

    @pytest.fixture
    def server(self, config) -> ReasoningGymResourcesServer:
        return ReasoningGymResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def _create_response(self, text: str, msg_id: str = "test_msg") -> NeMoGymResponse:
        return NeMoGymResponse(
            id="test_response_id",
            created_at=1234567890.0,
            model="test_model",
            object="response",
            output=[
                NeMoGymResponseOutputMessage(
                    id=msg_id,
                    role="assistant",
                    type="message",
                    content=[
                        NeMoGymResponseOutputText(
                            type="output_text",
                            text=text,
                            annotations=[],
                        )
                    ],
                )
            ],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    @pytest.mark.asyncio
    async def test_reasoning_gym_verify_correct_answer(self, server):
        dataset = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42)
        entry = dataset[0]

        response = self._create_response(text=entry["answer"], msg_id="test_msg_1")

        verify_request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=response,
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward >= 0.9, f"Expected high reward for correct answer, got {verify_response.reward}"

    @pytest.mark.asyncio
    async def test_reasoning_gym_verify_incorrect_answer(self, server):
        dataset = reasoning_gym.create_dataset("knights_knaves", size=1, seed=42)
        entry = dataset[0]

        response = self._create_response(text="This is completely wrong", msg_id="test_msg_2")

        verify_request = ReasoningGymVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": entry["question"]}]
            ),
            response=response,
            question=entry["question"],
            answer=entry["answer"],
            metadata=entry["metadata"],
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward <= 0.1, f"Expected low reward for incorrect answer, got {verify_response.reward}"

    @pytest.mark.asyncio
    async def test_reasoning_gym_verify_multiple_tasks(self, server):
        tasks_to_test = ["knights_knaves", "leg_counting", "basic_arithmetic"]

        for task_name in tasks_to_test:
            dataset = reasoning_gym.create_dataset(task_name, size=1, seed=42)
            entry = dataset[0]

            response = self._create_response(text=entry["answer"], msg_id=f"test_msg_{task_name}")

            verify_request = ReasoningGymVerifyRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[{"role": "user", "content": entry["question"]}]
                ),
                response=response,
                question=entry["question"],
                answer=entry["answer"],
                metadata=entry["metadata"],
            )

            verify_response = await server.verify(verify_request)

            assert verify_response.reward >= 0.9, f"Task {task_name} failed: reward={verify_response.reward}"
            assert verify_response.task_name == task_name
