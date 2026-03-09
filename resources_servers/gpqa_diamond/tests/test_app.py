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
from unittest.mock import MagicMock

from app import GPQADiamondResourcesServer

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.mcqa.app import MCQAResourcesServerConfig, MCQAVerifyRequest


class TestApp:
    def test_sanity(self) -> None:
        config = MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        GPQADiamondResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_verify_gpqa_diamond_format(self) -> None:
        server = GPQADiamondResourcesServer(
            config=MCQAResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        # Strict mode should reject an unboxed answer.
        unboxed_response = NeMoGymResponse(
            id="resp_unboxed",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_unboxed",
                    "content": [{"annotations": [], "text": "Final answer: C", "type": "output_text"}],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        verify_request = MCQAVerifyRequest(
            responses_create_params={
                "input": [
                    {
                        "role": "user",
                        "content": (
                            "You should output your final response letter inside \\boxed{} and nothing else "
                            "Question?\nA: optA\nB: optB\nC: optC\nD: optD"
                        ),
                    }
                ]
            },
            response=unboxed_response,
            options=[{"A": "optA"}, {"B": "optB"}, {"C": "optC"}, {"D": "optD"}],
            expected_answer="C",
            grading_mode="strict_single_letter_boxed",
        )
        result_unboxed = await server.verify(verify_request)
        assert result_unboxed.reward == 0.0

        boxed_response = unboxed_response.model_copy(
            update={
                "id": "resp_boxed",
                "output": [
                    {
                        "id": "msg_boxed",
                        "content": [{"annotations": [], "text": "Final: \\boxed{C}", "type": "output_text"}],
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                    }
                ],
            }
        )
        verify_request_boxed = verify_request.model_copy(update={"response": boxed_response})
        result_boxed = await server.verify(verify_request_boxed)

        assert result_boxed.reward == 1.0
        assert result_boxed.extracted_answer == "C"
