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

from app import StringMatchResourcesServer, StringMatchResourcesServerConfig, StringMatchVerifyRequest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "content": [
                    {
                        "annotations": [],
                        "text": text,
                        "type": "output_text",
                    }
                ],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_server() -> StringMatchResourcesServer:
    return StringMatchResourcesServer(
        config=StringMatchResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
        server_client=MagicMock(spec=ServerClient),
    )


def _make_request(
    text: str,
    expected_answer: str,
    extraction_mode: str = "final_answer",
    case_sensitive: bool = False,
) -> StringMatchVerifyRequest:
    return StringMatchVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "What is the answer?"}],
        },
        response=_make_response(text),
        expected_answer=expected_answer,
        extraction_mode=extraction_mode,
        case_sensitive=case_sensitive,
    )


class TestApp:
    def test_sanity(self) -> None:
        _make_server()

    async def test_final_answer_correct(self) -> None:
        server = _make_server()
        req = _make_request("I think it is lettuce.\n\nFinal answer: lettuce", "lettuce")
        result = await server.verify(req)
        assert result.reward == 1.0
        assert result.extracted_answer == "lettuce"

    async def test_final_answer_case_insensitive(self) -> None:
        server = _make_server()
        req = _make_request("Final answer: Yes", "yes")
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_final_answer_wrong(self) -> None:
        server = _make_server()
        req = _make_request("Final answer: No", "yes")
        result = await server.verify(req)
        assert result.reward == 0.0

    async def test_boxed_extraction(self) -> None:
        server = _make_server()
        req = _make_request(
            "The plant is \\boxed{lettuce}.",
            "lettuce",
            extraction_mode="boxed",
        )
        result = await server.verify(req)
        assert result.reward == 1.0
        assert result.extracted_answer == "lettuce"

    async def test_boxed_with_latex_text_wrapper(self) -> None:
        server = _make_server()
        req = _make_request(
            "The answer is \\boxed{\\text{Summer solstice}}",
            "Summer solstice",
            extraction_mode="boxed",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_final_answer_fallback_to_boxed(self) -> None:
        """In final_answer mode, if no 'Final answer:' found, fall back to boxed."""
        server = _make_server()
        req = _make_request(
            "After analysis, \\boxed{Dog2}",
            "Dog2",
            extraction_mode="final_answer",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_last_line_extraction(self) -> None:
        server = _make_server()
        req = _make_request(
            "Let me think about this.\nThe answer is clear.\nYes",
            "Yes",
            extraction_mode="last_line",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_full_response_extraction(self) -> None:
        server = _make_server()
        req = _make_request("No", "No", extraction_mode="full_response")
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_case_sensitive_mode(self) -> None:
        server = _make_server()
        req = _make_request(
            "Final answer: YES",
            "Yes",
            case_sensitive=True,
        )
        result = await server.verify(req)
        assert result.reward == 0.0

    async def test_answer_colon_variant(self) -> None:
        """'Answer: X' should also be captured by final_answer mode."""
        server = _make_server()
        req = _make_request("After reasoning... Answer: Throughput", "Throughput")
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_unicode_normalization(self) -> None:
        server = _make_server()
        req = _make_request("Final answer: caf\u00e9", "cafe\u0301")
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_whitespace_normalization(self) -> None:
        server = _make_server()
        req = _make_request("Final answer:   Summer   solstice  ", "Summer solstice")
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_no_extraction_returns_zero(self) -> None:
        server = _make_server()
        req = _make_request(
            "I'm not sure about this question.",
            "Yes",
            extraction_mode="boxed",
        )
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.extracted_answer is None

    async def test_last_boxed_wins(self) -> None:
        server = _make_server()
        req = _make_request(
            "First try \\boxed{wrong}. Wait, \\boxed{correct}",
            "correct",
            extraction_mode="boxed",
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_last_final_answer_wins(self) -> None:
        server = _make_server()
        req = _make_request(
            "Final answer: wrong\nActually, Final answer: correct",
            "correct",
        )
        result = await server.verify(req)
        assert result.reward == 1.0
