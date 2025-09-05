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
from app import (
    CompCodingResourcesServer,
    CompCodingResourcesServerConfig,
    CompCodingVerifyRequest,
)
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        cfg = CompCodingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="")
        CompCodingResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))

    async def test_verify_pass_via_response(self) -> None:
        # Assistant returns a python code block that squares the input
        response = NeMoGymResponse(
            id="resp_ok",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_ok",
                    "content": [
                        {
                            "annotations": [],
                            "text": "```python\nn=int(input())\nprint(n*n)\n```",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

        server = CompCodingResourcesServer(
            config=CompCodingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_req = CompCodingVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Read n and print n^2."}],
                "temperature": 0,
                "parallel_tool_calls": False,
            },
            response=response,
            verifier_metadata={"unit_tests": {"inputs": ["2\n", "5\n"], "outputs": ["4", "25"]}},
        )

        res = await server.verify(verify_req)
        assert res.reward == 1.0, res.reason

    async def test_verify_fail_wrong_answer(self) -> None:
        # Assistant prints n+1 instead of n*n
        response_bad = NeMoGymResponse(
            id="resp_bad",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_bad",
                    "content": [
                        {
                            "annotations": [],
                            "text": "```python\nn=int(input())\nprint(n+1)\n```",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

        server = CompCodingResourcesServer(
            config=CompCodingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_req_bad = CompCodingVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "square n"}]},
            response=response_bad,
            verifier_metadata={"unit_tests": {"inputs": ["3\n"], "outputs": ["9"]}},
        )

        res2 = await server.verify(verify_req_bad)
        assert res2.reward == 0.0 and "FAILED" in res2.reason

    async def test_verify_missing_response_validation_error(self) -> None:
        """Omitting `response` should fail request validation (schema requires it)."""
        _ = CompCodingResourcesServer(
            config=CompCodingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint=""),
            server_client=MagicMock(spec=ServerClient),
        )

        with pytest.raises(ValidationError):
            _ = CompCodingVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "anything"}]},
                # response is intentionally omitted
                verifier_metadata={"unit_tests": {"inputs": ["1\n"], "outputs": ["1"]}},
            )

    async def test_verify_no_code_block(self) -> None:
        """Test when response contains no code block - should extract raw text"""
        response = NeMoGymResponse(
            id="resp_no_block",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_no_block",
                    "content": [
                        {
                            "annotations": [],
                            "text": "n=int(input())\nprint(n*n)",  # No ```python``` wrapper
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

        server = CompCodingResourcesServer(
            config=CompCodingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_req = CompCodingVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Read n and print n^2."}],
            },
            response=response,
            verifier_metadata={"unit_tests": {"inputs": ["2\n"], "outputs": ["4"]}},
        )

        res = await server.verify(verify_req)
        assert res.reward == 1.0, res.reason

    async def test_verify_syntax_error(self) -> None:
        """Code has a syntax error -> should report ERROR and reward 0.0"""
        server = CompCodingResourcesServer(
            config=CompCodingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint=""),
            server_client=MagicMock(spec=ServerClient),
        )

        response = NeMoGymResponse(
            id="resp_syntax_error",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_bad_syntax",
                    "content": [
                        {
                            "annotations": [],
                            "text": "```python\nprint('hello'  # Missing closing parenthesis\n```",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

        verify_req = CompCodingVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Print hello"}]},
            response=response,
            verifier_metadata={"unit_tests": {"inputs": ["\n"], "outputs": ["hello"]}},
        )

        res = await server.verify(verify_req)
        assert res.reward == 0.0 and "ERROR" in res.reason

    async def test_verify_runtime_error(self) -> None:
        server = CompCodingResourcesServer(
            config=CompCodingResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint=""),
            server_client=MagicMock(spec=ServerClient),
        )

        response = NeMoGymResponse(
            id="resp_runtime_error",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_runtime_error",
                    "content": [
                        {
                            "annotations": [],
                            "text": "```python\nn=int(input())\nprint(n/0)\n```",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )

        verify_req = CompCodingVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "Divide by zero"}]},
            response=response,
            verifier_metadata={"unit_tests": {"inputs": ["5\n"], "outputs": ["error"]}},
        )

        res = await server.verify(verify_req)
        assert res.reward == 0.0 and "ERROR" in res.reason
