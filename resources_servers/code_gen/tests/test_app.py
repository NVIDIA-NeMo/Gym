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

from typing import Generator
from unittest.mock import MagicMock

import pytest
import ray
from app import (
    CompCodingResourcesServer,
    CompCodingResourcesServerConfig,
    CompCodingVerifyRequest,
    CompCodingVerifyResponse,
)
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


class TestApp:
    @pytest.fixture(scope="module")
    def code_gen_resources_server_client(self) -> Generator[TestClient, None, None]:
        ray.init(num_cpus=1)
        server = CompCodingResourcesServer(
            config=CompCodingResourcesServerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                num_processes=1,
                unit_test_timeout_secs=10,
                debug=False,
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        app = server.setup_webserver()
        with TestClient(app) as client:
            yield client

    async def test_verify_pass_via_response(self, code_gen_resources_server_client: TestClient) -> None:
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

        verify_req = CompCodingVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Read n and print n^2."}],
                "temperature": 0,
                "parallel_tool_calls": False,
            },
            response=response,
            verifier_metadata={"unit_tests": {"inputs": ["2\n", "5\n"], "outputs": ["4", "25"]}},
        )

        response = code_gen_resources_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CompCodingVerifyResponse.model_validate(response.json())
        assert res.reward == 1.0

    async def test_verify_fail_wrong_answer(self, code_gen_resources_server_client: TestClient) -> None:
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

        verify_req_bad = CompCodingVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "square n"}]},
            response=response_bad,
            verifier_metadata={"unit_tests": {"inputs": ["3\n"], "outputs": ["9"]}},
        )

        response = code_gen_resources_server_client.post(
            url="/verify",
            json=verify_req_bad.model_dump(),
        )
        res = CompCodingVerifyResponse.model_validate(response.json())
        assert res.reward == 0.0 and res.metadata["error_message"] == "Wrong answer at output_line_idx=0: 4 != 9"

    def test_verify_missing_response_validation_error(self) -> None:
        """Omitting `response` should fail request validation (schema requires it)."""
        with pytest.raises(ValidationError):
            CompCodingVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "anything"}]},
                # response is intentionally omitted
                verifier_metadata={"unit_tests": {"inputs": ["1\n"], "outputs": ["1"]}},
            )

    async def test_verify_no_code_block(self, code_gen_resources_server_client: TestClient) -> None:
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

        verify_req = CompCodingVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Read n and print n^2."}],
            },
            response=response,
            verifier_metadata={"unit_tests": {"inputs": ["2\n"], "outputs": ["4"]}},
        )

        response = code_gen_resources_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CompCodingVerifyResponse.model_validate(response.json())
        # LCB code extraction only accepts fenced code blocks.
        assert res.reward == 0.0 and res.extracted_model_output and not res.extracted_model_code

    async def test_verify_syntax_error(self, code_gen_resources_server_client: TestClient) -> None:
        """Code has a syntax error -> should report ERROR and reward 0.0"""
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

        response = code_gen_resources_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CompCodingVerifyResponse.model_validate(response.json())
        assert (
            res.reward == 0.0
            and res.metadata["error_message"] == "Error during testing: '(' was never closed (<string>, line 1)"
        )

    async def test_verify_runtime_error(self, code_gen_resources_server_client: TestClient) -> None:
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

        response = code_gen_resources_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CompCodingVerifyResponse.model_validate(response.json())
        assert res.reward == 0.0 and res.metadata["error_message"] == "Runtime Error"


def _make_server():
    return CompCodingResourcesServer(
        config=CompCodingResourcesServerConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="code_gen",
            num_processes=1,
            unit_test_timeout_secs=10,
            debug=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


class TestComputeMetrics:
    def test_produces_pass_at_k_and_subsets(self) -> None:
        server = _make_server()
        tasks = [
            [
                {"reward": 1.0, "extracted_model_code": "print(1)", "difficulty": "easy"},
                {"reward": 1.0, "extracted_model_code": "print(1)", "difficulty": "easy"},
            ],
            [
                {"reward": 0.0, "extracted_model_code": "print(2)", "difficulty": "hard"},
                {"reward": 1.0, "extracted_model_code": "print(2b)", "difficulty": "hard"},
            ],
            [
                {"reward": 1.0, "extracted_model_code": "print(3)", "difficulty": "easy"},
                {"reward": 0.0, "extracted_model_code": None, "difficulty": "easy"},
            ],
        ]
        m = server.compute_metrics(tasks)

        # Overall metrics
        assert "pass@1/accuracy" in m
        assert "pass@2/accuracy" in m
        assert "majority@2/accuracy" in m
        assert "pass@1[avg-of-2]/accuracy" in m
        assert "pass@1[avg-of-2]/accuracy/avg_sample_std_dev" in m

        # Per-difficulty subsets
        assert "easy/pass@1/accuracy" in m
        assert "hard/pass@1/accuracy" in m
        assert m["easy/pass@1/accuracy"] > m["hard/pass@1/accuracy"]

    def test_no_answer_tracked(self) -> None:
        server = _make_server()
        tasks = [
            [{"reward": 1.0, "extracted_model_code": "x"}, {"reward": 0.0, "extracted_model_code": None}],
        ]
        m = server.compute_metrics(tasks)
        assert "pass@1[avg-of-2]/no_answer" in m
        assert m["pass@1[avg-of-2]/no_answer"] == pytest.approx(50.0)


class TestGetKeyMetrics:
    @pytest.mark.asyncio
    async def test_selects_headline_metrics(self) -> None:
        server = _make_server()
        responses = []
        for task_idx in range(4):
            for rollout_idx in range(3):
                responses.append(
                    {
                        TASK_INDEX_KEY_NAME: task_idx,
                        ROLLOUT_INDEX_KEY_NAME: rollout_idx,
                        "reward": float((task_idx + rollout_idx) % 2),
                        "extracted_model_code": f"print({task_idx})" if rollout_idx < 2 else None,
                        "difficulty": ["easy", "easy", "hard", "hard"][task_idx],
                    }
                )

        result = await server.aggregate_metrics(AggregateMetricsRequest(verify_responses=responses))
        km = result.key_metrics

        # Should have token stats, overall pass/majority, and per-subset
        assert "pass@3/accuracy" in km
        assert "majority@3/accuracy" in km
        assert "pass@1[avg-of-3]/accuracy" in km
        assert any(k.startswith("easy/") for k in km)
        assert any(k.startswith("hard/") for k in km)
        # Should NOT have stat suffixes or no_answer
        assert not any("std_dev" in k for k in km)
        assert not any("no_answer" in k for k in km)
