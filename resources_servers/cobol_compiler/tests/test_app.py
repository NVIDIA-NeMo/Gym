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

import shutil
from typing import Generator
from unittest.mock import MagicMock

import pytest
import ray
from app import (
    CobolCompilerConfig,
    CobolCompilerServer,
    CobolVerifyRequest,
    CobolVerifyResponse,
)
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


# Skip all tests if cobc is not available
pytestmark = pytest.mark.skipif(
    shutil.which("cobc") is None,
    reason="GnuCOBOL (cobc) not installed",
)

# A simple COBOL program that reads a number and prints its square
COBOL_SQUARE = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SQUARE.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-LINE       PIC X(20).
       01  WS-N          PIC S9(9).
       01  WS-RESULT     PIC S9(18).
       01  WS-DISPLAY    PIC -(18)9.
       PROCEDURE DIVISION.
           ACCEPT WS-LINE
           MOVE FUNCTION NUMVAL(FUNCTION TRIM(WS-LINE))
               TO WS-N
           COMPUTE WS-RESULT = WS-N * WS-N
           MOVE WS-RESULT TO WS-DISPLAY
           DISPLAY FUNCTION TRIM(WS-DISPLAY)
           STOP RUN.
"""

# A COBOL program that has a compilation error
COBOL_BAD_SYNTAX = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. BAD.
       PROCEDURE DIVISION.
           DISPLAY HELLO
           STOP RUN.
"""

# A COBOL program that compiles but produces wrong output
COBOL_WRONG_OUTPUT = """\
       IDENTIFICATION DIVISION.
       PROGRAM-ID. WRONG.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-LINE       PIC X(20).
       01  WS-N          PIC S9(9).
       01  WS-RESULT     PIC S9(18).
       01  WS-DISPLAY    PIC -(18)9.
       PROCEDURE DIVISION.
           ACCEPT WS-LINE
           MOVE FUNCTION NUMVAL(FUNCTION TRIM(WS-LINE))
               TO WS-N
           COMPUTE WS-RESULT = WS-N + 1
           MOVE WS-RESULT TO WS-DISPLAY
           DISPLAY FUNCTION TRIM(WS-DISPLAY)
           STOP RUN.
"""


def _make_response(text: str) -> NeMoGymResponse:
    """Build a NeMoGymResponse wrapping the given assistant text."""
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
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )


SQUARE_TEST_CASES = {
    "test_cases": [
        {"input": "3", "expected_output": "9", "test_id": 0},
        {"input": "5", "expected_output": "25", "test_id": 1},
    ],
    "task_id": "test/001",
    "entry_point": "square",
}


class TestApp:
    @pytest.fixture(scope="module")
    def cobol_server_client(self) -> Generator[TestClient, None, None]:
        ray.init(num_cpus=1, ignore_reinit_error=True)
        server = CobolCompilerServer(
            config=CobolCompilerConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                num_processes=1,
                timeout_secs=30,
                debug=False,
            ),
            server_client=MagicMock(spec=ServerClient),
        )
        app = server.setup_webserver()
        with TestClient(app) as client:
            yield client

    async def test_verify_pass(self, cobol_server_client: TestClient) -> None:
        """COBOL code that squares correctly should get reward 1.0."""
        verify_req = CobolVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Square a number"}],
            },
            response=_make_response(f"```cobol\n{COBOL_SQUARE}\n```"),
            verifier_metadata=SQUARE_TEST_CASES,
        )
        response = cobol_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CobolVerifyResponse.model_validate(response.json())
        assert res.reward == 1.0
        assert res.compilation_success is True
        assert res.tests_passed == 2
        assert res.tests_total == 2

    async def test_verify_fail_compile_error(self, cobol_server_client: TestClient) -> None:
        """Code with syntax errors should get reward 0.0 and compilation_success=False."""
        verify_req = CobolVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Square a number"}],
            },
            response=_make_response(f"```cobol\n{COBOL_BAD_SYNTAX}\n```"),
            verifier_metadata=SQUARE_TEST_CASES,
        )
        response = cobol_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CobolVerifyResponse.model_validate(response.json())
        assert res.reward == 0.0
        assert res.compilation_success is False
        assert res.compilation_errors is not None
        assert len(res.compilation_errors) > 0

    async def test_verify_fail_wrong_output(self, cobol_server_client: TestClient) -> None:
        """Code that compiles but gives wrong answers should get reward 0.0."""
        verify_req = CobolVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Square a number"}],
            },
            response=_make_response(f"```cobol\n{COBOL_WRONG_OUTPUT}\n```"),
            verifier_metadata=SQUARE_TEST_CASES,
        )
        response = cobol_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CobolVerifyResponse.model_validate(response.json())
        assert res.reward == 0.0
        assert res.compilation_success is True
        assert res.tests_passed < res.tests_total

    async def test_verify_no_code(self, cobol_server_client: TestClient) -> None:
        """Response with no extractable COBOL code should get reward 0.0."""
        verify_req = CobolVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Square a number"}],
            },
            response=_make_response("I cannot write COBOL code for this task."),
            verifier_metadata=SQUARE_TEST_CASES,
        )
        response = cobol_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CobolVerifyResponse.model_validate(response.json())
        assert res.reward == 0.0
        assert res.extracted_code is None

    async def test_verify_empty_response(self, cobol_server_client: TestClient) -> None:
        """Empty model output should get reward 0.0."""
        verify_req = CobolVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Square a number"}],
            },
            response=_make_response(""),
            verifier_metadata=SQUARE_TEST_CASES,
        )
        response = cobol_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CobolVerifyResponse.model_validate(response.json())
        assert res.reward == 0.0

    def test_verify_missing_response_validation_error(self) -> None:
        """Omitting `response` should fail request validation."""
        with pytest.raises(ValidationError):
            CobolVerifyRequest(
                responses_create_params={"input": [{"role": "user", "content": "anything"}]},
                verifier_metadata=SQUARE_TEST_CASES,
            )

    async def test_verify_code_without_fences(self, cobol_server_client: TestClient) -> None:
        """Raw COBOL code without markdown fences should still be extracted."""
        verify_req = CobolVerifyRequest(
            responses_create_params={
                "input": [{"role": "user", "content": "Square a number"}],
            },
            response=_make_response(COBOL_SQUARE),
            verifier_metadata=SQUARE_TEST_CASES,
        )
        response = cobol_server_client.post(
            url="/verify",
            json=verify_req.model_dump(),
        )
        res = CobolVerifyResponse.model_validate(response.json())
        assert res.reward == 1.0
        assert res.extracted_code is not None
