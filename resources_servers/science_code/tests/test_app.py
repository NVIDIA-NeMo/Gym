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
"""
Tests for ScienceCode resources server.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.science_code.app import (
    FailureCode,
    ScienceCodeResourcesServer,
    ScienceCodeResourcesServerConfig,
    ScienceCodeVerifyRequest,
    extract_code_from_response,
)


class TestExtractCodeFromResponse:
    """Tests for code extraction from model responses."""

    def test_extract_from_python_code_block(self):
        text = """Here's the solution:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

This implements the recursive Fibonacci."""
        result = extract_code_from_response(text)
        assert result is not None
        assert "def fibonacci(n):" in result
        assert "return fibonacci(n-1)" in result

    def test_extract_from_generic_code_block(self):
        text = """```
import numpy as np

def solve(x):
    return np.sqrt(x)
```"""
        result = extract_code_from_response(text)
        assert result is not None
        assert "import numpy as np" in result

    def test_extract_raw_python_def(self):
        text = """The solution is:

def compute(x, y):
    return x + y

That should work."""
        result = extract_code_from_response(text)
        assert result is not None
        assert "def compute(x, y):" in result

    def test_extract_raw_python_import(self):
        text = """import numpy as np
from scipy import linalg

def solve(A, b):
    return linalg.solve(A, b)"""
        result = extract_code_from_response(text)
        assert result is not None
        assert "import numpy" in result

    def test_no_code_found(self):
        text = "This response contains no Python code, just plain text."
        result = extract_code_from_response(text)
        assert result is None

    def test_empty_text(self):
        assert extract_code_from_response("") is None
        assert extract_code_from_response(None) is None

    def test_multiple_code_blocks_returns_last(self):
        text = """First attempt:
```python
def wrong():
    pass
```

Better solution:
```python
def correct(x):
    return x * 2
```"""
        result = extract_code_from_response(text)
        assert result is not None
        assert "def correct(x):" in result
        assert "wrong" not in result

    def test_extract_multiline_function(self):
        text = """```python
import numpy as np

def trapezoidal(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    return h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
```"""
        result = extract_code_from_response(text)
        assert result is not None
        assert "def trapezoidal" in result
        assert "np.linspace" in result


class TestScienceCodeResourcesServerVerify:
    """Tests for the verify method."""

    @pytest.fixture
    def resources_server(self) -> ScienceCodeResourcesServer:
        """Create a ScienceCodeResourcesServer instance for testing."""
        config = ScienceCodeResourcesServerConfig(
            host="127.0.0.1",
            port=20002,
            entrypoint="",
            name="science_code_test_server",
            judge_model_server={"name": "test_judge", "type": "responses_api_models"},
            judge_responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content="test")]
            ),
            check_twice_swap=False,
        )

        server = ScienceCodeResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        return server

    def _create_verify_request(
        self,
        model_output: str,
        problem: str = "Implement a function that computes the factorial",
        solution: str = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    ) -> ScienceCodeVerifyRequest:
        """Helper to create a ScienceCodeVerifyRequest."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text=model_output)],
        )
        response = NeMoGymResponse(
            id="test_response",
            created_at=1000,
            model="test_model",
            object="response",
            output=[output_message],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        return ScienceCodeVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[NeMoGymEasyInputMessage(role="user", content=problem)]
            ),
            response=response,
            problem=problem,
            solution=solution,
        )

    @pytest.mark.asyncio
    async def test_verify_no_code_extracted(self, resources_server: ScienceCodeResourcesServer):
        """Test verify returns reward=0.0 when no code is found."""
        request = self._create_verify_request(
            model_output="This response has no code at all.",
        )

        result = await resources_server.verify(request)

        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.NO_CODE_EXTRACTED
        assert result.extracted_code is None
        assert result.judge_passed is False

    @pytest.mark.asyncio
    async def test_verify_judge_passes(self, resources_server: ScienceCodeResourcesServer):
        """Test verify returns reward=1.0 when judge passes."""
        resources_server.config.check_twice_swap = False

        request = self._create_verify_request(
            model_output="```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n```",
        )

        # Mock judge to return equal
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        result = await resources_server.verify(request)

        assert result.reward == 1.0
        assert result.judge_passed is True
        assert result.failure_reason == FailureCode.NONE
        assert result.extracted_code is not None

    @pytest.mark.asyncio
    async def test_verify_judge_fails(self, resources_server: ScienceCodeResourcesServer):
        """Test verify returns reward=0.0 when judge fails."""
        request = self._create_verify_request(
            model_output="```python\ndef wrong(n):\n    return 0\n```",
        )

        # Mock judge to return not equal
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A!=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        result = await resources_server.verify(request)

        assert result.reward == 0.0
        assert result.judge_passed is False
        assert result.failure_reason == FailureCode.JUDGE_EVALUATION_FAILED

    @pytest.mark.asyncio
    async def test_verify_with_swap_check(self, resources_server: ScienceCodeResourcesServer):
        """Test verify with swap check enabled."""
        resources_server.config.check_twice_swap = True

        request = self._create_verify_request(
            model_output="```python\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)\n```",
        )

        # Mock judge to return equal for both passes
        mock_response = MagicMock()
        mock_response.json = AsyncMock(
            return_value={
                "id": "judge_resp",
                "created_at": 1000,
                "model": "judge",
                "object": "response",
                "output": [
                    {
                        "id": "msg_judge",
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "[[A=B]]", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            }
        )
        resources_server.server_client.post = AsyncMock(return_value=mock_response)

        result = await resources_server.verify(request)

        assert result.reward == 1.0
        assert result.judge_passed is True
        assert len(result.judge_evaluations) == 2

    def test_verify_missing_problem_field(self):
        """Test that missing problem field raises ValidationError."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text="test")],
        )
        response = NeMoGymResponse(
            id="test_response",
            created_at=1000,
            model="test_model",
            object="response",
            output=[output_message],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        with pytest.raises(ValidationError, match="problem"):
            ScienceCodeVerifyRequest(
                solution="def test(): pass",
                response=response,
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(role="user", content="test")]
                ),
            )

    def test_verify_missing_solution_field(self):
        """Test that missing solution field raises ValidationError."""
        output_message = NeMoGymResponseOutputMessage(
            id="msg_1",
            content=[NeMoGymResponseOutputText(annotations=[], text="test")],
        )
        response = NeMoGymResponse(
            id="test_response",
            created_at=1000,
            model="test_model",
            object="response",
            output=[output_message],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        with pytest.raises(ValidationError, match="solution"):
            ScienceCodeVerifyRequest(
                problem="Test problem",
                response=response,
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                    input=[NeMoGymEasyInputMessage(role="user", content="test")]
                ),
            )
