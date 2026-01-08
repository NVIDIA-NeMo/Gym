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

from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.math_formal_lean.app import (
    MathFormalLeanResourcesServer,
    MathFormalLeanResourcesServerConfig,
    MathFormalLeanVerifyRequest,
)


class TestMathFormalLeanApp:
    @pytest.fixture
    def config(self) -> MathFormalLeanResourcesServerConfig:
        return MathFormalLeanResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="math_formal_lean",
            sandbox_host="127.0.0.1",
            sandbox_port=6000,
            compilation_timeout=30.0,
        )

    @pytest.fixture
    def server(self, config) -> MathFormalLeanResourcesServer:
        return MathFormalLeanResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

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
    async def test_verify_successful_proof(self, server):
        """Test that a successful proof compilation returns reward 1.0."""
        # Mock the sandbox client to return a successful compilation
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={"process_status": "completed", "stdout": "", "stderr": ""}
        )

        # Create a mock model response with a valid proof
        generation = """Here's the proof:
```lean4
theorem test : True := by
  trivial
```"""
        response = self._create_response(text=generation)

        verify_request = MathFormalLeanVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Prove True"}]
            ),
            response=response,
            header="import Mathlib\n\n",
            formal_statement="theorem test : True := by\n",
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward == 1.0
        assert verify_response.proof_status == "completed"
        assert "trivial" in verify_response.predicted_proof

    @pytest.mark.asyncio
    async def test_verify_failed_proof(self, server):
        """Test that a failed proof compilation returns reward 0.0."""
        # Mock the sandbox client to return an error
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={"process_status": "error", "stdout": "", "stderr": "type mismatch"}
        )

        generation = """```lean4
theorem test : True := by
  wrong_tactic
```"""
        response = self._create_response(text=generation)

        verify_request = MathFormalLeanVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Prove True"}]
            ),
            response=response,
            header="import Mathlib\n\n",
            formal_statement="theorem test : True := by\n",
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward == 0.0
        assert verify_response.proof_status == "error"

    @pytest.mark.asyncio
    async def test_verify_timeout(self, server):
        """Test that a timeout returns reward 0.0."""
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={"process_status": "timeout", "stdout": "", "stderr": "Client timed out"}
        )

        generation = "simp"
        response = self._create_response(text=generation)

        verify_request = MathFormalLeanVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Prove something"}]
            ),
            response=response,
            header="import Mathlib\n\n",
            formal_statement="theorem test : True := by\n",
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward == 0.0
        assert verify_response.proof_status == "timeout"

    @pytest.mark.asyncio
    async def test_verify_has_sorry(self, server):
        """Test that a proof with sorry returns reward 0.0."""
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={
                "process_status": "completed",
                "stdout": "warning: declaration uses 'sorry'",
                "stderr": "",
            }
        )

        generation = "sorry"
        response = self._create_response(text=generation)

        verify_request = MathFormalLeanVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Prove something"}]
            ),
            response=response,
            header="import Mathlib\n\n",
            formal_statement="theorem test : True := by\n",
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward == 0.0
        assert verify_response.proof_status == "has_sorry"

    @pytest.mark.asyncio
    async def test_verify_empty_generation(self, server):
        """Test that empty generation returns reward 0.0."""
        response = self._create_response(text="")

        verify_request = MathFormalLeanVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Prove something"}]
            ),
            response=response,
            header="import Mathlib\n\n",
            formal_statement="theorem test : True := by\n",
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward == 0.0
        assert verify_response.proof_status == "empty_generation"

    @pytest.mark.asyncio
    async def test_verify_builds_correct_proof(self, server):
        """Test that the proof is built correctly with header and formal statement."""
        captured_code = None

        async def capture_code(code, timeout):
            nonlocal captured_code
            captured_code = code
            return {"process_status": "completed", "stdout": "", "stderr": ""}

        server._sandbox_client.execute_lean4 = capture_code

        generation = """```lean4
theorem test (n : Nat) : n + 0 = n := by
  simp
```"""
        response = self._create_response(text=generation)

        verify_request = MathFormalLeanVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Prove n + 0 = n"}]
            ),
            response=response,
            header="import Mathlib\nimport Aesop\n\n",
            formal_statement="theorem test (n : Nat) : n + 0 = n := by\n",
        )

        await server.verify(verify_request)

        # Verify the built proof contains all expected parts
        assert captured_code is not None
        assert "import Mathlib" in captured_code
        assert "import Aesop" in captured_code
        assert "theorem test (n : Nat) : n + 0 = n := by" in captured_code
        assert "simp" in captured_code

    @pytest.mark.asyncio
    async def test_verify_compiler_output_included(self, server):
        """Test that compiler output is included in the response."""
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={
                "process_status": "error",
                "stdout": "some output",
                "stderr": "error: unknown identifier 'wrong_tactic'",
            }
        )

        response = self._create_response(text="wrong_tactic")

        verify_request = MathFormalLeanVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Prove something"}]
            ),
            response=response,
            header="import Mathlib\n\n",
            formal_statement="theorem test : True := by\n",
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.compiler_output is not None
        assert verify_response.compiler_output.process_status == "error"
        assert verify_response.compiler_output.stdout == "some output"
        assert "unknown identifier" in verify_response.compiler_output.stderr
