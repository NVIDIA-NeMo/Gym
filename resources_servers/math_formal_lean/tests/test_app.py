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
    REFINEMENT_PROMPT_TEMPLATE,
    MathFormalLeanResourcesServer,
    MathFormalLeanResourcesServerConfig,
    MathFormalLeanVerifyRequest,
    build_correction_prompt,
    format_error_feedback,
    get_error_str,
    parse_error,
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


class TestMultiTurnSelfCorrection:
    """Tests for multi-turn self-correction functionality.

    The resource server is stateless - it always provides error feedback on failure.
    The agent is responsible for controlling the retry loop and turn counting.
    """

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
    async def test_success_returns_no_correction_needed(self, server):
        """Test that success returns reward 1.0 with no correction needed."""
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={"process_status": "completed", "stdout": "", "stderr": ""}
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
            turn_index=0,
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward == 1.0
        assert verify_response.proof_status == "completed"
        assert verify_response.needs_correction is False
        assert verify_response.turn_index == 0
        assert verify_response.correction_prompt is None

    @pytest.mark.asyncio
    async def test_failure_always_returns_needs_correction(self, server):
        """Test that failure always returns needs_correction=True with error feedback."""
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={
                "process_status": "error",
                "stdout": "",
                "stderr": "/lean4/my_project/file.lean:5:10: error: unknown identifier 'wrong'",
            }
        )

        generation = "wrong_tactic"
        response = self._create_response(text=generation)

        verify_request = MathFormalLeanVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Prove something"}]
            ),
            response=response,
            header="import Mathlib\n\n",
            formal_statement="theorem test : True := by\n",
            turn_index=0,
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward == 0.0
        assert verify_response.needs_correction is True
        assert verify_response.turn_index == 0
        assert verify_response.correction_prompt is not None
        assert verify_response.error_feedback is not None
        assert "proof attempt" in verify_response.correction_prompt.lower()

    @pytest.mark.asyncio
    async def test_turn_index_echoed_back(self, server):
        """Test that turn_index from request is echoed back in response."""
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={"process_status": "error", "stdout": "", "stderr": "type mismatch"}
        )

        generation = "wrong_tactic"
        response = self._create_response(text=generation)

        # Test with turn_index=2 (agent tracks this, server just echoes)
        verify_request = MathFormalLeanVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input=[{"role": "user", "content": "Prove something"}]
            ),
            response=response,
            header="import Mathlib\n\n",
            formal_statement="theorem test : True := by\n",
            turn_index=2,
        )

        verify_response = await server.verify(verify_request)

        # Server is stateless - always provides correction prompt on failure
        assert verify_response.reward == 0.0
        assert verify_response.needs_correction is True
        assert verify_response.turn_index == 2
        assert verify_response.correction_prompt is not None

    @pytest.mark.asyncio
    async def test_success_on_correction_turn(self, server):
        """Test that success on any turn returns reward 1.0."""
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={"process_status": "completed", "stdout": "", "stderr": ""}
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
            turn_index=1,  # This is a correction turn
        )

        verify_response = await server.verify(verify_request)

        assert verify_response.reward == 1.0
        assert verify_response.proof_status == "completed"
        assert verify_response.needs_correction is False
        assert verify_response.turn_index == 1


class TestErrorParsing:
    """Tests for error parsing utilities."""

    def test_parse_error_basic(self):
        """Test parsing a basic Lean4 error."""
        log_string = "/lean4/my_project/Main.lean:10:5: error: unknown identifier 'foo'"
        errors = parse_error(log_string)

        assert len(errors) == 1
        assert errors[0]["pos"]["line"] == 10
        assert errors[0]["pos"]["column"] == 5
        assert "unknown identifier" in errors[0]["data"]

    def test_parse_error_multiple(self):
        """Test parsing multiple errors."""
        log_string = """/lean4/my_project/Main.lean:10:5: error: unknown identifier 'foo'
/lean4/my_project/Main.lean:15:3: error: type mismatch"""
        errors = parse_error(log_string)

        assert len(errors) == 2
        assert errors[0]["pos"]["line"] == 10
        assert errors[1]["pos"]["line"] == 15

    def test_parse_error_no_errors(self):
        """Test parsing output with no errors."""
        log_string = "All good, no errors here"
        errors = parse_error(log_string)

        assert len(errors) == 0

    def test_get_error_str_basic(self):
        """Test formatting errors with code context."""
        code = """import Mathlib

theorem test : True := by
  wrong_tactic
  sorry"""
        errors = [{"pos": {"line": 4, "column": 2}, "endPos": None, "data": "unknown tactic 'wrong_tactic'"}]

        error_str = get_error_str(code, errors)

        assert "Error 1:" in error_str
        assert "wrong_tactic" in error_str
        assert "unknown tactic" in error_str

    def test_format_error_feedback_timeout(self):
        """Test formatting timeout error."""
        compiler_output = {"process_status": "timeout", "stdout": "", "stderr": ""}
        feedback = format_error_feedback(compiler_output, "some code")

        assert "timed out" in feedback.lower()

    def test_format_error_feedback_fallback(self):
        """Test fallback when no structured errors found."""
        compiler_output = {
            "process_status": "error",
            "stdout": "some stdout",
            "stderr": "some stderr without structured errors",
        }
        feedback = format_error_feedback(compiler_output, "some code")

        assert "some stdout" in feedback or "some stderr" in feedback


class TestCorrectionPrompt:
    """Tests for correction prompt building."""

    def test_build_correction_prompt_default_template(self):
        """Test building correction prompt with default template."""
        proof_attempt = "wrong_tactic"
        error_message = "unknown tactic 'wrong_tactic'"

        prompt = build_correction_prompt(proof_attempt, error_message)

        assert "proof attempt" in prompt.lower()
        assert "wrong_tactic" in prompt
        assert "unknown tactic" in prompt
        assert "fix this proof" in prompt.lower()

    def test_build_correction_prompt_custom_template(self):
        """Test building correction prompt with custom template."""
        custom_template = "Fix this: {proof_attempt}\nError: {error_message}"
        proof_attempt = "broken_code"
        error_message = "syntax error"

        prompt = build_correction_prompt(proof_attempt, error_message, custom_template)

        assert prompt == "Fix this: broken_code\nError: syntax error"

    def test_refinement_prompt_template_has_placeholders(self):
        """Test that the default template has the expected placeholders."""
        assert "{proof_attempt}" in REFINEMENT_PROMPT_TEMPLATE
        assert "{error_message}" in REFINEMENT_PROMPT_TEMPLATE
