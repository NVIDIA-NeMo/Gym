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
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    EpisodeId,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    TaskId,
)
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from resources_servers.math_formal_lean.app import (
    PROOF_REFINEMENT_RESPONSE_KEY,
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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

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

        await server.verify(MagicMock(session={}), verify_request)

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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

        assert verify_response.compiler_output is not None
        assert verify_response.compiler_output.process_status == "error"
        assert verify_response.compiler_output.stdout == "some output"
        assert "unknown identifier" in verify_response.compiler_output.stderr


class TestMultiTurnSelfCorrection:
    """Tests for multi-turn self-correction functionality.

    Verification reads only the request and always provides error feedback on failure.
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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

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

        verify_response = await server.verify(MagicMock(session={}), verify_request)

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


_HEADER = "import Mathlib\n\n"
_STATEMENT = "theorem test : True := by\n"
_FAILED_COMPILE = {
    "process_status": "error",
    "stdout": "",
    "stderr": "/lean4/my_project/test.lean:3:2: error: unknown tactic",
}


def _response_json(text: str) -> dict:
    return {
        "id": "resp",
        "created_at": 1.0,
        "model": "test_model",
        "object": "response",
        "output": [
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def _episode_seed(resources_session_id: str = "resources-session", **task_data: object) -> dict:
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="math_formal_lean", task_id="0"),
        task_data={"header": _HEADER, "formal_statement": _STATEMENT} | task_data,
    ).model_dump(mode="json")


def _verify_body(text: str) -> dict:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "Prove True"}]},
        "response": _response_json(text),
        "header": _HEADER,
        "formal_statement": _STATEMENT,
    }


class TestEpisodeSessions:
    """Episode sessions keep the task for /check_proof; /verify still scores from the request."""

    @pytest.fixture
    def server(self) -> MathFormalLeanResourcesServer:
        server = MathFormalLeanResourcesServer(
            config=MathFormalLeanResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="lean"),
            server_client=MagicMock(spec=ServerClient),
        )
        server._sandbox_client.execute_lean4 = AsyncMock(return_value=_FAILED_COMPILE)
        return server

    def test_follows_the_session_contract(self, server) -> None:
        check_resources_session_contract(
            server.setup_webserver(),
            ResourcesSeedSessionRequest.model_validate(_episode_seed("contract-session")),
            keeps_state=True,
            holds_state=lambda resources_session_id: (
                resources_session_id in server._episode_tasks or resources_session_id in server._passed_checks
            ),
            verify_body=_verify_body("```lean4\ntrivial\n```"),
        )
        # A verify after close is masked without compiling anything.
        server._sandbox_client.execute_lean4.assert_not_awaited()

    def test_check_proof_reports_what_verify_scores(self, server) -> None:
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=_episode_seed()).status_code == 200
        generation = "Proof:\n```lean4\ntheorem test : True := by\n  wrong_tactic\n```"

        check = client.post("/check_proof", json={"proof": generation})
        verify = client.post("/verify", json=_verify_body(generation))

        assert check.status_code == 200 and verify.status_code == 200
        checked, verified = check.json(), verify.json()
        assert "reward" not in checked
        assert verified["reward"] == 0.0
        for key in ("proof_status", "predicted_proof", "compiler_output", "needs_correction", "error_feedback"):
            assert checked[key] == verified[key], key
        assert checked["correction_prompt"] == verified["correction_prompt"]
        assert checked["needs_correction"] is True
        # Both compiled the same file, built from the seeded task and the generation.
        compiled = [call.kwargs["code"] for call in server._sandbox_client.execute_lean4.await_args_list]
        assert compiled == [_HEADER + _STATEMENT + "  wrong_tactic"] * 2

    def test_check_proof_needs_an_open_episode_session(self, server) -> None:
        client = TestClient(server.setup_webserver())
        # An agent's /run seeds with its row and keeps nothing to check against.
        assert (
            client.post("/seed_session", json={"header": _HEADER, "formal_statement": _STATEMENT}).status_code == 200
        )
        refused = client.post("/check_proof", json={"proof": "trivial"})
        assert refused.status_code == 404
        assert refused.json()["detail"]["failure_kind"] == "session_lost"

        client.cookies.clear()
        seed = _episode_seed()
        assert client.post("/seed_session", json=seed).status_code == 200
        close = ResourcesCloseSessionRequest(
            resources_session_id=seed["resources_session_id"], episode_id=seed["episode_id"]
        ).model_dump(mode="json")
        assert client.post("/close_session", json=close).status_code == 200
        assert client.post("/check_proof", json={"proof": "trivial"}).status_code == 404
        server._sandbox_client.execute_lean4.assert_not_awaited()

    def test_seed_refuses_a_task_without_a_theorem(self, server) -> None:
        client = TestClient(server.setup_webserver())
        seed = _episode_seed()
        del seed["task_data"]["formal_statement"]
        refused = client.post("/seed_session", json=seed)
        assert refused.status_code == 422
        assert "formal_statement" in refused.text

    @pytest.mark.parametrize("malformed", ["no_task_id", "extra_field"])
    def test_seed_refuses_a_malformed_typed_seed(self, server, malformed: str) -> None:
        # A typed seed that fails validation must not fall through to the cookie seed, which keeps nothing.
        client = TestClient(server.setup_webserver())
        seed = _episode_seed()
        if malformed == "no_task_id":
            del seed["task_id"]
        else:
            seed["unexpected"] = True
        assert client.post("/seed_session", json=seed).status_code == 422
        assert server._episode_tasks == {}

    async def test_verify_reports_the_attempts_of_a_session_activation(self, server) -> None:
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={"process_status": "completed", "stdout": "", "stderr": ""}
        )
        first = {"turn_index": 0, "generation": "bad", "proof_status": "error", "reward": 0.0}
        last = {"turn_index": 1, "generation": "good"}
        body = _verify_body("```lean4\ntrivial\n```")
        body["response"][PROOF_REFINEMENT_RESPONSE_KEY] = {"total_turns": 2, "all_attempts": [first, last]}

        result = await server.verify(MagicMock(session={}), MathFormalLeanVerifyRequest.model_validate(body))

        assert result.reward == 1.0
        assert result.total_turns == 2
        assert result.turn_index == 1
        # The last attempt is graded by this verification; earlier ones keep what their check reported.
        assert result.all_attempts == [
            first,
            last | {"proof_status": "completed", "reward": 1.0, "error_feedback": None, "correction_prompt": None},
        ]
        dumped = result.model_dump(mode="json")
        assert PROOF_REFINEMENT_RESPONSE_KEY not in dumped["response"]

    async def test_verify_without_session_attempts_adds_no_fields(self, server) -> None:
        result = await server.verify(
            MagicMock(session={}), MathFormalLeanVerifyRequest.model_validate(_verify_body("trivial"))
        )
        dumped = result.model_dump(mode="json")
        assert "total_turns" not in dumped and "all_attempts" not in dumped

    def test_verify_reuses_a_pass_that_check_proof_compiled(self, server) -> None:
        client = TestClient(server.setup_webserver())
        assert client.post("/seed_session", json=_episode_seed()).status_code == 200
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={"process_status": "completed", "stdout": "", "stderr": ""}
        )
        passing = "```lean4\ntheorem test : True := by\n  trivial\n```"

        assert client.post("/check_proof", json={"proof": passing}).json()["proof_status"] == "completed"
        verified = client.post("/verify", json=_verify_body(passing)).json()

        assert verified["reward"] == 1.0
        # A sandbox failure on a second compile could fail a proof that passed, so there is none.
        assert server._sandbox_client.execute_lean4.await_count == 1
        # Another file is compiled.
        assert client.post("/verify", json=_verify_body("```lean4\nsimp\n```")).json()["reward"] == 1.0
        assert server._sandbox_client.execute_lean4.await_count == 2

    async def test_retire_releases_the_episode_records(self, server) -> None:
        client = TestClient(server.setup_webserver())
        seed = _episode_seed()
        assert client.post("/seed_session", json=seed).status_code == 200
        server._sandbox_client.execute_lean4 = AsyncMock(
            return_value={"process_status": "completed", "stdout": "", "stderr": ""}
        )
        assert client.post("/check_proof", json={"proof": "trivial"}).status_code == 200
        sid = seed["resources_session_id"]
        assert sid in server._episode_tasks and sid in server._passed_checks

        await server.retire_session_state(sid)

        assert sid not in server._episode_tasks and sid not in server._passed_checks
        # Retiring does not fence the ID: a restore may seed it again.
        assert client.post("/seed_session", json=seed).status_code == 200
