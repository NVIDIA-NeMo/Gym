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
import json
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from app import (
    MathWithCodeResourcesServer,
    MathWithCodeResourcesServerConfig,
    PythonMathVerifyRequest,
    _answers_match,
    _extract_boxed_answer,
    _normalize_answer,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)


class TestApp:
    """Tests for the math_with_code grader."""

    SERVER_NAME = "math_with_code"

    def test_sanity(self) -> None:
        """Basic instantiation test - always runs."""
        config = MathWithCodeResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        MathWithCodeResourcesServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            (r"The answer is \boxed{42}.", "42"),
            (r"First \boxed{41}, then \boxed{42}.", "42"),
            (r"The answer is \boxed{\frac{1}{2}}.", r"\frac{1}{2}"),
            (r"The answer is \boxed{}.", None),
            (r"The answer is \boxed{42", None),
            ("The answer is 42.", None),
        ],
        ids=[
            "single-boxed-answer",
            "last-boxed-answer",
            "nested-braces",
            "empty-box",
            "unclosed-box",
            "missing-box",
        ],
    )
    def test_extract_boxed_answer(
        self,
        text: str,
        expected: str | None,
    ) -> None:
        assert _extract_boxed_answer(text) == expected

    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("42", "42"),
            ("  42  ", "42"),
            (r"\(42\)", "42"),
            ("$42$", "42"),
            (r"\text{42}", "42"),
            ("1   2", "1 2"),
        ],
        ids=[
            "plain",
            "whitespace",
            "latex-parentheses",
            "dollar-delimiters",
            "text-wrapper",
            "internal-whitespace",
        ],
    )
    def test_normalize_answer(
        self,
        answer: str,
        expected: str,
    ) -> None:
        assert _normalize_answer(answer) == expected

    @pytest.mark.parametrize(
        ("actual", "expected", "matches"),
        [
            ("42", "42", True),
            (r"\(42\)", "42", True),
            ("42", "42.0", True),
            ("0.5", ".5", True),
            ("43", "42", False),
            ("not-a-number", "42", False),
            (None, "42", False),
        ],
        ids=[
            "exact-match",
            "normalized-exact-match",
            "numeric-equivalent",
            "decimal-equivalent",
            "numeric-mismatch",
            "non-numeric-mismatch",
            "missing-answer",
        ],
    )
    def test_answers_match(
        self,
        actual: str | None,
        expected: str,
        matches: bool,
    ) -> None:
        assert _answers_match(actual, expected) is matches


def _session_server() -> tuple[MathWithCodeResourcesServer, FastAPI]:
    config = MathWithCodeResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="math_with_code")
    server = MathWithCodeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return server, server.setup_webserver()


EPISODE_ID = EpisodeId(rollout_id="rollout", attempt=0)


def _seed(resources_session_id: str) -> Dict[str, Any]:
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=EPISODE_ID,
        task_id=TaskId(taskset="math_with_code", task_id="0"),
        task_data={"expected_result": "4"},
    ).model_dump(mode="json")


def _close(resources_session_id: str) -> Dict[str, Any]:
    return ResourcesCloseSessionRequest(resources_session_id=resources_session_id, episode_id=EPISODE_ID).model_dump(
        mode="json"
    )


def _verify_body(*output: dict[str, Any]) -> Dict[str, Any]:
    return {
        "responses_create_params": {"input": [{"role": "user", "content": "What is 2 + 2?"}]},
        "response": {
            "id": "resp_test",
            "created_at": 0.0,
            "model": "dummy",
            "object": "response",
            "output": list(output),
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        },
        "expected_result": "4",
    }


def _message(text: str) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "content": [{"annotations": [], "text": text, "type": "output_text"}],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }


def _tool_output(output: str) -> dict[str, Any]:
    return {"call_id": "call-1", "output": output, "type": "function_call_output"}


VERIFY_BODY = _verify_body(_message("\\boxed{4}"))
# What Simple Agent's execute_python Python tool returns: the JSON this server's own tool returned before it moved.
PYTHON_TOOL_OUTPUT = json.dumps(
    {"success": True, "stdout": "\\boxed{4}\n", "stderr": "", "error_message": None, "result": None},
    separators=(",", ":"),
)


class TestResourcesSessions:
    def test_the_server_runs_no_code(self) -> None:
        """The Python tool moved to the agent's sandbox: this server serves no tool, only grading and sessions."""
        _, app = _session_server()
        paths = {route.path for route in app.routes}
        assert "/verify" in paths
        assert not {"/execute_python", "/end_session"} & paths
        assert TestClient(app).post("/execute_python", json={"code": "1 + 1"}).status_code == 404

    @pytest.mark.parametrize(
        ("output", "reward", "extracted"),
        [
            pytest.param([_message("\\boxed{4}")], 1.0, "4", id="assistant-boxed"),
            # The model may print the boxed answer from its code instead of writing it.
            pytest.param([_tool_output(PYTHON_TOOL_OUTPUT), _message("Done.")], 1.0, "4", id="python-tool-stdout"),
            # The assistant's last boxed answer wins over one its code printed.
            pytest.param([_tool_output(PYTHON_TOOL_OUTPUT), _message("\\boxed{5}")], 0.0, "5", id="assistant-wins"),
            pytest.param([_tool_output("not json \\boxed{4}")], 1.0, "4", id="plain-text-tool-output"),
            pytest.param([_message("no answer")], 0.0, None, id="no-answer"),
        ],
    )
    def test_verify_grades_the_final_answer(self, output: list, reward: float, extracted: str | None) -> None:
        _, app = _session_server()
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200

        verified = client.post("/verify", json=_verify_body(*output)).json()

        assert (verified["reward"], verified["extracted_answer"]) == (reward, extracted)
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200

    async def test_request_scored_verify_retries_after_scoring_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        server, _ = _session_server()
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "rs-1"}
        await server.seed_session(ResourcesSeedSessionRequest.model_validate(_seed("rs-1")), request=request)
        calls = 0

        def fail_once_then_score(actual, expected):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ValueError("scoring unavailable")
            return _answers_match(actual, expected)

        monkeypatch.setattr("app._answers_match", fail_once_then_score)
        body = PythonMathVerifyRequest.model_validate(VERIFY_BODY)
        with pytest.raises(ValueError, match="scoring unavailable"):
            await server.verify(request, body)
        retried = await server.verify(request, body)
        assert retried.reward == 1.0 and not retried.mask_sample
        assert calls == 2
        await server.close_resources_session(ResourcesCloseSessionRequest.model_validate(_close("rs-1")))

    def test_the_seed_declares_that_the_verifier_executes_nothing(self) -> None:
        """Only this declaration lets an Environment Server run /verify outside the attempt's sandbox."""
        _, app = _session_server()
        seeded = TestClient(app).post("/seed_session", json=_seed("rs-1"))
        assert seeded.status_code == 200 and seeded.json()["verifier_executes_nothing"] is True

    def test_session_contract(self) -> None:
        server, app = _session_server()

        check_resources_session_contract(
            app,
            ResourcesSeedSessionRequest.model_validate(_seed("contract-session")),
            keeps_state=True,
            holds_state=lambda resources_session_id: resources_session_id in server._session_episodes,
            verify_body=VERIFY_BODY,
        )

    def test_verify_then_close(self) -> None:
        server, app = _session_server()
        client = TestClient(app)
        assert client.post("/seed_session", json=_seed("rs-1")).status_code == 200

        verified = client.post("/verify", json=VERIFY_BODY)

        assert verified.status_code == 200 and verified.json()["reward"] == 1.0
        assert client.post("/close_session", json=_close("rs-1")).status_code == 200
        assert "rs-1" not in server._session_episodes

    def test_cookie_seed_keeps_working(self) -> None:
        server, app = _session_server()
        client = TestClient(app)

        # An Agent's /run seeds with the row itself.
        seeded = client.post("/seed_session", json=VERIFY_BODY)
        assert seeded.status_code == 200 and seeded.json() == {}
        assert server._session_episodes == {}
        assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 1.0
