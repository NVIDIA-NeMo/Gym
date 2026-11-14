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
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi.testclient import TestClient
from multidict import CIMultiDict, CIMultiDictProxy
from pydantic import BaseModel
from yarl import URL

from nemo_gym.agents import (
    AGENT_OUTCOME_METADATA_KEY,
    AgentSeedSessionRequest,
    OutcomeAction,
    agent_outcome_policy,
    check_agent_seed_rejected,
)
from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import (
    EpisodeId,
    FailureBoundary,
    FailureEvent,
    FailureHeader,
    FailureReport,
    NeMoGymResponseCreateParamsNonStreaming,
    TaskId,
)
from nemo_gym.server_runtime import BaseSeedSessionRequest, ResourcesSeedSessionRequest
from resources_servers.math_formal_lean.app import (
    CheckProofRequest,
    MathFormalLeanResourcesServer,
    MathFormalLeanResourcesServerConfig,
    MathFormalLeanVerifyRequest,
)
from responses_api_agents.proof_refinement_agent.app import (
    PROOF_REFINEMENT_RESPONSE_KEY,
    ModelServerRef,
    ProofRefinementAgent,
    ProofRefinementAgentConfig,
    ProofRefinementRunRequest,
    ResourcesServerRef,
)


class _FakeResp:
    ok = True
    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.cookies: dict[str, str] = {}

    async def json(self) -> dict[str, Any]:
        return self._payload

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode()


class _FakeErrorResp(_FakeResp):
    ok = False

    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        super().__init__(payload)
        self.status = status
        self.content = self

    def raise_for_status(self) -> None:
        raise RuntimeError(f"HTTP {self.status}")


class _FakeModelErrorResp(_FakeErrorResp):
    """A model server reply with an HTTP error status, raised as aiohttp raises it, with an optional failure report."""

    def __init__(self, status: int, payload: dict[str, Any], *, report: FailureReport | None = None) -> None:
        super().__init__(status, payload)
        self.report = report

    def raise_for_status(self) -> None:
        url = URL("http://policy:8000/v1/responses")
        raise ClientResponseError(
            RequestInfo(url=url, method="POST", headers=CIMultiDictProxy(CIMultiDict()), real_url=url),
            (),
            status=self.status,
            message="Bad Request" if self.status < 500 else "Service Unavailable",
            headers=FailureHeader.encode(self.report) if self.report is not None else None,
        )


def _model_report(http_status: int | None) -> FailureReport:
    """A model server's report whose selected failure is a model request that ended with ``http_status``."""
    event = FailureEvent(
        boundary=FailureBoundary.MODEL,
        failure_kind="model_request_error",
        failure_reason="model request failed",
        operation="model_request",
        operation_id="op",
        attempt_id="op-1",
        http_status=http_status,
    )
    return FailureReport(failure_id=event.event_id, events={event.event_id: event})


def _usage(input_tokens: int, output_tokens: int, *, cached: int = 0, reasoning: int = 0) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": reasoning},
        "total_tokens": input_tokens + output_tokens,
    }


def _model_response(response_id: str, text: str, usage: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "policy-model",
        "object": "response",
        "output": [
            {
                "id": f"msg-{response_id}",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage,
    }


class TestApp:
    def test_sanity(self) -> None:
        config = ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="",
            ),
        )
        ProofRefinementAgent(config=config, server_client=MagicMock(spec=ServerClient))

    def test_config_defaults(self) -> None:
        """Test that config has correct default values."""
        config = ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            resources_server=ResourcesServerRef(
                type="resources_servers",
                name="math_formal_lean",
            ),
            model_server=ModelServerRef(
                type="responses_api_models",
                name="policy_model",
            ),
        )
        assert config.max_correction_turns == 0  # Default: single-turn
        assert config.include_all_attempts is True  # Default: include all attempts

    async def test_run_sums_usage_across_correction_turns(self) -> None:
        config = ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="proof_refinement_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="math_formal_lean"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            max_correction_turns=2,
        )
        body = ProofRefinementRunRequest(responses_create_params={"input": "Prove it."})
        first = _model_response("resp-0", "proof attempt", _usage(300, 1400, cached=100, reasoning=1200))
        second = _model_response("resp-1", "fixed proof", _usage(1818, 24, cached=300))

        def verify_result(response: dict[str, Any], *, needs_correction: bool) -> dict[str, Any]:
            # math_formal_lean's /verify echoes the request body, including the model response.
            return body.model_dump() | {
                "response": response,
                "reward": 0.0 if needs_correction else 1.0,
                "proof_status": "error" if needs_correction else "success",
                "needs_correction": needs_correction,
                "correction_prompt": "Fix the proof." if needs_correction else None,
            }

        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            side_effect=[
                _FakeResp({}),  # /seed_session
                _FakeResp(first),
                _FakeResp(verify_result(first, needs_correction=True)),
                _FakeResp(second),
                _FakeResp(verify_result(second, needs_correction=False)),
            ]
        )
        agent = ProofRefinementAgent(config=config, server_client=server_client)

        result = await agent.run(MagicMock(cookies={}), body)

        assert result.total_turns == 2
        assert [attempt["response"]["id"] for attempt in result.all_attempts] == ["resp-0", "resp-1"]
        assert result.response.id == "resp-1"
        # Usage covers every model call in the episode, not just the last one.
        expected_usage = _usage(2118, 1424, cached=400, reasoning=1200)
        expected_usage["input_tokens_details"]["cache_write_tokens"] = None
        assert result.response.usage.model_dump() == expected_usage

    async def test_correction_turns_keep_the_callers_sampling_params(self) -> None:
        config = ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="proof_refinement_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="math_formal_lean"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            max_correction_turns=1,
        )
        body = ProofRefinementRunRequest(
            responses_create_params={"input": "Prove it.", "max_output_tokens": 1024, "temperature": 0.6, "top_p": 0.9}
        )
        first = _model_response("resp-0", "proof attempt", _usage(10, 20))
        second = _model_response("resp-1", "fixed proof", _usage(30, 40))
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            side_effect=[
                _FakeResp({}),  # /seed_session
                _FakeResp(first),
                _FakeResp(
                    body.model_dump()
                    | {"response": first, "reward": 0.0, "needs_correction": True, "correction_prompt": "Fix it."}
                ),
                _FakeResp(second),
                _FakeResp(body.model_dump() | {"response": second, "reward": 1.0, "needs_correction": False}),
            ]
        )
        agent = ProofRefinementAgent(config=config, server_client=server_client)

        await agent.run(MagicMock(cookies={}), body)

        model_calls = [
            c.kwargs["json"]
            for c in server_client.post.await_args_list
            if c.kwargs["url_path"].endswith("/v1/responses")
        ]
        correction_request = model_calls[1]
        assert correction_request["input"] == [{"role": "user", "content": "Fix it."}]
        # The output cap must survive the correction turn (Responses calls it max_output_tokens, not max_tokens).
        assert correction_request["max_output_tokens"] == 1024
        assert correction_request["temperature"] == 0.6
        assert correction_request["top_p"] == 0.9


def _agent(*, max_correction_turns: int = 2, server_client: Any = None) -> ProofRefinementAgent:
    return ProofRefinementAgent(
        config=ProofRefinementAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="proof_refinement_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="lean"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            max_correction_turns=max_correction_turns,
        ),
        server_client=server_client or MagicMock(spec=ServerClient),
    )


_GRANT = {
    "kind": "direct_http",
    "name": "lean.direct_http",
    "required": True,
    "base_url": "http://lean:9000/ng-rollout/r",
    "cookies": {"lean_cookie": "episode"},
}


def _agent_seed(*tool_accesses: dict[str, Any]) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session",
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="math_formal_lean", task_id="0"),
        tool_accesses=list(tool_accesses),
    )


def _check(*, needs_correction: bool) -> dict[str, Any]:
    return {
        "proof_status": "error" if needs_correction else "completed",
        "predicted_proof": "theorem t : True := by ...",
        "needs_correction": needs_correction,
        "error_feedback": "Error 1: unknown tactic" if needs_correction else None,
        "correction_prompt": "Fix the proof." if needs_correction else None,
    }


class TestAgentSessions:
    def test_seed_needs_the_direct_http_grant_to_correct(self) -> None:
        detail = check_agent_seed_rejected(_agent().setup_webserver(), _agent_seed())
        assert "resources_tool_transports: [direct_http]" in detail

        # Without corrections there is nothing to check, so no grant is needed.
        agent = _agent(max_correction_turns=0)
        with TestClient(agent.setup_webserver()) as client:
            assert client.post("/v1/agent_sessions", json=_agent_seed().model_dump(mode="json")).status_code == 200

    @pytest.mark.parametrize("kind", ["mcp", "sandbox", "two_direct_http"])
    def test_seed_refuses_grants_it_cannot_use(self, kind: str) -> None:
        seed = _agent_seed(_GRANT).model_dump(mode="json")
        if kind == "mcp":
            seed["tool_accesses"].append(
                {
                    "kind": "mcp",
                    "name": "search",
                    "required": True,
                    "connection": {"transport": "streamable_http", "url": "http://search:9000/mcp"},
                }
            )
        elif kind == "sandbox":
            seed["sandbox_access"] = {
                "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {}},
                "workdir": "/workspace",
            }
        else:
            seed["tool_accesses"].append(_GRANT | {"name": "other.direct_http"})
        agent = _agent()
        check_agent_seed_rejected(agent.setup_webserver(), AgentSeedSessionRequest.model_validate(seed))
        assert not agent._session_records

    def test_activation_runs_the_correction_loop(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            side_effect=[
                _FakeResp(_model_response("resp-0", "bad proof", _usage(10, 20))),
                _FakeResp(_model_response("resp-1", "good proof", _usage(30, 40))),
            ]
        )
        agent = _agent(server_client=server_client)
        seed = _agent_seed(_GRANT)
        path = f"/ng-rollout/{seed.episode_id.capture_key}/v1/responses"
        body = {"input": [{"role": "user", "content": "Prove it."}], "max_output_tokens": 256}
        check = AsyncMock(
            side_effect=[_FakeResp(_check(needs_correction=True)), _FakeResp(_check(needs_correction=False))]
        )
        with (
            patch("responses_api_agents.proof_refinement_agent.app.http_request", check),
            TestClient(agent.setup_webserver()) as client,
        ):
            assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).status_code == 200
            assert client.post("/ng-rollout/wrong/v1/responses", json=body).status_code == 409
            result = client.post(path, json=body)
            assert result.status_code == 200, result.text
            # The same activation again joins the first: nothing runs twice.
            assert client.post(path, json=body).json() == result.json()
            capture_path = path.replace("/v1/responses", "/training-token-capture/v1/responses")
            assert client.post(capture_path, json=body).status_code == 409
            assert client.post(path, json={"input": "different"}).status_code == 409
            close = {"agent_session_id": seed.agent_session_id, "episode_id": seed.episode_id.model_dump()}
            closed = client.post("/v1/agent_sessions/close", json=close)

        response = result.json()
        assert response["id"] == "resp-1"
        assert response["usage"] == _usage(40, 60)
        refinement = response[PROOF_REFINEMENT_RESPONSE_KEY]
        assert refinement["total_turns"] == 2
        attempts = refinement["all_attempts"]
        assert [attempt["generation"] for attempt in attempts] == ["bad proof", "good proof"]
        assert [attempt["reward"] for attempt in attempts] == [0.0, 1.0]
        assert attempts[1]["input"] == {
            "input": [{"role": "user", "content": "Fix the proof."}],
            "model": None,
            "max_output_tokens": 256,
        }
        # Every model call goes to the policy under the rollout's path, without the agent's or the grant's cookies.
        assert [call.kwargs["url_path"] for call in server_client.post.await_args_list] == [path, path]
        assert all(not call.kwargs["cookies"] for call in server_client.post.await_args_list)
        # Each check goes through the grant: its base URL, cookies and headers.
        assert [call.kwargs["json"] for call in check.await_args_list] == [
            {"proof": "bad proof"},
            {"proof": "good proof"},
        ]
        for call in check.await_args_list:
            assert call.kwargs["url"] == "http://lean:9000/ng-rollout/r/check_proof"
            assert call.kwargs["cookies"] == {"lean_cookie": "episode"}
        assert closed.status_code == 200, closed.text
        assert closed.json()["resources_cookies"] is None
        # Close returns what the activation did: its two model calls and the two checks between them.
        observations = closed.json()["agent_observations"]
        invocation, *checks = observations["records"]
        assert invocation["kind"] == "agent_invocation"
        assert invocation["status"] == "completed"
        assert [call["response_id"] for call in invocation["model_calls"]] == ["resp-0", "resp-1"]
        assert [(c["tool_call_id"], c["tool_name"], c["status"]) for c in checks] == [
            ("check_proof-0", "check_proof", "completed"),
            ("check_proof-1", "check_proof", "completed"),
        ]
        assert all(c["completed_at"] >= c["started_at"] for c in checks)
        assert observations["gaps"] == []

    @pytest.mark.parametrize(
        ("check_reply", "status"),
        [
            # A resources server without /check_proof: the pairing can never work, so the episode fails once.
            (_FakeErrorResp(404, {"detail": "Not Found"}), 422),
            (_FakeErrorResp(405, {"detail": "Method Not Allowed"}), 422),
            # math_formal_lean lost the session: an infrastructure failure, which stays retryable.
            (_FakeErrorResp(404, {"detail": {"failure_kind": "session_lost", "message": "gone"}}), 500),
        ],
    )
    def test_a_refused_check_fails_the_activation(self, check_reply: _FakeErrorResp, status: int) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=_FakeResp(_model_response("resp-0", "proof", _usage(1, 1))))
        agent = _agent(server_client=server_client)
        seed = _agent_seed(_GRANT)
        path = f"/ng-rollout/{seed.episode_id.capture_key}/v1/responses"
        with (
            patch("responses_api_agents.proof_refinement_agent.app.http_request", AsyncMock(return_value=check_reply)),
            TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client,
        ):
            assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).status_code == 200
            result = client.post(path, json={"input": "Prove it."})
            close = {"agent_session_id": seed.agent_session_id, "episode_id": seed.episode_id.model_dump()}
            closed = client.post("/v1/agent_sessions/close", json=close)

        assert result.status_code == status, result.text
        if status == 422:
            assert "pair only with math_formal_lean" in result.text
        # The failed activation and its refused check still come back at close.
        invocation, check = closed.json()["agent_observations"]["records"]
        assert (invocation["status"], invocation["error_type"]) == ("failed", check["error_type"])
        assert check["status"] == "failed"

    def test_a_session_closed_before_its_activation_reports_a_gap(self) -> None:
        agent = _agent()
        seed = _agent_seed(_GRANT)
        with TestClient(agent.setup_webserver()) as client:
            assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).status_code == 200
            close = {"agent_session_id": seed.agent_session_id, "episode_id": seed.episode_id.model_dump()}
            closed = client.post("/v1/agent_sessions/close", json=close)

        observations = closed.json()["agent_observations"]
        assert observations["records"] == []
        assert [gap["code"] for gap in observations["gaps"]] == ["observation_capture_failed"]

    @pytest.mark.parametrize(
        ("status", "action"),
        [
            # The environment server's default outcome policy grades a refused request and retries an outage.
            (400, OutcomeAction.VERIFY),
            (503, OutcomeAction.RETRY),
        ],
    )
    async def test_a_failed_model_call_returns_its_outcome_and_the_earlier_attempts(
        self, status: int, action: OutcomeAction
    ) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            side_effect=[
                _FakeResp(_model_response("resp-0", "bad proof", _usage(10, 20))),
                _FakeModelErrorResp(status, {"error": {"message": "prompt is too long"}}),
            ]
        )
        agent = _agent(server_client=server_client)
        state = await agent._seed_agent_session_state(_agent_seed(_GRANT))
        check = AsyncMock(return_value=_FakeResp(_check(needs_correction=True)))
        with patch("responses_api_agents.proof_refinement_agent.app.http_request", check):
            response = await agent._refine(
                state,
                NeMoGymResponseCreateParamsNonStreaming(input="Prove it.", metadata={"row": "7"}),
                model_url_path="/v1/responses",
            )

        assert response.status == "failed"
        assert response.output == []
        assert f"Model HTTP {status}" in response.error.message and "prompt is too long" in response.error.message
        assert response.metadata["row"] == "7"
        assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == {
            "failure_kind": "model_request_error",
            "status_code": status,
        }
        assert agent_outcome_policy(response) == action
        # The usage of the call that answered.
        assert response.usage.model_dump() == _usage(10, 20)
        refinement = getattr(response, PROOF_REFINEMENT_RESPONSE_KEY)
        assert refinement["total_turns"] == 2
        first, failed = refinement["all_attempts"]
        assert (first["generation"], first["proof_status"]) == ("bad proof", "error")
        assert failed["input"]["input"] == [{"role": "user", "content": "Fix the proof."}]
        assert failed["generation"] == "" and failed["response"]["status"] == "failed"
        # No check of the failed turn; close reports the failed invocation with the call that answered.
        assert check.await_count == 1
        invocation, checked = state.observations.records
        assert (invocation.status, invocation.error_type) == ("failed", "model_request_error")
        assert [call.response_id for call in invocation.model_calls] == ["resp-0"]
        assert checked.status == "completed"

    async def test_a_refused_first_call_reports_zero_usage(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(return_value=_FakeModelErrorResp(400, {"error": {"message": "too long"}}))
        agent = _agent(server_client=server_client)
        state = await agent._seed_agent_session_state(_agent_seed(_GRANT))
        check = AsyncMock()
        with patch("responses_api_agents.proof_refinement_agent.app.http_request", check):
            response = await agent._refine(
                state, NeMoGymResponseCreateParamsNonStreaming(input="Prove it."), model_url_path="/v1/responses"
            )

        assert response.status == "failed"
        assert response.usage.model_dump() == _usage(0, 0)
        refinement = getattr(response, PROOF_REFINEMENT_RESPONSE_KEY)
        assert refinement["total_turns"] == 1
        assert [attempt["generation"] for attempt in refinement["all_attempts"]] == [""]
        assert check.await_count == 0
        assert state.observations.records[0].model_calls == []

    @pytest.mark.parametrize(
        "report",
        [
            # The model's headers came back 2xx and its body failed: it may already have answered.
            _model_report(200),
            # A report that was too large to send: what it selected cannot be read.
            FailureReport(omitted=True),
        ],
    )
    async def test_a_reply_that_may_hold_an_answer_propagates(self, report: FailureReport) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            side_effect=[
                _FakeResp(_model_response("resp-0", "bad proof", _usage(1, 1))),
                _FakeModelErrorResp(500, {"error": {"message": "body read failed"}}, report=report),
            ]
        )
        agent = _agent(server_client=server_client)
        state = await agent._seed_agent_session_state(_agent_seed(_GRANT))
        check = AsyncMock(return_value=_FakeResp(_check(needs_correction=True)))
        with (
            patch("responses_api_agents.proof_refinement_agent.app.http_request", check),
            pytest.raises(ClientResponseError) as raised,
        ):
            await agent._refine(
                state, NeMoGymResponseCreateParamsNonStreaming(input="Prove it."), model_url_path="/v1/responses"
            )

        # The error goes on with the model server's report, which the agent's reply forwards.
        assert FailureHeader.decode(raised.value.headers) == report
        invocation = state.observations.records[0]
        assert (invocation.status, invocation.error_type) == ("failed", "ClientResponseError")

    async def test_a_reported_provider_refusal_still_returns_its_outcome(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            return_value=_FakeModelErrorResp(400, {"error": {"message": "too long"}}, report=_model_report(400))
        )
        agent = _agent(server_client=server_client)
        state = await agent._seed_agent_session_state(_agent_seed(_GRANT))

        response = await agent._refine(
            state, NeMoGymResponseCreateParamsNonStreaming(input="Prove it."), model_url_path="/v1/responses"
        )

        assert response.status == "failed"
        assert agent_outcome_policy(response) == OutcomeAction.VERIFY

    async def test_last_allowed_attempt_is_left_to_verify(self) -> None:
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(
            side_effect=[
                _FakeResp(_model_response("resp-0", "bad proof", _usage(1, 1))),
                _FakeResp(_model_response("resp-1", "still bad", _usage(1, 1))),
            ]
        )
        agent = _agent(max_correction_turns=1, server_client=server_client)
        state = await agent._seed_agent_session_state(_agent_seed(_GRANT))
        check = AsyncMock(return_value=_FakeResp(_check(needs_correction=True)))
        with patch("responses_api_agents.proof_refinement_agent.app.http_request", check):
            response = await agent._refine(
                state, NeMoGymResponseCreateParamsNonStreaming(input="Prove it."), model_url_path="/v1/responses"
            )

        assert check.await_count == 1
        refinement = getattr(response, PROOF_REFINEMENT_RESPONSE_KEY)
        assert refinement["total_turns"] == 2
        # The unchecked attempt carries no grading fields: math_formal_lean's /verify adds them.
        assert "proof_status" not in refinement["all_attempts"][-1]


# ------------------------------------------------------------------------------------------------
# Reward parity: the same scripted model and Lean compiler through /run and through an agent session
# ------------------------------------------------------------------------------------------------

_ROW = {
    "responses_create_params": {
        "input": [{"role": "user", "content": "Complete the Lean 4 proof."}],
        "temperature": 0.6,
        "max_output_tokens": 512,
    },
    "header": "import Mathlib\n\n",
    "formal_statement": "theorem t : True := by\n",
    "name": "t",
}


def _wire(value: Any) -> Any:
    """A request body as ServerClient sends it."""
    if isinstance(value, BaseModel):
        value = value.model_dump(exclude_unset=True)
    return json.loads(json.dumps(value, default=lambda item: item.model_dump(mode="json")))


class _Episode:
    """A scripted policy and Lean compiler. The policy answers its calls in order; Lean passes `good_tactic`."""

    def __init__(self, proofs: list[str], *, refused_at: int | None = None) -> None:
        self.proofs = proofs
        # The policy call that the model server refuses with HTTP 400, if any.
        self.refused_at = refused_at
        self.model_requests: list[dict[str, Any]] = []
        self.compiled: list[str] = []
        self.lean = MathFormalLeanResourcesServer(
            config=MathFormalLeanResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="lean"),
            server_client=MagicMock(spec=ServerClient),
        )
        self.lean._sandbox_client.execute_lean4 = AsyncMock(side_effect=self._compile)
        server_client = MagicMock(spec=ServerClient)
        server_client.post = AsyncMock(side_effect=self._post)
        self.agent = _agent(max_correction_turns=2, server_client=server_client)

    async def _compile(self, code: str, timeout: float) -> dict[str, Any]:
        self.compiled.append(code)
        if "good_tactic" in code:
            return {"process_status": "completed", "stdout": "", "stderr": ""}
        line = code.count("\n") + 1
        return {"process_status": "error", "stdout": "", "stderr": f"/lean4/my_project/t.lean:{line}:2: error: no"}

    async def _post(self, *, server_name: str, url_path: str, json: Any, cookies: Any = None, **_: Any) -> _FakeResp:
        body = _wire(json)
        if server_name == "policy_model":
            index = len(self.model_requests)
            self.model_requests.append(body)
            if index == self.refused_at:
                return _FakeModelErrorResp(400, {"error": {"message": "prompt is too long"}})
            proof = f"```lean4\ntheorem t : True := by\n  {self.proofs[index]}\n```"
            return _FakeResp(_model_response(f"resp-{index}", proof, _usage(100 * (index + 1), 10)))
        if server_name == "proof_refinement_agent":  # /run's call to the agent's own /v1/responses
            params = NeMoGymResponseCreateParamsNonStreaming.model_validate(body)
            response = await self.agent.responses(MagicMock(cookies={}, session={}), MagicMock(), params)
            return _FakeResp(response.model_dump(mode="json"))
        if url_path == "/seed_session":
            await self.lean.seed_session(MagicMock(session={}), BaseSeedSessionRequest.model_validate(body))
            return _FakeResp({})
        assert url_path == "/verify"
        result = await self.lean.verify(MagicMock(session={}), MathFormalLeanVerifyRequest.model_validate(body))
        return _FakeResp(result.model_dump(mode="json"))

    async def run(self) -> dict[str, Any]:
        """The legacy path: the agent's /run, behind a legacy_agent relay."""
        result = await self.agent.run(MagicMock(cookies={}), ProofRefinementRunRequest.model_validate(_ROW))
        return result.model_dump(mode="json")

    async def session(self) -> dict[str, Any]:
        """The calls single_agent_turn_legacy makes: seed, seed the agent with the grant, one activation, verify."""
        task_data = {key: value for key, value in _ROW.items() if key != "responses_create_params"}
        seed_request = MagicMock(session={})
        await self.lean.seed_session(
            seed_request,
            ResourcesSeedSessionRequest(
                resources_session_id="resources-session",
                episode_id=EpisodeId(rollout_id="rollout", attempt=0),
                task_id=TaskId(taskset="math_formal_lean", task_id="0"),
                task_data=task_data,
            ),
        )
        resources_request = MagicMock(session=dict(seed_request.session))
        state = await self.agent._seed_agent_session_state(_agent_seed(_GRANT))

        async def check_proof(*, url: str, json: dict[str, Any], **_: Any) -> _FakeResp:
            assert url.endswith("/check_proof")
            result = await self.lean.check_proof(resources_request, CheckProofRequest.model_validate(json))
            return _FakeResp(result.model_dump(mode="json"))

        params = NeMoGymResponseCreateParamsNonStreaming.model_validate(_ROW["responses_create_params"])
        with patch("responses_api_agents.proof_refinement_agent.app.http_request", AsyncMock(side_effect=check_proof)):
            response = await self.agent._refine(state, params, model_url_path="/v1/responses")
        verify_body = task_data | {
            "responses_create_params": params.model_dump(mode="json", exclude_unset=True),
            "response": response.model_dump(mode="json"),
        }
        result = await self.lean.verify(resources_request, MathFormalLeanVerifyRequest.model_validate(verify_body))
        assert resources_request.session[SESSION_ID_KEY] == "resources-session"
        return result.model_dump(mode="json")


class TestRewardParity:
    @pytest.mark.parametrize(
        ("proofs", "reward", "total_turns"),
        [
            (["good_tactic"], 1.0, 1),
            (["bad_tactic", "good_tactic"], 1.0, 2),
            (["bad_tactic", "worse_tactic", "still_bad"], 0.0, 3),
            # The last allowed attempt is never checked, so /verify compiles and grades it.
            (["bad_tactic", "worse_tactic", "good_tactic"], 1.0, 3),
        ],
    )
    async def test_session_result_equals_the_run_result(self, proofs: list[str], reward: float, total_turns: int):
        legacy, session = _Episode(proofs), _Episode(proofs)

        run_result = await legacy.run()
        session_result = await session.session()

        assert run_result["reward"] == reward
        assert run_result["total_turns"] == total_turns
        # The whole result row is the same: reward, final verification, attempts, response and its summed usage.
        assert session_result == run_result
        # The policy saw the same requests, correction prompts included.
        assert session.model_requests == legacy.model_requests
        # Lean compiled the same files: /verify reuses a pass that /check_proof compiled instead of compiling it again.
        assert session.compiled == legacy.compiled


async def test_a_graded_model_refusal_scores_zero_and_keeps_the_attempts() -> None:
    """A correction turn the model server refuses with HTTP 400, graded as the default outcome policy says."""
    episode = _Episode(["bad_tactic", "good_tactic"], refused_at=1)

    result = await episode.session()

    assert result["reward"] == 0.0
    assert result["response"]["status"] == "failed"
    assert result["total_turns"] == 2
    first, refused = result["all_attempts"]
    assert first["proof_status"] == "error"
    # /verify grades the refused turn as an empty generation, without compiling anything.
    assert (refused["generation"], refused["proof_status"], refused["reward"]) == ("", "empty_generation", 0.0)
    assert len(episode.compiled) == 1


def test_attempt_records_keep_the_last_messages_text_as_run_always_did() -> None:
    from responses_api_agents.proof_refinement_agent.app import generation_text

    response = _model_response("resp", "first", _usage(1, 1))
    second = json.loads(json.dumps(response["output"][0]))
    second["content"][0]["text"] = "second"
    response["output"].append(second)
    assert generation_text(response) == "second"
    assert generation_text({"output": []}) == ""
