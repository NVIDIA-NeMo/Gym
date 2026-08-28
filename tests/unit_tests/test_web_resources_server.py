# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.web.api_models import (
    WebEvaluateResponse,
    WebSeedSessionResponse,
    WebSessionStatusResponse,
    WebStepResponse,
    WebVerifyRequest,
)
from nemo_gym.web.models import (
    WebArtifactRef,
    WebBenchmark,
    WebObservation,
    WebVerifierResult,
)
from nemo_gym.web.resource_config import WebResourcesServerConfig
from nemo_gym.web.resources_server import WebResourcesServer, _error_response
from nemo_gym.web.session import (
    BenchmarkPreconditionError,
    CapacityUnavailableError,
    EvaluatorConfigurationError,
    EvaluatorInfrastructureError,
    SessionConflictError,
    SessionNotFoundError,
)


class FakeManager:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.closed: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self.failures: dict[str, Exception] = {}
        self.verifier_result = WebVerifierResult(reward=1.0, raw_score=0.75, task_success=True)

    def _record(self, method: str, session_id: str) -> None:
        self.calls.append((method, session_id))
        if method in self.failures:
            raise self.failures[method]

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def seed_session(self, session_id, body):
        self._record("seed_session", session_id)
        return WebSeedSessionResponse(
            session_id=session_id,
            task_id=body.task.task_id,
            status="ready",
            observation=WebObservation(url="https://example.test/start"),
            info={"seeded": True},
        )

    async def session_status(self, session_id):
        self._record("session_status", session_id)
        return WebSessionStatusResponse(
            session_id=session_id,
            task_id="7",
            benchmark="webarena",
            status="ready",
            created_at=1.0,
            last_access_at=2.0,
            site_lease_id="lease-1",
        )

    async def reset_session(self, session_id, body):
        self._record("reset_session", session_id)
        return WebSeedSessionResponse(
            session_id=session_id,
            task_id=body.task.task_id,
            status="ready",
            observation=WebObservation(url="https://example.test/reset"),
        )

    async def observe(self, session_id):
        self._record("observe", session_id)
        return WebObservation(url="https://example.test/observe")

    async def step(self, session_id, body):
        self._record("step", session_id)
        return WebStepResponse(
            operation_id=body.operation_id,
            observation=WebObservation(url="https://example.test/step"),
            execution_ok=True,
        )

    async def evaluate(self, session_id, final_answer=None):
        del final_answer
        self._record("evaluate", session_id)
        return WebEvaluateResponse(result=self.verifier_result)

    async def close_session(self, session_id):
        self._record("close_session", session_id)
        self.closed.append(session_id)
        return True

    async def recording_artifacts(self, session_id):
        self._record("recording_artifacts", session_id)
        return [
            WebArtifactRef(
                uri="file:///recording.webm",
                mime_type="video/webm",
                size_bytes=1,
                sha256="0" * 64,
            )
        ]

    async def health(self):
        if "health" in self.failures:
            raise self.failures["health"]
        return {"status": "ok", "sessions": 0}


def _config(**updates: Any) -> WebResourcesServerConfig:
    values = {
        "name": "web",
        "host": "localhost",
        "port": 8000,
        "entrypoint": "app.py",
        "domain": "agent",
    }
    values.update(updates)
    return WebResourcesServerConfig(**values)


def _server(manager: FakeManager, **config_updates: Any) -> WebResourcesServer:
    class TestWebResourcesServer(WebResourcesServer):
        def make_session_manager(self):
            return manager

    return TestWebResourcesServer(
        config=_config(**config_updates),
        server_client=MagicMock(spec=ServerClient),
    )


def _task_json(task_id: str = "7") -> dict[str, Any]:
    return {"benchmark": WebBenchmark.WEBARENA.value, "task_id": task_id}


def _verify_request() -> WebVerifyRequest:
    return WebVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="task"),
        response=NeMoGymResponse(
            id="response-1",
            created_at=1.0,
            model="test-model",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        ),
        final_answer="done",
    )


def test_error_response_has_stable_machine_readable_shape() -> None:
    response = _error_response(status_code=503, detail="busy", error_kind="capacity", retryable=True)
    assert response.status_code == 503
    assert response.body == b'{"detail":"busy","error_kind":"capacity","retryable":true}'


def test_abstract_manager_factory_fails_explicitly() -> None:
    with pytest.raises(NotImplementedError):
        WebResourcesServer.make_session_manager(object())


def test_http_contract_routes_lifecycle_and_session_cookie() -> None:
    manager = FakeManager()
    app = _server(manager).setup_webserver()
    assert {
        "/seed_session",
        "/verify",
        "/healthz",
        "/session",
        "/reset",
        "/observe",
        "/step",
        "/evaluate",
        "/close",
    }.issubset({route.path for route in app.routes})

    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        seed = client.post("/seed_session", json={"task": _task_json()})
        status = client.get("/session")
        observe = client.get("/observe")
        reset = client.post("/reset", json={"task": _task_json()})
        step = client.post(
            "/step",
            json={
                "operation_id": "operation-1",
                "action": {"name": "noop", "script": "noop()"},
            },
        )
        evaluate = client.post("/evaluate", json={"final_answer": "done"})
        close = client.post("/close")

    assert manager.started == 1
    assert manager.stopped == 1
    assert seed.status_code == status.status_code == observe.status_code == 200
    assert reset.status_code == step.status_code == evaluate.status_code == close.status_code == 200
    assert seed.json()["task_id"] == "7"
    assert observe.json()["url"].endswith("/observe")
    assert step.json()["operation_id"] == "operation-1"
    assert evaluate.json()["result"]["reward"] == 1.0
    assert close.json()["recording_artifacts"][0]["mime_type"] == "video/webm"
    session_ids = {session_id for _method, session_id in manager.calls}
    assert len(session_ids) == 1


@pytest.mark.parametrize(
    ("method", "path", "body", "error", "status", "kind", "retryable"),
    [
        ("session_status", "/session", None, SessionNotFoundError("gone"), 404, "session_not_found", True),
        (
            "reset_session",
            "/reset",
            {"task": _task_json()},
            SessionConflictError("busy"),
            409,
            "session_conflict",
            True,
        ),
        ("observe", "/observe", None, CapacityUnavailableError("full"), 503, "capacity_unavailable", True),
        (
            "step",
            "/step",
            {"operation_id": "op", "action": {"name": "noop", "script": "noop()"}},
            BenchmarkPreconditionError("bad task"),
            422,
            "benchmark_precondition",
            False,
        ),
        (
            "evaluate",
            "/evaluate",
            {},
            EvaluatorConfigurationError("judge missing"),
            422,
            "evaluator_configuration",
            False,
        ),
        (
            "observe",
            "/observe",
            None,
            EvaluatorInfrastructureError("judge down"),
            502,
            "evaluator_infrastructure",
            True,
        ),
        ("observe", "/observe", None, ValueError("invalid"), 400, "invalid_task", False),
    ],
)
def test_http_exception_handlers(
    method: str,
    path: str,
    body: dict[str, Any] | None,
    error: Exception,
    status: int,
    kind: str,
    retryable: bool,
) -> None:
    manager = FakeManager()
    manager.failures[method] = error
    with TestClient(_server(manager).setup_webserver(), raise_server_exceptions=False) as client:
        response = client.request("POST" if body is not None else "GET", path, json=body)

    assert response.status_code == status
    assert response.json()["error_kind"] == kind
    assert response.json()["retryable"] is retryable
    if isinstance(error, SessionNotFoundError):
        assert response.json()["detail"] == "unknown session: gone"
    else:
        assert response.json()["detail"] == str(error)


def test_bearer_auth_protects_stateful_routes_but_not_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WEB_TEST_TOKEN", "expected-token")
    manager = FakeManager()
    app = _server(
        manager,
        require_auth=True,
        auth_token_env="WEB_TEST_TOKEN",
    ).setup_webserver()

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        for headers in ({}, {"Authorization": "Basic nope"}, {"Authorization": "Bearer wrong"}):
            response = client.get("/session", headers=headers)
            assert response.status_code == 401
            assert response.json() == {
                "detail": "invalid bearer token",
                "error_kind": "authentication_error",
                "retryable": False,
            }
        assert client.get("/session", headers={"Authorization": "bearer expected-token"}).status_code == 200


def test_required_auth_token_is_checked_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("WEB_MISSING_TOKEN", raising=False)
    app = _server(
        FakeManager(),
        require_auth=True,
        auth_token_env="WEB_MISSING_TOKEN",
    ).setup_webserver()

    with pytest.raises(RuntimeError, match="WEB_MISSING_TOKEN must be set"):
        with TestClient(app):
            pass


def test_session_id_requires_middleware_cookie() -> None:
    assert WebResourcesServer._session_id(SimpleNamespace(session={SESSION_ID_KEY: 123})) == "123"
    with pytest.raises(HTTPException, match="Gym session cookie is missing"):
        WebResourcesServer._session_id(SimpleNamespace(session={}))


@pytest.mark.asyncio
async def test_verify_maps_valid_invalid_and_infrastructure_results() -> None:
    manager = FakeManager()
    server = _server(manager)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-a"})
    body = _verify_request()

    valid = await server.verify(request, body)
    assert valid.reward == 1.0
    assert valid.raw_score == 0.75
    assert valid.task_success is True
    assert valid.mask_sample is False

    manager.verifier_result = WebVerifierResult(
        reward=1.0,
        raw_score=0.25,
        task_success=False,
        valid_sample=False,
        failure_kind="site_failure",
    )
    invalid = await server.verify(request, body)
    assert invalid.reward == 0.0
    assert invalid.mask_sample is True
    assert invalid.failure_kind == "site_failure"

    manager.failures["evaluate"] = RuntimeError("judge offline")
    failed = await server.verify(request, body)
    assert failed.reward == 0.0
    assert failed.raw_score == 0.0
    assert failed.task_success is False
    assert failed.mask_sample is True
    assert failed.failure_kind == "verifier_error:RuntimeError:judge offline"
    assert manager.closed == ["session-a", "session-a", "session-a"]


@pytest.mark.asyncio
async def test_verify_cleanup_failure_does_not_replace_successful_verdict() -> None:
    manager = FakeManager()
    manager.failures["close_session"] = RuntimeError("lease release failed")
    server = _server(manager)
    request = SimpleNamespace(session={SESSION_ID_KEY: "session-a"})

    result = await server.verify(request, _verify_request())

    assert result.reward == 1.0
    assert result.raw_score == 0.75
    assert result.task_success is True
    assert result.mask_sample is False
