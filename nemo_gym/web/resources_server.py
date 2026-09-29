# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral HTTP resource server for stateful web rollouts."""

from __future__ import annotations

import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.failure_kinds import SESSION_LOST, SESSION_RELEASE_FAILED, VERIFIER_ERROR
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.web.api_models import (
    WebCloseResponse,
    WebEvaluateRequest,
    WebEvaluateResponse,
    WebResetRequest,
    WebSeedSessionRequest,
    WebSeedSessionResponse,
    WebSessionIdentity,
    WebSessionStatusResponse,
    WebStepRequest,
    WebStepResponse,
    WebVerifyRequest,
    WebVerifyResponse,
)
from nemo_gym.web.resource_config import WebResourcesServerConfig
from nemo_gym.web.session import (
    BenchmarkPreconditionError,
    CapacityUnavailableError,
    EvaluatorConfigurationError,
    EvaluatorInfrastructureError,
    SessionConflictError,
    SessionNotFoundError,
)
from nemo_gym.web.session_control import SessionIdentityError, WebSessionControl
from nemo_gym.web.session_manager import WebSessionManager


LOG = logging.getLogger("nemo_gym.web.resources_server")


def _error_response(*, status_code: int, detail: str, error_kind: str, retryable: bool) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "error_kind": error_kind,
            "retryable": retryable,
        },
    )


class WebResourcesServer(SimpleResourcesServer):
    """Expose any common-protocol web backend through Gym's session API."""

    config: WebResourcesServerConfig
    _manager: WebSessionManager = PrivateAttr()
    _session_control: WebSessionControl = PrivateAttr()

    def model_post_init(self, _context) -> None:
        self._manager = self.make_session_manager()
        self._session_control = WebSessionControl(self._manager, lifetime_seconds=self.config.session_ttl_seconds)

    def make_session_manager(self) -> WebSessionManager:
        """Create the backend-specific manager used by the common HTTP API."""

        raise NotImplementedError

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if self.config.require_auth and not self.config.auth_token():
                raise RuntimeError(f"{self.config.auth_token_env} must be set when require_auth=true")
            await self._manager.start()
            self._session_control.start()
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self._session_control.stop(timeout=self.config.browser_release_timeout_seconds)
                await self._manager.stop()

        app.router.lifespan_context = lifespan

        @app.middleware("http")
        async def bearer_auth(request: Request, call_next):
            if not self.config.require_auth or request.url.path in {
                "/",
                "/healthz",
                "/docs",
                "/openapi.json",
                "/redoc",
            }:
                return await call_next(request)
            expected = self.config.auth_token()
            authorization = request.headers.get("authorization", "")
            supplied = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
            if not supplied or not hmac.compare_digest(supplied, expected):
                return _error_response(
                    status_code=401,
                    detail="invalid bearer token",
                    error_kind="authentication_error",
                    retryable=False,
                )
            return await call_next(request)

        @app.exception_handler(SessionNotFoundError)
        async def session_not_found(_request, exc: SessionNotFoundError):
            return _error_response(
                status_code=404,
                detail=f"unknown session: {exc.args[0]}",
                error_kind=SESSION_LOST,
                retryable=True,
            )

        @app.exception_handler(SessionIdentityError)
        async def identity_error(_request, exc: SessionIdentityError):
            return _error_response(
                status_code=403, detail=str(exc), error_kind="authentication_error", retryable=False
            )

        @app.exception_handler(RequestValidationError)
        async def invalid_body(_request, exc: RequestValidationError):
            # Pydantic normally echoes invalid input, including a short or
            # malformed close capability. Return locations/messages only.
            return JSONResponse(
                status_code=422,
                content={
                    "detail": [
                        {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                        for error in exc.errors()
                    ]
                },
            )

        @app.exception_handler(SessionConflictError)
        async def session_conflict(_request, exc: SessionConflictError):
            return _error_response(
                status_code=409,
                detail=str(exc),
                error_kind="session_conflict",
                retryable=True,
            )

        @app.exception_handler(CapacityUnavailableError)
        async def capacity_unavailable(_request, exc: CapacityUnavailableError):
            return _error_response(
                status_code=503,
                detail=str(exc),
                error_kind="capacity_unavailable",
                retryable=True,
            )

        @app.exception_handler(BenchmarkPreconditionError)
        async def benchmark_precondition(_request, exc: BenchmarkPreconditionError):
            return _error_response(
                status_code=422,
                detail=str(exc),
                error_kind="benchmark_precondition",
                retryable=False,
            )

        @app.exception_handler(EvaluatorConfigurationError)
        async def evaluator_configuration(_request, exc: EvaluatorConfigurationError):
            return _error_response(
                status_code=422,
                detail=str(exc),
                error_kind="evaluator_configuration",
                retryable=False,
            )

        @app.exception_handler(EvaluatorInfrastructureError)
        async def evaluator_infrastructure(_request, exc: EvaluatorInfrastructureError):
            return _error_response(
                status_code=502,
                detail=str(exc),
                error_kind="evaluator_infrastructure",
                retryable=True,
            )

        @app.exception_handler(ValueError)
        async def invalid_request(_request, exc: ValueError):
            return _error_response(
                status_code=400,
                detail=str(exc),
                error_kind="invalid_task",
                retryable=False,
            )

        app.get("/healthz")(self.healthz)
        app.get("/session")(self.session_status)
        app.post("/reset")(self.reset_session)
        app.get("/observe")(self.observe)
        app.post("/step")(self.step)
        app.post("/evaluate")(self.evaluate)
        app.post("/close", response_model=WebCloseResponse)(self.close_session)
        return app

    @staticmethod
    def _session_id(request: Request) -> str:
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            raise HTTPException(status_code=400, detail="Gym session cookie is missing")
        return str(session_id)

    async def seed_session(
        self,
        request: Request,
        body: WebSeedSessionRequest,
    ) -> WebSeedSessionResponse:
        if body.session_identity is None:
            return await self._manager.seed_session(self._session_id(request), body)
        self._check_identity_cookie(request, body)
        result = await self._session_control.seed(body)
        request.session[SESSION_ID_KEY] = body.session_identity
        request.session["web_identity_bound"] = True
        return result

    @staticmethod
    def _check_identity_cookie(request: Request, body: WebSessionIdentity) -> None:
        if request.session.get("web_identity_bound") and request.session.get(SESSION_ID_KEY) != body.session_identity:
            raise SessionConflictError("cookie and caller session identities do not match")

    async def session_status(self, request: Request) -> WebSessionStatusResponse:
        return await self._manager.session_status(self._session_id(request))

    async def reset_session(
        self,
        request: Request,
        body: WebResetRequest,
    ) -> WebSeedSessionResponse:
        return await self._manager.reset_session(self._session_id(request), body)

    async def observe(self, request: Request):
        return await self._manager.observe(self._session_id(request))

    async def step(
        self,
        request: Request,
        body: WebStepRequest,
    ) -> WebStepResponse:
        return await self._manager.step(self._session_id(request), body)

    async def evaluate(
        self,
        request: Request,
        body: WebEvaluateRequest,
    ) -> WebEvaluateResponse:
        return await self._manager.evaluate(self._session_id(request), body.final_answer)

    async def close_session(
        self, request: Request, body: WebSessionIdentity | None = None
    ) -> WebCloseResponse | JSONResponse:
        if body is not None and body.session_identity is not None:
            self._check_identity_cookie(request, body)
            session_id = body.session_identity
            closed = await self._session_control.close(body)
        else:
            session_id = self._session_id(request)
            closed = await self._manager.close_session(session_id)
        if not closed:
            return JSONResponse(
                status_code=503,
                content=WebCloseResponse(
                    closed=False,
                    session_id=session_id,
                    failure_kind=SESSION_RELEASE_FAILED,
                    failure_reason="session cleanup is incomplete; retry close with the same identity",
                ).model_dump(),
            )
        recordings = await self._manager.recording_artifacts(session_id)
        return WebCloseResponse(
            closed=True,
            session_id=session_id,
            recording_artifacts=recordings,
        )

    async def verify(
        self,
        request: Request,
        body: WebVerifyRequest,
    ) -> WebVerifyResponse:
        """Run the colocated benchmark evaluator and always release the browser."""

        session_id = self._session_id(request)
        response: WebVerifyResponse | None = None
        try:
            evaluation = await self._manager.evaluate(session_id, body.final_answer)
            result = evaluation.result
            response = WebVerifyResponse(
                **body.model_dump(),
                reward=result.reward if result.valid_sample else 0.0,
                raw_score=result.raw_score,
                task_success=result.task_success,
                mask_sample=not result.valid_sample,
                failure_kind=result.failure_kind,
            )
        except Exception as exc:  # noqa: BLE001 - verifier infrastructure errors must be masked.
            LOG.exception("Web verifier failed for session=%s", session_id)
            response = WebVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                raw_score=0.0,
                task_success=False,
                mask_sample=True,
                failure_kind=SESSION_LOST if isinstance(exc, SessionNotFoundError) else VERIFIER_ERROR,
                failure_reason=f"{type(exc).__name__}: {exc}",
            )
        finally:
            cleanup_reason = None
            try:
                if not await self._manager.close_session(session_id):
                    cleanup_reason = "session cleanup is incomplete; retry close"
            except Exception as exc:  # noqa: BLE001 - cleanup must not replace a decided verdict.
                cleanup_reason = f"session cleanup failed: {type(exc).__name__}"
                LOG.exception("Web verifier cleanup failed for session=%s", session_id)
            if response is not None and cleanup_reason is not None:
                response.cleanup_failure_kind = SESSION_RELEASE_FAILED
                response.cleanup_failure_reason = cleanup_reason
        return response

    async def healthz(self):
        return await self._manager.health()
