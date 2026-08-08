# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful BrowserGym resource server for WebArena-family rollouts."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig
from resources_servers.browsergym_web.models import (
    WebCloseResponse,
    WebEvaluateRequest,
    WebEvaluateResponse,
    WebResetRequest,
    WebSeedSessionRequest,
    WebSeedSessionResponse,
    WebSessionStatusResponse,
    WebStepRequest,
    WebStepResponse,
    WebVerifyRequest,
    WebVerifyResponse,
)
from resources_servers.browsergym_web.session_manager import (
    BenchmarkPreconditionError,
    BrowserGymSessionManager,
    CapacityUnavailableError,
    SessionConflictError,
    SessionNotFoundError,
)


def _error_response(*, status_code: int, detail: str, error_kind: str, retryable: bool) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "detail": detail,
            "error_kind": error_kind,
            "retryable": retryable,
        },
    )


class BrowserGymWebResourcesServer(SimpleResourcesServer):
    """Own live BrowserGym environments using Gym's signed session cookie."""

    config: BrowserGymWebResourcesServerConfig
    _manager: BrowserGymSessionManager = PrivateAttr()

    def model_post_init(self, _context) -> None:
        self._manager = BrowserGymSessionManager(self.config)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if self.config.require_auth and not self.config.auth_token():
                raise RuntimeError(f"{self.config.auth_token_env} must be set when require_auth=true")
            await self._manager.start()
            try:
                async with parent_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
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
                error_kind="session_not_found",
                retryable=True,
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
        app.post("/close")(self.close_session)
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
        return await self._manager.seed_session(self._session_id(request), body)

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

    async def close_session(self, request: Request) -> WebCloseResponse:
        await self._manager.close_session(self._session_id(request))
        return WebCloseResponse(closed=True)

    async def verify(
        self,
        request: Request,
        body: WebVerifyRequest,
    ) -> WebVerifyResponse:
        """Run the colocated benchmark evaluator and always release the browser."""

        session_id = self._session_id(request)
        try:
            evaluation = await self._manager.evaluate(session_id, body.final_answer)
            result = evaluation.result
            return WebVerifyResponse(
                **body.model_dump(),
                reward=result.reward if result.valid_sample else 0.0,
                raw_score=result.raw_score,
                task_success=result.task_success,
                mask_sample=not result.valid_sample,
                failure_kind=result.failure_kind,
            )
        except Exception as exc:  # noqa: BLE001 - verifier infrastructure errors must be masked.
            return WebVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                raw_score=0.0,
                task_success=False,
                mask_sample=True,
                failure_kind=f"verifier_error:{type(exc).__name__}",
            )
        finally:
            await self._manager.close_session(session_id)

    async def healthz(self):
        return await self._manager.health()


if __name__ == "__main__":
    BrowserGymWebResourcesServer.run_webserver()
