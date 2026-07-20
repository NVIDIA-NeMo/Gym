# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym-native OSWorld Resources Server backed by remote Docker workers."""

from __future__ import annotations

import hmac
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import PrivateAttr

from nemo_gym.base_resources_server import SimpleResourcesServer
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.osworld.config import OSWorldResourcesServerConfig
from resources_servers.osworld.models import (
    OSWorldCloseResponse,
    OSWorldEvaluateResponse,
    OSWorldObservation,
    OSWorldResetRequest,
    OSWorldSeedSessionRequest,
    OSWorldSeedSessionResponse,
    OSWorldSessionStatusResponse,
    OSWorldStepRequest,
    OSWorldStepResponse,
    OSWorldVerifyRequest,
    OSWorldVerifyResponse,
)
from resources_servers.osworld.session_manager import (
    CapacityUnavailableError,
    OSWorldSessionManager,
    SessionConflictError,
    SessionNotFoundError,
)


class OSWorldResourcesServer(SimpleResourcesServer):
    """Own stateful OSWorld environments using Gym's signed session cookie."""

    config: OSWorldResourcesServerConfig
    _manager: OSWorldSessionManager = PrivateAttr()

    def model_post_init(self, _context) -> None:
        self._manager = OSWorldSessionManager(self.config)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            if self.config.require_auth and not self.config.auth_token():
                raise RuntimeError(
                    f"{self.config.auth_token_env} must be set when require_auth=true"
                )
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
                return JSONResponse(status_code=401, content={"detail": "invalid bearer token"})
            return await call_next(request)

        @app.exception_handler(SessionNotFoundError)
        async def session_not_found(_request, exc: SessionNotFoundError):
            return JSONResponse(status_code=404, content={"detail": f"unknown session: {exc.args[0]}"})

        @app.exception_handler(SessionConflictError)
        async def session_conflict(_request, exc: SessionConflictError):
            return JSONResponse(status_code=409, content={"detail": str(exc)})

        @app.exception_handler(CapacityUnavailableError)
        async def capacity_unavailable(_request, exc: CapacityUnavailableError):
            return JSONResponse(status_code=503, content={"detail": str(exc)})

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
        body: OSWorldSeedSessionRequest,
    ) -> OSWorldSeedSessionResponse:
        return await self._manager.seed_session(self._session_id(request), body)

    async def session_status(
        self,
        request: Request,
    ) -> OSWorldSessionStatusResponse:
        return await self._manager.session_status(self._session_id(request))

    async def reset_session(
        self,
        request: Request,
        body: OSWorldResetRequest,
    ) -> OSWorldSeedSessionResponse:
        return await self._manager.reset_session(self._session_id(request), body)

    async def observe(self, request: Request) -> OSWorldObservation:
        return await self._manager.observe(self._session_id(request))

    async def step(
        self,
        request: Request,
        body: OSWorldStepRequest,
    ) -> OSWorldStepResponse:
        return await self._manager.step(self._session_id(request), body)

    async def evaluate(self, request: Request) -> OSWorldEvaluateResponse:
        return await self._manager.evaluate(self._session_id(request))

    async def close_session(self, request: Request) -> OSWorldCloseResponse:
        await self._manager.close_session(self._session_id(request))
        return OSWorldCloseResponse(closed=True)

    async def verify(
        self,
        request: Request,
        body: OSWorldVerifyRequest,
    ) -> OSWorldVerifyResponse:
        """Standard Gym verifier endpoint; evaluation always owns final cleanup."""

        session_id = self._session_id(request)
        try:
            result = await self._manager.evaluate(session_id)
            return OSWorldVerifyResponse(
                **body.model_dump(),
                reward=1.0 if result.score >= 1.0 else 0.0,
                osworld_score=result.score,
                mask_sample=False,
            )
        except Exception:
            return OSWorldVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                osworld_score=0.0,
                mask_sample=True,
            )
        finally:
            await self._manager.close_session(session_id)

    async def healthz(self):
        return await self._manager.health()


if __name__ == "__main__":
    OSWorldResourcesServer.run_webserver()
