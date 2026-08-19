# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authenticated, retry-safe HTTP control plane for rollout capture custody."""

from __future__ import annotations

import asyncio
import hmac
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.token_id_capture.gate import (
    CONTROL_ROUTE_PREFIX,
    GateError,
    GateStateError,
    OperationConflictError,
    RolloutCaptureGate,
    RolloutRegistration,
    UnknownRolloutError,
)
from nemo_gym.token_id_capture.staging.records import RolloutReceipt


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    owner_id: str = Field(min_length=1, max_length=256)
    operation_id: str = Field(min_length=1, max_length=256)


class SealRequest(RegisterRequest):
    reward: float | None = None
    terminal_logical_request_id: str = Field(min_length=1, max_length=256)


class FailRequest(RegisterRequest):
    reason: str = Field(min_length=1, max_length=512)


def _http_error(error: GateError) -> HTTPException:
    if isinstance(error, UnknownRolloutError):
        return HTTPException(status_code=404, detail=str(error))
    if isinstance(error, (GateStateError, OperationConflictError)):
        return HTTPException(status_code=409, detail=str(error))
    return HTTPException(status_code=400, detail=str(error))


def install_rollout_control_routes(
    app: Any,
    gate: RolloutCaptureGate,
    *,
    auth_token: str,
    expiry_sweep_interval_s: float,
) -> None:
    """Install bearer-protected control routes and the live TTL scheduler."""
    if not auth_token:
        raise ValueError("rollout control routes require a non-empty auth token")
    if expiry_sweep_interval_s <= 0:
        raise ValueError("expiry_sweep_interval_s must be positive")
    expected = f"Bearer {auth_token}"
    router = APIRouter(prefix=CONTROL_ROUTE_PREFIX)

    def check_auth(authorization: str | None) -> None:
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="missing or invalid control-plane bearer token",
            )

    @router.put("/rollouts/{rollout_id}")
    async def register_rollout(
        rollout_id: str,
        body: RegisterRequest,
        authorization: str | None = Header(default=None),
    ) -> dict:
        check_auth(authorization)
        try:
            registration = await gate.register_rollout(
                rollout_id,
                owner_id=body.owner_id,
                operation_id=body.operation_id,
            )
        except GateError as error:
            raise _http_error(error) from error
        return registration.model_dump()

    @router.post("/rollouts/{rollout_id}/seal")
    async def seal_rollout(
        rollout_id: str,
        body: SealRequest,
        authorization: str | None = Header(default=None),
    ) -> dict:
        check_auth(authorization)
        try:
            receipt = await gate.seal_rollout(
                rollout_id,
                owner_id=body.owner_id,
                operation_id=body.operation_id,
                reward=body.reward,
                terminal_logical_request_id=body.terminal_logical_request_id,
            )
        except GateError as error:
            raise _http_error(error) from error
        return receipt.model_dump()

    @router.post("/rollouts/{rollout_id}/fail")
    async def fail_rollout(
        rollout_id: str,
        body: FailRequest,
        authorization: str | None = Header(default=None),
    ) -> dict:
        check_auth(authorization)
        try:
            cleanup = await gate.fail_rollout(
                rollout_id,
                owner_id=body.owner_id,
                operation_id=body.operation_id,
                reason=body.reason,
            )
        except GateError as error:
            raise _http_error(error) from error
        return cleanup.model_dump()

    @router.get("/cleanup")
    async def drain_cleanup(
        authorization: str | None = Header(default=None),
    ) -> list[dict]:
        check_auth(authorization)
        return [manifest.model_dump() for manifest in await gate.drain_cleanup_manifests()]

    @router.get("/metrics")
    async def gate_metrics(
        authorization: str | None = Header(default=None),
    ) -> dict:
        check_auth(authorization)
        return await gate.snapshot_metrics()

    app.include_router(router)

    original_lifespan = app.router.lifespan_context

    async def expiry_loop() -> None:
        while True:
            await asyncio.sleep(expiry_sweep_interval_s)
            await gate.expire_stale()

    @asynccontextmanager
    async def gate_lifespan(application: Any):
        task = asyncio.create_task(expiry_loop())
        try:
            async with original_lifespan(application) as state:
                yield state
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    app.router.lifespan_context = gate_lifespan


class RolloutControlClient:
    """Framework-side client with bounded, owner-bound idempotent operations."""

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: str,
        owner_id: str,
        request_timeout_s: float,
    ) -> None:
        if not auth_token or not owner_id or request_timeout_s <= 0:
            raise ValueError("control client requires auth token, owner ID, and positive timeout")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {auth_token}"}
        self._owner_id = owner_id
        self._request_timeout_s = request_timeout_s

    async def register(self, rollout_id: str, *, operation_id: str) -> RolloutRegistration:
        response = await self._request(
            "PUT",
            f"/rollouts/{rollout_id}",
            json={"owner_id": self._owner_id, "operation_id": operation_id},
        )
        await self._require_success(response, rollout_id, "registration")
        return RolloutRegistration.model_validate(await response.json())

    async def seal(
        self,
        rollout_id: str,
        *,
        operation_id: str,
        terminal_logical_request_id: str,
        reward: float | None,
    ) -> RolloutReceipt:
        response = await self._request(
            "POST",
            f"/rollouts/{rollout_id}/seal",
            json={
                "owner_id": self._owner_id,
                "operation_id": operation_id,
                "terminal_logical_request_id": terminal_logical_request_id,
                "reward": reward,
            },
        )
        await self._require_success(response, rollout_id, "seal")
        return RolloutReceipt.model_validate(await response.json())

    async def fail(
        self,
        rollout_id: str,
        *,
        operation_id: str,
        reason: str,
    ) -> dict:
        response = await self._request(
            "POST",
            f"/rollouts/{rollout_id}/fail",
            json={
                "owner_id": self._owner_id,
                "operation_id": operation_id,
                "reason": reason,
            },
        )
        await self._require_success(response, rollout_id, "failure")
        return await response.json()

    async def cleanup(self) -> list[dict]:
        response = await self._request("GET", "/cleanup")
        await self._require_success(response, "gate", "cleanup")
        return await response.json()

    async def metrics(self) -> dict:
        response = await self._request("GET", "/metrics")
        await self._require_success(response, "gate", "metrics")
        return await response.json()

    async def _request(self, method: str, path: str, **kwargs: Any):
        # Deferred because server_utils loads Gym's aiohttp/server stack.
        from nemo_gym.server_utils import request

        kwargs.setdefault("headers", {}).update(self._headers)
        try:
            return await asyncio.wait_for(
                request(
                    method,
                    f"{self._base_url}{CONTROL_ROUTE_PREFIX}{path}",
                    **kwargs,
                ),
                timeout=self._request_timeout_s,
            )
        except asyncio.TimeoutError as error:
            raise RuntimeError(f"gate control request {method} {path} exceeded {self._request_timeout_s}s") from error

    @staticmethod
    async def _require_success(response: Any, rollout_id: str, operation: str) -> None:
        if response.status != 200:
            raise RuntimeError(
                f"rollout {rollout_id} {operation} failed: HTTP {response.status} {await response.text()}"
            )
