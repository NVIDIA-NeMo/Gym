# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only HTTP control plane for the rollout capture ledger.

The single route exposes the token-free per-rollout manifest so the framework
never reads ledger files directly — ``LineageStore`` implementations remain
swappable. Rollout lifecycle (receipt assembly, staging cleanup) is
framework-owned; there are no register/seal/fail transitions.
"""

from __future__ import annotations

import asyncio
import hmac
from typing import Any

from fastapi import APIRouter, Header, HTTPException

from nemo_gym.token_id_capture.protocols import LineageStore
from nemo_gym.token_id_capture.staging.records import RolloutManifest


CONTROL_ROUTE_PREFIX = "/training-token-capture/control"


def install_rollout_control_routes(
    app: Any,
    lineage_store: LineageStore,
    *,
    auth_token: str,
) -> None:
    """Install the bearer-protected manifest route."""
    if not auth_token:
        raise ValueError("rollout control routes require a non-empty auth token")
    expected = f"Bearer {auth_token}"
    router = APIRouter(prefix=CONTROL_ROUTE_PREFIX)

    def check_auth(authorization: str | None) -> None:
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="missing or invalid control-plane bearer token",
            )

    @router.get("/rollouts/{rollout_id}/manifest")
    async def rollout_manifest(
        rollout_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict:
        check_auth(authorization)
        try:
            return await lineage_store.manifest(rollout_id)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    app.include_router(router)


class RolloutControlClient:
    """Framework-side client for the read-only manifest route."""

    def __init__(
        self,
        base_url: str,
        *,
        auth_token: str,
        request_timeout_s: float,
    ) -> None:
        if not auth_token or request_timeout_s <= 0:
            raise ValueError("control client requires an auth token and a positive timeout")
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {auth_token}"}
        self._request_timeout_s = request_timeout_s

    async def manifest(self, rollout_id: str) -> RolloutManifest:
        response = await self._request("GET", f"/rollouts/{rollout_id}/manifest")
        if response.status != 200:
            raise RuntimeError(
                f"rollout {rollout_id} manifest fetch failed: HTTP {response.status} {await response.text()}"
            )
        return RolloutManifest.model_validate(await response.json())

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
            raise RuntimeError(
                f"ledger control request {method} {path} exceeded {self._request_timeout_s}s"
            ) from error
