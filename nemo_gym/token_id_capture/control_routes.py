# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Expose rollout capture manifests and ledger removal over HTTP.

The training framework reads one manifest for each rollout.
The endpoint returns call metadata and does not read staged token data.
The framework builds receipts and removes staged data after use.
It then retires the rollout's ledger, which Gym cannot otherwise know is no longer needed.
"""

from __future__ import annotations

import asyncio
import hmac
from collections.abc import Sequence
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.token_id_capture.protocols import CaptureLedger
from nemo_gym.token_id_capture.staging.records import RolloutManifest, RolloutRemoval


CONTROL_ROUTE_PREFIX = "/training-token-capture/control"
# Bounds one retire or delete request so it finishes well within a control timeout.
# The client splits larger batches.
MAX_LEDGER_BATCH = 4096


class LedgerBatchRequest(BaseModel):
    """Body of the retire and delete routes."""

    model_config = ConfigDict(extra="forbid")

    rollout_ids: list[str] = Field(min_length=1, max_length=MAX_LEDGER_BATCH)


def install_rollout_control_routes(
    app: Any,
    lineage_store: CaptureLedger,
    *,
    auth_token: str,
) -> None:
    """Install the bearer-protected manifest, retire, and delete routes."""
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

    @router.post("/rollouts/retire")
    async def retire_rollouts(
        body: LedgerBatchRequest,
        authorization: str | None = Header(default=None),
    ) -> dict:
        check_auth(authorization)
        try:
            return await lineage_store.retire(body.rollout_ids)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/rollouts/delete")
    async def delete_rollouts(
        body: LedgerBatchRequest,
        authorization: str | None = Header(default=None),
    ) -> dict:
        check_auth(authorization)
        try:
            return await lineage_store.delete(body.rollout_ids)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    app.include_router(router)


class RolloutControlClient:
    """Framework-side client for the manifest, retire, and delete routes."""

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

    async def retire(self, rollout_ids: Sequence[str]) -> RolloutRemoval:
        """Retire ledgers; see ``CaptureLedger.retire``. Retrying after a failure is safe."""
        return await self._remove("retire", rollout_ids)

    async def delete(self, rollout_ids: Sequence[str]) -> RolloutRemoval:
        """Delete ledgers and their fences; see ``CaptureLedger.delete``. Retrying after a failure is safe."""
        return await self._remove("delete", rollout_ids)

    async def _remove(self, action: str, rollout_ids: Sequence[str]) -> RolloutRemoval:
        rollout_ids = list(rollout_ids)
        result = RolloutRemoval()
        for start in range(0, len(rollout_ids), MAX_LEDGER_BATCH):
            batch = rollout_ids[start : start + MAX_LEDGER_BATCH]
            response = await self._request("POST", f"/rollouts/{action}", json={"rollout_ids": batch})
            if response.status != 200:
                raise RuntimeError(
                    f"{action} of {len(batch)} rollout ledgers failed: HTTP {response.status} {await response.text()}"
                )
            removal = RolloutRemoval.model_validate(await response.json())
            result.removed.extend(removal.removed)
            result.absent.extend(removal.absent)
        return result

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
