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
from collections.abc import Callable, Sequence
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.token_id_capture.protocols import CaptureLedger, RolloutRemovalPayload, RolloutRetiredError
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
    refuse_removal: Callable[[], str | None] | None = None,
) -> None:
    """Install the bearer-protected manifest, retire, and delete routes.

    ``refuse_removal`` returns a reason to refuse retire and delete for now, with 409; the caller retries later.
    """
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

    def check_removal() -> None:
        reason = refuse_removal() if refuse_removal is not None else None
        if reason is not None:
            raise HTTPException(status_code=409, detail=reason)

    @router.get("/rollouts/{rollout_id}/manifest")
    async def rollout_manifest(
        rollout_id: str,
        authorization: str | None = Header(default=None),
    ) -> dict:
        check_auth(authorization)
        try:
            return await lineage_store.manifest(rollout_id)
        except RolloutRetiredError as error:
            # Gone, not empty: an empty manifest would read as a rollout that made no calls.
            raise HTTPException(status_code=410, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/rollouts/retire")
    async def retire_rollouts(
        body: LedgerBatchRequest,
        authorization: str | None = Header(default=None),
    ) -> RolloutRemovalPayload:
        check_auth(authorization)
        check_removal()
        try:
            return await lineage_store.retire(body.rollout_ids)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @router.post("/rollouts/delete")
    async def delete_rollouts(
        body: LedgerBatchRequest,
        authorization: str | None = Header(default=None),
    ) -> RolloutRemovalPayload:
        check_auth(authorization)
        check_removal()
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
        """Fetch a rollout's manifest. A retired rollout raises ``RolloutRetiredError``."""
        from nemo_gym.token_id_capture.store import validate_rollout_id

        # Validate before building the URL: dot segments are normalized, so "../rollouts/r1" would fetch r1.
        validate_rollout_id(rollout_id)
        response = await self._request("GET", f"/rollouts/{rollout_id}/manifest")
        if response.status == 410:
            raise RolloutRetiredError(f"rollout {rollout_id} is retired: {await response.text()}")
        if response.status != 200:
            raise RuntimeError(
                f"rollout {rollout_id} manifest fetch failed: HTTP {response.status} {await response.text()}"
            )
        manifest = RolloutManifest.model_validate(await response.json())
        if manifest.rollout_id != rollout_id:
            raise ValueError(f"asked for the manifest of rollout {rollout_id} but got rollout {manifest.rollout_id}")
        return manifest

    async def retire(self, rollout_ids: Sequence[str]) -> RolloutRemoval:
        """Retire ledgers; see ``CaptureLedger.retire``. Retrying after a failure is safe."""
        return await self._remove("retire", rollout_ids)

    async def delete(self, rollout_ids: Sequence[str]) -> RolloutRemoval:
        """Delete ledgers and their fences; see ``CaptureLedger.delete``. Retrying after a failure is safe."""
        return await self._remove("delete", rollout_ids)

    async def _remove(self, action: str, rollout_ids: Sequence[str]) -> RolloutRemoval:
        from nemo_gym.token_id_capture.store import validate_rollout_ids

        # Validate and deduplicate the whole list before the first request. Batch by batch, a bare string
        # would become one-character IDs, an invalid ID in a later batch would fail after earlier batches
        # had changed state, and an ID repeated across batches would be reported both removed and absent.
        rollout_ids = validate_rollout_ids(rollout_ids)
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
