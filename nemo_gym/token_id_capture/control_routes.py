# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The gate's control plane: register before dispatch, seal (or fail) after.

The RL controller talks to these routes through ``RolloutControlClient`` —
the only HTTP surface a framework needs beyond the model endpoints
themselves. Registration is create-only (a NaN-retry re-dispatch fails
loudly, § 7); sealing returns the token-free ``RolloutReceipt`` and drops
every byte of gate state for the rollout.

Routes are installed only when the gate is enabled, so a default run exposes
nothing. The routes live outside any sandbox-facing namespace: agents never
see them (agents only ever POST model endpoints).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from nemo_gym.token_id_capture.gate import RolloutCaptureGate
from nemo_gym.token_id_capture.staging.lineage import (
    DuplicateRolloutError,
    LineageStateError,
    UnknownRolloutError,
)
from nemo_gym.token_id_capture.staging.records import RolloutReceipt


class SealRequest(BaseModel):
    reward: Optional[float] = None
    terminal_call_id: Optional[str] = None


class FailRequest(BaseModel):
    reason: str = "controller_abort"


def install_rollout_control_routes(app: Any, gate: RolloutCaptureGate) -> None:
    """Attach the register/seal/fail control API for one gate."""
    router = APIRouter()

    @router.put("/ng-control/rollouts/{rollout_id}")
    async def register_rollout(rollout_id: str) -> dict:
        try:
            gate.register_rollout(rollout_id)
        except DuplicateRolloutError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"rollout_id": rollout_id, "registered": True}

    @router.post("/ng-control/rollouts/{rollout_id}/seal")
    async def seal_rollout(rollout_id: str, body: SealRequest) -> dict:
        try:
            receipt = gate.seal_rollout(rollout_id, reward=body.reward, terminal_call_id=body.terminal_call_id)
        except UnknownRolloutError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except LineageStateError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return receipt.model_dump()

    @router.post("/ng-control/rollouts/{rollout_id}/fail")
    async def fail_rollout(rollout_id: str, body: FailRequest) -> dict:
        try:
            gate.fail_rollout(rollout_id, reason=body.reason)
        except UnknownRolloutError:
            # Idempotent: sealing/TTL may already have dropped it; a second
            # fail from a cancelled dispatch must not error.
            return {"rollout_id": rollout_id, "failed": False}
        return {"rollout_id": rollout_id, "failed": True}

    @router.get("/ng-control/metrics")
    async def gate_metrics() -> dict:
        return dict(gate.metrics)

    app.include_router(router)


class RolloutControlClient:
    """The framework-side client for the control routes (aiohttp via Gym's
    shared client; retry/pooling semantics inherited from server_utils)."""

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    async def register(self, rollout_id: str) -> None:
        response = await self._request("PUT", f"/ng-control/rollouts/{rollout_id}")
        if response.status != 200:
            raise RuntimeError(
                f"rollout {rollout_id} registration failed: HTTP {response.status} {await response.text()}"
            )

    async def seal(
        self,
        rollout_id: str,
        *,
        reward: Optional[float] = None,
        terminal_call_id: Optional[str] = None,
    ) -> RolloutReceipt:
        response = await self._request(
            "POST",
            f"/ng-control/rollouts/{rollout_id}/seal",
            json={"reward": reward, "terminal_call_id": terminal_call_id},
        )
        if response.status != 200:
            raise RuntimeError(f"rollout {rollout_id} seal failed: HTTP {response.status} {await response.text()}")
        return RolloutReceipt.model_validate(await response.json())

    async def fail(self, rollout_id: str, *, reason: str = "controller_abort") -> None:
        response = await self._request("POST", f"/ng-control/rollouts/{rollout_id}/fail", json={"reason": reason})
        if response.status != 200:
            raise RuntimeError(f"rollout {rollout_id} fail failed: HTTP {response.status} {await response.text()}")

    async def metrics(self) -> dict:
        response = await self._request("GET", "/ng-control/metrics")
        if response.status != 200:
            raise RuntimeError(f"gate metrics failed: HTTP {response.status}")
        return await response.json()

    async def _request(self, method: str, path: str, **kwargs: Any):
        # Deferred: server_utils pulls the aiohttp/server stack, which must
        # not load merely by importing this module's route models.
        from nemo_gym.server_utils import request

        return await request(method, f"{self._base_url}{path}", **kwargs)
