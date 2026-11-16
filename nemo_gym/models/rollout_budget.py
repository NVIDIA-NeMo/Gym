# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA NeMo Gym (Apache-2.0), original path: nemo_gym/models/rollout_budget.py
# Source: https://gitlab-master.nvidia.com/tongc/gym/-/tree/d8c548915431e59169890d34cd8cea56ef2a905e
"""A durable generation admission limit shared by model-server workers.

All policy model servers in a run must use the same state directory. Calls without
an idempotency guarantee consume separate slots: identical prompts are not retries.
Reservations are not refunded after dispatch, because a transport error cannot prove
that the backend did not generate. Such errors remain infrastructure failures.
"""

import asyncio
import os
from pathlib import Path

from aiohttp import ClientResponseError
from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.client.http_client import raise_for_status
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.episode_context import current_rollout_id
from nemo_gym.models.token_capture.sink import current_capture_context
from nemo_gym.models.token_capture.store import TokenCaptureStore, validate_rollout_id


BUDGET_STOP_REASON = "model_call_budget_exhausted"


class RolloutGenerationBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_calls: int = Field(gt=0)
    state_dir: Path
    agent_servers: list[str] = Field(min_length=1)


class RolloutGenerationBudget:
    def __init__(self, config: RolloutGenerationBudgetConfig) -> None:
        self.config = config
        self.store = TokenCaptureStore(config.state_dir)

    def reserve(self, rollout_id: str) -> int:
        """Return the admitted 1-based ordinal, or zero when exhausted, atomically.

        Reuse the capture store's filesystem lock protocol, in a separate directory.
        Keep exhausted records until the run is retired; deleting them would reopen
        admission for late retries.
        """
        path = self.store.root / f"{validate_rollout_id(rollout_id)}.generation-count"
        with self.store._locked(rollout_id):
            used = int(path.read_text()) if path.exists() else 0
            if used < 0:
                raise ValueError("Generation budget state has a negative count")
            if used >= self.config.max_calls:
                return 0
            temporary = path.with_suffix(".next")
            with temporary.open("w") as handle:
                handle.write(str(used + 1))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(self.store.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return used + 1

    async def admit(self, request: Request, server_client: ServerClient) -> int:
        rollout_id = current_rollout_id()
        if rollout_id is None:
            raise HTTPException(422, "Generation budget requires rollout correlation")
        ordinal = await asyncio.to_thread(self.reserve, rollout_id)
        if ordinal:
            return ordinal
        # No backend call or capture intent exists for this request. Do not poison
        # an otherwise complete training ledger with a missing-generation failure.
        context = current_capture_context()
        if context is not None:
            context.refused = True
        for agent in self.config.agent_servers:
            try:
                response = await server_client.post(
                    server_name=agent,
                    url_path=f"/rollouts/{rollout_id}/stop",
                    json={"reason": BUDGET_STOP_REASON},
                )
                await raise_for_status(response)
            except ClientResponseError as error:
                if error.status == 404:
                    continue
                raise
            stopped = await response.json()
            if stopped.get("rollout_id") != rollout_id:
                raise RuntimeError("Budget stop returned a different rollout identity")
            # expire() acknowledges the stop marker before the supervisor kills
            # the CLI. Keep its request waiting so it cannot turn an HTTP refusal
            # into a model error before normal timeout/file collection completes.
            for _ in range(300):
                if await request.is_disconnected():
                    break
                await asyncio.sleep(0.1)
            raise HTTPException(409, {"code": BUDGET_STOP_REASON, "max_calls": self.config.max_calls})
        raise HTTPException(503, "Generation budget exhausted but no agent owns this rollout")
