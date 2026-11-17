# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise a verifier's stateless checkpoint contract through its HTTP routes."""

import time
from pathlib import Path
from unittest.mock import MagicMock

import httpx
from omegaconf import OmegaConf

from nemo_gym.client.server_client import ServerClient
from nemo_gym.server_runtime._checkpoint.steps import (
    CHECKPOINT_CLOSE_FOLLOWS_HEADER,
    CHECKPOINT_RESTART_HEADER,
    CHECKPOINT_VERIFY_HEADER,
)
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    SimpleResourcesServer,
)


async def assert_stateless_checkpoint_recovery(
    server_type: type[SimpleResourcesServer],
    config: BaseResourcesServerConfig,
    request: BaseVerifyRequest,
    checkpoint_dir: Path,
    *,
    expected_reward: float,
) -> None:
    """Checkpoint a seeded verifier, restore in a fresh instance, and replay its verification."""

    def client() -> httpx.AsyncClient:
        server_client = MagicMock(spec=ServerClient)
        server_client.global_config_dict = OmegaConf.create(
            {"checkpoint": {"enabled": True, "control_auth_token": "test-token"}}
        )
        server = server_type(config=config, server_client=server_client)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://test")

    def control(**extra: object) -> dict[str, object]:
        return {"checkpoint_id": "visual-verifier", "deadline_ts": time.time() + 10, **extra}

    auth = {"authorization": "Bearer test-token"}
    scope = [{"rollout_id": "visual", "attempt": 1}]
    payload = request.model_dump(mode="json")
    async with client() as source:
        seeded = await source.post("/ng-rollout/visual-a1/seed_session", json=request.model_dump(exclude={"response"}))
        assert seeded.status_code == 200, seeded.text
        assert CHECKPOINT_RESTART_HEADER not in seeded.headers
        assert seeded.headers[CHECKPOINT_VERIFY_HEADER] == "replay"
        # A checkpoint may keep the caller's boundary before a verification that already returned.
        # The caller closes afterwards, so this completed verification must leave the session live.
        before = await source.post(
            "/ng-rollout/visual-a1/verify", json=payload, headers={CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}
        )
        assert before.status_code == 200, before.text
        assert before.json()["reward"] == expected_reward

        prepared = await source.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=auth)
        assert prepared.status_code == 200, prepared.text
        assert prepared.json()["phase"] == "prepared"
        assert prepared.json()["report"]["restarts"] == []

        committed = await source.post(
            "/ng-control/v1/checkpoint/commit",
            json=control(checkpoint_dir=str(checkpoint_dir), episode_ids=scope),
            headers=auth,
        )
        assert committed.status_code == 200, committed.text
        # Stateless resources still save their session identities for routing after a restart.
        assert committed.json()["manifest"]["record_count"] == 1
        cookies = dict(source.cookies)
        resumed = await source.post("/ng-control/v1/checkpoint/resume", json=control(), headers=auth)
        assert resumed.status_code == 200, resumed.text

    async with client() as restored:
        restored.cookies.update(cookies)
        loaded = await restored.post(
            "/ng-control/v1/checkpoint/restore",
            json=control(checkpoint_dir=str(checkpoint_dir), episode_ids=scope),
            headers=auth,
        )
        assert loaded.status_code == 200, loaded.text
        resumed = await restored.post("/ng-control/v1/checkpoint/resume", json=control(), headers=auth)
        assert resumed.status_code == 200, resumed.text
        after = await restored.post("/ng-rollout/visual-a2/verify", json=payload)
        assert after.status_code == 200, after.text
        assert after.json() == before.json()
