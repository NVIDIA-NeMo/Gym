# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from nemo_gym._checkpoint import ResourceSnapshot
from nemo_gym.base_resources_server import BaseSeedSessionRequest
from nemo_gym.rollout_correlation import rollout_context
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.workplace_assistant.app import (
    WorkbenchRequest,
    WorkbenchResourcesServerConfig,
)
from resources_servers.workplace_assistant.checkpoint_test_app import (
    CheckpointTestWorkbenchResourcesServer,
)


@pytest.mark.asyncio
async def test_checkpoint_fixture_audits_mutation_and_restored_state(tmp_path, monkeypatch) -> None:
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setenv("NEMO_GYM_CHECKPOINT_TEST_EVENTS", str(audit))
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    server = CheckpointTestWorkbenchResourcesServer(
        config=WorkbenchResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="resources",
        ),
        server_client=client,
    )
    request = Request(scope={"type": "http", "session": {SESSION_ID_KEY: "session-a"}})

    with rollout_context(
        "rollout-a-a0",
        attempt_index=0,
        logical_rollout_id="rollout-a",
    ):
        await server.seed_session(request, BaseSeedSessionRequest())
        await server.route_to_python_function(
            "calendar_create_event",
            WorkbenchRequest(
                event_name="NeMo RL checkpoint recovery sentinel",
                participant_email="checkpoint-recovery@example.com",
                event_start="2025-01-15 10:00:00",
                duration="30",
            ),
            request,
        )

    state = await server.export_checkpoint_state("rollout-a", 0)
    await server.restore_checkpoint_states(
        [
            ResourceSnapshot(
                rollout_id="rollout-a",
                attempt_index=1,
                state_revision=2,
                state=state,
            )
        ]
    )

    events = [json.loads(line) for line in audit.read_text().splitlines()]
    assert events == [
        {
            "attempt_index": 0,
            "event": "mutation_applied",
            "rollout_id": "rollout-a",
            "sentinel_count": 1,
        },
        {
            "attempt_index": 1,
            "event": "state_restored",
            "rollout_id": "rollout-a",
            "sentinel_count": 1,
        },
    ]
