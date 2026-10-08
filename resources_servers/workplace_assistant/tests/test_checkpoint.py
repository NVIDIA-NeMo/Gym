# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Workplace Assistant sessions checkpoint through the real control routes and continue in a fresh server."""

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI
from omegaconf import DictConfig

from nemo_gym.server_utils import ServerClient
from resources_servers.workplace_assistant import app as workplace_app
from resources_servers.workplace_assistant.app import (
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
    _export_tool_env,
)


AUTH = {"authorization": "Bearer t"}
SEED = {"responses_create_params": {"input": "hi"}}
# Every restore below continues the episodes its source checkpoint exported.
SCOPE = [
    {"rollout_id": "r"},
    {"rollout_id": "r", "attempt": 1},
    {"rollout_id": "q"},
    {"rollout_id": "q", "attempt": 1},
]

# Steps before the checkpoint mutate three different tables, one of which (the CRM) holds empty CSV cells.
BEFORE = [
    (
        "calendar_create_event",
        {
            "event_name": "Checkpoint sync",
            "participant_email": "raj.patel@atlas.com",
            "event_start": "2023-12-01 10:00:00",
            "duration": "30",
        },
    ),
    (
        "customer_relationship_manager_add_customer",
        {"customer_name": "Checkpoint Corp", "assigned_to_email": "raj.patel@atlas.com", "status": "Lead"},
    ),
    ("email_send_email", {"recipient": "raj.patel@atlas.com", "subject": "Checkpoint", "body": "Before the cut"}),
]
# Steps after the checkpoint only give the same answers if the earlier mutations survived.
AFTER = [
    (
        "calendar_create_event",
        {
            "event_name": "Checkpoint sync",
            "participant_email": "raj.patel@atlas.com",
            "event_start": "2023-12-02 10:00:00",
            "duration": "45",
        },
    ),
    ("calendar_search_events", {"query": "Checkpoint sync"}),
    ("customer_relationship_manager_search_customers", {"customer_name": "Checkpoint Corp"}),
    ("email_search_emails", {"query": "Before the cut"}),
]


def make_server() -> tuple[WorkbenchResourcesServer, FastAPI]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = WorkbenchResourcesServerConfig(host="", port=0, entrypoint="", name="workplace_assistant")
    server = WorkbenchResourcesServer(config=config, server_client=server_client)
    return server, server.setup_webserver()


def client_for(app: FastAPI, cookies: dict[str, str] | None = None) -> httpx.AsyncClient:
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r")
    client.cookies.update(cookies or {})
    return client


def control(checkpoint_id: str = "c1", **extra: Any) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


async def call_tools(client: httpx.AsyncClient, steps: list[tuple[str, dict[str, str]]]) -> list[Any]:
    outputs = []
    for name, arguments in steps:
        response = await client.post(f"/{name}", json=arguments)
        assert response.status_code == 200, response.text
        outputs.append(response.json()["output"])
    return outputs


def verify_body() -> dict[str, Any]:
    calls = [
        {"type": "function_call", "call_id": f"c{i}", "name": name, "arguments": json.dumps(arguments)}
        for i, (name, arguments) in enumerate(BEFORE + AFTER[:1])
    ]
    return {
        "responses_create_params": {"input": "hi"},
        "response": {
            "id": "resp",
            "created_at": 0.0,
            "model": "m",
            "object": "response",
            "output": calls,
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
        "ground_truth": [{"name": name, "arguments": json.dumps(arguments)} for name, arguments in BEFORE + AFTER[:1]],
        "id": 0,
        "category": "workplace_assistant_calendar",
        "environment_name": "workplace_assistant",
    }


async def checkpoint(client: httpx.AsyncClient, checkpoint_dir: Path) -> dict[str, Any]:
    prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
    assert prepared.json()["phase"] == "prepared"
    commit = await client.post(
        "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(checkpoint_dir)), headers=AUTH
    )
    assert commit.status_code == 200, commit.text
    return commit.json()


async def restore(client: httpx.AsyncClient, checkpoint_dir: Path) -> httpx.Response:
    return await client.post(
        "/ng-control/v1/checkpoint/restore",
        json=control("r1", checkpoint_dir=str(checkpoint_dir), episode_ids=SCOPE),
        headers=AUTH,
    )


async def test_a_restored_session_continues_exactly_like_an_uncheckpointed_run(tmp_path: Path) -> None:
    reference, reference_app = make_server()
    async with client_for(reference_app) as client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await call_tools(client, BEFORE)
        expected_outputs = await call_tools(client, AFTER)
        [expected_tables] = [_export_tool_env(env) for env in reference.session_id_to_tool_env.values()]
        expected_verify = (await client.post("/verify", json=verify_body())).json()

    _, source_app = make_server()
    async with client_for(source_app) as client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await call_tools(client, BEFORE)
        commit = await checkpoint(client, tmp_path)
        cookies = dict(client.cookies)

    restored, restored_app = make_server()
    async with client_for(restored_app, cookies) as client:
        assert (await restore(client, tmp_path)).json()["phase"] == "restored"
        await client.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        outputs = await call_tools(client, AFTER)
        [tables] = [_export_tool_env(env) for env in restored.session_id_to_tool_env.values()]
        verified = (await client.post("/verify", json=verify_body())).json()
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

    assert commit["manifest"]["record_count"] == 1
    # The second event's ID follows the first one's, and the searches find what was written before the cut.
    assert outputs == expected_outputs
    assert outputs[0] == "00000301"
    assert outputs[1]["pagination"]["total_events"] == 2
    assert outputs[2]["pagination"]["total_customers"] == 1
    assert tables == expected_tables
    assert verified["reward"] == expected_verify["reward"] == 1.0
    # Verify ends the session and discards its tool environment.
    assert restored.session_id_to_tool_env == {}
    assert status["report"]["counts"]["sessions"] == 0


async def test_invalid_restored_state_installs_nothing(tmp_path: Path) -> None:
    source, source_app = make_server()
    async with client_for(source_app) as healthy, client_for(source_app) as broken:
        await healthy.post("/ng-rollout/r/seed_session", json=SEED)
        await broken.post("/ng-rollout/q/seed_session", json=SEED)
        await call_tools(healthy, BEFORE)
        broken_env = source.session_id_to_tool_env[next(reversed(source.session_id_to_tool_env))]
        del broken_env["containers"]["analytics"]._plots_data
        commit = await checkpoint(healthy, tmp_path)
        cookies = dict(healthy.cookies)

    restored, restored_app = make_server()
    async with client_for(restored_app, cookies) as client:
        with pytest.raises(ValueError, match="invalid workplace assistant state.*analytics tables"):
            await restore(client, tmp_path)
        status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

    assert commit["manifest"]["record_count"] == 2
    assert restored.session_id_to_tool_env == {}
    assert status["phase"] == "idle" and status["report"]["counts"]["sessions"] == 0


async def test_a_retired_session_is_gone(tmp_path: Path) -> None:
    server, app = make_server()
    async with client_for(app) as client:
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await call_tools(client, BEFORE)
        await client.post(
            "/ng-control/v1/checkpoint/retire", json=control(episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )
        commit = await checkpoint(client, tmp_path)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        stale = await client.post("/calendar_search_events", json={"query": "Checkpoint sync"})

    assert commit["manifest"]["record_count"] == 0
    # The retire stopped and released the session; a late call is not served from it and does not recreate it.
    assert stale.status_code >= 400
    assert server.session_id_to_tool_env == {}


async def test_a_session_dropped_by_a_failed_verify_is_left_out_of_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> bool:
        raise RuntimeError("verifier failed")

    source, source_app = make_server()
    async with client_for(source_app) as live, client_for(source_app) as failed:
        await live.post("/ng-rollout/r/seed_session", json=SEED)
        await failed.post("/ng-rollout/q/seed_session", json=SEED)
        await call_tools(live, BEFORE)
        await call_tools(failed, BEFORE)
        with monkeypatch.context() as patch:
            patch.setattr(workplace_app, "is_correct", fail)
            with pytest.raises(RuntimeError, match="verifier failed"):
                await failed.post("/verify", json=verify_body())
        commit = await checkpoint(live, tmp_path)
        live_cookies, failed_cookies = dict(live.cookies), dict(failed.cookies)

    restored, restored_app = make_server()
    async with client_for(restored_app, live_cookies) as live, client_for(restored_app, failed_cookies) as failed:
        await restore(live, tmp_path)
        await live.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        continued = await call_tools(live, AFTER[:1])
        gone = await failed.post("/calendar_search_events", json={"query": "Checkpoint sync"})

    assert len(source.session_id_to_tool_env) == 1
    assert commit["episode_ids"] == ["r"]
    assert commit["manifest"]["record_count"] == 1
    assert len(restored.session_id_to_tool_env) == 1
    assert continued == ["00000301"]
    assert gone.status_code == 400
