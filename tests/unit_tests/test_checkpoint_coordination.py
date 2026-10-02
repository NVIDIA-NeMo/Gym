# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import orjson
import pytest
from fastapi import FastAPI
from omegaconf import OmegaConf

from nemo_gym._checkpoint import coordination
from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    PrepareReport,
    install_participant,
    next_attempt,
)
from nemo_gym._checkpoint.coordination import CoordinationError
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import BaseServerConfig, ServerClient


TOKEN = "t"


class Recorder(CheckpointParticipant):
    """A participant that is ready unless told otherwise and records the calls it receives."""

    record_model = CheckpointRecord

    def __init__(self, kind: str, events: list[str], name: str) -> None:
        super().__init__()
        self.kind = kind
        self.name = name
        self.events = events
        self.blockers: list[str] = []
        self.fail_restore = False
        self.live: set[str] = set()

    async def close_admission(self, request: CheckpointRequest) -> None:
        self.events.append(f"close {self.name}")

    async def open_admission(self) -> None:
        self.events.append(f"open {self.name}")

    def readiness(self) -> PrepareReport:
        return PrepareReport(ready=not self.blockers, blockers=self.blockers)

    async def retire(self, episode_id: EpisodeId) -> None:
        self.events.append(f"retire {self.name}")

        def covered(key: str) -> bool:
            other = EpisodeId.from_capture_key(key)
            return other.rollout_id == episode_id.rollout_id and other.attempt <= episode_id.attempt

        self.blockers = [key for key in self.blockers if not covered(key)]
        self.live = {key for key in self.live if not covered(key)}

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        return [CheckpointRecord(episode_id=episode_id) for episode_id in episode_ids or []]

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        if self.fail_restore:
            raise ValueError("corrupt state")
        self.live = {next_attempt(record.episode_id).capture_key for record in records}


class _Response:
    def __init__(self, response: httpx.Response) -> None:
        self.status = response.status_code
        self._content = response.content

    async def read(self) -> bytes:
        return self._content


class InProcessClient(ServerClient):
    """Route ServerClient calls to in-process ASGI apps by server name."""

    apps: dict[str, Any]

    async def request(self, server_name: str, url_path: str, method: str, **kwargs: Any) -> _Response:
        transport = httpx.ASGITransport(app=self.apps[server_name])
        async with httpx.AsyncClient(transport=transport, base_url=f"http://{server_name}") as client:
            return _Response(
                await client.request(method, url_path, json=kwargs.get("json"), headers=kwargs["headers"])
            )


SERVER_TYPES = {
    "environment": "environment_servers",
    "model": "responses_api_models",
    "agent": "responses_api_agents",
    "resources": "resources_servers",
}


def deployment(*servers: tuple[str, Optional[str]]) -> tuple[InProcessClient, dict[str, Recorder], list[str]]:
    """Build one app per ``(name, kind)``; a kind of ``None`` is a server without a participant."""
    events: list[str] = []
    apps: dict[str, Any] = {}
    recorders: dict[str, Recorder] = {}
    global_config: dict[str, Any] = {}
    for name, kind in servers:
        app = FastAPI()
        if kind is not None:
            recorders[name] = Recorder(kind, events, name)
            install_participant(app, recorders[name], auth_token=TOKEN, instance_name=name, lease_grace_seconds=60)
        apps[name] = app
        global_config[name] = {SERVER_TYPES[kind or "model"]: {name: {}}}
    client = InProcessClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(global_config),
        apps=apps,
    )
    return client, recorders, events


FULL = (("res", "resources"), ("agent", "agent"), ("policy", "model"), ("env", "environment"), ("judge", None))


def deadline(seconds: float = 5) -> float:
    return time.time() + seconds


async def test_discover_orders_participants_and_skips_servers_without_control_routes() -> None:
    client, _, _ = deployment(*FULL)

    participants = await coordination.discover(client, auth_token=TOKEN)

    assert [(member.server_name, member.kind) for member in participants.members] == [
        ("env", "environment"),
        ("policy", "model"),
        ("agent", "agent"),
        ("res", "resources"),
    ]


async def test_discover_refuses_a_deployment_without_a_policy_model() -> None:
    client, _, _ = deployment(("env", "environment"), ("judge", None))

    with pytest.raises(CoordinationError, match="no policy model"):
        await coordination.discover(client, auth_token=TOKEN)


async def test_prepare_runs_in_gym_order_and_resume_releases_in_reverse() -> None:
    client, _, events = deployment(*FULL)
    participants = await coordination.discover(client, auth_token=TOKEN)

    result = await coordination.prepare(participants, "c1", deadline_ts=deadline())
    await coordination.resume(participants, "c1", deadline_ts=deadline())

    assert result.prepared
    assert events == [
        "close env",
        "close policy",
        "close agent",
        "close res",
        "open res",
        "open agent",
        "open policy",
        "open env",
    ]


async def test_prepare_stops_at_an_unready_stage_until_the_controller_retires_its_blockers() -> None:
    client, recorders, events = deployment(*FULL)
    participants = await coordination.discover(client, auth_token=TOKEN)
    recorders["agent"].blockers = ["r-a1"]

    missed = await coordination.prepare(participants, "c1", deadline_ts=deadline(0.2))
    later_stage_untouched = "close res" not in events
    # A straggler is retired after the checkpoint is abandoned, never while it is open.
    with pytest.raises(CoordinationError, match="invalid_phase"):
        await coordination.retire(participants, "c1", [EpisodeId(rollout_id="r", attempt=1)], deadline_ts=deadline())
    await coordination.resume(participants, "c1", deadline_ts=deadline())
    await coordination.retire(participants, "retire", [EpisodeId(rollout_id="r", attempt=1)], deadline_ts=deadline())
    prepared = await coordination.prepare(participants, "c2", deadline_ts=deadline())

    assert not missed.prepared
    assert missed.blockers() == {"agent": ["r-a1"]}
    assert later_stage_untouched
    assert prepared.prepared


async def test_resume_after_a_partial_prepare_fences_the_participants_it_never_reached() -> None:
    client, recorders, _ = deployment(*FULL)
    participants = await coordination.discover(client, auth_token=TOKEN)
    recorders["agent"].blockers = ["r"]

    await coordination.prepare(participants, "c1", deadline_ts=deadline(0.2))
    await coordination.resume(participants, "c1", deadline_ts=deadline())
    # A prepare for the aborted checkpoint that arrives late must not close the resources server.
    res_only = coordination.Participants(
        client=client, auth_token=TOKEN, members=tuple(m for m in participants.members if m.kind == "resources")
    )

    with pytest.raises(CoordinationError, match="stale_checkpoint"):
        await coordination.prepare(res_only, "c1", deadline_ts=deadline())


async def test_commit_scopes_every_participant_to_the_continued_episodes(tmp_path: Path) -> None:
    client, _, _ = deployment(*FULL)
    participants = await coordination.discover(client, auth_token=TOKEN)
    await coordination.prepare(participants, "c1", deadline_ts=deadline())

    replies = await coordination.commit(
        participants, "c1", str(tmp_path), [EpisodeId(rollout_id="r")], deadline_ts=deadline()
    )

    assert {server: reply["episode_ids"] for server, reply in replies.items()} == {
        "env": ["r"],
        "policy": ["r"],
        "agent": ["r"],
        "res": ["r"],
    }


async def test_restore_is_all_or_nothing(tmp_path: Path) -> None:
    source, _, _ = deployment(*FULL)
    participants = await coordination.discover(source, auth_token=TOKEN)
    await coordination.prepare(participants, "c1", deadline_ts=deadline())
    await coordination.commit(participants, "c1", str(tmp_path), [EpisodeId(rollout_id="r")], deadline_ts=deadline())

    fresh, recorders, _ = deployment(*FULL)
    recorders["agent"].fail_restore = True
    restarted = await coordination.discover(fresh, auth_token=TOKEN)

    with pytest.raises(CoordinationError, match="restore failed on 1"):
        await coordination.restore(restarted, "r1", str(tmp_path), [EpisodeId(rollout_id="r")], deadline_ts=deadline())
    # Nothing continues from a partial restore: the replacement attempt is gone and fenced everywhere.
    assert {name: recorder.live for name, recorder in recorders.items()} == {
        "env": set(),
        "policy": set(),
        "agent": set(),
        "res": set(),
    }
    # The retire stopped and freed the attempt, so nothing about it remains to refuse later requests.
    assert all(len(recorder.attempts) == 0 for recorder in recorders.values())


async def test_renew_extends_every_participants_lease() -> None:
    client, _, _ = deployment(*FULL)
    participants = await coordination.discover(client, auth_token=TOKEN)
    await coordination.prepare(participants, "c1", deadline_ts=deadline(5))
    before = await _leases(client, participants)
    await coordination.renew(participants, "c1", deadline_ts=deadline(100))
    after = await _leases(client, participants)
    await coordination.resume(participants, "c1", deadline_ts=deadline())

    assert all(after[name] > before[name] + 90 for name in before)


async def _leases(client: "InProcessClient", participants: coordination.Participants) -> dict[str, float]:
    leases = {}
    for member in participants.members:
        response = await client.request(
            member.server_name, "/ng-control/v1/checkpoint/status", "GET", headers={"Authorization": f"Bearer {TOKEN}"}
        )
        leases[member.server_name] = orjson.loads(await response.read())["lease_expires_at"]
    return leases


async def test_retire_stops_callers_before_the_servers_they_call() -> None:
    client, _, events = deployment(*FULL)
    participants = await coordination.discover(client, auth_token=TOKEN)

    await coordination.retire(participants, "retire", [EpisodeId(rollout_id="r")], deadline_ts=deadline())

    retires = [event for event in events if event.startswith("retire ")]
    assert retires[:2] == ["retire env", "retire agent"]
    assert sorted(retires[2:]) == ["retire policy", "retire res"]
