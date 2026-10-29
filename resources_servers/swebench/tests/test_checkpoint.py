# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE-bench server on the sandbox owner base: typed sessions, capture-once verify, and checkpoints.

The fake provider behaves like OpenSandbox with durable snapshots: a checkpoint snapshots the agent's sandbox and
leaves it running, and a restore forks the snapshot under a new id.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.checkpoint_conformance import (
    TOKEN,
    FakeSnapshotProvider,
    check_sandbox_owner,
    checkpointing_config,
)
from resources_servers.swebench.app import SwebenchResourcesServer, SwebenchResourcesServerConfig


AUTH = {"Authorization": f"Bearer {TOKEN}"}
INSTANCE = {
    "repo": "astropy/astropy",
    "instance_id": "astropy__astropy-12907",
    "base_commit": "d16bfe05a744909de4b27f5875fe0d4ed41ce607",  # pragma: allowlist secret
    "patch": "diff --git a/x b/x",
    "test_patch": "diff --git a/t b/t",
    "problem_statement": "Something is wrong.",
    "hints_text": "",
    "created_at": "2022-03-03T15:14:54Z",
    "version": "4.3",
    "FAIL_TO_PASS": "[]",
    "PASS_TO_PASS": "[]",
    "environment_setup_commit": "298ccb478e6bf092953bca67a3d29dc6c35f6752",  # pragma: allowlist secret
    "difficulty": "<15 min fix",
    "subset": "verified",
    "split": "test",
}
RESPONSE = {
    "output": [],
    "id": "",
    "created_at": 0,
    "model": "",
    "object": "response",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}
VERIFY = {**INSTANCE, "responses_create_params": {"input": []}, "response": RESPONSE}
SEED_HEAD = "5eed5eed"
PATCH = "diff --git a/f b/f\n+fixed\n"


def git_results(command: str) -> Optional[SandboxExecResult]:
    if command.strip() == "git rev-parse HEAD":
        return SandboxExecResult(stdout=f"{SEED_HEAD}\n", stderr=None, return_code=0)
    if "git --no-pager diff --cached" in command:
        return SandboxExecResult(stdout=PATCH, stderr=None, return_code=0)
    return None


class _Swebench(SwebenchResourcesServer):
    """The SWE-bench server on the provider a test gives it."""

    provider: Any = None

    def shared_provider(self) -> Any:
        return self.provider


@pytest.fixture(autouse=True)
def offline(monkeypatch: MonkeyPatch) -> AsyncMock:
    """No Gym config, no multilingual patches, and a grading that resolves every patch it is given."""
    monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
    monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr("resources_servers.swebench.app.patch_swebench_multilingual_sandbox", AsyncMock())
    grading = AsyncMock(return_value={"resolved": True, "completed": True})
    monkeypatch.setattr("resources_servers.swebench.app.run_instance", grading)
    return grading


def make_server(provider: Any, global_config: dict, **config: Any) -> _Swebench:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = global_config
    server = _Swebench(
        config=SwebenchResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="swebench",
            sandbox_provider="sandbox",
            sandbox_config={"ttl_s": 18000, "resources": {"cpu": 2}},
            sandbox_checkpoint={"snapshot_retry_delay_s": 0},
            apply_anti_cheating=False,
            **config,
        ),
        server_client=client,
    )
    server.provider = provider
    return server


def typed_seed(attempt: int = 0, session_id: str = "rs-1") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="r", attempt=attempt),
        task_id=TaskId(taskset="swebench", task_id=INSTANCE["instance_id"]),
        task_data={**INSTANCE, "responses_create_params": {"input": []}},
    )


def http(server: _Swebench) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://swebench")


def control(checkpoint_id: str, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


def diffs(provider: FakeSnapshotProvider) -> int:
    return sum(
        "git --no-pager diff --cached" in command for box in provider.boxes.values() for command in box["commands"]
    )


async def seed(client: httpx.AsyncClient, body: ResourcesSeedSessionRequest) -> httpx.Response:
    reply = await client.post(
        f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
    )
    assert reply.status_code == 200, reply.text
    return reply


async def test_the_swebench_server_conforms_to_the_sandbox_owner_contract(tmp_path: Path) -> None:
    async def capture(server: _Swebench, session_id: str) -> None:
        await server.capture_once(
            session_id, "patch", lambda: server._capture_patch(session_id), release_sandbox_after=True
        )

    def factory(provider: Any, global_config: dict) -> _Swebench:
        provider._exec_results = git_results
        return make_server(provider, global_config)

    await check_sandbox_owner(factory, typed_seed(), tmp_path / "state", capture=capture)


async def test_a_typed_seed_lends_the_sandbox_and_keeps_the_seed_commit_in_the_session_data(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        reply = await seed(client, typed_seed())
        retried = await seed(client, typed_seed())

    access = reply.json()["sandbox_access"]
    assert access["workdir"] == "/testbed"
    assert access["connection"]["provider_config_ref"] == "sandbox"
    assert retried.json() == reply.json()
    assert server.session_data("rs-1") == {"seed_head": SEED_HEAD}
    # The verify step replays: the patch is captured once, so a repeated verify grades the same patch.
    assert reply.headers["x-ng-checkpoint-verify"] == "replay"
    assert len(provider.boxes) == 1


async def test_a_typed_seed_whose_task_does_not_match_its_task_id_is_refused(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(provider, checkpointing_config(tmp_path))
    mismatched = typed_seed().model_copy(update={"task_id": TaskId(taskset="swebench", task_id="other")})
    async with http(server) as client:
        reply = await client.post("/ng-rollout/r/seed_session", json=mismatched.model_dump(mode="json"))

    assert reply.status_code == 422
    assert provider.boxes == {}


async def test_verify_captures_the_patch_once_against_the_seed_commit_and_grades_it_in_a_fresh_sandbox(
    tmp_path: Path, offline: AsyncMock
) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        await seed(client, typed_seed())
        first = await client.post("/ng-rollout/r/verify", json=VERIFY, headers={"x-ng-checkpoint-close-follows": "1"})
        second = await client.post("/ng-rollout/r/verify", json=VERIFY, headers={"x-ng-checkpoint-close-follows": "1"})

    assert first.status_code == second.status_code == 200, first.text
    assert first.json()["model_patch"] == second.json()["model_patch"] == PATCH
    assert first.json()["reward"] == 1.0
    assert diffs(provider) == 1, "the patch is captured once per session"
    [command] = [
        command for box in provider.boxes.values() for command in box["commands"] if "git --no-pager diff" in command
    ]
    # Staged and committed fixes count: the index is diffed against the commit the session was seeded at.
    assert "git add -u" in command and f"diff --cached {SEED_HEAD}" in command
    # The agent's sandbox stopped after the capture; each verify graded in its own sandbox, under its own run id.
    assert [sandbox for sandbox in provider.live()] == []
    run_ids = [call.kwargs["run_id"] for call in offline.await_args_list]
    assert len(set(run_ids)) == 2
    assert [call.kwargs["pred"]["model_patch"] for call in offline.await_args_list] == [PATCH, PATCH]


async def test_a_failed_capture_leaves_the_agent_sandbox_running_for_a_retry(tmp_path: Path) -> None:
    def failing(command: str) -> Optional[SandboxExecResult]:
        if "git --no-pager diff" in command:
            return SandboxExecResult(stdout=None, stderr="exec lost", return_code=-1, error_type="timeout")
        return git_results(command)

    provider = FakeSnapshotProvider(failing)
    server = make_server(provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        await seed(client, typed_seed())
        reply = await client.post("/ng-rollout/r/verify", json=VERIFY)

    assert reply.status_code == 200
    assert reply.json()["model_patch"] is None
    assert "patch" not in server.session_data("rs-1")
    assert server.session_sandbox("rs-1") is not None


async def test_a_grading_that_fails_still_stops_its_sandbox(tmp_path: Path, offline: AsyncMock) -> None:
    offline.side_effect = RuntimeError("harness crashed")
    provider = FakeSnapshotProvider(git_results)
    server = make_server(provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        await seed(client, typed_seed())
        with pytest.raises(RuntimeError):
            await client.post("/ng-rollout/r/verify", json=VERIFY)

    assert provider.live() == []


async def test_a_commit_during_grading_exports_the_patch_alone_and_a_restore_grades_it_again(
    tmp_path: Path, offline: AsyncMock
) -> None:
    grading, finish = asyncio.Event(), asyncio.Event()

    async def slow_grading(**kwargs: Any) -> dict:
        grading.set()
        await finish.wait()
        return {"resolved": True, "completed": True}

    offline.side_effect = slow_grading
    provider = FakeSnapshotProvider(git_results)
    config = checkpointing_config(tmp_path / "state")
    server = make_server(provider, config)
    async with http(server) as client:
        await seed(client, typed_seed())
        verifying = asyncio.create_task(
            client.post("/ng-rollout/r/verify", json=VERIFY, headers={"x-ng-checkpoint-close-follows": "1"})
        )
        await asyncio.wait_for(grading.wait(), 5)
        # A replayable verify does not hold up the checkpoint, and the patch is already captured.
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control("c1"), headers=AUTH)
        committed = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control("c1", checkpoint_dir=str(tmp_path / "ckpt")), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control("c1"), headers=AUTH)
        finish.set()
        await verifying
        # The Environment Server keeps the session cookie in its bookmark and sends it again after a restore.
        cookies = dict(client.cookies)
    assert prepared.json()["phase"] == "prepared", prepared.text
    assert committed.status_code == 200, committed.text
    assert provider.snapshots == {}, "a session past its capture exports its data, not a sandbox"

    finish.clear()
    offline.side_effect = None
    restored = make_server(provider, config)
    async with http(restored) as client:
        client.cookies.update(cookies)
        for operation, body in (
            ("restore", control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])),
            ("resume", control("r1")),
        ):
            reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
            assert reply.status_code == 200, reply.text
        regraded = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)

    assert regraded.status_code == 200, regraded.text
    assert regraded.json()["model_patch"] == PATCH
    assert diffs(provider) == 1, "the restored session grades the patch it captured before the crash"
    assert restored.session_data("rs-1") == {"seed_head": SEED_HEAD, "patch": PATCH}


async def test_an_agent_s_own_run_still_seeds_with_the_flat_instance(tmp_path: Path, offline: AsyncMock) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(provider, {})
    async with http(server) as client:
        seeded = await client.post("/seed_session", json={**INSTANCE, "responses_create_params": {"input": []}})
        verified = await client.post("/verify", json=VERIFY)

    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["sandbox_handle"].startswith("sb-")
    assert verified.status_code == 200 and verified.json()["model_patch"] == PATCH
    assert provider.live() == []


def test_a_golden_patch_server_holds_no_session_state(tmp_path: Path) -> None:
    server = make_server(
        FakeSnapshotProvider(git_results),
        {"checkpoint": {"enabled": True, "control_auth_token": TOKEN}},
        is_verifying_golden_patch=True,
    )
    # No state_dir is needed: the server never seeds, so it checkpoints nothing.
    server.setup_webserver()
    assert server.checkpoint_participant.mode == "restart_only"
