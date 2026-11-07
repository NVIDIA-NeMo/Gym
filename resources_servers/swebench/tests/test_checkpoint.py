# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE-bench server's checkpoints: episode sessions over a task sandbox, snapshotted, forked and graded once.

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

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import (
    FakeSandboxProvider,
    FakeSnapshotProvider,
    ResourcesSeedSessionRequest,
    check_sandbox_session_checkpoints,
    checkpointing_config,
)
from resources_servers.swebench.app import SwebenchResourcesServer, SwebenchResourcesServerConfig


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
    """No Gym config, no multilingual patches, grading sandboxes on a provider of their own, and a grading that
    resolves every patch it is given."""
    graders = FakeSandboxProvider()
    monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
    monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_config", lambda *_: graders)
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


def episode_seed(attempt: int = 0, session_id: str = "rs-1") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="r", attempt=attempt),
        task_id=TaskId(taskset="swebench", task_id=INSTANCE["instance_id"]),
        task_data={**INSTANCE, "responses_create_params": {"input": []}},
    )


def http(server: _Swebench) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://swebench")


def auth(config: dict) -> dict:
    return {"Authorization": f"Bearer {config['checkpoint']['control_auth_token']}"}


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


async def test_the_swebench_server_conforms_to_the_sandbox_session_checkpoint_contract(tmp_path: Path) -> None:
    def factory(provider: Any, global_config: dict) -> _Swebench:
        provider._exec_results = git_results
        return make_server(provider, global_config)

    await check_sandbox_session_checkpoints(factory, episode_seed(), tmp_path / "state")


async def test_an_episode_seed_lends_the_task_sandbox_and_says_its_verify_replays(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        reply = await seed(client, episode_seed())
        retried = await seed(client, episode_seed())

    access = reply.json()["sandbox_access"]
    assert access["workdir"] == "/testbed"
    assert access["connection"]["provider_config_ref"] == "sandbox"
    assert retried.json() == reply.json()
    assert server._export_task_sandbox_state("rs-1") == {"seed_head": SEED_HEAD}
    # The patch is captured once per session, so a verify repeated after a restore grades the same patch.
    assert reply.headers["x-ng-checkpoint-verify"] == "replay"
    assert len(provider.boxes) == 1


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
        await seed(client, episode_seed())
        verifying = asyncio.create_task(
            client.post("/ng-rollout/r/verify", json=VERIFY, headers={"x-ng-checkpoint-close-follows": "1"})
        )
        await asyncio.wait_for(grading.wait(), 5)
        # A replayable verify does not hold up the checkpoint, and the patch is already captured.
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control("c1"), headers=auth(config))
        committed = await client.post(
            "/ng-control/v1/checkpoint/commit",
            json=control("c1", checkpoint_dir=str(tmp_path / "ckpt")),
            headers=auth(config),
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control("c1"), headers=auth(config))
        finish.set()
        await verifying
        # The environment server keeps the session cookie in its bookmark and sends it again after a restore.
        cookies = dict(client.cookies)
    assert prepared.json()["phase"] == "prepared", prepared.text
    assert committed.status_code == 200, committed.text
    assert provider.snapshots == {}, "a session past its capture exports its patch, not a sandbox"

    offline.side_effect = None
    restored = make_server(provider, config)
    async with http(restored) as client:
        client.cookies.update(cookies)
        for operation, body in (
            ("restore", control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])),
            ("resume", control("r1")),
        ):
            reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=auth(config))
            assert reply.status_code == 200, reply.text
        regraded = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)

    assert regraded.status_code == 200, regraded.text
    assert regraded.json()["model_patch"] == PATCH
    assert diffs(provider) == 1, "the restored session grades the patch it captured before the crash"
    # Each grading ran under its own run id, so a replayed one never shares log files with the first.
    run_ids = [call.kwargs["run_id"] for call in offline.await_args_list]
    assert len(run_ids) == 2 and len(set(run_ids)) == 2


async def test_a_restored_session_captures_its_patch_against_the_commit_it_was_seeded_at(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider(git_results)
    config = checkpointing_config(tmp_path / "state")
    async with http(make_server(provider, config)) as client:
        await seed(client, episode_seed())
        for operation, body in (
            ("prepare", control("c1")),
            ("commit", control("c1", checkpoint_dir=str(tmp_path / "ckpt"))),
            ("resume", control("c1")),
        ):
            reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=auth(config))
            assert reply.status_code == 200, reply.text

    restored = make_server(provider, config)
    async with http(restored) as client:
        for operation, body in (
            ("restore", control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])),
            ("resume", control("r1")),
        ):
            reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=auth(config))
            assert reply.status_code == 200, reply.text
        await seed(client, episode_seed(attempt=1))
        verified = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)

    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"] == PATCH
    [command] = [
        command for box in provider.boxes.values() for command in box["commands"] if "git --no-pager diff" in command
    ]
    assert f"diff --cached {SEED_HEAD}" in command
    assert provider.created_from_image() == 1, "the replacement attempt's seed does not set the task up again"


def test_a_golden_patch_server_holds_no_session_state() -> None:
    server = make_server(
        FakeSnapshotProvider(git_results),
        {"checkpoint": {"enabled": True, "control_auth_token": "t"}},
        is_verifying_golden_patch=True,
    )
    # No state_dir is needed: the server never seeds an episode session, so it checkpoints nothing.
    server.setup_webserver()
    assert server.checkpoint_participant.mode == "restart_only"
