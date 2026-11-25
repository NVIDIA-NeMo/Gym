# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial-rollout checkpoints of the episode sessions, through the server's routes and fake providers.

The agent sandbox runs on the provider every task sandbox shares, a fake with durable snapshots; verifier sandboxes
start on a provider of their own. The patch is collected with the task's real collect hook; grading is stubbed and
records the patch it is given.
"""

import hashlib
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import resources_servers.deepswe_external1.app as module
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import ResourcesSeedSessionRequest, seed_restarts
from resources_servers.deepswe.app import VerifierResult
from resources_servers.deepswe.validate_golden import _empty_response
from resources_servers.deepswe_external1.app import DeepsweExternal1ResourcesServer, saved_capture
from resources_servers.deepswe_external1.inline_task import InlineTask
from resources_servers.deepswe_external1.prepare_examples import task_row
from resources_servers.swebench.testing.checkpoint_conformance import (
    FakeSandboxProvider,
    FakeSnapshotProvider,
    check_sandbox_session_checkpoints,
    checkpointing_config,
)


TOKEN = checkpointing_config(Path("/state"))["checkpoint"]["control_auth_token"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}
# The agent's committed work as the collect hook writes it; not UTF-8, as a patch of a Latin-1 file is not.
PATCH = b"diff --git a/main.go b/main.go\n--- a/main.go\n+++ b/main.go\n@@ -1 +1 @@\n-bug\n+caf\xe9\n"
CLOSE_FOLLOWS = {"x-ng-checkpoint-close-follows": "1"}


class AgentProvider(FakeSnapshotProvider):
    """The provider the agent sandboxes share: the collect hook's model.patch holds ``PATCH``."""

    async def download_file(self, handle: Any, source_path: str, target_path: Path) -> None:
        self._box(handle.sandbox_id)
        target_path.write_bytes(PATCH if source_path == "/logs/artifacts/model.patch" else b"")


class VerifierProvider(FakeSandboxProvider):
    """Where verifier sandboxes start, with IDs of their own so none matches an agent sandbox's."""

    def _next_id(self, prefix: str) -> str:
        return super()._next_id("verifier")


@pytest.fixture
def verifiers(monkeypatch: pytest.MonkeyPatch) -> VerifierProvider:
    provider = VerifierProvider()
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(module, "resolve_provider_config", lambda *_: provider)
    monkeypatch.setattr(module, "resolve_provider_metadata", lambda *_: {})
    return provider


def make_server(provider: Any, global_config: dict, **config: Any) -> DeepsweExternal1ResourcesServer:
    class OnProvider(DeepsweExternal1ResourcesServer):
        """The server on the provider a test gives it."""

        provider: Any = None

        def shared_provider(self) -> Any:
            return self.provider

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = global_config
    server = OnProvider(
        config=module.DeepsweExternal1ResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="deepswe_external1",
            sandbox_provider="sandbox",
            sandbox_config={"ttl_s": 18000},
            sandbox_checkpoint={"snapshot_retry_delay_s": 0},
            logs_dir=Path.cwd() / "logs",
            **config,
        ),
        server_client=client,
    )
    server.provider = provider
    server._run_verifier = AsyncMock(return_value=VerifierResult(evaluation_completed=True, reward=1.0))
    return server


def episode_seed(task: InlineTask, attempt: int = 0) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id="rs-1",
        episode_id=EpisodeId(rollout_id="r", attempt=attempt),
        task_id=TaskId(taskset="deepswe_external1", task_id=task.data.task_id),
        task_data=task_row(task.data, "original instruction\n"),
    )


def verify_body(task: InlineTask) -> dict[str, Any]:
    return task_row(task.data, "original instruction\n") | {"response": _empty_response()}


def http(server: DeepsweExternal1ResourcesServer) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://deepswe")


def control(checkpoint_id: str, **extra: Any) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


async def seed(client: httpx.AsyncClient, body: ResourcesSeedSessionRequest) -> httpx.Response:
    reply = await client.post(
        f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
    )
    assert reply.status_code == 200, reply.text
    return reply


async def checkpoint(client: httpx.AsyncClient, checkpoint_dir: Path) -> None:
    for operation, body in (
        ("prepare", control("c1")),
        ("commit", control("c1", checkpoint_dir=str(checkpoint_dir))),
        ("resume", control("c1")),
    ):
        reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
        assert reply.status_code == 200, reply.text


async def restore(client: httpx.AsyncClient, checkpoint_dir: Path) -> None:
    for operation, body in (
        ("restore", control("r1", checkpoint_dir=str(checkpoint_dir), episode_ids=[{"rollout_id": "r"}])),
        ("resume", control("r1")),
    ):
        reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
        assert reply.status_code == 200, reply.text


def collections(provider: FakeSandboxProvider, sandbox_id: str | None = None) -> int:
    """The collect hooks run in ``sandbox_id``, or in every sandbox, a fork counting the commands it inherited."""
    boxes = [provider.boxes[sandbox_id]] if sandbox_id else provider.boxes.values()
    return sum("> /logs/artifacts/model.patch" in command for box in boxes for command in box["commands"])


def graded_patches(server: DeepsweExternal1ResourcesServer) -> list[bytes]:
    return [call.args[2] for call in server._run_verifier.await_args_list]


@pytest.mark.asyncio
async def test_the_server_conforms_to_the_sandbox_session_checkpoint_contract(
    task: InlineTask, tmp_path: Path, verifiers: VerifierProvider
) -> None:
    # Without verify_body: a verify reports sandbox IDs, timings and a log directory, which differ by run.
    await check_sandbox_session_checkpoints(make_server, episode_seed(task), tmp_path / "state")


@pytest.mark.asyncio
async def test_an_episode_seed_says_a_verify_replays(
    task: InlineTask, tmp_path: Path, verifiers: VerifierProvider
) -> None:
    server = make_server(AgentProvider(), checkpointing_config(tmp_path))
    async with http(server) as client:
        reply = await seed(client, episode_seed(task))

    assert reply.json()["sandbox_access"]["workdir"] == task.data.workdir
    assert not seed_restarts(reply.headers)
    # A replayed verify grades the saved patch, so the checkpoint does not wait for grading.
    assert reply.headers["x-ng-checkpoint-verify"] == "replay"
    await server.shutdown()


@pytest.mark.parametrize("mode", ["golden", "null"])
def test_validation_modes_hold_no_session_state(mode: str) -> None:
    server = make_server(
        AgentProvider(),
        {"checkpoint": {"enabled": True, "control_auth_token": TOKEN}},
        is_verifying_golden_patch=mode == "golden",
        is_verifying_null_patch=mode == "null",
    )
    # No state_dir is needed: golden- and null-patch validation seed no episode sessions, so they checkpoint nothing.
    server.setup_webserver()
    assert server.checkpoint_participant.mode == "restart_only"


@pytest.mark.asyncio
async def test_a_restore_continues_in_a_fork_and_verify_grades_the_patch_collected_there(
    task: InlineTask, tmp_path: Path, verifiers: VerifierProvider
) -> None:
    provider = AgentProvider()
    config = checkpointing_config(tmp_path / "state")
    async with http(make_server(provider, config)) as client:
        await seed(client, episode_seed(task))
        await checkpoint(client, tmp_path / "ckpt")
    assert len(provider.snapshots) == 1

    restored = make_server(provider, config)
    async with http(restored) as client:
        await restore(client, tmp_path / "ckpt")
        reseeded = await seed(client, episode_seed(task, attempt=1))
        verified = await client.post("/ng-rollout/r-a1/verify", json=verify_body(task))

    fork = reseeded.json()["sandbox_access"]["connection"]["descriptor"]["sandbox_id"]
    assert provider.boxes[fork]["from_snapshot"] is not None, "the replacement attempt continues in the fork"
    assert provider.created_from_image() == 1, "the fork is not set up again"
    assert verified.status_code == 200, verified.text
    result = verified.json()
    assert result["reward"] == 1.0 and not result["mask_sample"] and result["cleanup_errors"] == []
    assert collections(provider, fork) == 1, "the patch is collected from the fork"
    assert result["agent_sandbox_id"] == fork and result["verifier_sandbox_id"] in verifiers.boxes
    assert result["model_patch_sha256"] == hashlib.sha256(PATCH).hexdigest()
    assert graded_patches(restored) == [PATCH]
    assert provider.live() == [] and verifiers.live() == []
    await restored.shutdown()


@pytest.mark.asyncio
async def test_a_commit_after_the_capture_exports_the_patch_alone_and_a_restore_grades_it_again(
    task: InlineTask, tmp_path: Path, verifiers: VerifierProvider
) -> None:
    provider = AgentProvider()
    config = checkpointing_config(tmp_path / "state")
    server = make_server(provider, config)
    async with http(server) as client:
        await seed(client, episode_seed(task))
        # The header keeps the session tracked until its close, so the commit below still exports it.
        first = await client.post("/ng-rollout/r/verify", json=verify_body(task), headers=CLOSE_FOLLOWS)
        await checkpoint(client, tmp_path / "ckpt")
        # The environment server keeps the session cookie in its bookmark and sends it again after a restore.
        cookies = dict(client.cookies)
    assert first.status_code == 200, first.text
    [agent_sandbox] = provider.boxes
    assert first.json()["agent_sandbox_id"] == agent_sandbox
    assert provider.snapshots == {}, "a session past its capture exports its patch, not a sandbox"
    exported = (await server.export_session_states(["rs-1"], checkpoint_id="c2"))["rs-1"]
    assert exported["capture"] == saved_capture(agent_sandbox, PATCH), "the capture is stored as a JSON value"
    await server.shutdown()

    restored = make_server(provider, config)
    async with http(restored) as client:
        client.cookies.update(cookies)
        await restore(client, tmp_path / "ckpt")
        again = await client.post("/ng-rollout/r-a1/verify", json=verify_body(task))

    assert again.status_code == 200, again.text
    assert collections(provider) == 1, "the restored session grades the patch collected before the crash"
    assert graded_patches(restored) == [PATCH], "the saved patch is graded byte for byte"
    for field in ("reward", "agent_sandbox_id", "model_patch", "model_patch_sha256", "model_patch_bytes"):
        assert again.json()[field] == first.json()[field], field
    assert provider.live() == [] and verifiers.live() == []
    await restored.shutdown()


@pytest.mark.asyncio
async def test_a_restore_forks_the_agent_sandbox_with_the_restoring_job_s_egress_rule(
    task: InlineTask, tmp_path: Path, verifiers: VerifierProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saved rule allows the host of the model server that ran when the checkpoint was taken."""
    model_host = {"now": "10.0.0.1"}
    monkeypatch.setattr("resources_servers.deepswe.app.get_global_config_dict", lambda: {})
    monkeypatch.setattr(
        "resources_servers.deepswe.app.get_first_server_config_dict",
        lambda _config, _name: {"host": model_host["now"]},
    )

    class RecordingProvider(AgentProvider):
        """Keeps the egress rule each sandbox was created with."""

        def __init__(self) -> None:
            super().__init__()
            self.rules: dict[str, Any] = {}

        async def create(self, spec: Any) -> Any:
            handle = await super().create(spec)
            self.rules[handle.sandbox_id] = spec.provider_options.get("network_policy")
            return handle

        async def create_from_snapshot(self, snapshot_id: str, spec: Any) -> Any:
            handle = await super().create_from_snapshot(snapshot_id, spec)
            self.rules[handle.sandbox_id] = spec.provider_options.get("network_policy")
            return handle

    def allowing(host: str) -> dict[str, Any]:
        return {"defaultAction": "deny", "egress": [{"action": "allow", "target": host}]}

    provider = RecordingProvider()
    config = checkpointing_config(tmp_path / "state")
    model = {"sandbox_model_server": {"type": "responses_api_models", "name": "policy_model"}}
    async with http(make_server(provider, config, **model)) as client:
        await seed(client, episode_seed(task))
        await checkpoint(client, tmp_path / "ckpt")
    [seeded] = provider.rules
    assert provider.rules[seeded] == allowing("10.0.0.1")

    # The job is resubmitted on other nodes: its model server has another host.
    model_host["now"] = "10.0.0.2"
    restored = make_server(provider, config, **model)
    async with http(restored) as client:
        await restore(client, tmp_path / "ckpt")

    [fork] = [sandbox_id for sandbox_id in provider.rules if sandbox_id != seeded]
    assert provider.boxes[fork]["from_snapshot"] is not None
    assert provider.rules[fork] == allowing("10.0.0.2"), "the fork can reach the model server that now runs"
    await restored.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpointing", [True, False])
async def test_an_agent_run_seed_still_grades_its_patch_and_is_a_restart_under_checkpoints(
    task: InlineTask, tmp_path: Path, verifiers: VerifierProvider, checkpointing: bool
) -> None:
    provider = AgentProvider()
    config = checkpointing_config(tmp_path / "state") if checkpointing else {}
    server = make_server(provider, config)
    # With checkpointing on, every resources call of an episode carries its rollout prefix.
    prefix = "/ng-rollout/r" if checkpointing else ""
    async with http(server) as client:
        seeded = await client.post(f"{prefix}/seed_session", json=task_row(task.data, "original instruction\n"))
        assert seeded.status_code == 200, seeded.text
        if checkpointing:
            prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control("c1"), headers=AUTH)
            # The episode is a restart, so the controller leaves it out of the commit's scope.
            commit = control("c1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[])
            committed = await client.post("/ng-control/v1/checkpoint/commit", json=commit, headers=AUTH)
            await client.post("/ng-control/v1/checkpoint/resume", json=control("c1"), headers=AUTH)
            assert prepared.json()["report"]["restarts"] == ["r"]
            assert committed.status_code == 200 and committed.json()["manifest"]["record_count"] == 0
        handle = seeded.json()["sandbox_handle"]
        verified = await client.post(f"{prefix}/verify", json=verify_body(task) | {"sandbox_handle": handle})

    # No restore could install a session without an episode seed, so its reply says the episode starts over.
    assert seed_restarts(seeded.headers) is checkpointing
    assert seeded.json()["sandbox_descriptor"] == {"sandbox_id": handle, "workdir": task.data.workdir}
    assert provider.snapshots == {} and server._checkpointer().keys == []
    assert verified.status_code == 200, verified.text
    assert verified.json()["reward"] == 1.0 and verified.json()["agent_sandbox_id"] == handle
    assert graded_patches(server) == [PATCH]
    assert provider.live() == [] and verifiers.live() == []
    await server.shutdown()
