# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""swe_rebench, scale_swe, swemer_v1, swemer_v2 and swe_next on the repository-task sandbox owner base.

Every server runs the same contract against a fake provider that behaves like OpenSandbox with durable snapshots:
the typed seed lends the task sandbox and keeps the base commit in the session data, ``/verify`` captures the
agent's patch once and grades it in a fresh sandbox, and a checkpoint snapshots the agent's sandbox or, after the
capture, exports the captured patch alone.
"""

import asyncio
import importlib
import time
from dataclasses import dataclass, field
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


AUTH = {"Authorization": f"Bearer {TOKEN}"}
SEED_HEAD = "5eed5eed"
ROW_BASE = "0bad0bad"
# The agent's change, plus a file the checkout held untracked before the agent started: the capture drops it.
AGENT_SECTION = "diff --git a/src.py b/src.py\n--- a/src.py\n+++ b/src.py\n@@ -1 +1 @@\n-bug\n+fix\n"
PRISTINE_SECTION = (
    "diff --git a/notes.txt b/notes.txt\nnew file mode 100644\n--- /dev/null\n+++ b/notes.txt\n@@ -0,0 +1 @@\n+x\n"
)
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


@dataclass(frozen=True)
class Case:
    name: str
    server: str
    config: str
    instance: dict[str, Any]
    workdir: str
    base_commit: str = SEED_HEAD
    extra_config: dict[str, Any] = field(default_factory=dict)

    @property
    def module(self) -> Any:
        return importlib.import_module(f"resources_servers.{self.name}.app")


CASES = [
    Case(
        "swe_rebench",
        "SWERebenchResourcesServer",
        "SWERebenchResourcesServerConfig",
        {
            "instance_id": "elastic__synthetics-1",
            "repo": "elastic/synthetics",
            "base_commit": ROW_BASE,
            "image_name": "i",
        },
        workdir="/synthetics",
        # The row names the commit its tests are written against.
        base_commit=ROW_BASE,
    ),
    Case(
        "scale_swe",
        "ScaleSWEResourcesServer",
        "ScaleSWEResourcesServerConfig",
        {"instance_id": "scale-1", "workdir": "/app", "image_url": "i", "pre_commands": "git checkout -q parent"},
        workdir="/app",
    ),
    Case(
        "swemer_v1",
        "SwemerV1ResourcesServer",
        "SwemerV1ResourcesServerConfig",
        {"instance_id": "swemer-1", "image_ref": "i", "test_framework": "pytest", "test_command": "pytest"},
        workdir="/workspace/repo",
    ),
    Case(
        "swemer_v2",
        "SwemerV2ResourcesServer",
        "SwemerV2ResourcesServerConfig",
        {"instance_id": "swemer-2", "image_ref": "i", "test_framework": "pytest", "test_command": "pytest"},
        workdir="/workspace/repo",
    ),
    Case(
        "swe_next",
        "SWENextResourcesServer",
        "SWENextResourcesServerConfig",
        {"instance_id": "n-1", "image_ref": "i"},
        "/testbed",
    ),
]
BY_NAME = {case.name: case for case in CASES}


def git_results(command: str) -> Optional[SandboxExecResult]:
    """A repository at SEED_HEAD whose agent changed src.py and that held notes.txt untracked from the start."""

    def ok(stdout: str) -> SandboxExecResult:
        return SandboxExecResult(stdout=stdout, stderr=None, return_code=0)

    if "HEAD_SHA" in command:  # the committed-work listing: nothing committed since the base
        return ok(f"HEAD_SHA {SEED_HEAD}\n")
    if command.rstrip().endswith("rev-parse HEAD"):
        return ok(f"{SEED_HEAD}\n")
    if "ls-files --others --exclude-standard" in command:
        return ok("notes.txt\n")
    if "status --porcelain" in command:
        return ok(" M src.py\n?? notes.txt\n")
    if "add -N ." in command:
        return ok(AGENT_SECTION + PRISTINE_SECTION)
    return None


def captures(provider: FakeSnapshotProvider) -> int:
    return sum("add -N ." in command for box in provider.boxes.values() for command in box["commands"])


@pytest.fixture(params=[case.name for case in CASES])
def case(request: pytest.FixtureRequest, monkeypatch: MonkeyPatch) -> Case:
    """One server, with no Gym config and a grading that resolves every patch it is given."""
    case = BY_NAME[request.param]
    module = f"resources_servers.{case.name}.app"
    monkeypatch.setattr(f"{module}.get_global_config_dict", lambda: {})
    monkeypatch.setattr(f"{module}.resolve_provider_metadata", lambda *_: {})
    if case.name == "swe_rebench":
        monkeypatch.setattr(f"{module}.resolve_parser", lambda *_: object())
    verification = importlib.import_module(f"resources_servers.{case.name}.verification")
    passing = verification.VerificationResult(
        completed=True, resolved=True, patch_applied=True, test_results={}, test_output="ok", error=None
    )
    monkeypatch.setattr(f"{module}.run_verification", AsyncMock(return_value=passing))
    return case


def graded_patches(case: Case) -> list[str]:
    return [call.kwargs["inputs"].patch for call in case.module.run_verification.await_args_list]


def make_server(case: Case, provider: Any, global_config: dict, **config: Any) -> Any:
    base = getattr(case.module, case.server)

    class Owner(base):
        """The server on the provider a test gives it."""

        provider: Any = None

        def shared_provider(self) -> Any:
            return self.provider

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = global_config
    server = Owner(
        config=getattr(case.module, case.config)(
            host="",
            port=0,
            entrypoint="",
            name=case.name,
            sandbox_provider="sandbox",
            sandbox_config={"ttl_s": 18000, "resources": {"cpu": 2}},
            sandbox_checkpoint={"snapshot_retry_delay_s": 0},
            **case.extra_config,
            **config,
        ),
        server_client=client,
    )
    server.provider = provider
    return server


def typed_seed(case: Case, attempt: int = 0, session_id: str = "rs-1", **task: Any) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="r", attempt=attempt),
        task_id=TaskId(taskset=case.name, task_id=case.instance["instance_id"]),
        task_data={**case.instance, "responses_create_params": {"input": []}, **task},
    )


def verify_body(case: Case) -> dict[str, Any]:
    return {**case.instance, "responses_create_params": {"input": []}, "response": RESPONSE}


def http(server: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://owner")


def control(checkpoint_id: str, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


async def seed(client: httpx.AsyncClient, body: ResourcesSeedSessionRequest) -> httpx.Response:
    reply = await client.post(
        f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
    )
    assert reply.status_code == 200, reply.text
    return reply


async def test_each_server_conforms_to_the_sandbox_owner_contract(case: Case, tmp_path: Path) -> None:
    async def capture(server: Any, session_id: str) -> None:
        await server.capture_once(
            session_id, "patch", lambda: server._capture_patch(session_id), release_sandbox_after=True
        )

    def factory(provider: Any, global_config: dict) -> Any:
        provider._exec_results = git_results
        return make_server(case, provider, global_config)

    await check_sandbox_owner(factory, typed_seed(case), tmp_path / "state", capture=capture)


async def test_a_typed_seed_lends_the_prepared_checkout_and_keeps_what_grading_needs(
    case: Case, tmp_path: Path
) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        reply = await seed(client, typed_seed(case))

    access = reply.json()["sandbox_access"]
    assert access["workdir"] == case.workdir
    assert access["connection"]["provider_config_ref"] == "sandbox"
    assert server.session_data("rs-1") == {"base_commit": case.base_commit, "pristine_untracked": ["notes.txt"]}
    # A replayed verify grades the stored patch, so the checkpoint does not wait for grading.
    assert reply.headers["x-ng-checkpoint-verify"] == "replay"
    [box] = provider.boxes.values()
    # The repository is prepared (anti-cheat scrub, committer identity) before the agent gets the sandbox.
    assert any("anti_cheat" in command for command in box["commands"]), box["commands"]
    assert any("git config --global user.email" in command for command in box["commands"])


async def test_a_typed_seed_whose_task_is_not_an_instance_is_refused_without_a_sandbox(
    case: Case, tmp_path: Path
) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    mismatched = typed_seed(case).model_copy(update={"task_id": TaskId(taskset=case.name, task_id="other")})
    invalid = typed_seed(case, session_id="rs-2")
    invalid = invalid.model_copy(update={"task_data": {"responses_create_params": {"input": []}}})
    async with http(server) as client:
        for body in (mismatched, invalid):
            reply = await client.post("/ng-rollout/r/seed_session", json=body.model_dump(mode="json"))
            assert reply.status_code == 422, reply.text

    assert provider.boxes == {}


async def test_verify_captures_the_patch_once_and_grades_it_in_fresh_sandboxes(case: Case, tmp_path: Path) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        await seed(client, typed_seed(case))
        first = await client.post("/ng-rollout/r/verify", json=verify_body(case))
        second = await client.post("/ng-rollout/r/verify", json=verify_body(case))

    assert first.status_code == second.status_code == 200, first.text
    assert first.json()["reward"] == second.json()["reward"] == 1.0
    assert first.json()["model_patch"] == second.json()["model_patch"] == AGENT_SECTION
    assert first.json()["patch_source"] == "worktree"
    assert captures(provider) == 1, "the patch is captured once per session"
    [diff] = [command for box in provider.boxes.values() for command in box["commands"] if "add -N ." in command]
    assert f"diff {case.base_commit}" in diff and case.workdir in diff
    assert graded_patches(case) == [AGENT_SECTION, AGENT_SECTION]
    # The agent's sandbox stopped after the capture, and each grading ran in its own sandbox, which it stopped.
    assert len(provider.boxes) == 3
    assert provider.live() == []


async def test_a_failed_capture_grades_an_empty_patch_and_keeps_the_agent_sandbox(case: Case, tmp_path: Path) -> None:
    def failing(command: str) -> Optional[SandboxExecResult]:
        if "status --porcelain" in command:
            return SandboxExecResult(stdout="", stderr="fatal: not a git repository", return_code=128)
        return git_results(command)

    provider = FakeSnapshotProvider(failing)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        await seed(client, typed_seed(case))
        reply = await client.post("/ng-rollout/r/verify", json=verify_body(case))

    assert reply.status_code == 200, reply.text
    assert reply.json()["patch_source"] == "none"
    assert reply.json()["error"].startswith("Failed to extract model patch")
    assert graded_patches(case) == [""]
    assert "patch" not in server.session_data("rs-1"), "a failed capture stores nothing, so a retry captures again"
    assert server.session_sandbox("rs-1") is not None


async def test_a_commit_during_grading_exports_the_patch_alone_and_a_restore_grades_it_again(
    case: Case, tmp_path: Path
) -> None:
    grading, finish = asyncio.Event(), asyncio.Event()
    passing = case.module.run_verification.return_value

    async def slow_grading(**kwargs: Any) -> Any:
        grading.set()
        await finish.wait()
        return passing

    case.module.run_verification.side_effect = slow_grading
    provider = FakeSnapshotProvider(git_results)
    config = checkpointing_config(tmp_path / "state")
    server = make_server(case, provider, config)
    async with http(server) as client:
        await seed(client, typed_seed(case))
        verifying = asyncio.create_task(
            client.post("/ng-rollout/r/verify", json=verify_body(case), headers={"x-ng-checkpoint-close-follows": "1"})
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
        cookies = dict(client.cookies)
    assert prepared.json()["phase"] == "prepared", prepared.text
    assert committed.status_code == 200, committed.text
    assert provider.snapshots == {}, "a session past its capture exports its data, not a sandbox"

    case.module.run_verification.side_effect = None
    restored = make_server(case, provider, config)
    async with http(restored) as client:
        client.cookies.update(cookies)
        for operation, body in (
            ("restore", control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])),
            ("resume", control("r1")),
        ):
            reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
            assert reply.status_code == 200, reply.text
        regraded = await client.post("/ng-rollout/r-a1/verify", json=verify_body(case))

    assert regraded.status_code == 200, regraded.text
    assert regraded.json()["model_patch"] == AGENT_SECTION
    assert captures(provider) == 1, "the restored session grades the patch it captured before the crash"
    assert restored.session_data("rs-1")["patch"]["patch"] == AGENT_SECTION


async def test_a_restore_continues_in_a_fork_and_captures_against_the_saved_base(case: Case, tmp_path: Path) -> None:
    provider = FakeSnapshotProvider(git_results)
    config = checkpointing_config(tmp_path / "state")
    server = make_server(case, provider, config)
    async with http(server) as client:
        await seed(client, typed_seed(case))
        for operation, body in (
            ("prepare", control("c1")),
            ("commit", control("c1", checkpoint_dir=str(tmp_path / "ckpt"))),
            ("resume", control("c1")),
        ):
            reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
            assert reply.status_code == 200, reply.text
    assert len(provider.snapshots) == 1

    restored = make_server(case, provider, config)
    async with http(restored) as client:
        for operation, body in (
            ("restore", control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])),
            ("resume", control("r1")),
        ):
            reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
            assert reply.status_code == 200, reply.text
        reseeded = await seed(client, typed_seed(case, attempt=1))
        verified = await client.post("/ng-rollout/r-a1/verify", json=verify_body(case))

    fork = reseeded.json()["sandbox_access"]["connection"]["descriptor"]["sandbox_id"]
    assert provider.boxes[fork]["from_snapshot"] is not None, "the replacement attempt continues in the fork"
    assert provider.created_from_image() == 1 + len(graded_patches(case)), "the fork is not set up again"
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"] == AGENT_SECTION
    [diff] = [command for command in provider.boxes[fork]["commands"] if "add -N ." in command]
    assert f"diff {case.base_commit}" in diff, "the base commit survives the restore in the session data"


async def test_an_agent_s_own_run_still_seeds_with_the_flat_instance(case: Case) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(case, provider, {})
    async with http(server) as client:
        seeded = await client.post("/seed_session", json={**case.instance, "responses_create_params": {"input": []}})
        verified = await client.post("/verify", json=verify_body(case))

    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["sandbox_handle"].startswith("sb-")
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"] == AGENT_SECTION
    assert provider.live() == []


def test_a_golden_patch_server_holds_no_session_state(case: Case) -> None:
    server = make_server(
        case,
        FakeSnapshotProvider(git_results),
        {"checkpoint": {"enabled": True, "control_auth_token": TOKEN}},
        is_verifying_golden_patch=True,
    )
    # No state_dir is needed: the server never seeds, so it checkpoints nothing.
    server.setup_webserver()
    assert server.checkpoint_participant.mode == "restart_only"


@pytest.mark.parametrize("case", ["swemer_v1"], indirect=True)
async def test_the_agent_sandbox_gets_the_build_mirror_files(case: Case) -> None:
    verification = importlib.import_module("resources_servers.swemer_v1.verification")
    provider = FakeSnapshotProvider(git_results)
    async with http(make_server(case, provider, {})) as client:
        await seed(client, typed_seed(case))

    [box] = provider.boxes.values()
    assert f"upload {verification.GRADLE_INIT_PATH}" in box["commands"]
    assert f"upload {verification.MAVEN_SETTINGS_PATH}" in box["commands"]


@pytest.mark.parametrize("case", ["swe_next"], indirect=True)
async def test_swe_next_links_testbed_in_the_agent_and_grading_sandboxes(case: Case) -> None:
    provider = FakeSnapshotProvider(git_results)
    async with http(make_server(case, provider, {})) as client:
        await seed(client, typed_seed(case))
        await client.post("/ng-rollout/r/verify", json=verify_body(case))

    agent_box, grading_box = provider.boxes.values()
    # The link comes before the repository is prepared in the agent's sandbox, and before grading in the other.
    assert agent_box["commands"][0] == "ln -sfn /workspace /testbed", agent_box["commands"][:3]
    assert "ln -sfn /workspace /testbed" in grading_box["commands"]
