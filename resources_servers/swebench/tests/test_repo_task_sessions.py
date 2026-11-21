# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoints of swe_rebench, swemer_v1, swemer_v2 and swe_next (``RepoTaskSessionResourcesServer``).

Every server runs the same contract against a fake provider that behaves like OpenSandbox with durable snapshots:
the episode seed lends the task sandbox and keeps what the patch capture needs, ``/verify`` captures the agent's
patch once and grades it in a fresh sandbox on a provider of its own, and a checkpoint snapshots the agent's sandbox
or, after the capture, exports the captured patch alone.
"""

import asyncio
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pytest import MonkeyPatch

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import ResourcesSeedSessionRequest
from resources_servers.swebench.anti_cheat import AntiCheatSetupError
from resources_servers.swebench.testing.checkpoint_conformance import (
    FakeSandboxProvider,
    FakeSnapshotProvider,
    check_sandbox_session_checkpoints,
    checkpointing_config,
)


TOKEN = checkpointing_config(Path("/state"))["checkpoint"]["control_auth_token"]
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
    # The commit the patch is taken against, and whether the seed records it (and a checkpoint exports it).
    base_commit: str = SEED_HEAD
    records_base_commit: bool = True

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
        records_base_commit=False,
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


def captures(provider: FakeSandboxProvider) -> int:
    return sum("add -N ." in command for box in provider.boxes.values() for command in box["commands"])


@pytest.fixture
def graders() -> FakeSandboxProvider:
    """The provider grading sandboxes start on, apart from the one the task sandboxes share."""
    return FakeSandboxProvider()


@pytest.fixture(params=[case.name for case in CASES])
def case(request: pytest.FixtureRequest, monkeypatch: MonkeyPatch, graders: FakeSandboxProvider) -> Case:
    """One server, with no Gym config and a grading that resolves every patch it is given."""
    case = BY_NAME[request.param]
    module = f"resources_servers.{case.name}.app"
    monkeypatch.setattr(f"{module}.get_global_config_dict", lambda: {})
    monkeypatch.setattr(f"{module}.resolve_provider_config", lambda *_: graders)
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

    class OnProvider(base):
        """The server on the provider a test gives it."""

        provider: Any = None

        def shared_provider(self) -> Any:
            return self.provider

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = global_config
    server = OnProvider(
        config=getattr(case.module, case.config)(
            host="",
            port=0,
            entrypoint="",
            name=case.name,
            sandbox_provider="sandbox",
            sandbox_config={"ttl_s": 18000, "resources": {"cpu": 2}},
            sandbox_checkpoint={"snapshot_retry_delay_s": 0},
            **config,
        ),
        server_client=client,
    )
    server.provider = provider
    return server


def episode_seed(case: Case, attempt: int = 0, session_id: str = "rs-1") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="r", attempt=attempt),
        task_id=TaskId(taskset=case.name, task_id=case.instance["instance_id"]),
        task_data={**case.instance, "responses_create_params": {"input": []}},
    )


def capture_state(case: Case) -> dict[str, Any]:
    """What a checkpoint exports for a seeded session's patch capture."""
    state: dict[str, Any] = {"pristine_untracked": ["notes.txt"]}
    if case.records_base_commit:
        state["base_commit"] = case.base_commit
    return state


def verify_body(case: Case) -> dict[str, Any]:
    return {**case.instance, "responses_create_params": {"input": []}, "response": RESPONSE}


def http(server: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://repo")


def control(checkpoint_id: str, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


async def seed(client: httpx.AsyncClient, body: ResourcesSeedSessionRequest) -> httpx.Response:
    reply = await client.post(
        f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
    )
    assert reply.status_code == 200, reply.text
    return reply


async def test_each_server_conforms_to_the_sandbox_session_checkpoint_contract(case: Case, tmp_path: Path) -> None:
    def factory(provider: Any, global_config: dict) -> Any:
        provider._exec_results = git_results
        return make_server(case, provider, global_config)

    await check_sandbox_session_checkpoints(factory, episode_seed(case), tmp_path / "state")


async def test_an_episode_seed_lends_the_prepared_checkout_and_keeps_what_the_capture_needs(
    case: Case, tmp_path: Path
) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        reply = await seed(client, episode_seed(case))

    access = reply.json()["sandbox_access"]
    assert access["workdir"] == case.workdir
    assert access["connection"]["provider_config_ref"] == "sandbox"
    assert server._export_task_sandbox_state("rs-1") == capture_state(case)
    # A replayed verify grades the saved patch, so the checkpoint does not wait for grading.
    assert reply.headers["x-ng-checkpoint-verify"] == "replay"
    [box] = provider.boxes.values()
    # The repository is prepared (anti-cheat scrub, committer identity) before the agent gets the sandbox.
    assert any("anti_cheat" in command for command in box["commands"]), box["commands"]
    assert any("git config --global user.email" in command for command in box["commands"])


async def test_an_episode_seed_whose_task_data_is_not_a_row_is_refused_without_a_sandbox(
    case: Case, tmp_path: Path
) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    invalid = episode_seed(case).model_copy(update={"task_data": {"responses_create_params": {"input": []}}})
    async with http(server) as client:
        reply = await client.post("/ng-rollout/r/seed_session", json=invalid.model_dump(mode="json"))

    assert reply.status_code == 422, reply.text
    assert provider.boxes == {}


async def test_verify_captures_the_patch_once_grades_it_in_a_fresh_sandbox_and_replays_the_verdict(
    case: Case, tmp_path: Path, graders: FakeSandboxProvider
) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        await seed(client, episode_seed(case))
        first = await client.post("/ng-rollout/r/verify", json=verify_body(case))
        second = await client.post("/ng-rollout/r/verify", json=verify_body(case))

    assert first.status_code == second.status_code == 200, first.text
    assert first.json() == second.json()
    assert first.json()["reward"] == 1.0
    assert first.json()["model_patch"] == AGENT_SECTION
    assert first.json()["patch_source"] == "worktree"
    assert captures(provider) == 1, "the patch is captured once per session"
    [diff] = [command for box in provider.boxes.values() for command in box["commands"] if "add -N ." in command]
    assert f"diff {case.base_commit}" in diff and case.workdir in diff
    assert graded_patches(case) == [AGENT_SECTION], "an identical retry gets the first verdict"
    # The agent's sandbox stopped after the capture; the grading ran in a sandbox of its own, which it stopped.
    assert len(provider.boxes) == 1 and provider.live() == []
    assert len(graders.boxes) == 1 and graders.live() == []


async def test_a_capture_git_refuses_grades_an_empty_patch_and_stops_the_agent_sandbox(
    case: Case, tmp_path: Path
) -> None:
    def failing(command: str) -> Optional[SandboxExecResult]:
        if "status --porcelain" in command:
            return SandboxExecResult(stdout="", stderr="fatal: not a git repository", return_code=128)
        return git_results(command)

    provider = FakeSnapshotProvider(failing)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        await seed(client, episode_seed(case))
        reply = await client.post("/ng-rollout/r/verify", json=verify_body(case))

    assert reply.status_code == 200, reply.text
    assert reply.json()["patch_source"] == "none"
    assert reply.json()["error"].startswith("Failed to extract model patch")
    assert graded_patches(case) == [""]
    assert provider.live() == []


async def test_a_capture_the_sandbox_cannot_run_is_masked_and_a_commit_stores_it(case: Case, tmp_path: Path) -> None:
    def unreachable(command: str) -> Optional[SandboxExecResult]:
        if "status --porcelain" in command:
            return SandboxExecResult(stdout="", stderr="gateway timeout", return_code=125, error_type="sandbox")
        return git_results(command)

    provider = FakeSnapshotProvider(unreachable)
    config = checkpointing_config(tmp_path / "state")
    server = make_server(case, provider, config)
    async with http(server) as client:
        await seed(client, episode_seed(case))
        reply = await client.post("/ng-rollout/r/verify", json=verify_body(case))
        for operation, body in (
            ("prepare", control("c1")),
            ("commit", control("c1", checkpoint_dir=str(tmp_path / "ckpt"))),
            ("resume", control("c1")),
        ):
            committed = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
            assert committed.status_code == 200, committed.text

    # Not evidence of an empty patch: the verdict is masked and nothing is graded.
    assert reply.status_code == 200, reply.text
    assert reply.json()["mask_sample"] is True and reply.json()["evaluation_completed"] is False
    assert graded_patches(case) == []
    # The commit could store the capture, and a session past its capture has no snapshot.
    assert provider.snapshots == {}
    exported = (await server.export_session_states(["rs-1"], checkpoint_id="c2"))["rs-1"]
    assert exported["capture"]["incomplete"] is True
    assert exported["capture"]["capture"]["source"] == "none"


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
        await seed(client, episode_seed(case))
        verifying = asyncio.create_task(
            client.post("/ng-rollout/r/verify", json=verify_body(case), headers={"x-ng-checkpoint-close-follows": "1"})
        )
        try:
            async with asyncio.timeout(30):
                await grading.wait()
                # A replayable verify does not hold up the checkpoint, and the patch is already captured.
                prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control("c1"), headers=AUTH)
                committed = await client.post(
                    "/ng-control/v1/checkpoint/commit",
                    json=control("c1", checkpoint_dir=str(tmp_path / "ckpt")),
                    headers=AUTH,
                )
                await client.post("/ng-control/v1/checkpoint/resume", json=control("c1"), headers=AUTH)
        finally:
            # Also when a line above fails: the grading must not stay parked behind the test.
            finish.set()
        async with asyncio.timeout(30):
            graded = await verifying
        assert graded.status_code == 200, graded.text
        # The environment server keeps the session cookie in its bookmark and sends it again after a restore.
        cookies = dict(client.cookies)
    assert prepared.json()["phase"] == "prepared", prepared.text
    assert committed.status_code == 200, committed.text
    assert provider.snapshots == {}, "a session past its capture exports its patch, not a sandbox"

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
    assert graded_patches(case) == [AGENT_SECTION, AGENT_SECTION]


@pytest.mark.parametrize("outcome", ["git_failure", "unavailable"])
async def test_a_restore_keeps_the_outcome_of_a_capture_that_failed(case: Case, tmp_path: Path, outcome: str) -> None:
    def failing(command: str) -> Optional[SandboxExecResult]:
        if "status --porcelain" in command:
            if outcome == "git_failure":
                return SandboxExecResult(stdout="", stderr="fatal: not a git repository", return_code=128)
            return SandboxExecResult(stdout="", stderr="gateway timeout", return_code=125, error_type="sandbox")
        return git_results(command)

    provider = FakeSnapshotProvider(failing)
    config = checkpointing_config(tmp_path / "state")
    server = make_server(case, provider, config)
    async with http(server) as client:
        await seed(client, episode_seed(case))
        # The header keeps the session tracked until its close, so the commit below still exports it.
        first = await client.post(
            "/ng-rollout/r/verify", json=verify_body(case), headers={"x-ng-checkpoint-close-follows": "1"}
        )
        for operation, body in (
            ("prepare", control("c1")),
            ("commit", control("c1", checkpoint_dir=str(tmp_path / "ckpt"))),
            ("resume", control("c1")),
        ):
            reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
            assert reply.status_code == 200, reply.text
        cookies = dict(client.cookies)
    assert first.status_code == 200, first.text
    assert provider.snapshots == {}, "a session past its capture exports its capture, not a sandbox"

    # A capture run again would now succeed, so a verdict that changes shows the saved capture was not used.
    provider._exec_results = git_results
    restored = make_server(case, provider, config)
    async with http(restored) as client:
        client.cookies.update(cookies)
        for operation, body in (
            ("restore", control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])),
            ("resume", control("r1")),
        ):
            reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
            assert reply.status_code == 200, reply.text
        async with asyncio.timeout(30):
            again = await client.post("/ng-rollout/r-a1/verify", json=verify_body(case))

    assert again.status_code == 200, again.text
    attempts = sum("status --porcelain" in c for box in provider.boxes.values() for c in box["commands"])
    assert attempts == 1, "the restored session does not capture again"
    assert provider.created_from_image() == 1 and provider.live() == []
    if outcome == "git_failure":
        # A git failure is the agent's: the empty patch is graded again.
        assert again.json()["patch_source"] == "none"
        assert again.json()["error"].startswith("Failed to extract model patch")
        assert again.json()["evaluation_completed"] is True
        assert graded_patches(case) == ["", ""]
    else:
        # A capture the sandbox could not run is not evidence of an empty patch, before or after a restore.
        assert again.json()["mask_sample"] is True and again.json()["evaluation_completed"] is False
        assert graded_patches(case) == []


async def test_a_restore_continues_in_a_fork_and_captures_against_the_saved_base(case: Case, tmp_path: Path) -> None:
    provider = FakeSnapshotProvider(git_results)
    config = checkpointing_config(tmp_path / "state")
    server = make_server(case, provider, config)
    async with http(server) as client:
        await seed(client, episode_seed(case))
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
        reseeded = await seed(client, episode_seed(case, attempt=1))
        verified = await client.post("/ng-rollout/r-a1/verify", json=verify_body(case))

    fork = reseeded.json()["sandbox_access"]["connection"]["descriptor"]["sandbox_id"]
    assert provider.boxes[fork]["from_snapshot"] is not None, "the replacement attempt continues in the fork"
    assert provider.created_from_image() == 1, "the fork is not set up again"
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"] == AGENT_SECTION
    [diff] = [command for command in provider.boxes[fork]["commands"] if "add -N ." in command]
    assert f"diff {case.base_commit}" in diff, "the patch is taken against the same base after a restore"
    assert AGENT_SECTION in verified.json()["model_patch"] and "notes.txt" not in verified.json()["model_patch"], (
        "the files untracked before the agent started are restored with the session"
    )


async def test_an_agent_s_own_run_still_seeds_with_the_flat_row(case: Case, graders: FakeSandboxProvider) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(case, provider, {})
    async with http(server) as client:
        seeded = await client.post("/seed_session", json={**case.instance, "responses_create_params": {"input": []}})
        verified = await client.post("/verify", json=verify_body(case))

    assert seeded.status_code == 200, seeded.text
    assert seeded.json()["sandbox_handle"] in provider.boxes
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"] == AGENT_SECTION
    assert provider.live() == [] and graders.live() == []


def test_a_golden_patch_server_holds_no_session_state(case: Case) -> None:
    server = make_server(
        case,
        FakeSnapshotProvider(git_results),
        {"checkpoint": {"enabled": True, "control_auth_token": TOKEN}},
        is_verifying_golden_patch=True,
    )
    # No state_dir is needed: golden-patch grading seeds no episode sessions, so it checkpoints nothing.
    server.setup_webserver()
    assert server.checkpoint_participant.mode == "restart_only"


@pytest.mark.parametrize("case", ["swemer_v1"], indirect=True)
async def test_the_agent_sandbox_gets_the_build_mirror_files(case: Case) -> None:
    verification = importlib.import_module("resources_servers.swemer_v1.verification")
    provider = FakeSnapshotProvider(git_results)
    async with http(make_server(case, provider, {})) as client:
        await seed(client, episode_seed(case))

    [box] = provider.boxes.values()
    assert f"upload {verification.GRADLE_INIT_PATH}" in box["commands"]
    assert f"upload {verification.MAVEN_SETTINGS_PATH}" in box["commands"]


@pytest.mark.parametrize("case", ["swe_next"], indirect=True)
async def test_swe_next_links_testbed_in_the_agent_and_grading_sandboxes(
    case: Case, graders: FakeSandboxProvider
) -> None:
    provider = FakeSnapshotProvider(git_results)
    async with http(make_server(case, provider, {})) as client:
        await seed(client, episode_seed(case))
        await client.post("/ng-rollout/r/verify", json=verify_body(case))

    [agent_box] = provider.boxes.values()
    [grading_box] = graders.boxes.values()
    # The link comes before the repository is prepared in the agent's sandbox, and before grading in the other.
    assert agent_box["commands"][0] == "ln -sfn /workspace /testbed", agent_box["commands"][:3]
    assert "ln -sfn /workspace /testbed" in grading_box["commands"]


async def test_a_required_scrub_that_fails_fails_the_seed_and_stops_the_sandbox(case: Case, tmp_path: Path) -> None:
    def failing(command: str) -> Optional[SandboxExecResult]:
        if "anti_cheat_setup.sh" in command:
            return SandboxExecResult(stdout="", stderr="cannot rewrite refs", return_code=1)
        return git_results(command)

    provider = FakeSnapshotProvider(failing)
    server = make_server(case, provider, checkpointing_config(tmp_path), anti_cheat_required=True)
    body = episode_seed(case)
    async with http(server) as client:
        # The test transport re-raises the server's error instead of answering 500 as the server would.
        with pytest.raises(AntiCheatSetupError):
            await client.post(
                f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
            )

    assert provider.live() == []
    assert "rs-1" not in server._session_id_to_sandbox


async def test_a_scrub_that_fails_is_best_effort_unless_required(case: Case, tmp_path: Path) -> None:
    def failing(command: str) -> Optional[SandboxExecResult]:
        if "anti_cheat_setup.sh" in command:
            return SandboxExecResult(stdout="", stderr="cannot rewrite refs", return_code=1)
        return git_results(command)

    provider = FakeSnapshotProvider(failing)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        reply = await seed(client, episode_seed(case))

    assert reply.status_code == 200 and len(provider.live()) == 1


def _no_retry_delay(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("nemo_gym.agents.utils.setup_retry.asyncio", SimpleNamespace(sleep=AsyncMock()))


async def test_a_seed_retries_setup_reads_a_new_sandbox_could_not_run(
    case: Case, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A new sandbox can refuse its first commands (a gateway 502); the read-only setup reads are run again."""
    _no_retry_delay(monkeypatch)
    raised: list[str] = []

    def flaky(command: str) -> Optional[SandboxExecResult]:
        for read in ("rev-parse HEAD", "ls-files --others"):
            if read in command and read not in raised:
                raised.append(read)
                raise ConnectionError("502: The sandbox could not be reached")
        return git_results(command)

    server = make_server(case, FakeSnapshotProvider(flaky), checkpointing_config(tmp_path))
    async with http(server) as client:
        reply = await seed(client, episode_seed(case))

    assert reply.status_code == 200, reply.text
    # swe_rebench takes the base commit from the row, so its seed does not read HEAD.
    assert sorted(raised) == sorted((["rev-parse HEAD"] if case.records_base_commit else []) + ["ls-files --others"])
    assert server._export_task_sandbox_state("rs-1") == capture_state(case)


async def test_a_seed_whose_untracked_files_cannot_be_listed_fails_rather_than_list_none(
    case: Case, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # An empty list would make the patch capture count the image's own untracked files as the agent's.
    _no_retry_delay(monkeypatch)

    def unreachable(command: str) -> Optional[SandboxExecResult]:
        if "ls-files --others" in command:
            raise ConnectionError("502: The sandbox could not be reached")
        return git_results(command)

    provider = FakeSnapshotProvider(unreachable)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    body = episode_seed(case)
    async with http(server) as client:
        # The test transport re-raises the server's error instead of answering 500 as the server would.
        with pytest.raises(ConnectionError, match="could not be reached"):
            await client.post(
                f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
            )

    assert sum("ls-files --others" in c for box in provider.boxes.values() for c in box["commands"]) == 3
    assert provider.live() == []


async def test_a_listing_the_provider_interrupted_fails_the_seed(case: Case, tmp_path: Path) -> None:
    def interrupted(command: str) -> Optional[SandboxExecResult]:
        if "ls-files --others" in command:
            return SandboxExecResult(stdout="", stderr="", return_code=124, error_type="timeout")
        return git_results(command)

    provider = FakeSnapshotProvider(interrupted)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    body = episode_seed(case)
    async with http(server) as client:
        with pytest.raises(RuntimeError, match="could not list its pristine untracked files: timeout"):
            await client.post(
                f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
            )

    assert provider.live() == []


async def test_a_listing_git_refuses_fails_the_seed(case: Case, tmp_path: Path) -> None:
    # No list is not an empty list: the capture would count the image's own untracked files as the agent's.
    def refused(command: str) -> Optional[SandboxExecResult]:
        if "ls-files --others" in command:
            return SandboxExecResult(stdout="", stderr="fatal: detected dubious ownership", return_code=128)
        return git_results(command)

    provider = FakeSnapshotProvider(refused)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    body = episode_seed(case)
    async with http(server) as client:
        with pytest.raises(RuntimeError, match="could not list its pristine untracked files: exit 128"):
            await client.post(
                f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
            )

    assert provider.live() == []
    assert "rs-1" not in server._session_id_to_sandbox


RECORDING = [case.name for case in CASES if case.records_base_commit]


@pytest.mark.parametrize("case", RECORDING, indirect=True)
@pytest.mark.parametrize(
    "head",
    [
        SandboxExecResult(stdout="", stderr="fatal: bad revision 'HEAD'", return_code=128),
        SandboxExecResult(stdout="", stderr="", return_code=124, error_type="timeout"),
        SandboxExecResult(stdout="\n", stderr=None, return_code=0),
    ],
    ids=["git-fails", "interrupted", "empty"],
)
async def test_a_seed_that_reads_no_base_commit_fails_and_stops_the_sandbox(
    case: Case, tmp_path: Path, head: SandboxExecResult
) -> None:
    def no_head(command: str) -> Optional[SandboxExecResult]:
        if command.rstrip().endswith("rev-parse HEAD"):
            return head
        return git_results(command)

    provider = FakeSnapshotProvider(no_head)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    body = episode_seed(case)
    async with http(server) as client:
        # The test transport re-raises the server's error instead of answering 500 as the server would.
        with pytest.raises(RuntimeError, match="did not produce a base commit"):
            await client.post(
                f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
            )

    assert provider.live() == []
    assert "rs-1" not in server._session_id_to_base_commit and "rs-1" not in server._session_id_to_sandbox


@pytest.mark.parametrize("case", RECORDING, indirect=True)
async def test_a_seed_whose_base_commit_read_keeps_raising_fails_after_three_attempts(
    case: Case, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _no_retry_delay(monkeypatch)

    def unreachable(command: str) -> Optional[SandboxExecResult]:
        if command.rstrip().endswith("rev-parse HEAD"):
            raise ConnectionError("502: The sandbox could not be reached")
        return git_results(command)

    provider = FakeSnapshotProvider(unreachable)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    body = episode_seed(case)
    async with http(server) as client:
        with pytest.raises(ConnectionError, match="could not be reached"):
            await client.post(
                f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
            )

    reads = sum(c.rstrip().endswith("rev-parse HEAD") for box in provider.boxes.values() for c in box["commands"])
    assert reads == 3
    assert provider.live() == []


@pytest.mark.parametrize("case", ["swemer_v1", "swemer_v2"], indirect=True)
async def test_a_repository_check_the_provider_interrupted_fails_the_seed_and_initialises_nothing(
    case: Case, tmp_path: Path
) -> None:
    # Not an answer: initialising here would commit a snapshot on top of a repository that may exist.
    def interrupted(command: str) -> Optional[SandboxExecResult]:
        if "rev-parse --git-dir" in command:
            return SandboxExecResult(stdout="", stderr="", return_code=124, error_type="timeout")
        return git_results(command)

    provider = FakeSnapshotProvider(interrupted)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    body = episode_seed(case)
    async with http(server) as client:
        with pytest.raises(RuntimeError, match="could not check for a git repository: timeout"):
            await client.post(
                f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
            )

    assert not any("git init" in c for box in provider.boxes.values() for c in box["commands"])
    assert provider.live() == []


async def test_a_seed_retries_the_first_command_a_new_sandbox_could_not_run(
    case: Case, tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The very first command of a new sandbox is the one most likely to meet a gateway 502."""
    _no_retry_delay(monkeypatch)
    monkeypatch.setattr("resources_servers.swebench.anti_cheat.ANTI_CHEAT_RETRY_DELAY_S", 0.0)
    refused: list[str] = []

    def first_refused(command: str) -> Optional[SandboxExecResult]:
        if not refused:
            refused.append(command)
            raise ConnectionError("502: The sandbox could not be reached")
        return git_results(command)

    provider = FakeSnapshotProvider(first_refused)
    server = make_server(case, provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        reply = await seed(client, episode_seed(case))

    assert reply.status_code == 200, reply.text
    [box] = provider.boxes.values()
    # The command the sandbox refused was run again; swe_rebench's first operation is the retried scrub.
    assert box["commands"].count(refused[0]) == 2, box["commands"][:4]
    assert server._export_task_sandbox_state("rs-1") == capture_state(case)
