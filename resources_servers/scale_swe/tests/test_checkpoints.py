# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""scale_swe's episode sessions in partial-rollout checkpoints (``RepoTaskSessionResourcesServer``).

As for the other repository-task servers, against a fake provider with durable snapshots: the episode seed lends the
prepared task sandbox and keeps what the patch capture needs, ``/verify`` captures the agent's patch once and grades it
in a fresh sandbox on a provider of its own, and a checkpoint snapshots the agent's sandbox or, after the capture,
exports the capture alone, with scale_swe's own capture outcomes.
"""

import asyncio
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import resources_servers.scale_swe.app as scale_swe_app
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import ResourcesSeedSessionRequest
from resources_servers.scale_swe.app import ScaleSWEResourcesServer, ScaleSWEResourcesServerConfig
from resources_servers.scale_swe.verification import TEST_OUTPUT_BEGIN, VerificationResult
from resources_servers.swebench.testing.checkpoint_conformance import (
    FakeSandboxProvider,
    FakeSnapshotProvider,
    check_sandbox_session_checkpoints,
    checkpointing_config,
)


TOKEN = checkpointing_config(Path("/state"))["checkpoint"]["control_auth_token"]
AUTH = {"Authorization": f"Bearer {TOKEN}"}
# What `git rev-parse HEAD` prints in the task sandbox: a seed refuses anything but a full object name.
BASE_COMMIT = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # pragma: allowlist secret
WORKDIR = "/repo"
ROW = {"instance_id": "task", "workdir": WORKDIR, "image_url": "image", "FAIL_TO_PASS": '["tests/test_x.py::test_a"]'}
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
# A verifier run that reached pytest, so its outcome is a verdict about the patch.
PASSING = VerificationResult(
    completed=True,
    resolved=True,
    patch_applied=True,
    test_results={"resolved": True},
    test_output=f"{TEST_OUTPUT_BEGIN}\nPASSED tests/test_x.py::test_a\n",
)


def git_results(command: str) -> SandboxExecResult | None:
    """A repository at BASE_COMMIT whose agent changed src.py and that held notes.txt untracked from the start."""

    def ok(stdout: str) -> SandboxExecResult:
        return SandboxExecResult(stdout=stdout, stderr=None, return_code=0)

    if "HEAD_SHA" in command:  # the committed-work listing: nothing committed since the base
        return ok(f"HEAD_SHA {BASE_COMMIT}\n")
    if command.rstrip().endswith("rev-parse HEAD"):
        return ok(f"{BASE_COMMIT}\n")
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


@pytest.fixture(autouse=True)
def grading(monkeypatch: pytest.MonkeyPatch, graders: FakeSandboxProvider) -> AsyncMock:
    """No Gym config, and a grading that resolves every patch it is given."""
    monkeypatch.setattr(scale_swe_app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(scale_swe_app, "resolve_provider_config", lambda *_: graders)
    monkeypatch.setattr(scale_swe_app, "resolve_provider_metadata", lambda *_: {})
    # The reply's timings would differ between two runs of the same verification; a restore is compared with none.
    monkeypatch.setattr(scale_swe_app, "time", lambda: 0.0)
    run = AsyncMock(return_value=PASSING)
    monkeypatch.setattr(scale_swe_app, "run_verification", run)
    return run


def graded_patches(grading: AsyncMock) -> list[str]:
    return [call.kwargs["inputs"].patch for call in grading.await_args_list]


class _OnProvider(ScaleSWEResourcesServer):
    """The server on the provider a test gives it."""

    provider: Any = None

    def shared_provider(self) -> Any:
        return self.provider


def make_server(provider: Any, global_config: dict, **config: Any) -> ScaleSWEResourcesServer:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = global_config
    server = _OnProvider(
        config=ScaleSWEResourcesServerConfig(
            host="",
            port=0,
            entrypoint="",
            name="scale_swe",
            sandbox_provider="sandbox",
            sandbox_config={"ttl_s": 18000, "resources": {"cpu": 2}},
            sandbox_checkpoint={"snapshot_retry_delay_s": 0},
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
        task_id=TaskId(taskset="scale_swe", task_id=ROW["instance_id"]),
        task_data={**ROW, "responses_create_params": {"input": []}},
    )


def verify_body() -> dict[str, Any]:
    return {**ROW, "responses_create_params": {"input": []}, "response": RESPONSE}


def http(server: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://scale")


def control(checkpoint_id: str, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


async def seed(client: httpx.AsyncClient, body: ResourcesSeedSessionRequest) -> httpx.Response:
    reply = await client.post(
        f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
    )
    assert reply.status_code == 200, reply.text
    return reply


async def checkpoint(client: httpx.AsyncClient, checkpoint_id: str, checkpoint_dir: Path) -> None:
    for operation, body in (
        ("prepare", control(checkpoint_id)),
        ("commit", control(checkpoint_id, checkpoint_dir=str(checkpoint_dir))),
        ("resume", control(checkpoint_id)),
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


async def test_scale_swe_conforms_to_the_sandbox_session_checkpoint_contract(tmp_path: Path) -> None:
    def factory(provider: Any, global_config: dict) -> Any:
        provider._exec_results = git_results
        return make_server(provider, global_config)

    await check_sandbox_session_checkpoints(factory, episode_seed(), tmp_path / "state", verify_body=verify_body())


async def test_an_episode_seed_lends_the_prepared_checkout_and_keeps_what_the_capture_needs(tmp_path: Path) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        reply = await seed(client, episode_seed())

    access = reply.json()["sandbox_access"]
    assert access["workdir"] == WORKDIR
    assert access["connection"]["provider_config_ref"] == "sandbox"
    assert server._export_task_sandbox_state("rs-1") == {
        "pristine_untracked": ["notes.txt"],
        "base_commit": BASE_COMMIT,
    }
    # A replayed verify grades the saved patch, so the checkpoint does not wait for grading.
    assert reply.headers["x-ng-checkpoint-verify"] == "replay"
    [box] = provider.boxes.values()
    # The repository is prepared (anti-cheat scrub, committer identity) before the agent gets the sandbox.
    assert any("anti_cheat" in command for command in box["commands"]), box["commands"]
    assert any("git config --global user.email" in command for command in box["commands"])


async def test_verify_captures_the_patch_once_grades_it_in_a_fresh_sandbox_and_replays_the_verdict(
    tmp_path: Path, graders: FakeSandboxProvider, grading: AsyncMock
) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(provider, checkpointing_config(tmp_path))
    async with http(server) as client:
        await seed(client, episode_seed())
        first = await client.post("/ng-rollout/r/verify", json=verify_body())
        second = await client.post("/ng-rollout/r/verify", json=verify_body())

    assert first.status_code == second.status_code == 200, first.text
    assert first.json() == second.json()
    assert first.json()["reward"] == 1.0
    assert first.json()["model_patch"] == AGENT_SECTION
    assert captures(provider) == 1, "the patch is captured once per session"
    [diff] = [command for box in provider.boxes.values() for command in box["commands"] if "add -N ." in command]
    assert f"diff {BASE_COMMIT}" in diff and WORKDIR in diff
    assert graded_patches(grading) == [AGENT_SECTION], "an identical retry gets the first verdict"
    # The agent's sandbox stopped after the capture; the grading ran in a sandbox of its own, which it stopped.
    assert len(provider.boxes) == 1 and provider.live() == []
    assert len(graders.boxes) == 1 and graders.live() == []


async def test_a_capture_the_sandbox_cannot_run_is_masked_and_a_commit_stores_it(
    tmp_path: Path, grading: AsyncMock
) -> None:
    def unreachable(command: str) -> SandboxExecResult | None:
        if "status --porcelain" in command:
            return SandboxExecResult(stdout="", stderr="gateway timeout", return_code=125, error_type="sandbox")
        return git_results(command)

    provider = FakeSnapshotProvider(unreachable)
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    async with http(server) as client:
        await seed(client, episode_seed())
        reply = await client.post("/ng-rollout/r/verify", json=verify_body())
        await checkpoint(client, "c1", tmp_path / "ckpt")

    # Not evidence of an empty patch: the verdict is masked and nothing is graded.
    assert reply.status_code == 200, reply.text
    assert reply.json()["mask_sample"] is True and reply.json()["evaluation_completed"] is False
    assert reply.json()["failure_kind"] == "verifier_error"
    assert graded_patches(grading) == []
    # The commit could store the capture with its failure kind, and a session past its capture has no snapshot.
    assert provider.snapshots == {}
    exported = (await server.export_session_states(["rs-1"], checkpoint_id="c2"))["rs-1"]
    assert exported["capture"]["incomplete"] is True and exported["capture"]["failure_kind"] == "verifier_error"
    assert exported["capture"]["capture"]["source"] == "none"
    assert exported["sandboxes"] == {"task": None}


async def test_a_commit_during_grading_exports_the_patch_alone_and_a_restore_grades_it_again(
    tmp_path: Path, grading: AsyncMock
) -> None:
    started, finish = asyncio.Event(), asyncio.Event()

    async def slow_grading(**kwargs: Any) -> VerificationResult:
        started.set()
        await finish.wait()
        return PASSING

    grading.side_effect = slow_grading
    provider = FakeSnapshotProvider(git_results)
    config = checkpointing_config(tmp_path / "state")
    server = make_server(provider, config)
    async with http(server) as client:
        await seed(client, episode_seed())
        verifying = asyncio.create_task(
            client.post("/ng-rollout/r/verify", json=verify_body(), headers={"x-ng-checkpoint-close-follows": "1"})
        )
        try:
            async with asyncio.timeout(30):
                await started.wait()
                # A replayable verify does not hold up the checkpoint, and the patch is already captured.
                await checkpoint(client, "c1", tmp_path / "ckpt")
        finally:
            # Also when a line above fails: the grading must not stay parked behind the test.
            finish.set()
        async with asyncio.timeout(30):
            graded = await verifying
        assert graded.status_code == 200, graded.text
        # The environment server keeps the session cookie in its bookmark and sends it again after a restore.
        cookies = dict(client.cookies)
    assert provider.snapshots == {}, "a session past its capture exports its patch, not a sandbox"

    grading.side_effect = None
    async with http(make_server(provider, config)) as client:
        client.cookies.update(cookies)
        await restore(client, tmp_path / "ckpt")
        regraded = await client.post("/ng-rollout/r-a1/verify", json=verify_body())

    assert regraded.status_code == 200, regraded.text
    assert regraded.json()["model_patch"] == AGENT_SECTION
    assert regraded.json() == graded.json(), "the restored session's verdict is the one the crash interrupted"
    assert captures(provider) == 1, "the restored session grades the patch it captured before the crash"
    assert graded_patches(grading) == [AGENT_SECTION, AGENT_SECTION]


@pytest.mark.parametrize("outcome", ["git_failure", "unavailable"])
async def test_a_restore_keeps_the_outcome_of_a_capture_that_failed(
    tmp_path: Path, grading: AsyncMock, outcome: str
) -> None:
    def failing(command: str) -> SandboxExecResult | None:
        if "status --porcelain" in command:
            if outcome == "git_failure":
                return SandboxExecResult(stdout="", stderr="fatal: not a git repository", return_code=128)
            return SandboxExecResult(stdout="", stderr="gateway timeout", return_code=125, error_type="sandbox")
        return git_results(command)

    provider = FakeSnapshotProvider(failing)
    config = checkpointing_config(tmp_path / "state")
    async with http(make_server(provider, config)) as client:
        await seed(client, episode_seed())
        # The header keeps the session tracked until its close, so the commit below still exports it.
        first = await client.post(
            "/ng-rollout/r/verify", json=verify_body(), headers={"x-ng-checkpoint-close-follows": "1"}
        )
        await checkpoint(client, "c1", tmp_path / "ckpt")
        cookies = dict(client.cookies)
    assert first.status_code == 200, first.text
    assert provider.snapshots == {}, "a session past its capture exports its capture, not a sandbox"

    # A capture run again would now succeed, so a verdict that changes shows the saved capture was not used.
    provider._exec_results = git_results
    async with http(make_server(provider, config)) as client:
        client.cookies.update(cookies)
        await restore(client, tmp_path / "ckpt")
        async with asyncio.timeout(30):
            again = await client.post("/ng-rollout/r-a1/verify", json=verify_body())

    assert again.status_code == 200, again.text
    attempts = sum("status --porcelain" in c for box in provider.boxes.values() for c in box["commands"])
    assert attempts == 1, "the restored session does not capture again"
    assert provider.created_from_image() == 1 and provider.live() == []
    # Neither outcome is graded: there is no patch to grade.
    assert graded_patches(grading) == []
    for reply in (first.json(), again.json()):
        assert reply["patch_source"] == "none"
        if outcome == "git_failure":
            # A repository git cannot read is the agent's doing: a measured 0, before and after a restore.
            assert reply["evaluation_completed"] is True and reply["mask_sample"] is False
            assert reply["failure_reason"].startswith("Failed to extract model patch: git exited 128")
        else:
            # A capture the sandbox could not run is not evidence of an empty patch, before or after a restore.
            assert reply["evaluation_completed"] is False and reply["mask_sample"] is True
            assert reply["failure_kind"] == "verifier_error"


async def test_a_restore_continues_in_a_fork_and_captures_against_the_saved_base(
    tmp_path: Path, grading: AsyncMock
) -> None:
    provider = FakeSnapshotProvider(git_results)
    config = checkpointing_config(tmp_path / "state")
    async with http(make_server(provider, config)) as client:
        await seed(client, episode_seed())
        await checkpoint(client, "c1", tmp_path / "ckpt")
    assert len(provider.snapshots) == 1

    async with http(make_server(provider, config)) as client:
        await restore(client, tmp_path / "ckpt")
        reseeded = await seed(client, episode_seed(attempt=1))
        verified = await client.post("/ng-rollout/r-a1/verify", json=verify_body())

    fork = reseeded.json()["sandbox_access"]["connection"]["descriptor"]["sandbox_id"]
    assert provider.boxes[fork]["from_snapshot"] is not None, "the replacement attempt continues in the fork"
    assert provider.created_from_image() == 1, "the fork is not set up again"
    assert verified.status_code == 200, verified.text
    [diff] = [command for command in provider.boxes[fork]["commands"] if "add -N ." in command]
    assert f"diff {BASE_COMMIT}" in diff, "the patch is taken against the same base after a restore"
    assert verified.json()["model_patch"] == AGENT_SECTION, (
        "the files untracked before the agent started are restored with the session"
    )
    assert graded_patches(grading) == [AGENT_SECTION]


async def test_a_restore_forks_the_agent_sandbox_with_the_restoring_job_s_egress_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The saved rule allows the host of the model server that ran when the checkpoint was taken."""
    model_url = {"now": "http://10.0.0.1:8000"}
    for module in ("resources_servers.scale_swe.app", "resources_servers.swebench.repo_task_sessions"):
        monkeypatch.setattr(f"{module}.resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr("resources_servers.swebench.repo_task_sessions.get_global_config_dict", lambda: {})
    monkeypatch.setattr("resources_servers.swebench.agent_network.resolve_provider_config", lambda *_: ["opensandbox"])
    monkeypatch.setattr("resources_servers.swebench.agent_network.get_server_url", lambda _: model_url["now"])

    class RecordingProvider(FakeSnapshotProvider):
        """Keeps the egress rule each sandbox was created with."""

        def __init__(self) -> None:
            super().__init__(git_results)
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
    blocked = {
        "enforce_agent_no_network": True,
        "sandbox_model_server": {"type": "responses_api_models", "name": "policy_model"},
    }
    async with http(make_server(provider, config, **blocked)) as client:
        await seed(client, episode_seed())
        await checkpoint(client, "c1", tmp_path / "ckpt")
    [seeded] = provider.rules
    assert provider.rules[seeded] == allowing("10.0.0.1")

    # The job is resubmitted on other nodes: its model server has another host.
    model_url["now"] = "http://10.0.0.2:8000"
    async with http(make_server(provider, config, **blocked)) as client:
        reply = await client.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        assert reply.status_code == 200, reply.text

    [fork] = [sandbox_id for sandbox_id in provider.rules if sandbox_id != seeded]
    assert provider.boxes[fork]["from_snapshot"] is not None
    assert provider.rules[fork] == allowing("10.0.0.2"), "the fork can reach the model server that now runs"


async def test_an_agent_s_own_run_still_seeds_with_the_flat_row(graders: FakeSandboxProvider) -> None:
    provider = FakeSnapshotProvider(git_results)
    async with http(make_server(provider, {})) as client:
        seeded = await client.post("/seed_session", json={**ROW, "responses_create_params": {"input": []}})
        verified = await client.post("/verify", json=verify_body())

    assert seeded.status_code == 200, seeded.text
    # The task sandbox is on the provider every task sandbox shares.
    assert seeded.json()["sandbox_handle"] in provider.boxes
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"] == AGENT_SECTION
    assert provider.live() == [] and graders.live() == []


async def test_an_agent_s_own_run_cannot_replace_an_episode_session(graders: FakeSandboxProvider) -> None:
    provider = FakeSnapshotProvider(git_results)
    server = make_server(provider, {})
    async with http(server) as client:
        episode = await client.post("/seed_session", json=episode_seed().model_dump(mode="json"))
        assert episode.status_code == 200, episode.text
        # The same cookie, now with a flat row, as an agent's own /run would seed.
        agent_run = await client.post("/seed_session", json={**ROW, "responses_create_params": {"input": []}})

    assert agent_run.status_code == 409, agent_run.text
    [sandbox_id] = provider.boxes
    assert provider.live() == [sandbox_id], "the episode session's sandbox is neither stopped nor replaced"
    await server.shutdown()


class _SlowStops(FakeSnapshotProvider):
    """A provider whose stops wait for ``release`` once ``hold`` is set."""

    def __init__(self) -> None:
        super().__init__(git_results)
        self.hold = False
        self.stopping = asyncio.Event()
        self.release = asyncio.Event()

    async def close(self, handle: Any) -> None:
        if self.hold:
            self.stopping.set()
            await self.release.wait()
        await super().close(handle)


async def test_a_commit_during_the_stop_after_capture_waits_and_snapshots_nothing(tmp_path: Path) -> None:
    provider = _SlowStops()
    server = make_server(provider, checkpointing_config(tmp_path / "state"))
    async with http(server) as client:
        await seed(client, episode_seed())
        provider.hold = True
        verifying = asyncio.create_task(
            client.post("/ng-rollout/r/verify", json=verify_body(), headers={"x-ng-checkpoint-close-follows": "1"})
        )
        try:
            async with asyncio.timeout(30):
                await provider.stopping.wait()
            prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control("c1"), headers=AUTH)
            committing = asyncio.create_task(
                client.post(
                    "/ng-control/v1/checkpoint/commit",
                    json=control("c1", checkpoint_dir=str(tmp_path / "ckpt")),
                    headers=AUTH,
                )
            )
            # Give the commit time to reach the session while the stop is still running.
            await asyncio.sleep(0.5)
            snapshots_during_stop = dict(provider.snapshots)
        finally:
            provider.release.set()
        async with asyncio.timeout(30):
            committed = await committing
            graded = await verifying
        await client.post("/ng-control/v1/checkpoint/resume", json=control("c1"), headers=AUTH)

    assert prepared.status_code == 200, prepared.text
    assert graded.status_code == 200, graded.text
    assert snapshots_during_stop == {}, "a commit snapshotted the task sandbox while its stop was running"
    assert committed.status_code == 200, committed.text
    assert provider.snapshots == {}, "a session past its capture exports its capture, not a sandbox"
    await server.shutdown()


def test_a_golden_patch_server_holds_no_session_state() -> None:
    server = make_server(
        FakeSnapshotProvider(git_results),
        {"checkpoint": {"enabled": True, "control_auth_token": TOKEN}},
        is_verifying_golden_patch=True,
    )
    # No state_dir is needed: golden-patch grading seeds no episode sessions, so it checkpoints nothing.
    server.setup_webserver()
    assert server.checkpoint_participant.mode == "restart_only"
