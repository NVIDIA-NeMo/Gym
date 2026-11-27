# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the checkpoints of a repository-task SWE server (``RepoTaskSessionResourcesServer``) through its routes.

The server runs on ``FakeSnapshotProvider``, a provider with durable snapshots, over the first row of its
``data/example.jsonl``. The task sandbox holds a repository at ``SEED_HEAD`` whose agent changed ``src.py`` and that
held ``notes.txt`` untracked before the agent started. Grading runs on a provider of its own and resolves every patch.
"""

import importlib
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
from pytest import MonkeyPatch

from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_runtime._checkpoint.steps import CHECKPOINT_CLOSE_FOLLOWS_HEADER, CHECKPOINT_VERIFY_HEADER
from nemo_gym.server_runtime.base_resources_server import ResourcesSeedSessionRequest
from resources_servers.swebench.testing.checkpoint_conformance import (
    TOKEN,
    FakeSandboxProvider,
    FakeSnapshotProvider,
    check_sandbox_session_checkpoints,
    checkpointing_config,
)


AUTH = {"Authorization": f"Bearer {TOKEN}"}
SESSION_ID = "rs-1"
SEED_HEAD = "5eed5eed"
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
SCENARIOS = [
    "conformance",
    "seed_keeps_what_the_capture_needs",
    "restored_verify_grades_the_saved_patch",
    "fork_captures_against_the_saved_base",
    "agent_run_seed",
    "golden_patch_restart_only",
]


def git_results(command: str) -> SandboxExecResult | None:
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


def captures(provider: FakeSandboxProvider) -> list[str]:
    return [command for box in provider.boxes.values() for command in box["commands"] if "add -N ." in command]


def control(checkpoint_id: str, **extra: Any) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


async def checkpoint(client: httpx.AsyncClient, operations: list[tuple[str, dict[str, Any]]]) -> None:
    for operation, body in operations:
        reply = await client.post(f"/ng-control/v1/checkpoint/{operation}", json=body, headers=AUTH)
        assert reply.status_code == 200, f"{operation}: {reply.text}"


async def check_repo_task_checkpoints(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    prefix: str,
    scenario: str,
    *,
    records_base_commit: bool,
    base_commit: str | None = None,
) -> None:
    """Run ``scenario`` (one of ``SCENARIOS``) against ``resources_servers.<module_name>.app``.

    ``records_base_commit`` says the seed reads the commit the patch is taken against, and a checkpoint exports it.
    Otherwise the patch is taken against ``base_commit``, or the row's ``base_commit`` when that is ``None``.
    """
    app = importlib.import_module(f"resources_servers.{module_name}.app")
    verification = importlib.import_module(f"resources_servers.{module_name}.verification")
    row = json.loads((Path(app.__file__).parent / "data/example.jsonl").read_text().splitlines()[0])
    row = row.get("verifier_metadata", row) | {"responses_create_params": {"input": []}}
    verify = row | {"response": RESPONSE}
    # The server's fixed checkout, or the row's.
    workdir = getattr(app, "REPO_DIRECTORY", None) or row["workdir"]
    capture_base = SEED_HEAD if records_base_commit else (base_commit or row["base_commit"])

    graders = FakeSandboxProvider()
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: graders)
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
    # The timings in a verify reply are the only part two gradings of one patch do not share.
    monkeypatch.setattr(app, "time", lambda: 0.0)
    passing = verification.VerificationResult(
        completed=True, resolved=True, patch_applied=True, test_results={}, test_output="ok", error=None
    )
    grade = AsyncMock(return_value=passing)
    monkeypatch.setattr(app, "run_verification", grade)

    class OnProvider(getattr(app, prefix + "ResourcesServer")):
        """The server on the provider the scenario gives it."""

        provider: Any = None

        def shared_provider(self) -> Any:
            return self.provider

    def make(provider: Any, global_config: dict, **config: Any) -> Any:
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = global_config
        server = OnProvider(
            config=getattr(app, prefix + "ResourcesServerConfig")(
                host="",
                port=0,
                entrypoint="",
                name=module_name,
                sandbox_provider="sandbox",
                sandbox_config={"ttl_s": 18000, "resources": {"cpu": 2}},
                sandbox_checkpoint={"snapshot_retry_delay_s": 0},
                **config,
            ),
            server_client=client,
        )
        server.provider = provider
        return server

    def http(server: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://repo")

    def episode_seed(attempt: int = 0) -> ResourcesSeedSessionRequest:
        return ResourcesSeedSessionRequest(
            resources_session_id=SESSION_ID,
            episode_id=EpisodeId(rollout_id="r", attempt=attempt),
            task_id=TaskId(taskset=module_name, task_id=row["instance_id"]),
            task_data=row,
        )

    async def seed(client: httpx.AsyncClient, attempt: int = 0) -> httpx.Response:
        body = episode_seed(attempt)
        reply = await client.post(
            f"/ng-rollout/{body.episode_id.capture_key}/seed_session", json=body.model_dump(mode="json")
        )
        assert reply.status_code == 200, reply.text
        return reply

    def graded_patches() -> list[str]:
        return [call.kwargs["inputs"].patch for call in grade.await_args_list]

    capture_state: dict[str, Any] = {"pristine_untracked": ["notes.txt"]}
    if records_base_commit:
        capture_state["base_commit"] = SEED_HEAD
    commit = [
        ("prepare", control("c1")),
        ("commit", control("c1", checkpoint_dir=str(tmp_path / "ckpt"))),
        ("resume", control("c1")),
    ]
    restore = [
        ("restore", control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r"}])),
        ("resume", control("r1")),
    ]

    if scenario == "conformance":

        def factory(provider: Any, global_config: dict) -> Any:
            provider._exec_results = git_results
            return make(provider, global_config)

        await check_sandbox_session_checkpoints(factory, episode_seed(), tmp_path / "state", verify_body=verify)

    elif scenario == "seed_keeps_what_the_capture_needs":
        provider = FakeSnapshotProvider(git_results)
        server = make(provider, checkpointing_config(tmp_path))
        async with http(server) as client:
            reply = await seed(client)
        assert reply.json()["sandbox_access"]["workdir"] == workdir
        assert server._export_task_sandbox_state(SESSION_ID) == capture_state
        # A replayed verify grades the saved patch, so a checkpoint does not wait for grading.
        assert reply.headers[CHECKPOINT_VERIFY_HEADER] == "replay"
        [box] = provider.boxes.values()
        assert any("anti_cheat" in command for command in box["commands"]), box["commands"]

    elif scenario == "restored_verify_grades_the_saved_patch":
        provider = FakeSnapshotProvider(git_results)
        config = checkpointing_config(tmp_path / "state")
        async with http(make(provider, config)) as client:
            await seed(client)
            # The header keeps the session tracked until its close, so the commit below still exports it.
            first = await client.post(
                "/ng-rollout/r/verify", json=verify, headers={CHECKPOINT_CLOSE_FOLLOWS_HEADER: "1"}
            )
            await checkpoint(client, commit)
            # The environment server keeps the session cookie and sends it again after a restore.
            cookies = dict(client.cookies)
        assert first.status_code == 200, first.text
        assert provider.snapshots == {}, "a session past its capture exports its patch, not a sandbox"

        async with http(make(provider, config)) as client:
            client.cookies.update(cookies)
            await checkpoint(client, restore)
            again = await client.post("/ng-rollout/r-a1/verify", json=verify)
        assert again.status_code == 200, again.text
        assert first.json()["model_patch"] == AGENT_SECTION and first.json()["reward"] == 1.0
        assert again.json() == first.json(), "a verify after a restore replies as the one before the crash"
        assert len(captures(provider)) == 1, "the restored session grades the patch it captured before the crash"
        assert graded_patches() == [AGENT_SECTION, AGENT_SECTION]
        assert provider.live() == []

    elif scenario == "fork_captures_against_the_saved_base":
        provider = FakeSnapshotProvider(git_results)
        config = checkpointing_config(tmp_path / "state")
        async with http(make(provider, config)) as client:
            await seed(client)
            await checkpoint(client, commit)
        assert len(provider.snapshots) == 1

        restored = make(provider, config)
        async with http(restored) as client:
            await checkpoint(client, restore)
            reseeded = await seed(client, attempt=1)
            verified = await client.post("/ng-rollout/r-a1/verify", json=verify)
        fork = reseeded.json()["sandbox_access"]["connection"]["descriptor"]["sandbox_id"]
        assert provider.boxes[fork]["from_snapshot"] is not None, "the replacement attempt continues in the fork"
        assert provider.created_from_image() == 1, "the fork is not set up again"
        assert verified.status_code == 200, verified.text
        # The files untracked before the agent started are restored with the session, so the capture drops them.
        assert verified.json()["model_patch"] == AGENT_SECTION
        [diff] = [command for command in provider.boxes[fork]["commands"] if "add -N ." in command]
        assert f"diff {capture_base}" in diff and workdir in diff, "the patch is taken against the same base"

    elif scenario == "agent_run_seed":
        # Without checkpoints an agent's own run seeds with the flat row and gets the sandbox handle, as before.
        provider = FakeSnapshotProvider(git_results)
        async with http(make(provider, {})) as client:
            seeded = await client.post("/seed_session", json=row)
            verified = await client.post("/verify", json=verify)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_handle"] in provider.boxes
        assert verified.status_code == 200, verified.text
        assert verified.json()["model_patch"] == AGENT_SECTION
        [diff] = captures(provider)
        assert f"diff {capture_base}" in diff
        assert provider.live() == [] and graders.live() == []

    elif scenario == "restore_applies_the_restoring_job_s_egress_rule":
        # For a server whose config has AgentNetworkConfig. The saved rule allows the host of the model server that
        # ran when the checkpoint was taken; a job resubmitted on other nodes serves its model from another host.
        model_url = {"now": "http://10.0.0.1:8000"}
        for module in (app, importlib.import_module("resources_servers.swebench.repo_task_sessions")):
            monkeypatch.setattr(module, "resolve_provider_config", lambda *_: {"opensandbox": {}})
        monkeypatch.setattr("resources_servers.swebench.repo_task_sessions.get_global_config_dict", lambda: {})
        monkeypatch.setattr(
            "resources_servers.swebench.agent_network.resolve_provider_config", lambda *_: ["opensandbox"]
        )
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
        async with http(make(provider, config, **blocked)) as client:
            await seed(client)
            await checkpoint(client, commit)
        [seeded] = provider.rules
        assert provider.rules[seeded] == allowing("10.0.0.1")

        model_url["now"] = "http://10.0.0.2:8000"
        async with http(make(provider, config, **blocked)) as client:
            await checkpoint(client, restore[:1])
        [fork] = [sandbox_id for sandbox_id in provider.rules if sandbox_id != seeded]
        assert provider.boxes[fork]["from_snapshot"] is not None
        assert provider.rules[fork] == allowing("10.0.0.2"), "the fork can reach the model server that now runs"

    elif scenario == "golden_patch_restart_only":
        server = make(
            FakeSnapshotProvider(git_results),
            {"checkpoint": {"enabled": True, "control_auth_token": TOKEN}},
            is_verifying_golden_patch=True,
        )
        # No state_dir is needed: golden-patch grading seeds no episode sessions, so it checkpoints nothing.
        server.setup_webserver()
        assert server.checkpoint_participant.mode == "restart_only"

    else:
        raise ValueError(f"unknown scenario {scenario!r}")
