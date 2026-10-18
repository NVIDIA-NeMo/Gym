# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SWE-bench server's sandboxes through the real resources checkpoint participant.

The fake provider behaves like OpenSandbox with a durable snapshot store: a snapshot is an explicit copy of a
running sandbox that lives until deleted, and a sandbox can be created from it. A checkpoint snapshots the
agent's sandbox and leaves it running; a restore forks the snapshot under a new id.

The last tests run the real OpenCode agent against this server, as a checkpointed rollout runs it.
"""

import asyncio
import json
import time
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from omegaconf import DictConfig
from pytest import MonkeyPatch

from nemo_gym._checkpoint.agent import AgentSessionRecord
from nemo_gym._checkpoint.control import CheckpointRequest
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.checkpoint import SandboxCheckpointState
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.swebench.app import SwebenchResourcesServer, SwebenchResourcesServerConfig
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
)


AUTH = {"authorization": "Bearer t"}
SCOPE = [{"rollout_id": "r"}, {"rollout_id": "r", "attempt": 1}]
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
SEED = {**INSTANCE, "responses_create_params": {"input": []}}
VERIFY = {
    **INSTANCE,
    "responses_create_params": {"input": []},
    "response": {
        "output": [],
        "id": "",
        "created_at": 0,
        "model": "",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
}


class SnapshotProvider:
    """Explicit snapshots copy the sandbox's command history; create forks one; connect is by id."""

    name = "fake-opensandbox"

    def __init__(self) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self.fail_snapshot = False
        self._counter = 0

    def _box(self, sandbox_id: str) -> dict[str, Any]:
        box = self.boxes.get(sandbox_id)
        if box is None or box["state"] == "stopped":
            raise RuntimeError(f"sandbox {sandbox_id} is gone")
        return box

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        self._counter += 1
        sandbox_id = f"sb-{self._counter}"
        self.calls.append(("create", sandbox_id))
        snapshot_id = spec.provider_options.get("snapshot_id")
        if (spec.image is None) == (snapshot_id is None):
            raise ValueError("exactly one of image or snapshot_id must be specified")
        commands = list(self.snapshots[snapshot_id]["commands"]) if snapshot_id else []
        self.boxes[sandbox_id] = {"state": "running", "commands": commands, "from_snapshot": snapshot_id}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        box = self._box(handle.sandbox_id)
        box["commands"].append(command)
        if command == "pwd":
            return SandboxExecResult(stdout="/testbed\n", stderr=None, return_code=0)
        if "git --no-pager diff" in command:
            return SandboxExecResult(stdout="diff --git a/f b/f\n+fixed\n", stderr=None, return_code=0)
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self._box(handle.sandbox_id)["commands"].append(f"upload {target_path}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        pass

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        box = self.boxes.get(handle.sandbox_id)
        if box is None or box["state"] == "stopped":
            return SandboxStatus.STOPPED
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        self.calls.append(("close", handle.sandbox_id))
        if handle.sandbox_id in self.boxes:
            self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def snapshot(self, handle: SandboxHandle, *, name: str | None = None) -> str:
        self.calls.append(("snapshot", handle.sandbox_id))
        if self.fail_snapshot:
            raise RuntimeError("registry push failed")
        box = self._box(handle.sandbox_id)
        self._counter += 1
        snapshot_id = f"snap-{self._counter}"
        self.snapshots[snapshot_id] = {"sandbox_id": handle.sandbox_id, "commands": list(box["commands"])}
        return snapshot_id

    async def wait_snapshot_ready(self, snapshot_id: str) -> None:
        self.calls.append(("wait_snapshot_ready", snapshot_id))

    async def delete_snapshot(self, snapshot_id: str) -> None:
        self.calls.append(("delete_snapshot", snapshot_id))
        self.snapshots.pop(snapshot_id, None)

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        self._box(sandbox_id)
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)


def ops(provider: SnapshotProvider, op: str) -> list[str]:
    return [sandbox_id for name, sandbox_id in provider.calls if name == op]


def control(checkpoint_id: str = "c1", *, timeout: float = 5, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


@pytest.fixture
def provider(monkeypatch: MonkeyPatch) -> SnapshotProvider:
    provider = SnapshotProvider()
    monkeypatch.setattr("resources_servers.swebench.app.get_global_config_dict", lambda: {})
    monkeypatch.setattr("nemo_gym.sandbox.owner.get_global_config_dict", lambda: {})
    monkeypatch.setattr("nemo_gym.sandbox.owner.resolve_provider_config", lambda *_: provider)
    monkeypatch.setattr("resources_servers.swebench.app.resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr("resources_servers.swebench.app.patch_swebench_multilingual_sandbox", AsyncMock())
    monkeypatch.setattr(
        "resources_servers.swebench.app.run_instance", AsyncMock(return_value={"resolved": True, "completed": True})
    )
    return provider


def make_server(
    state_dir: Path, *, parallelism: int = 4, on_stop: str = "kill"
) -> tuple[SwebenchResourcesServer, httpx.AsyncClient]:
    """A server indexing its snapshots under ``state_dir``; a second one over the same directory is a restart."""
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "t", "state_dir": str(state_dir)}}
    )
    config = SwebenchResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="swebench",
        sandbox_provider="sandbox",
        sandbox_config={"ttl_s": 18000, "resources": {"cpu": 2}},
        sandbox_checkpoint={"parallelism": parallelism, "on_stop": on_stop},
        apply_anti_cheating=False,
        clear_swebench_debug_logs=True,
    )
    server = SwebenchResourcesServer(config=config, server_client=server_client)
    app = server.setup_webserver()
    return server, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r", timeout=30)


async def test_a_commit_snapshots_the_agents_sandbox_and_leaves_it_running(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    server, client = make_server(tmp_path / "state")
    async with client:
        seeded = await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        assert seeded.status_code == 200, seeded.text
        assert seeded.json()["sandbox_handle"] == "sb-1"
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        at_commit = provider.boxes["sb-1"]["state"]
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        verified = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)

    assert commit.status_code == 200, commit.text
    assert commit.json()["manifest"]["record_count"] == 1
    assert at_commit == "running" and ops(provider, "snapshot") == ["sb-1"]
    [record_path] = (tmp_path / "gym").rglob("records-*.jsonl")
    [record] = [json.loads(line) for line in record_path.read_text().splitlines()]
    state = SandboxCheckpointState.model_validate(record["state"]["sandbox"])
    assert state.descriptor["sandbox_id"] == "sb-1" and state.snapshot_id == "snap-2"
    assert record["state"]["workdir"] == "/testbed"
    # Verify extracted the patch from the agent's sandbox, graded in a fresh one, and stopped the agent's: the
    # episode is over. Its snapshot stays, indexed under the checkpoint, until a release drops that checkpoint.
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"] == "diff --git a/f b/f\n+fixed\n" and verified.json()["reward"] == 1
    assert len(ops(provider, "create")) == 2, "the only new sandbox is the verifier's"
    assert provider.boxes["sb-1"]["state"] == "stopped" and list(provider.snapshots) == ["snap-2"]
    assert server._checkpointer().indexed_snapshots == {"c1": ["snap-2"]}
    assert "sb-1" not in server._checkpointer().session_ids


async def test_a_fresh_server_forks_the_snapshot_and_the_replacement_reseeds_into_the_fork(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    _, client = make_server(tmp_path / "state")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        # The episode continues in the live sandbox, which moves past the checkpoint before the crash.
        cookies = dict(client.cookies)
    provider.boxes["sb-1"]["commands"].append("echo drift > file")

    fresh_server, fresh = make_server(tmp_path / "state")
    async with fresh:
        fresh.cookies.update(cookies)
        restore = await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=SCOPE),
            headers=AUTH,
        )
        assert restore.status_code == 200, restore.text
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        # The replacement attempt re-seeds the session: it gets the fork, and no other sandbox is created.
        reseeded = await fresh.post("/ng-rollout/r-a2/seed_session", json=SEED)
        access = await fresh.post("/ng-rollout/r-a2/sandbox_access")
        verified = await fresh.post("/ng-rollout/r-a2/verify", json=VERIFY)

    assert reseeded.status_code == 200, reseeded.text
    fork = reseeded.json()["sandbox_handle"]
    assert fork == "sb-3" and provider.boxes[fork]["from_snapshot"] == "snap-2"
    assert "echo drift > file" not in provider.boxes[fork]["commands"], "the fork holds the checkpoint, not the drift"
    assert access.json()["connection"]["descriptor"]["sandbox_id"] == fork and access.json()["workdir"] == "/testbed"
    await fresh_server._checkpointer().settle()
    assert provider.boxes["sb-1"]["state"] == "stopped", "the superseded sandbox is stopped"
    assert verified.status_code == 200, verified.text
    assert verified.json()["model_patch"].endswith("+fixed\n")
    assert len(ops(provider, "create")) == 3, "the original, the fork, and the verifier's"
    assert provider.boxes[fork]["state"] == "stopped"
    assert fresh_server._checkpointer().session_ids == []


async def test_a_snapshot_failure_fails_the_commit_without_publishing(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    _, client = make_server(tmp_path / "state")
    provider.fail_snapshot = True
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        provider.fail_snapshot = False
        verified = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)

    assert commit.status_code == 409 and commit.json()["error"]["code"] == "export_failed", commit.text
    assert "could not be snapshotted" in commit.json()["error"]["detail"]
    assert not list(tmp_path.rglob("manifest*"))
    assert verified.status_code == 200, "the episode continues in its sandbox"


async def test_a_commit_the_controller_stops_after_kills_the_sandbox(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    server, client = make_server(tmp_path / "state")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path), stop=True), headers=AUTH
        )

    assert commit.status_code == 200, commit.text
    assert provider.boxes["sb-1"]["state"] == "stopped" and server._checkpointer().session_ids == []
    assert len(provider.snapshots) == 1, "the checkpoint restores the session from its snapshot"


async def test_on_stop_none_leaves_the_sandbox_running(provider: SnapshotProvider, tmp_path: Path) -> None:
    _, client = make_server(tmp_path / "state", on_stop="none")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path), stop=True), headers=AUTH
        )

    assert commit.status_code == 200, commit.text
    assert provider.boxes["sb-1"]["state"] == "running"


async def test_retiring_the_attempt_stops_its_sandbox_and_keeps_its_snapshots(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    server, client = make_server(tmp_path / "state")
    async with client:
        # Attempt 0 of rollout r; a retire names the attempt and everything before it.
        await client.post("/ng-rollout/r/seed_session", json=SEED)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH)
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)
        retired = await client.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]), headers=AUTH
        )

    assert retired.status_code == 200, retired.text
    assert provider.boxes["sb-1"]["state"] == "stopped"
    assert server._checkpointer().session_ids == [] and server._workdirs == {}
    # The checkpoint still names the snapshot; a release deletes it once the controller drops the checkpoint.
    assert len(provider.snapshots) == 1 and ops(provider, "delete_snapshot") == []


async def test_a_verify_for_a_session_without_a_sandbox_is_refused_not_graded(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    _, client = make_server(tmp_path / "state")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        first = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)
        creates_after_first = len(ops(provider, "create"))
        # The client resends a /verify whose reply it lost: the first one already stopped the agent's sandbox.
        again = await client.post("/ng-rollout/r-a1/verify", json=VERIFY)

    assert first.status_code == 200 and first.json()["reward"] == 1, first.text
    # An empty patch would grade as reward 0, as if the model had changed nothing; the resend fails instead.
    assert again.status_code == 409, again.text
    assert "has no sandbox" in again.json()["detail"]
    assert len(ops(provider, "create")) == creates_after_first, "a refused verify starts no grading sandbox"


async def test_prepared_verification_captures_then_grades_while_a_checkpoint_is_open(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    server, client = make_server(tmp_path / "state")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        prepared = await client.post("/ng-rollout/r-a1/prepare_verify", json=VERIFY)
        ended = (server._checkpointer().session_ids, provider.boxes["sb-1"]["state"])
        # The reply was lost: the retry gets the same patch although the sandbox is gone.
        retried = await client.post("/ng-rollout/r-a1/prepare_verify", json=VERIFY)
        await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        # Grading a captured patch changes no session state: it runs while the checkpoint is open.
        graded = await client.post(
            "/ng-rollout/r-a1/verify_prepared", json=VERIFY | {"model_patch": prepared.json()["model_patch"]}
        )
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        await client.post("/ng-control/v1/checkpoint/resume", json=control(), headers=AUTH)

    assert prepared.status_code == 200, prepared.text
    assert prepared.json() == {"model_patch": "diff --git a/f b/f\n+fixed\n"}
    assert ended == ([], "stopped"), "the prepare ended the session and stopped the agent's sandbox"
    assert retried.status_code == 200 and retried.json() == prepared.json(), retried.text
    assert graded.status_code == 200, graded.text
    assert graded.json()["reward"] == 1 and graded.json()["model_patch"] == prepared.json()["model_patch"]
    assert commit.status_code == 200 and commit.json()["manifest"]["record_count"] == 0, commit.text


async def test_a_prepare_for_a_session_without_a_sandbox_is_refused(provider: SnapshotProvider, tmp_path: Path) -> None:
    _, client = make_server(tmp_path / "state")
    async with client:
        refused = await client.post("/ng-rollout/r-a1/prepare_verify", json=VERIFY)

    assert refused.status_code == 409 and "has no sandbox" in refused.json()["detail"], refused.text


async def test_sandbox_access_names_the_sessions_sandbox_and_checkout(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    _, client = make_server(tmp_path / "state")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        access = await client.post("/sandbox_access")

    assert access.status_code == 200, access.text
    assert access.json() == {
        # The descriptor is the sandbox's own serialization: the provider's id plus the spec's workdir.
        "connection": {
            "kind": "direct",
            "provider_config_ref": "sandbox",
            "descriptor": {"sandbox_id": "sb-1", "workdir": None},
        },
        "workdir": "/testbed",
    }


async def test_a_release_deletes_the_snapshots_of_dropped_checkpoints_even_after_a_restart(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    server, client = make_server(tmp_path / "state")
    async with client:
        await client.post("/ng-rollout/r-a1/seed_session", json=SEED)
        for checkpoint_id in ("c1", "c2"):
            await client.post("/ng-control/v1/checkpoint/prepare", json=control(checkpoint_id), headers=AUTH)
            await client.post(
                "/ng-control/v1/checkpoint/commit",
                json=control(checkpoint_id, checkpoint_dir=str(tmp_path / checkpoint_id)),
                headers=AUTH,
            )
            await client.post("/ng-control/v1/checkpoint/resume", json=control(checkpoint_id), headers=AUTH)
        released = await client.post(
            "/ng-control/v1/checkpoint/release", json=control("c2", kept_checkpoint_ids=["c2"]), headers=AUTH
        )
        await server._checkpointer().settle()
        after_first = set(provider.snapshots)

    # A restarted server has created nothing yet; the index tells it what the previous process took.
    restarted, fresh = make_server(tmp_path / "state")
    async with fresh:
        await fresh.post(
            "/ng-control/v1/checkpoint/release", json=control("done", kept_checkpoint_ids=[]), headers=AUTH
        )
        await restarted._checkpointer().settle()

    assert released.status_code == 200, released.text
    assert after_first == {"snap-3"}, "c1's snapshot went, c2's stayed"
    assert provider.snapshots == {} and restarted._checkpointer().indexed_snapshots == {}


def test_a_golden_patch_instance_lends_no_sandbox_and_starts_restart_only_with_eight_workers(
    provider: SnapshotProvider,
) -> None:
    """Training configs load the whole swebench yaml, so its grade-only golden-patch instance starts with
    checkpointing on too: it holds no session state and needs neither a state_dir nor a single worker."""
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = SwebenchResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="swebench_golden",
        sandbox_provider="sandbox",
        sandbox_config={},
        is_verifying_golden_patch=True,
        num_workers=8,
    )

    server = SwebenchResourcesServer(config=config, server_client=server_client)

    assert server.checkpoint_mode_in_effect() == "restart_only"


def test_checkpointing_without_a_state_dir_fails_at_startup(provider: SnapshotProvider) -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = SwebenchResourcesServerConfig(
        host="", port=0, entrypoint="", name="swebench", sandbox_provider="sandbox", sandbox_config={}
    )

    with pytest.raises(ValueError, match="checkpoint.state_dir is required"):
        SwebenchResourcesServer(config=config, server_client=server_client)


# -- with the real OpenCode agent ----------------------------------------------------------------------------
# The OpenCode agent borrows the session's sandbox. Its calls reach this server's app the way ServerClient sends
# them (under the attempt's /ng-rollout prefix, with the session cookie), and its sandbox client runs commands in
# the fake backend's sandboxes: ``opencode run`` runs until it is signalled or the test lets it finish, and a launch
# leaves its session in the sandbox's history, so a snapshot carries it into a fork.

OPENCODE_EXPORT = {
    "messages": [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "fix the bug"}]},
        {"info": {"role": "assistant", "time": {"created": 1, "completed": 2}}, "parts": []},
    ]
}
AGENT_CLOSE = CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 3600)


def exec_result(stdout: str = "", return_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr="", return_code=return_code, error_type=None)


def launches_in(provider: SnapshotProvider, sandbox_id: str) -> list[str]:
    commands = provider.boxes[sandbox_id]["commands"]
    return [c for c in commands if "opencode run" in c or "nemo-gym-opencode-resume.py" in c]


class AgentSandbox:
    """The agent's client for one sandbox of the fake backend."""

    def __init__(self, provider: SnapshotProvider, sandbox_id: str, finish: asyncio.Event) -> None:
        self.provider = provider
        self.sandbox_id = sandbox_id
        self.finish = finish
        self.running: Optional[asyncio.Event] = None

    async def exec(self, command: str, timeout_s: Optional[float] = None, env: Any = None, **_: Any) -> Any:
        history = self.provider._box(self.sandbox_id)["commands"]  # a sandbox that is gone raises
        history.append(command)
        if "kill -INT" in command or "kill -KILL" in command:
            # The agent's stop ends the launch that is running; the kill after it ended finds nothing.
            if self.running is not None:
                self.running.set()
            return exec_result()
        if "opencode run" in command or "nemo-gym-opencode-resume.py" in command:
            self.running = stop = asyncio.Event()
            stopped = asyncio.ensure_future(stop.wait())
            finished = asyncio.ensure_future(self.finish.wait())
            try:
                await asyncio.wait({stopped, finished}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                stopped.cancel()
                finished.cancel()
                self.running = None
            if self.finish.is_set():
                return exec_result("Shell: /bin/bash\nOpenCode run finished")
            return exec_result("Shell: /bin/bash\n", return_code=130)
        data_home = (env or {}).get("XDG_DATA_HOME")
        saved = any(f"XDG_DATA_HOME={data_home}" in c and "opencode run" in c for c in history[:-1])
        if "session list" in command:
            return exec_result('[{"id": "ses_1"}]' if saved else "")
        if "opencode export" in command:
            # OpenCode reports progress on stderr, which OpenSandbox's background executions return inside stdout.
            progress = "" if "2>/dev/null" in command else "Exporting session: ses_1\n"
            return exec_result(progress + json.dumps(OPENCODE_EXPORT)) if saved else exec_result(return_code=1)
        if "opencode db path" in command:
            return exec_result("/tmp/db")
        return exec_result()

    async def upload(self, *_: Any, **__: Any) -> None:
        pass

    async def download(self, remote: str, local: Path) -> None:
        if str(remote).endswith("export.json"):
            Path(local).write_text(json.dumps(OPENCODE_EXPORT))

    async def stop(self) -> None:
        handle = SandboxHandle(sandbox_id=self.sandbox_id, provider_name=self.provider.name, raw=None)
        await self.provider.close(handle)

    async def disconnect(self) -> None:
        pass


class AppReply:
    """An httpx response as the agent reads an aiohttp one."""

    def __init__(self, response: httpx.Response) -> None:
        self.status = response.status_code
        self.ok = response.status_code < 400
        self.headers = {key.lower(): value for key, value in response.headers.items()}
        self.cookies = dict(response.cookies)
        self._body = response.content
        self.content = SimpleNamespace(read=AsyncMock(return_value=self._body))

    async def json(self) -> Any:
        return json.loads(self._body)

    async def text(self) -> str:
        return self._body.decode(errors="replace")

    async def read(self) -> bytes:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"swebench replied {self.status}: {self._body[:300]!r}")


class ResourcesLink:
    """The agent's ServerClient.post onto this server's app, for one attempt."""

    def __init__(self, client: httpx.AsyncClient, capture_key: str) -> None:
        self.client = client
        self.capture_key = capture_key
        self.calls: list[str] = []

    async def post(self, *, server_name: str = "", url_path: str, json: Any = None, cookies: Any = None, **_: Any):
        path = f"/ng-rollout/{self.capture_key}{url_path}"
        self.calls.append(path)
        self.client.cookies.clear()
        cookie_header = "; ".join(f"{name}={value}" for name, value in (cookies or {}).items())
        response = await self.client.post(path, json=json, headers={"cookie": cookie_header} if cookie_header else {})
        return AppReply(response)


def make_opencode_agent(
    tmp_path: Path, client: httpx.AsyncClient, capture_key: str, provider: SnapshotProvider, finish: asyncio.Event
) -> tuple[OpenCodeSandboxedAgent, ResourcesLink]:
    config = OpenCodeSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="opencode",
        resources_server=ResourcesServerRef(type="resources_servers", name="swebench"),
        model_server=ModelServerRef(type="responses_api_models", name="policy"),
        opencode_version="1.0",
        preinstalled_opencode=True,
        sandbox_provider="sandbox",
        sandbox_config={},
        sandbox_timeout=600,
        opencode_max_context_window=1000,
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    link = ResourcesLink(client, capture_key)
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    server_client.post = link.post
    agent = OpenCodeSandboxedAgent(config=config, server_client=server_client)
    agent.setup_webserver()
    agent._create_opencode_config = AsyncMock(return_value={})

    async def connect(sandbox_id: Optional[str] = None, workdir: Optional[str] = None) -> AgentSandbox:
        provider._box(sandbox_id or "")  # connecting to a sandbox that is gone fails, as on OpenSandbox
        return AgentSandbox(provider, sandbox_id, finish)

    agent._start_sandbox = connect
    return agent, link


class RunRequest:
    """The agent's /run request; its ``cookies`` follow ``_cookies``, as Starlette's do."""

    def __init__(self, session_id: str) -> None:
        self._cookies: dict[str, str] = {}
        self.session = {SESSION_ID_KEY: session_id}
        self.state = SimpleNamespace()

    @property
    def cookies(self) -> dict[str, str]:
        return self._cookies


def run_request(session_id: str) -> RunRequest:
    return RunRequest(session_id)


def run_body(attempt: int = 0) -> OpenCodeSandboxedAgentRunRequest:
    return OpenCodeSandboxedAgentRunRequest.model_validate(
        {
            **INSTANCE,
            "responses_create_params": {"input": [{"role": "user", "content": "fix the bug"}]},
            "_ng_rollout_id": EpisodeId(rollout_id="r", attempt=attempt).capture_key,
        }
    )


async def until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not met"
        await asyncio.sleep(0.01)


async def crash_and_continue(
    provider: SnapshotProvider, tmp_path: Path, record: AgentSessionRecord, finish: asyncio.Event
) -> tuple[Any, ResourcesLink, SwebenchResourcesServer]:
    """Fresh servers over the same state: restore c1, then run the replacement attempt to its end."""
    fresh_server, fresh = make_server(tmp_path / "state")
    agent, link = make_opencode_agent(tmp_path, fresh, "r-a1", provider, finish)
    async with fresh:
        # As rollout collection restores: the checkpointed attempt, which the replacement r-a1 follows.
        restore = await fresh.post(
            "/ng-control/v1/checkpoint/restore",
            json=control("r1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
            headers=AUTH,
        )
        assert restore.status_code == 200, restore.text
        await fresh.post("/ng-control/v1/checkpoint/resume", json=control("r1"), headers=AUTH)
        await agent.checkpoint_participant.install([record], [EpisodeId(rollout_id="r")])
        await agent.checkpoint_participant.open_admission()
        finish.set()
        result = await asyncio.wait_for(agent.run(run_request("agent-session-2"), run_body(attempt=1)), 30)
        await fresh_server._checkpointer().settle()
    return result, link, fresh_server


async def test_opencode_continues_in_the_fork_of_its_sandbox_after_a_crash(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    finish = asyncio.Event()
    _, client = make_server(tmp_path / "state")
    agent, link = make_opencode_agent(tmp_path, client, "r", provider, finish)
    participant = agent.checkpoint_participant
    async with client:
        run = asyncio.create_task(agent.run(run_request("agent-session"), run_body()))
        await until(lambda: "sb-1" in provider.boxes and launches_in(provider, "sb-1"))
        # Checkpoint c1 in coordination's order: the agent stage parks the run, then this server prepares.
        await participant.close_admission(AGENT_CLOSE)
        await until(lambda: participant.readiness().ready)
        prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH)
        [record] = await participant.export(None, checkpoint_id="c1")
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        # The process dies with the run parked; its sandbox goes on past the checkpoint.
        run.cancel()
        with suppress(asyncio.CancelledError):
            await run
    provider.boxes["sb-1"]["commands"].append("echo drift > file")

    result, replacement, _ = await crash_and_continue(provider, tmp_path, record, finish)

    assert prepared.status_code == 200 and commit.status_code == 200, (prepared.text, commit.text)
    assert record.episode["next"] == "harness" and record.episode["resume"] is True
    assert link.calls[0] == "/ng-rollout/r/seed_session"
    # The replacement asked which sandbox is current and continued OpenCode's session in the fork.
    assert replacement.calls == ["/ng-rollout/r-a1/sandbox_access", "/ng-rollout/r-a1/verify"], "no re-seed"
    fork = next(box for box, state in provider.boxes.items() if state["from_snapshot"] == "snap-2")
    relaunch = launches_in(provider, fork)[-1]
    # The saved user turn is re-entered: no new prompt, so no message the task never had.
    assert "nemo-gym-opencode-resume.py" in relaunch and "opencode run" not in relaunch
    assert f"XDG_DATA_HOME={record.episode['state']['data_home']}" in relaunch
    assert "echo drift > file" not in provider.boxes[fork]["commands"], "the fork holds the checkpoint"
    assert provider.boxes["sb-1"]["state"] == "stopped", "the superseded sandbox is stopped"
    assert result.reward == 1.0 and result.opencode_finished is True
    # Verify stopped the fork once it had the patch; the agent never stops the sandbox SWE-bench checkpoints.
    assert ops(provider, "close").count(fork) == 1


async def test_opencode_asks_for_the_fork_when_the_checkpoint_came_before_its_first_launch(
    provider: SnapshotProvider, tmp_path: Path
) -> None:
    """A checkpoint that opens while the sandbox is still being created parks the run at its first OpenCode
    boundary (``continue: false``). After a crash the replacement still asks which sandbox is current."""
    finish = asyncio.Event()
    created = asyncio.Event()
    create = provider.create

    async def slow_create(spec: SandboxSpec) -> SandboxHandle:
        await created.wait()
        return await create(spec)

    provider.create = slow_create
    _, client = make_server(tmp_path / "state")
    agent, link = make_opencode_agent(tmp_path, client, "r", provider, finish)
    participant = agent.checkpoint_participant
    async with client:
        run = asyncio.create_task(agent.run(run_request("agent-session"), run_body()))
        await until(lambda: link.calls == ["/ng-rollout/r/seed_session"])
        # The seed is a replay step: the agent stage is ready at once; this server waits for the seed it serves.
        await participant.close_admission(AGENT_CLOSE)
        await until(lambda: participant.readiness().ready)
        preparing = asyncio.create_task(client.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=AUTH))
        await asyncio.sleep(0.05)
        provider.create = create
        created.set()
        prepared = await preparing
        await until(lambda: (participant.legacy_episodes.exported().get("run:r") or {}).get("next") == "harness")
        [record] = await participant.export(None, checkpoint_id="c1")
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        run.cancel()
        with suppress(asyncio.CancelledError):
            await run

    result, replacement, _ = await crash_and_continue(provider, tmp_path, record, finish)

    assert prepared.status_code == 200 and commit.status_code == 200, (prepared.text, commit.text)
    assert record.episode["resume"] is False and record.episode["sandbox"]["descriptor"]["sandbox_id"] == "sb-1"
    assert launches_in(provider, "sb-1") == [], "OpenCode never started before the checkpoint"
    assert replacement.calls == ["/ng-rollout/r-a1/sandbox_access", "/ng-rollout/r-a1/verify"]
    fork = next(box for box, state in provider.boxes.items() if state["from_snapshot"] == "snap-2")
    [launch] = launches_in(provider, fork)
    assert "nemo-gym-opencode-resume.py" not in launch
    assert provider.boxes["sb-1"]["state"] == "stopped"
    assert result.reward == 1.0
