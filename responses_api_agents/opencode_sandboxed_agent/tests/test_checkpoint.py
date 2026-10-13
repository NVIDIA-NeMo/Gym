# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The OpenCode agent's legacy /run as an interruptible participant in partial-rollout checkpoints.

The fake sandbox's ``opencode run`` blocks until the agent's interrupt command arrives, as the real one runs
until its task is done or it is signalled. The resources server is faked at the HTTP seam.
"""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import DictConfig

from nemo_gym._checkpoint.agent import AgentSessionRecord
from nemo_gym._checkpoint.control import CheckpointRequest
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent.app import (
    _REMOTE_RESUME_HELPER,
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
)


CLOSE = CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 3600)
EXPORT = {
    "messages": [
        {"info": {"role": "user"}, "parts": [{"type": "text", "text": "fix the bug"}]},
        {
            "info": {
                "role": "assistant",
                "tokens": {"input": 10, "output": 5, "reasoning": 0, "total": 15, "cache": {"read": 0, "write": 0}},
            },
            "parts": [{"type": "text", "text": "done"}],
        },
    ]
}


class FakeResponse:
    def __init__(self, payload: dict, *, cookies: dict | None = None, status: int = 200) -> None:
        self._payload = payload
        self.cookies = cookies or {}
        self.status = status
        self.ok = status < 400
        self.headers: dict[str, str] = {}

    async def json(self) -> dict:
        return self._payload

    async def text(self) -> str:
        return json.dumps(self._payload)

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode()


class FakeResourcesServer:
    """The resources server at the HTTP seam: seed, sandbox access, and verify."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sandbox_running = True
        # The sandbox /sandbox_access names: after a restore, the fork the resources server rebuilt.
        self.sandbox_id = "sb-1"

    async def post(self, *, server_name: str = "", url_path: str, json: Any = None, cookies: Any = None, **_: Any):
        self.calls.append(url_path)
        if url_path == "/seed_session":
            return FakeResponse({"sandbox_handle": "sb-1", "workdir": "/testbed"}, cookies={"ng_session": "s"})
        if url_path == "/sandbox_access":
            return FakeResponse(
                {
                    "connection": {
                        "kind": "direct",
                        "provider_config_ref": "sandbox",
                        "descriptor": {"sandbox_id": self.sandbox_id},
                    },
                    "workdir": "/testbed",
                }
            )
        if url_path == "/verify":
            return FakeResponse({**json, "reward": 1.0})
        raise AssertionError(url_path)


class PreparedResourcesServer(FakeResourcesServer):
    """Preparation releases the original sandbox; tests can be blocked independently."""

    def __init__(self, patch: str, *, block_preparation: bool = False, block_verification: bool = False) -> None:
        super().__init__()
        self.patch = patch
        self.preparing = asyncio.Event()
        self.prepared = asyncio.Event()
        self.verifying = asyncio.Event()
        self.verified = asyncio.Event()
        self.verified_patches: list[str] = []
        if not block_preparation:
            self.prepared.set()
        if not block_verification:
            self.verified.set()

    async def post(self, *, url_path: str, json: Any = None, **kwargs: Any):
        if url_path == "/prepare_verify":
            self.calls.append(url_path)
            self.preparing.set()
            await self.prepared.wait()
            self.sandbox_running = False
            return FakeResponse({"model_patch": self.patch})
        if url_path == "/verify_prepared":
            self.calls.append(url_path)
            self.verified_patches.append(json["model_patch"])
            self.verifying.set()
            await self.verified.wait()
            return FakeResponse({**json, "reward": 1.0})
        return await super().post(url_path=url_path, json=json, **kwargs)


class FakeSandbox:
    """OpenCode runs until interrupted; everything else answers at once."""

    def __init__(self, resources: FakeResourcesServer, *, finish_runs_after: int = 1) -> None:
        self.resources = resources
        self.commands: list[str] = []
        self.started = asyncio.Event()
        self.killed = asyncio.Event()
        self.stopped = False
        self.disconnected = 0
        self.runs = 0
        # The n-th launch (1-based) and later finish on their own instead of blocking.
        self.finish_runs_after = finish_runs_after
        self.upload = AsyncMock()

    async def exec(self, command: str, timeout_s: float | None = None, env: Any = None, **_: Any) -> Any:
        self.commands.append(command)
        if not self.resources.sandbox_running:
            raise RuntimeError("sandbox is stopped")
        if "[o]pencode run" in command:
            self.killed.set()
            return SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None)
        if "opencode run" in command or f"python3 {_REMOTE_RESUME_HELPER}" in command:
            self.runs += 1
            self.started.set()
            if self.runs > self.finish_runs_after:
                return SimpleNamespace(
                    stdout="Shell: /bin/bash\nOpenCode run finished", stderr="", return_code=0, error_type=None
                )
            await self.killed.wait()
            self.killed.clear()
            return SimpleNamespace(stdout="Shell: /bin/bash\n", stderr="interrupted", return_code=130, error_type=None)
        if "session list" in command:
            return SimpleNamespace(stdout='[{"id": "ses_1"}]', stderr="", return_code=0, error_type=None)
        if "opencode db path" in command:
            return SimpleNamespace(stdout="/tmp/db", stderr="", return_code=0, error_type=None)
        return SimpleNamespace(stdout="", stderr="", return_code=0, error_type=None)

    async def download(self, remote: str, local: Path) -> None:
        if str(remote).endswith("export.json"):
            Path(local).write_text(json.dumps(EXPORT))

    async def stop(self) -> None:
        self.stopped = True

    async def disconnect(self) -> None:
        self.disconnected += 1


def make_agent(
    tmp_path: Path, resources: FakeResourcesServer, *, concurrency: int | None = None
) -> OpenCodeSandboxedAgent:
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
        concurrency=concurrency,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    client.post = resources.post
    agent = OpenCodeSandboxedAgent(config=config, server_client=client)
    agent.setup_webserver()
    agent._create_opencode_config = AsyncMock(return_value={})
    return agent


class FakeRequest:
    """A request whose ``cookies`` follow ``_cookies``, as Starlette's do: the run swaps them for the seeded ones."""

    def __init__(self, session_id: str, body: dict) -> None:
        self._cookies: dict[str, str] = {}
        self.session = {SESSION_ID_KEY: session_id}
        self.state = SimpleNamespace()
        self.json = AsyncMock(return_value=body)

    @property
    def cookies(self) -> dict[str, str]:
        return self._cookies


def make_request(session_id: str, body: dict) -> FakeRequest:
    return FakeRequest(session_id, body)


def body_for(rollout_id: str, attempt: int = 0) -> OpenCodeSandboxedAgentRunRequest:
    key = EpisodeId(rollout_id=rollout_id, attempt=attempt).capture_key
    return OpenCodeSandboxedAgentRunRequest.model_validate(
        {"responses_create_params": {"input": [{"role": "user", "content": "fix the bug"}]}, "_ng_rollout_id": key}
    )


def launch_commands(sandbox: FakeSandbox) -> list[str]:
    return [
        c
        for c in sandbox.commands
        if ("opencode run" in c or f"python3 {_REMOTE_RESUME_HELPER}" in c) and "[o]pencode" not in c
    ]


def data_home_of(command: str) -> str:
    return next(token.split("=", 1)[1] for token in command.split() if token.startswith("XDG_DATA_HOME="))


async def wait_until(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not met"
        await asyncio.sleep(0.01)


async def test_a_checkpoint_interrupts_opencode_and_the_resume_continues_the_session(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    assert participant.readiness().blockers == ["r"], "OpenCode is running: the run blocks prepare"

    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)
    [record] = await participant.export(None)
    # The resources server pauses the sandbox now; the run is parked at its boundary.
    resources.sandbox_running = False
    await participant.open_admission()
    result = await asyncio.wait_for(run, timeout=10)

    # The exported boundary continues the OpenCode step with the same data home and the sandbox handle.
    assert record.episode["next"] == "opencode" and record.episode["continue"] is True
    assert record.episode["sandbox_handle"] == "sb-1" and record.episode["workdir"] == "/testbed"
    assert record.episode["cookies"]["ng_session"] == "s"
    first, second = launch_commands(sandbox)
    assert "opencode run" in first and f"python3 {_REMOTE_RESUME_HELPER}" in second
    assert "--continue" not in second and "Your previous session was interrupted" not in second
    assert any(call.args[1] == _REMOTE_RESUME_HELPER for call in sandbox.upload.await_args_list)
    assert data_home_of(first) == data_home_of(second) == record.episode["data_home"]
    assert 'test -x "$HOME/.opencode/bin/opencode"' in second, "the install is skipped when the binary is present"
    interrupts = [c for c in sandbox.commands if "[o]pencode run" in c]
    assert len(interrupts) == 1 and "-INT" in interrupts[0]
    # Before relaunching, the run asked the resources server which sandbox to continue in.
    assert resources.calls == ["/seed_session", "/sandbox_access", "/verify"]
    assert sandbox.disconnected == 1, "the stale client was released before reconnecting"
    assert result.reward == 1.0 and sandbox.stopped, "the episode finished and the sandbox was stopped"


@pytest.mark.parametrize("restore", [False, True], ids=["live-resume", "cold-restore"])
async def test_a_queued_run_is_checkpointed_before_it_gets_a_concurrency_slot(tmp_path: Path, restore: bool) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources, concurrency=1)
    participant = agent.checkpoint_participant
    active_sandbox = FakeSandbox(resources)
    queued_sandbox = FakeSandbox(resources, finish_runs_after=0)
    agent._start_sandbox = AsyncMock(side_effect=[active_sandbox, active_sandbox, queued_sandbox])
    tasks = [asyncio.create_task(agent.run(make_request("active-session", {}), body_for("active")))]
    try:
        await asyncio.wait_for(active_sandbox.started.wait(), 5)
        tasks.append(asyncio.create_task(agent.run(make_request("queued-session", {}), body_for("queued"))))
        await wait_until(lambda: "run:queued" in participant.legacy_episodes.keys())
        assert resources.calls == ["/seed_session"], "the queued run has not created a sandbox"

        await participant.close_admission(CLOSE)
        await wait_until(lambda: participant.readiness().ready)
        records = await participant.export(None)
        assert {record.episode_id.rollout_id for record in records} == {"active", "queued"}
        queued_record = next(record for record in records if record.episode_id.rollout_id == "queued")
        assert queued_record.episode == {"next": "seed"}
        assert not tasks[1].done() and not queued_sandbox.started.is_set()

        if restore:
            for task in tasks:
                task.cancel()
            # The upstream episode protocol holds failed replay-step cleanup until checkpoint resume.
            # The exported record above is already the restore point; release the old test handlers to drain.
            await participant.open_admission()
            await asyncio.gather(*tasks, return_exceptions=True)
            replacement_resources = FakeResourcesServer()
            replacement = make_agent(tmp_path / "replacement", replacement_resources, concurrency=1)
            replacement_sandbox = FakeSandbox(replacement_resources, finish_runs_after=0)
            replacement._start_sandbox = AsyncMock(return_value=replacement_sandbox)
            await replacement.checkpoint_participant.install([queued_record], [queued_record.episode_id])
            await replacement.checkpoint_participant.open_admission()
            result = await asyncio.wait_for(
                replacement.run(make_request("replacement-session", {}), body_for("queued", attempt=1)), 10
            )
            assert result.reward == 1.0 and replacement_sandbox.stopped
            assert replacement_resources.calls == ["/seed_session", "/verify"]
        else:
            await participant.open_admission()
            results = await asyncio.wait_for(asyncio.gather(*tasks), 10)
            assert [result.reward for result in results] == [1.0, 1.0]
            assert queued_sandbox.started.is_set() and queued_sandbox.stopped
    finally:
        for task in tasks:
            task.cancel()
        await participant.open_admission()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_cancelling_a_queued_run_removes_its_checkpoint_record_without_releasing_another_runs_slot(
    tmp_path: Path,
) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources, concurrency=1)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)
    tasks = [asyncio.create_task(agent.run(make_request("active-session", {}), body_for("active")))]
    try:
        await asyncio.wait_for(sandbox.started.wait(), 5)
        queued = asyncio.create_task(agent.run(make_request("queued-session", {}), body_for("queued")))
        tasks.append(queued)
        await wait_until(lambda: "run:queued" in participant.legacy_episodes.keys())
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert "run:queued" not in participant.legacy_episodes.keys()
        assert agent._sem.locked(), "cancelling a waiter must not release the active run's slot"
        assert agent._start_sandbox.await_count == 1
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_a_slot_released_during_checkpointing_does_not_start_the_queued_task_before_resume(
    tmp_path: Path,
) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources, concurrency=1)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    agent._start_sandbox = AsyncMock(return_value=sandbox)
    await agent._sem.acquire()
    queued = asyncio.create_task(agent.run(make_request("queued-session", {}), body_for("queued")))
    try:
        await wait_until(lambda: "run:queued" in participant.legacy_episodes.keys())
        await participant.close_admission(CLOSE)
        assert participant.readiness().ready
        agent._sem.release()
        await wait_until(lambda: agent._sem.locked())
        [record] = await participant.export(None)
        assert record.episode == {"next": "seed"}
        assert participant.readiness().ready and not queued.done()
        assert resources.calls == [], "acquiring a slot must not seed a sandbox while the checkpoint is closed"
        await participant.open_admission()
        result = await asyncio.wait_for(queued, 10)
        assert result.reward == 1.0 and sandbox.stopped
        assert not agent._sem.locked(), "the finished run must release its slot"
    finally:
        queued.cancel()
        await asyncio.gather(queued, return_exceptions=True)


async def test_a_replacement_attempt_continues_from_the_opencode_boundary_without_reseeding(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    # The resources server restored the session by forking the checkpoint's snapshot under a new id.
    resources.sandbox_id = "sb-2"
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    boundary = {
        "next": "opencode",
        "cookies": {"ng_session": "s"},
        "verify_mode": "wait",
        "sandbox_handle": "sb-1",
        "workdir": "/testbed",
        "data_home": "/tmp/nemo-gym-opencode-fixed",
        "continue": True,
    }
    await participant.install(
        [
            AgentSessionRecord(
                session_key="run:r", episode_id=EpisodeId(rollout_id="r"), session={}, boundary=None, episode=boundary
            )
        ],
        [EpisodeId(rollout_id="r")],
    )
    await participant.open_admission()

    result = await asyncio.wait_for(agent.run(make_request("agent-session-2", {}), body_for("r", attempt=1)), 10)

    [launch] = launch_commands(sandbox)
    assert f"python3 {_REMOTE_RESUME_HELPER}" in launch
    assert data_home_of(launch) == "/tmp/nemo-gym-opencode-fixed"
    assert resources.calls == ["/sandbox_access", "/verify"], "no re-seed: the session is the checkpoint's"
    # The run continued in the fork the resources server named, not in the sandbox the boundary recorded.
    assert agent._start_sandbox.await_args.kwargs == {"sandbox_id": "sb-2", "workdir": "/testbed"}
    assert result.reward == 1.0 and sandbox.stopped


async def test_a_run_cancelled_while_parked_leaves_the_checkpointed_sandbox_alone(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    await participant.close_admission(CLOSE)
    await wait_until(lambda: participant.readiness().ready)

    # The caller goes away while the run is parked (the collector stopped after its checkpoint).
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert not sandbox.stopped, "OpenCode is already stopped and the sandbox is the resources server's to keep"
    assert agent._sandbox_id_to_sandbox == {}


async def test_a_run_cancelled_while_opencode_runs_still_stops_the_sandbox(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    sandbox = FakeSandbox(resources)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    run = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    await asyncio.wait_for(sandbox.started.wait(), timeout=5)
    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert sandbox.stopped, "a disconnected caller must not leave OpenCode generating in its pod"


async def test_without_checkpointing_the_run_is_unchanged(tmp_path: Path) -> None:
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    agent._checkpoint_participant = None
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    agent._start_sandbox = AsyncMock(return_value=sandbox)

    result = await asyncio.wait_for(agent.run(make_request("agent-session", {}), body_for("r")), 10)

    [launch] = launch_commands(sandbox)
    assert "--continue" not in launch
    assert resources.calls == ["/seed_session", "/verify"]
    assert result.reward == 1.0 and sandbox.stopped


@pytest.mark.parametrize("patch", ["diff --git a/f b/f\n+fixed\n", ""])
async def test_verification_cut_carries_the_patch_and_restores_without_opening_or_reseeding_a_sandbox(
    tmp_path: Path, patch: str
) -> None:
    resources = PreparedResourcesServer(patch, block_verification=True)
    agent = make_agent(tmp_path, resources)
    agent.config.prepare_verification = True
    sandbox = FakeSandbox(resources, finish_runs_after=0)
    agent._start_sandbox = AsyncMock(return_value=sandbox)
    participant = agent.checkpoint_participant
    pending = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    try:
        await asyncio.wait_for(resources.verifying.wait(), 5)
        assert not resources.sandbox_running
        await participant.close_admission(CLOSE)
        assert participant.readiness().ready, "a replayable test must not block checkpoint preparation"
        [record] = await participant.export(None)
        assert record.episode["next"] == "verify"
        assert record.episode["verify_mode"] == "replay" and record.episode["prepared_patch"] == patch
        assert not pending.done()
    finally:
        pending.cancel()
        await participant.open_admission()
        await asyncio.gather(pending, return_exceptions=True)

    replacement_resources = PreparedResourcesServer(patch)
    replacement_resources.sandbox_running = False
    replacement = make_agent(tmp_path / "restored", replacement_resources)
    # A saved prepared patch is authoritative even if the opt-in has since been disabled.
    assert not replacement.config.prepare_verification
    replacement._start_sandbox = AsyncMock(side_effect=AssertionError("the original sandbox is gone"))
    await replacement.checkpoint_participant.install([record], [record.episode_id])
    await replacement.checkpoint_participant.open_admission()
    result = await asyncio.wait_for(replacement.run(make_request("restored-session", {}), body_for("r", attempt=1)), 5)
    assert result.reward == 1.0
    assert replacement_resources.calls == ["/verify_prepared"]
    assert replacement_resources.verified_patches == resources.verified_patches == [patch]
    replacement._start_sandbox.assert_not_awaited()


async def test_checkpoint_preparation_waits_until_the_patch_has_been_received_and_recorded(tmp_path: Path) -> None:
    resources = PreparedResourcesServer("diff", block_preparation=True, block_verification=True)
    agent = make_agent(tmp_path, resources)
    agent.config.prepare_verification = True
    agent._start_sandbox = AsyncMock(return_value=FakeSandbox(resources, finish_runs_after=0))
    participant = agent.checkpoint_participant
    pending = asyncio.create_task(agent.run(make_request("agent-session", {}), body_for("r")))
    try:
        await asyncio.wait_for(resources.preparing.wait(), 5)
        await participant.close_admission(CLOSE)
        assert not participant.readiness().ready, "extraction/release is a wait step, not a replay step"
        resources.prepared.set()
        await wait_until(lambda: participant.readiness().ready)
        [record] = await participant.export(None)
        assert record.episode["prepared_patch"] == "diff" and record.episode["next"] == "verify"
        assert not resources.verifying.is_set(), "tests start only after the saved boundary resumes"
    finally:
        pending.cancel()
        await participant.open_admission()
        await asyncio.gather(pending, return_exceptions=True)


async def test_a_replacement_from_the_return_boundary_returns_the_stored_result_without_regrading(
    tmp_path: Path,
) -> None:
    """A run parked after verify (it replied during prepare, or was about to) is continued from "return"."""
    resources = FakeResourcesServer()
    agent = make_agent(tmp_path, resources)
    participant = agent.checkpoint_participant
    agent._start_sandbox = AsyncMock(side_effect=AssertionError("no sandbox is touched after verify"))

    stored_result = {
        **body_for("r").model_dump(mode="json"),
        "response": {
            "id": "resp_x",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [],
            "tool_choice": "auto",
            "tools": [],
            "parallel_tool_calls": False,
        },
        "reward": 1.0,
        "resolved": True,
    }
    run_result = {
        "opencode_failed": False,
        "opencode_exit_code": 0,
        "opencode_error_type": None,
        "opencode_results_fpath": "/x/export.json",
        "opencode_run_stdout": "Shell: /bin/bash\nOpenCode run finished",
        "opencode_run_stderr": "",
        "opencode_export_found": True,
        "opencode_finished": True,
    }
    await participant.install(
        [
            AgentSessionRecord(
                session_key="run:r",
                episode_id=EpisodeId(rollout_id="r"),
                session={},
                boundary=None,
                episode={"next": "return", "result": stored_result, "run_result": run_result},
            )
        ],
        [EpisodeId(rollout_id="r")],
    )
    await participant.open_admission()

    result = await asyncio.wait_for(agent.run(make_request("agent-session-3", {}), body_for("r", attempt=1)), 10)

    assert result.reward == 1.0 and result.opencode_finished is True
    assert result.opencode_results_fpath == "/x/export.json"
    assert resources.calls == [], "no re-seed, no re-grade"
