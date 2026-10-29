# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformance checks for harness adapters that checkpoint their agent sessions. Import them from tests.

:func:`check_harness_adapter` drives an adapter through its real routes and checkpoint control routes, in process,
with its real process supervisor, against :class:`LocalExecProvider`, a provider whose one sandbox is this machine.
An adapter's own tests call it with a :class:`HarnessFixture` that runs a test harness in place of the adapter's CLI.
"""

import asyncio
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus


TOKEN = "conformance-token"
_AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _control(checkpoint_id: str, **extra: Any) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


class LocalExecProvider:
    """A provider whose one "sandbox" is this machine: commands run in a local shell and files are local paths.

    For harness conformance tests, which need the real process supervisor, real signals and real files. Every
    handle names the same machine; ``connect`` accepts any descriptor this provider serialized.
    """

    name = "local-exec"

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return SandboxHandle(sandbox_id=f"local-{len(self.commands)}", provider_name=self.name, raw=None)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout_s: Optional[float] = None,
        user: Any = None,
    ) -> SandboxExecResult:
        import os
        import signal

        self.commands.append(command)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        except TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()
            return SandboxExecResult(stdout=None, stderr=None, return_code=-1, error_type="timeout")
        except BaseException:
            # A cancelled exec leaves nothing running here, as a provider's cancelled command would not.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        import shutil

        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        import shutil

        shutil.copyfile(source_path, target_path)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def serialize_handle(self, handle: SandboxHandle, *, scope: Optional[str] = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw=None)


class HarnessFixture:
    """What :func:`check_harness_adapter` needs to know about one harness adapter and its test harness.

    The adapter runs a test harness in place of its CLI, in a :class:`LocalExecProvider` sandbox. The harness waits
    for ``finish()`` before it ends; ``harness_pids()`` lists its running processes, and ``launches()`` how many times
    it was started, resumes included.
    """

    def make_agent(self, global_config: dict[str, Any]) -> Any:
        """The adapter, configured for ``global_config``, running the test harness in the local sandbox."""
        raise NotImplementedError

    def seed(self, agent_session_id: str, attempt: int) -> dict[str, Any]:
        """An agent seed body that borrows the local sandbox."""
        raise NotImplementedError

    def activation(self) -> dict[str, Any]:
        """The ``/v1/responses`` body of the session's activation."""
        raise NotImplementedError

    def started(self) -> int:
        """How many launches of the test harness have started so far."""
        raise NotImplementedError

    def harness_pids(self) -> list[int]:
        """The test harness processes that are still alive."""
        raise NotImplementedError

    def finish(self) -> None:
        """Let every running and future launch of the test harness end."""
        raise NotImplementedError

    def hold(self) -> None:
        """Make launches wait again until the next ``finish``."""
        raise NotImplementedError

    def break_next_pause(self, agent: Any) -> None:
        """Make the next pause of the harness fail to confirm, as a process stuck in disk I/O would."""
        raise NotImplementedError


def _process_state(pid: int) -> Optional[str]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[0]


async def _until(condition: Callable[[], Any], timeout: float = 30) -> None:
    deadline = time.monotonic() + timeout
    while True:
        value = condition()
        if asyncio.iscoroutine(value):
            value = await value
        if value:
            return
        if time.monotonic() > deadline:
            raise AssertionError("condition not reached in time")
        await asyncio.sleep(0.05)


async def check_harness_adapter(fixture: HarnessFixture, state_dir: Path) -> None:
    """Check a harness adapter's checkpoint contract with its real supervisor; raise ``AssertionError`` on a violation.

    Checked, through the adapter's routes and checkpoint control routes:

    - a checkpoint pauses the running harness in place (every harness process stopped) and parks the activation;
      the commit exports the session, and the resume continues the same launch, which then finishes;
    - a re-sent activation gets the reply it already gave, without another launch;
    - a pause that is not confirmed stops the harness and parks the activation, and the resume relaunches it;
    - after a crash, a restored session closed before its replacement attempt re-binds it returns its saved evidence;
    - after a crash, the replacement attempt's seed re-binds the restored session to the sandbox, and its re-sent
      activation relaunches the harness, which continues the saved conversation;
    - a session closed while its activation runs stops the harness.
    """
    import httpx

    config = {"checkpoint": {"enabled": True, "control_auth_token": TOKEN, "state_dir": str(state_dir)}}

    def client_of(app: Any) -> httpx.AsyncClient:
        # One client per agent session: the client keeps the session's cookie.
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent", timeout=120)

    async def control(app: Any, operation: str, checkpoint_id: str, **extra: Any) -> dict[str, Any]:
        async with client_of(app) as http:
            reply = await http.post(
                f"/ng-control/v1/checkpoint/{operation}", json=_control(checkpoint_id, **extra), headers=_AUTH
            )
        assert reply.status_code == 200, f"{operation}: {reply.status_code} {reply.text}"
        return reply.json()

    async def seed(http: httpx.AsyncClient, session_id: str, attempt: int = 0) -> None:
        reply = await http.post("/v1/agent_sessions", json=fixture.seed(session_id, attempt))
        assert reply.status_code == 200, f"seed: {reply.text}"

    async def activate(http: httpx.AsyncClient, attempt: int = 0) -> httpx.Response:
        rollout = "r" if attempt == 0 else f"r-a{attempt}"
        return await http.post(
            f"/ng-rollout/{rollout}/v1/responses", json=fixture.activation(), headers={"x-ng-agent-activation": "1"}
        )

    def stopped(pids: list[int]) -> bool:
        return bool(pids) and all(_process_state(pid) in ("T", "t") for pid in pids)

    # -- a checkpoint pauses the harness in place --------------------------------------------------------------
    fixture.hold()
    agent = fixture.make_agent(config)
    app = agent.setup_webserver()
    async with client_of(app) as http:
        await seed(http, "paused")
        running = asyncio.create_task(activate(http))
        await _until(lambda: fixture.started() == 1 and fixture.harness_pids())
        prepared = await control(app, "prepare", "c1")
        assert prepared["phase"] == "prepared", f"a running harness pauses in place: {prepared}"
        assert stopped(fixture.harness_pids()), "every harness process is stopped while the checkpoint is open"
        await control(app, "commit", "c1", checkpoint_dir=str(state_dir / "ckpt-c1"))
        await control(app, "resume", "c1")
        await _until(lambda: not stopped(fixture.harness_pids()))
        fixture.finish()
        reply = await running
        assert reply.status_code == 200, f"the paused launch finishes after resume: {reply.text}"
        assert fixture.started() == 1, "pausing in place relaunches nothing"
        again = await activate(http)
        assert again.status_code == 200 and again.json()["id"] == reply.json()["id"], "a re-sent activation"
        assert fixture.started() == 1, "a re-sent activation launches nothing"

    # -- a pause that is not confirmed stops the harness, and the resume relaunches it ------------------------------
    fixture.hold()
    agent = fixture.make_agent(config)
    app = agent.setup_webserver()
    async with client_of(app) as http:
        await seed(http, "fallback")
        before = fixture.started()
        running = asyncio.create_task(activate(http))
        await _until(lambda: fixture.started() == before + 1 and fixture.harness_pids())
        fixture.break_next_pause(agent)
        prepared = await control(app, "prepare", "c2")
        assert prepared["phase"] == "prepared", f"a harness whose pause failed is stopped and parks: {prepared}"
        await _until(lambda: not fixture.harness_pids())
        await control(app, "resume", "c2")
        fixture.finish()
        reply = await running
        assert reply.status_code == 200, f"the relaunched harness finishes: {reply.text}"
        assert fixture.started() == before + 2, "the stopped launch is relaunched once"

    # -- after a crash: a restored session closed before it is re-bound -------------------------------------------
    for pid in fixture.harness_pids():
        _kill(pid)
    restored = fixture.make_agent(config)
    app = restored.setup_webserver()
    scope = [{"rollout_id": "r"}]
    await control(app, "restore", "r1", checkpoint_dir=str(state_dir / "ckpt-c1"), episode_ids=scope)
    await control(app, "resume", "r1")
    async with client_of(app) as http:
        closed = await http.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "paused", "episode_id": {"rollout_id": "r", "attempt": 1}},
        )
    assert closed.status_code == 200, f"a session restored and never re-bound still closes: {closed.text}"
    assert closed.json().get("agent_observations") is not None, "its close returns the evidence the checkpoint kept"

    # -- after a crash: the replacement attempt re-binds and relaunches from the saved conversation -----------------
    fixture.hold()
    agent = fixture.make_agent(config)
    app = agent.setup_webserver()
    async with client_of(app) as http:
        await seed(http, "crashed")
        before = fixture.started()
        running = asyncio.create_task(activate(http))
        await _until(lambda: fixture.started() == before + 1 and fixture.harness_pids())
        await control(app, "prepare", "c3")
        await control(app, "commit", "c3", checkpoint_dir=str(state_dir / "ckpt-c3"))
        crashed = fixture.harness_pids()
        running.cancel()
    # The crash: the agent process and everything it ran are gone.
    for pid in crashed:
        _kill(pid)
    restored = fixture.make_agent(config)
    app = restored.setup_webserver()
    await control(app, "restore", "r3", checkpoint_dir=str(state_dir / "ckpt-c3"), episode_ids=scope)
    await control(app, "resume", "r3")
    async with client_of(app) as http:
        await seed(http, "crashed", attempt=1)
        fixture.finish()
        reply = await activate(http, attempt=1)
    assert reply.status_code == 200, f"the replacement attempt's activation relaunches the harness: {reply.text}"
    assert fixture.started() == before + 2, "the restored activation relaunches the harness once"

    # -- a close while the activation runs stops the harness ----------------------------------------------------
    fixture.hold()
    agent = fixture.make_agent(config)
    app = agent.setup_webserver()
    async with client_of(app) as http:
        await seed(http, "closed")
        before = fixture.started()
        running = asyncio.create_task(activate(http))
        await _until(lambda: fixture.started() == before + 1 and fixture.harness_pids())
        closed = await http.post(
            "/v1/agent_sessions/close", json={"agent_session_id": "closed", "episode_id": {"rollout_id": "r"}}
        )
        await asyncio.gather(running, return_exceptions=True)
    assert closed.status_code == 200, f"close: {closed.text}"
    await _until(lambda: not fixture.harness_pids(), timeout=30)


def _kill(pid: int) -> None:
    import os
    import signal

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
