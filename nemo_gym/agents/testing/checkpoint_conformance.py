# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformance checks for harness adapters that checkpoint their agent sessions. Import them from tests.

:func:`check_harness_checkpointing` drives an adapter through its real routes and checkpoint control routes, in
process, with its real process supervisor, in a sandbox that is this machine. An adapter's own tests call it with a
:class:`HarnessFixture` that runs a test harness in place of the adapter's CLI.
"""

import asyncio
import time
from collections.abc import Callable, Coroutine, Mapping
from functools import cached_property
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from pydantic import JsonValue

from nemo_gym.agents.base_responses_api_agent import AGENT_ACTIVATION_HEADER, SimpleResponsesAPIAgent
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
    SandboxStatus,
)


class HarnessFixture:
    """What :func:`check_harness_checkpointing` needs to know about one harness adapter and its test harness.

    The adapter runs a test harness in place of its CLI, in the sandbox of ``provider``. The harness waits for
    ``finish()`` before it ends; ``harness_pids()`` lists its running processes, and ``started()`` how many launches
    have started, relaunches included.
    """

    @cached_property
    def provider(self) -> SandboxProvider:
        """The sandbox provider the adapter must use: its one sandbox is this machine.

        Commands run in a local shell and files are local paths, so the adapter's real process supervisor, real
        signals and real files are exercised. Every agent ``make_agent`` returns must reach its sandbox through this
        provider: the crash check ends what the crashed agent ran through it.
        """
        return _LocalExecProvider()

    def make_agent(self, global_config: dict[str, JsonValue]) -> SimpleResponsesAPIAgent:
        """The adapter, configured for ``global_config``, running the test harness in the sandbox of ``provider``."""
        raise NotImplementedError

    def seed(self, agent_session_id: str, attempt: int) -> dict[str, JsonValue]:
        """An agent seed body that borrows the local sandbox."""
        raise NotImplementedError

    def activation(self) -> dict[str, JsonValue]:
        """The ``/v1/responses`` body of the session's activation."""
        raise NotImplementedError

    def started(self) -> int:
        """How many launches of the test harness have started so far, relaunches included."""
        raise NotImplementedError

    def harness_pids(self) -> list[int]:
        """The test harness processes that are still alive (see :func:`process_state`)."""
        raise NotImplementedError

    def finish(self) -> None:
        """Let every running and future launch of the test harness end."""
        raise NotImplementedError

    def hold(self) -> None:
        """Make launches wait again until the next ``finish``."""
        raise NotImplementedError

    def break_next_pause(self, agent: SimpleResponsesAPIAgent) -> None:
        """Make the next pause of the harness fail to confirm, as a process stuck in disk I/O would."""
        raise NotImplementedError

    def crash_recovery_assertion(self) -> Callable[[dict[str, JsonValue]], None] | None:
        """Record the conversation a crash is about to interrupt; return the check that recovery continued it.

        Called in the crash scenario only, once its checkpoint is committed and while the harness is still paused.
        The returned callable is called once, with the reply of the activation that recovered from the crash, and
        raises ``AssertionError`` unless that activation continued the conversation recorded here: the same
        conversation, no new user message, and progress made after the recovery. A harness that started the task
        from the beginning must fail it. Return ``None``, the default, to check only that the harness was relaunched
        once.
        """
        return None


async def check_harness_checkpointing(fixture: HarnessFixture, state_dir: Path) -> None:
    """Check a harness adapter's checkpoint contract with its real supervisor; raise ``AssertionError`` on a violation.

    Checked, through the adapter's routes and checkpoint control routes:

    - a checkpoint pauses the running harness in place (every harness process stopped) and parks the activation;
      the commit exports the session, and the resume continues the same launch, which then finishes;
    - a re-sent activation gets the reply it already gave, without another launch;
    - a pause that is not confirmed stops the harness and parks the activation, and the resume relaunches it;
    - after a crash, a restored session closed before its replacement attempt re-binds it returns its saved evidence;
    - after a crash, the replacement attempt's seed re-binds the restored session to the sandbox, and its re-sent
      activation relaunches the harness, which continues the saved conversation
      (``HarnessFixture.crash_recovery_assertion``);
    - a session closed while its activation runs stops the harness.

    The crash ends the agent that took the checkpoint before the session is restored: its sandbox handles stop
    working, every process it started in the sandbox is killed, and its tasks are cancelled and must end. Whether
    the checks pass or fail, no process started through ``fixture.provider`` and no harness process is left running.
    """
    import httpx

    provider = fixture.provider
    assert isinstance(provider, _LocalExecProvider), "HarnessFixture.provider must not be overridden"
    config: dict[str, JsonValue] = {
        "checkpoint": {"enabled": True, "control_auth_token": _TOKEN, "state_dir": str(state_dir)}
    }
    requests: list[asyncio.Task[httpx.Response]] = []

    def client_of(app: FastAPI) -> httpx.AsyncClient:
        # One client per agent session: the client keeps the session's cookie.
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent", timeout=120)

    async def control(app: FastAPI, operation: str, checkpoint_id: str, **extra: JsonValue) -> dict[str, JsonValue]:
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
            f"/ng-rollout/{rollout}/v1/responses", json=fixture.activation(), headers={AGENT_ACTIVATION_HEADER: "1"}
        )

    def in_background(request: Coroutine[Any, Any, httpx.Response]) -> asyncio.Task[httpx.Response]:
        task = asyncio.create_task(request)
        requests.append(task)
        return task

    def stopped(pids: list[int]) -> bool:
        return bool(pids) and all(process_state(pid) in ("T", "t") for pid in pids)

    try:
        # -- a checkpoint pauses the harness in place ----------------------------------------------------------
        fixture.hold()
        agent = fixture.make_agent(config)
        app = agent.setup_webserver()
        async with client_of(app) as http:
            await seed(http, "paused")
            running = in_background(activate(http))
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

        # -- a pause that is not confirmed stops the harness, and the resume relaunches it --------------------------
        fixture.hold()
        agent = fixture.make_agent(config)
        app = agent.setup_webserver()
        async with client_of(app) as http:
            await seed(http, "fallback")
            before = fixture.started()
            running = in_background(activate(http))
            await _until(lambda: fixture.started() == before + 1 and fixture.harness_pids())
            fixture.break_next_pause(agent)
            prepared = await control(app, "prepare", "c2")
            assert prepared["phase"] == "prepared", (
                f"a harness whose pause failed is stopped, and its activation parks: {prepared}"
            )
            await _until(lambda: not fixture.harness_pids())
            await control(app, "resume", "c2")
            fixture.finish()
            reply = await running
            assert reply.status_code == 200, f"the relaunched harness finishes: {reply.text}"
            assert fixture.started() == before + 2, "the stopped launch is relaunched once"

        # -- after a crash: a restored session closed before it is re-bound ---------------------------------------
        for pid in fixture.harness_pids():
            _kill(pid)
        restored = fixture.make_agent(config)
        app = restored.setup_webserver()
        scope: list[JsonValue] = [{"rollout_id": "r"}]
        await control(app, "restore", "r1", checkpoint_dir=str(state_dir / "ckpt-c1"), episode_ids=scope)
        await control(app, "resume", "r1")
        async with client_of(app) as http:
            closed = await http.post(
                "/v1/agent_sessions/close",
                json={"agent_session_id": "paused", "episode_id": {"rollout_id": "r", "attempt": 1}},
            )
        assert closed.status_code == 200, f"a session restored and never re-bound still closes: {closed.text}"
        assert closed.json().get("agent_observations") is not None, (
            "its close returns the evidence the checkpoint kept"
        )

        # -- after a crash: the replacement attempt re-binds and relaunches from the saved conversation -------------
        fixture.hold()
        # Every task that starts between here and the crash belongs to the agent that crashes.
        earlier = asyncio.all_tasks()
        agent = fixture.make_agent(config)
        app = agent.setup_webserver()
        async with client_of(app) as http:
            await seed(http, "crashed")
            before = fixture.started()
            in_background(activate(http))
            await _until(lambda: fixture.started() == before + 1 and fixture.harness_pids())
            await control(app, "prepare", "c3")
            await control(app, "commit", "c3", checkpoint_dir=str(state_dir / "ckpt-c3"))
            continued = fixture.crash_recovery_assertion()
            # The crash: the agent process and everything it ran are gone before the session is restored.
            await _crash(provider, fixture, earlier)
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
        if continued is not None:
            # Only here: a stop fallback that continued its conversation says nothing about recovery from a crash.
            continued(reply.json())

        # -- a close while the activation runs stops the harness ------------------------------------------------
        fixture.hold()
        agent = fixture.make_agent(config)
        app = agent.setup_webserver()
        async with client_of(app) as http:
            await seed(http, "closed")
            before = fixture.started()
            running = in_background(activate(http))
            await _until(lambda: fixture.started() == before + 1 and fixture.harness_pids())
            closed = await http.post(
                "/v1/agent_sessions/close", json={"agent_session_id": "closed", "episode_id": {"rollout_id": "r"}}
            )
            await asyncio.gather(running, return_exceptions=True)
        assert closed.status_code == 200, f"close: {closed.text}"
        await _until(lambda: not fixture.harness_pids())
    finally:
        # Also after a failed check: no waiting request, no paused harness, no supervisor waiting for a continue.
        waiting = [task for task in requests if not task.done()]
        for task in waiting:
            task.cancel()
        if waiting:
            await asyncio.wait(waiting, timeout=5)
        # An agent that still runs after a failed check cannot start anything again, nor wait on what is killed here.
        provider.disconnect_handles()
        await _kill_sandbox_processes(provider, fixture)


def process_state(pid: int) -> str | None:
    """The state letter of process ``pid`` in ``/proc``: ``R``, ``S``, ``T`` when stopped, ``Z`` for a zombie.

    ``None`` when there is no such process. For a fixture's ``harness_pids``: a process whose state is ``None``,
    ``Z`` or ``X`` is no longer alive.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return stat.rsplit(")", 1)[1].split()[0]


_TOKEN = "conformance-token"
_AUTH = {"Authorization": f"Bearer {_TOKEN}"}
# The environment variable that marks every process a local provider's commands start.
_MARKER_VARIABLE = "NEMO_GYM_CONFORMANCE_SANDBOX"
# How long the tasks of a crashed agent may take to end once they are cancelled.
_CRASH_TIMEOUT_S = 60.0


def _control(checkpoint_id: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 120, **extra}


class _LocalExecProvider:
    """A provider whose one "sandbox" is this machine: commands run in a local shell and files are local paths.

    For harness conformance tests, which need the real process supervisor, real signals and real files. Every
    handle names the same machine; ``connect`` accepts any descriptor this provider serialized. Every process a
    command starts inherits this provider's marker in its environment, so ``processes`` finds what is left of them.
    """

    name = "local-exec"

    def __init__(self) -> None:
        self.commands: list[str] = []
        self._marker = uuid4().hex
        # A handle carries the generation it was made in; ``disconnect_handles`` starts the next one.
        self._generation = 0

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return SandboxHandle(sandbox_id=f"local-{len(self.commands)}", provider_name=self.name, raw=self._generation)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        import os
        import signal

        self._require_connected(handle)
        self.commands.append(command)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            cwd=cwd,
            env={**os.environ, **(env or {}), _MARKER_VARIABLE: self._marker},
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

        self._require_connected(handle)
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target_path)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        import shutil

        self._require_connected(handle)
        shutil.copyfile(source_path, target_path)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def close(self, handle: SandboxHandle) -> None:
        pass

    async def aclose(self) -> None:
        pass

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, JsonValue]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        return SandboxHandle(sandbox_id=str(descriptor["sandbox_id"]), provider_name=self.name, raw=self._generation)

    def disconnect_handles(self) -> None:
        """Make every handle created or connected so far refuse commands and file transfers.

        This is what a crashed agent server holds: it can no longer change the sandbox. Handles made afterwards work.
        """
        self._generation += 1

    def processes(self) -> list[int]:
        """The processes this provider's commands started, with their descendants, that are still alive."""
        import os

        marker = f"{_MARKER_VARIABLE}={self._marker}".encode()
        pids = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                environment = (entry / "environ").read_bytes()
            except OSError:
                # The process ended meanwhile, or it is another user's.
                continue
            if marker in environment.split(b"\0") and process_state(int(entry.name)) not in (None, "Z", "X"):
                pids.append(int(entry.name))
        return pids

    def _require_connected(self, handle: SandboxHandle) -> None:
        if handle.raw != self._generation:
            raise ConnectionError("this sandbox handle belongs to an agent server that crashed")


async def _until(condition: Callable[[], object], timeout: float = 120) -> None:
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


def _kill(pid: int) -> None:
    import os
    import signal

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def _kill_sandbox_processes(
    provider: _LocalExecProvider, fixture: HarnessFixture, timeout: float = 30
) -> list[int]:
    """Kill every process the provider's commands started and every harness process, stopped ones included.

    Returns the processes still alive after ``timeout``: an empty list when all are gone.
    """
    deadline = time.monotonic() + timeout
    while True:
        pids = sorted({*provider.processes(), *fixture.harness_pids()})
        if not pids or time.monotonic() > deadline:
            return pids
        for pid in pids:
            _kill(pid)
        await asyncio.sleep(0.05)


async def _crash(provider: _LocalExecProvider, fixture: HarnessFixture, earlier: set[asyncio.Task[Any]]) -> None:
    """End the agent whose tasks started after ``earlier`` was taken, as a crash of its server and sandbox would.

    The agent's sandbox handles stop working first, so nothing it does while its tasks are cancelled can change the
    files the restored session finds. Then every process in the sandbox is killed, the harness and its supervisor
    among them, and every task the agent started is cancelled. Raises ``AssertionError`` if a process or a task is
    left.
    """
    provider.disconnect_handles()
    left = await _kill_sandbox_processes(provider, fixture)
    assert not left, f"processes of the crashed agent survived a kill: {left}"
    current = asyncio.current_task()
    cancelled: set[asyncio.Task[Any]] = set()
    deadline = time.monotonic() + _CRASH_TIMEOUT_S
    while True:
        # Read again every round: a task may start another while it handles its cancellation.
        running = {task for task in asyncio.all_tasks() - earlier if task is not current and not task.done()}
        if not running:
            break
        assert time.monotonic() < deadline, f"tasks of the crashed agent did not end when cancelled: {running}"
        for task in running - cancelled:
            task.cancel()
        cancelled |= running
        await asyncio.wait(running, timeout=1)
    for task in cancelled:
        # Read each task's error, so asyncio does not log it as never retrieved.
        if not task.cancelled():
            task.exception()
    left = await _kill_sandbox_processes(provider, fixture)
    assert not left, f"the crashed agent left processes behind: {left}"
