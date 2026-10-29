# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformance checks for servers that checkpoint the sandboxes they own.

:func:`check_sandbox_owner` drives a ``SandboxOwnerResourcesServer`` subclass through its real routes,
checkpoint control routes and middleware, in process, against :class:`FakeSnapshotProvider`, a provider with
durable snapshots that behaves like OpenSandbox's: a snapshot is an independent copy that lives until deleted,
and ``create`` with ``provider_options["snapshot_id"]`` forks it under a new id. A server's own tests call it with
a factory that builds the server on the provider it is given, and one seed request the server accepts.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus


TOKEN = "conformance-token"
_AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeSandboxProvider:
    """An in-process provider: each sandbox records the commands it ran and the files uploaded to it.

    ``exec_results`` answers a command; the default answers ``pwd`` with ``/testbed`` and everything else with an
    empty success. A stopped sandbox refuses every call, as a deleted one does.
    """

    name = "fake"

    def __init__(self, exec_results: Optional[Callable[[str], Optional[SandboxExecResult]]] = None) -> None:
        self.boxes: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self._exec_results = exec_results
        self._counter = 0

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def _box(self, sandbox_id: str) -> dict[str, Any]:
        box = self.boxes.get(sandbox_id)
        if box is None or box["state"] == "stopped":
            raise RuntimeError(f"sandbox {sandbox_id} is gone")
        return box

    def live(self) -> list[str]:
        """The sandboxes that are still running."""
        return sorted(sandbox_id for sandbox_id, box in self.boxes.items() if box["state"] == "running")

    def created_from_image(self) -> int:
        return sum(1 for box in self.boxes.values() if box["from_snapshot"] is None)

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        snapshot_id = spec.provider_options.get("snapshot_id")
        if (spec.image is None) == (snapshot_id is None):
            raise ValueError("exactly one of image or snapshot_id must be specified")
        commands = list(self._snapshot_commands(snapshot_id)) if snapshot_id is not None else []
        sandbox_id = self._next_id("sb")
        self.calls.append(("create", sandbox_id))
        self.boxes[sandbox_id] = {"state": "running", "commands": commands, "from_snapshot": snapshot_id}
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    def _snapshot_commands(self, snapshot_id: str) -> list[str]:
        raise RuntimeError(f"provider {self.name!r} has no snapshot {snapshot_id!r}")

    async def exec(self, handle: SandboxHandle, command: str, **kwargs: Any) -> SandboxExecResult:
        self._box(handle.sandbox_id)["commands"].append(command)
        if self._exec_results is not None:
            result = self._exec_results(command)
            if result is not None:
                return result
        if command.strip() == "pwd":
            return SandboxExecResult(stdout="/testbed\n", stderr=None, return_code=0)
        return SandboxExecResult(stdout="", stderr=None, return_code=0)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        self._box(handle.sandbox_id)["commands"].append(f"upload {target_path}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        self._box(handle.sandbox_id)
        target_path.write_bytes(b"")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        box = self.boxes.get(handle.sandbox_id)
        return SandboxStatus.RUNNING if box is not None and box["state"] == "running" else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        self.calls.append(("close", handle.sandbox_id))
        if handle.sandbox_id in self.boxes:
            self.boxes[handle.sandbox_id]["state"] = "stopped"

    async def aclose(self) -> None:
        pass

    async def serialize_handle(self, handle: SandboxHandle, *, scope: Optional[str] = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        self._box(sandbox_id)
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)


class FakeSnapshotProvider(FakeSandboxProvider):
    """A :class:`FakeSandboxProvider` with durable snapshots, listed by name; ``fail_snapshots`` fails the next ones."""

    name = "fake-snapshots"

    def __init__(self, exec_results: Optional[Callable[[str], Optional[SandboxExecResult]]] = None) -> None:
        super().__init__(exec_results)
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.fail_snapshots = 0

    def _snapshot_commands(self, snapshot_id: str) -> list[str]:
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            return super()._snapshot_commands(snapshot_id)
        return snapshot["commands"]

    async def snapshot(self, handle: SandboxHandle, *, name: Optional[str] = None) -> str:
        self.calls.append(("snapshot", handle.sandbox_id))
        if self.fail_snapshots:
            self.fail_snapshots -= 1
            raise RuntimeError("snapshot push failed")
        box = self._box(handle.sandbox_id)
        snapshot_id = self._next_id("snap")
        self.snapshots[snapshot_id] = {
            "name": name or snapshot_id,
            "sandbox_id": handle.sandbox_id,
            "commands": list(box["commands"]),
        }
        return snapshot_id

    async def wait_snapshot_ready(self, snapshot_id: str) -> None:
        if snapshot_id not in self.snapshots:
            raise RuntimeError(f"snapshot {snapshot_id} is gone")

    async def delete_snapshot(self, snapshot_id: str) -> None:
        self.calls.append(("delete_snapshot", snapshot_id))
        self.snapshots.pop(snapshot_id, None)

    async def list_snapshots(self, prefix: str) -> list[dict[str, Any]]:
        return [
            {"id": snapshot_id, "name": snapshot["name"]}
            for snapshot_id, snapshot in self.snapshots.items()
            if snapshot["name"].startswith(prefix)
        ]


def checkpointing_config(state_dir: Path) -> dict[str, Any]:
    """The global ``checkpoint`` block a conformance server runs with."""
    return {"checkpoint": {"enabled": True, "control_auth_token": TOKEN, "state_dir": str(state_dir)}}


def _control(checkpoint_id: str, **extra: Any) -> dict[str, Any]:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 30, **extra}


async def check_sandbox_owner(
    make_server: Callable[[FakeSandboxProvider, dict[str, Any]], Any],
    seed: ResourcesSeedSessionRequest,
    state_dir: Path,
    *,
    capture: Optional[Callable[[Any, str], Awaitable[Any]]] = None,
) -> None:
    """Check a sandbox owner's checkpoint contract; raise ``AssertionError`` on a violation.

    ``make_server(provider, global_config)`` builds the server on ``provider`` (``shared_provider()`` must return
    it) with ``global_config`` as its global configuration. ``seed`` is a typed seed the server accepts for a fresh
    ``resources_session_id``. ``capture(server, session_id)``, if given, captures the server's grading artifact the
    way its ``/verify`` does, with ``capture_once(..., release_sandbox_after=True)``.

    Checked: the typed seed lends the ``main`` sandbox and is idempotent, and a conflicting seed is refused; a
    commit snapshots the sandbox and indexes the snapshot; after a crash, a restore forks the snapshot, the
    replacement attempt's seed gets the fork without running the setup again and with the session's data, and a
    seed of any other attempt is refused; a restore whose fork fails leaves no fork running; ``capture_once`` runs
    its capture once, and a commit after it exports the session's data without a sandbox; a release deletes the
    snapshots of every checkpoint it does not keep, including ones a crashed process indexed and ones it never
    indexed; a close stops the sandbox and fences a late seed; shutdown stops what is left; and on a provider that
    cannot snapshot the server runs ``restart_only`` and says so on its seed reply.
    """
    import httpx

    seed_path = f"/ng-rollout/{seed.episode_id.capture_key}/seed_session"
    seed_body = seed.model_dump(mode="json")
    session_id = seed.resources_session_id
    provider = FakeSnapshotProvider()
    config = checkpointing_config(state_dir)

    def client_of(server: Any) -> httpx.AsyncClient:
        app = server.setup_webserver()
        assert server.shared_provider() is provider, "make_server must build the server on the given provider"
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://owner")

    async def control(http: httpx.AsyncClient, operation: str, checkpoint_id: str, **extra: Any) -> dict[str, Any]:
        reply = await http.post(
            f"/ng-control/v1/checkpoint/{operation}", json=_control(checkpoint_id, **extra), headers=_AUTH
        )
        assert reply.status_code == 200, f"{operation} failed: {reply.status_code} {reply.text}"
        return reply.json()

    # -- seed, commit -------------------------------------------------------------------------------------------
    first = make_server(provider, config)
    async with client_of(first) as http:
        seeded = await http.post(seed_path, json=seed_body)
        assert seeded.status_code == 200, f"typed seed failed: {seeded.text}"
        access = seeded.json().get("sandbox_access")
        assert access is not None, "a sandbox owner's seed reply lends the session's main sandbox"
        assert seeded.headers.get("x-ng-checkpoint-restart") is None, "a provider that snapshots exports sessions"
        if first.restore_policy == "before_agent_close":
            assert seeded.headers.get("x-ng-checkpoint-restore-policy") == "before_agent_close"
        retried = await http.post(seed_path, json=seed_body)
        assert retried.status_code == 200 and retried.json()["sandbox_access"] == access, "an identical retry"
        other = seed.model_copy(update={"episode_id": seed.episode_id.model_copy(update={"attempt": 7})})
        conflicting = await http.post(
            f"/ng-rollout/{other.episode_id.capture_key}/seed_session", json=other.model_dump(mode="json")
        )
        assert conflicting.status_code == 409, "a seed naming the session with another episode is refused"
        created = provider.created_from_image()

        calls: list[str] = []

        async def counted() -> str:
            calls.append("capture")
            return "artifact"

        assert await first.capture_once(session_id, "conformance", counted) == "artifact"
        assert await first.capture_once(session_id, "conformance", counted) == "artifact"
        assert calls == ["capture"], "capture_once runs its capture once per session"

        await control(http, "prepare", "c1")
        committed = await control(http, "commit", "c1", checkpoint_dir=str(state_dir / "ckpt-c1"))
        await control(http, "resume", "c1")
        assert committed["episode_ids"] == [seed.episode_id.capture_key]
        indexed = first._checkpointer().indexed_snapshots
        assert list(indexed) == ["c1"] and len(indexed["c1"]) == 1, f"the commit indexes its snapshot: {indexed}"
        assert len(provider.live()) == 1, "a commit leaves the sandbox running"
        crashed_sandbox = provider.live()[0]

    # -- a crash, then a restore into a fork ----------------------------------------------------------------------
    restored = make_server(provider, config)
    async with client_of(restored) as http:
        scope = [seed.episode_id.model_dump(mode="json")]
        await control(http, "restore", "r1", checkpoint_dir=str(state_dir / "ckpt-c1"), episode_ids=scope)
        await control(http, "resume", "r1")
        replacement = seed.model_copy(
            update={"episode_id": seed.episode_id.model_copy(update={"attempt": seed.episode_id.attempt + 1})}
        )
        reseeded = await http.post(
            f"/ng-rollout/{replacement.episode_id.capture_key}/seed_session", json=replacement.model_dump(mode="json")
        )
        assert reseeded.status_code == 200, f"the replacement attempt's seed failed: {reseeded.text}"
        fork = reseeded.json()["sandbox_access"]
        assert fork["connection"]["descriptor"] != access["connection"]["descriptor"], "the seed lends the fork"
        assert provider.created_from_image() == created, (
            "the replacement attempt's seed does not set the task up again"
        )
        assert restored.session_data(session_id).get("conformance") == "artifact", "session data survives a restore"
        later = seed.model_copy(
            update={"episode_id": seed.episode_id.model_copy(update={"attempt": seed.episode_id.attempt + 2})}
        )
        refused = await http.post(
            f"/ng-rollout/{later.episode_id.capture_key}/seed_session", json=later.model_dump(mode="json")
        )
        assert refused.status_code == 409, "only the attempt a restore continues gets the restored session"
        await restored._checkpointer().settle()
        assert crashed_sandbox not in provider.live(), "the restore stops the sandbox the fork supersedes"

        # A capture that releases the sandbox: the next commit exports the session's data alone.
        if capture is not None:
            await capture(restored, session_id)
            await control(http, "prepare", "c2")
            await control(http, "commit", "c2", checkpoint_dir=str(state_dir / "ckpt-c2"))
            await control(http, "resume", "c2")
            assert restored.session_sandbox(session_id) is None, "capture_once released the sandbox"
            assert "c2" not in restored._checkpointer().indexed_snapshots, "nothing is snapshotted after the release"

        # Release: a crashed process's indexed snapshot and an unindexed one with this server's prefix go too.
        stray = await provider.snapshot(
            await provider.create(SandboxSpec(image="stray")),
            # Named as this server names a snapshot of a checkpoint it never indexed, as a crash during a commit leaves.
            name=restored._checkpointer().snapshot_name("dropped", "stray", 1),
        )
        await control(http, "release", "c1", kept=[])
        await restored._checkpointer().settle()
        assert provider.snapshots == {}, f"release deletes every snapshot of dropped checkpoints: {provider.snapshots}"
        assert stray not in provider.snapshots
        assert restored._checkpointer().indexed_snapshots == {}, "release empties the index"

        closed = await http.post(
            f"/ng-rollout/{replacement.episode_id.capture_key}/close_session",
            json={"resources_session_id": session_id, "episode_id": replacement.episode_id.model_dump(mode="json")},
        )
        assert closed.status_code == 200, f"close failed: {closed.text}"
        assert restored.session_sandbox(session_id) is None
        late = await http.post(
            f"/ng-rollout/{replacement.episode_id.capture_key}/seed_session", json=replacement.model_dump(mode="json")
        )
        assert late.status_code == 409, "a seed after close is refused"

    # -- a restore whose fork fails leaves nothing running --------------------------------------------------------
    rollback_provider = FakeSnapshotProvider()
    rollback = make_server(rollback_provider, checkpointing_config(state_dir / "rollback"))
    good = await rollback_provider.snapshot(await rollback_provider.create(SandboxSpec(image="x")), name="good")

    def owned_state(snapshot_id: str) -> dict[str, Any]:
        sandbox = {
            "provider_name": rollback_provider.name,
            "descriptor": {"sandbox_id": "gone"},
            "snapshot_id": snapshot_id,
            "spec": {},
            "snapshot_at": 0.0,
        }
        return {
            "sandboxes": {"main": sandbox},
            "workdir": "/testbed",
            "data": {},
            "task_id": seed.task_id.model_dump(mode="json"),
            "episode_id": seed.episode_id.model_dump(mode="json"),
        }

    states = {"ok": owned_state(good), "missing": owned_state("snap-missing")}
    live_before = set(rollback_provider.live())
    try:
        await rollback.restore_session_states(states)
    except Exception:
        pass
    else:
        raise AssertionError("a restore whose fork fails must raise")
    assert set(rollback_provider.live()) == live_before, "a failed restore stops the forks it made"
    assert rollback.session_sandbox("ok") is None, "a failed restore installs nothing"

    # -- shutdown, and a provider that cannot snapshot ------------------------------------------------------------
    plain = FakeSandboxProvider()
    provider = plain  # client_of checks the server is built on this provider
    restart_only = make_server(plain, checkpointing_config(state_dir / "plain"))
    app_client = client_of(restart_only)
    async with app_client as http:
        status = await http.get("/ng-control/v1/checkpoint/status", headers=_AUTH)
        assert status.json()["mode"] == "restart_only", "a provider that cannot snapshot makes the server restart_only"
        seeded = await http.post(seed_path, json=seed_body)
        assert seeded.status_code == 200
        assert seeded.headers.get("x-ng-checkpoint-restart") == "1", "its seed reply makes the episode a restart"
    await restart_only.shutdown()
    assert plain.live() == [], "shutdown stops every live sandbox"
    await asyncio.sleep(0)


# -- harness adapters ------------------------------------------------------------------------------------------


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
