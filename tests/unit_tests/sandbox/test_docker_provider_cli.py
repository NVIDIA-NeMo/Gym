# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Docker provider, and the Compose adapter on it, against a fake ``docker`` executable first on ``PATH``.

Nothing in the provider is replaced: every call crosses the real subprocess boundary of ``DockerProvider._run``, its
``exec.concurrency`` semaphore included. The fake keeps each container as a folder and runs an exec's command on the
host with the command's absolute paths mapped into that folder; removing a container kills what its execs left
running."""

import asyncio
import contextlib
import json
import os
import shlex
import signal
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from nemo_gym.sandbox.adapters.docker_compose import AsyncSandboxCompose
from nemo_gym.sandbox.providers.base import SandboxSpec, SupportsSandboxDetachedExec
from nemo_gym.sandbox.providers.docker.provider import DockerProvider


pytestmark = [
    pytest.mark.sandbox,
    pytest.mark.skipif(sys.platform == "win32", reason="the fake docker CLI is a POSIX shell script"),
]

FAKE_DOCKER = r"""
import contextlib
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONTAINERS = ROOT / "containers"
PROBE = "printf docker-sandbox-ready"
# Absolute paths in a command: at its start or after whitespace, a quote, `=`, `(` or a redirection.
ABSOLUTE = re.compile(r"(?<![^\s'\"=(<>])/[^\s'\";|&()<>]*")
HOST_PATHS = ("/dev/", "/bin/", "/usr/", "/proc/")


@contextlib.contextmanager
def locked():
    with open(ROOT / "lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def state(name):
    path = CONTAINERS / name / "state.json"
    return json.loads(path.read_text()) if path.is_file() else None


def in_container(name, text):
    root = str(CONTAINERS / name / "fs")
    return ABSOLUTE.sub(lambda match: match[0] if match[0].startswith(HOST_PATHS) else root + match[0], text)


def no_such_container(name):
    print(f"Error response from daemon: No such container: {name}", file=sys.stderr)
    return 1


def group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def record_session(name, pgid, *, add):
    sessions = CONTAINERS / name / "sessions"
    if not sessions.parent.is_dir():
        return
    recorded = sessions.read_text().split() if sessions.exists() else []
    recorded = [*recorded, str(pgid)] if add else [entry for entry in recorded if entry != str(pgid)]
    sessions.write_text("".join(entry + "\n" for entry in recorded))


def run(args):
    entrypoint = args.index("--entrypoint")
    options, image = args[:entrypoint], args[entrypoint + 2]
    values = lambda flag: [options[index + 1] for index, token in enumerate(options) if token == flag]
    name = values("--name")[0]
    with locked():
        number = len(list(CONTAINERS.iterdir())) + len(list((ROOT / "removed").iterdir())) + 2
        (CONTAINERS / name / "fs" / "etc").mkdir(parents=True)
        (CONTAINERS / name / "fs" / "etc" / "hosts").write_text("127.0.0.1\tlocalhost\n")
        (CONTAINERS / name / "state.json").write_text(
            json.dumps(
                {
                    "image": image,
                    "env": dict(value.split("=", 1) for value in values("--env")),
                    "workdir": (values("-w") or ["/"])[-1],
                    "address": f"172.17.0.{number}",
                }
            )
        )
    print(name)
    return 0


def exec_(args):
    flags, env, index = {}, {}, 0
    while args[index].startswith("-"):
        if args[index] in ("-i", "-d"):
            flags[args[index]] = True
            index += 1
            continue
        if args[index] == "--env":
            key, _, value = args[index + 1].partition("=")
            env[key] = value
        else:
            flags[args[index]] = args[index + 1]
        index += 2
    name, shell, _, command = args[index : index + 4]
    container = state(name)
    if container is None:
        return no_such_container(name)
    if command == PROBE and "stalled-probe" in container["image"]:
        # The readiness probe never answers. It ends when the container is removed, as an exec in a removed
        # container does, or when the test is over.
        (CONTAINERS / name / "probe-waiting").touch()
        while state(name) is not None and not (ROOT / "shutdown").exists():
            time.sleep(0.02)
        return no_such_container(name)
    if flags.get("--user") == "no-such-user":
        print("unable to find user no-such-user: no matching entries in passwd file", file=sys.stderr)
        return 126
    with locked():
        if state(name) is None:
            return no_such_container(name)
        cwd = Path(in_container(name, flags.get("-w", container["workdir"])))
        cwd.mkdir(parents=True, exist_ok=True)
        output = subprocess.DEVNULL if "-d" in flags else subprocess.PIPE
        process = subprocess.Popen(
            [shell, "-c", in_container(name, command)],
            cwd=cwd,
            env={"PATH": os.environ["PATH"], **container["env"], **env},
            stdin=None if "-i" in flags else subprocess.DEVNULL,
            stdout=output,
            stderr=output,
            # What the command starts belongs to the container, not to this client: removing the container ends
            # it, and killing the client (a timed-out exec) does not.
            start_new_session=True,
        )
        record_session(name, process.pid, add=True)
    if "-d" in flags:
        # Detached: Docker answers once the command started.
        return 0
    out, err = process.communicate()
    if not group_alive(process.pid):
        with locked():
            record_session(name, process.pid, add=False)
    sys.stdout.buffer.write(out)
    sys.stderr.buffer.write(err)
    return process.returncode


def cp(args):
    source, target = args
    name, path = target.split(":", 1)
    if state(name) is None:
        return no_such_container(name)
    Path(in_container(name, path)).write_text(in_container(name, Path(source).read_text()))
    return 0


def inspect(args):
    name = args[-1]
    form = args[args.index("--format") + 1] if "--format" in args else args[args.index("-f") + 1]
    container = state(name)
    if container is None:
        print(f"Error: No such object: {name}", file=sys.stderr)
        return 1
    if "NetworkSettings.Networks" in form:
        print(json.dumps({"bridge": {"IPAddress": container["address"], "GlobalIPv6Address": ""}}))
    else:
        print("running")
    return 0


def rm(args):
    name = args[-1]
    with locked():
        container = state(name)
        if container is None:
            return no_such_container(name)
        if "slow-rm" in container["image"]:
            (CONTAINERS / name / "rm-started").touch()
            time.sleep(0.5)
        sessions = CONTAINERS / name / "sessions"
        for pgid in sessions.read_text().split() if sessions.exists() else []:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(int(pgid), signal.SIGKILL)
        (CONTAINERS / name).rename(ROOT / "removed" / name)
    print(name)
    return 0


def running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main(args):
    me = ROOT / "active" / str(os.getpid())
    with locked():
        with open(ROOT / "calls.jsonl", "a") as calls:
            calls.write(json.dumps(args) + "\n")
        for other in (ROOT / "active").iterdir():
            if not running(int(other.name)):
                other.unlink()
        me.touch()
        count = len(list((ROOT / "active").iterdir()))
        peak = ROOT / "peak"
        if count > int(peak.read_text() if peak.exists() else "0"):
            peak.write_text(str(count))
    try:
        commands = {"run": run, "exec": exec_, "cp": cp, "inspect": inspect, "rm": rm}
        if args[0] not in commands:
            print(f"fake docker: unsupported command {args}", file=sys.stderr)
            return 2
        return commands[args[0]](args[1:])
    finally:
        me.unlink()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
"""


class FakeDocker:
    """What the fake ``docker`` CLI of :func:`fake_docker` recorded: its containers, calls and peak concurrency."""

    def __init__(self, root: Path):
        self.root = root

    def containers(self) -> set[str]:
        """The containers that exist now."""
        return {path.name for path in (self.root / "containers").iterdir()}

    def removed(self) -> set[str]:
        """The containers ``docker rm`` removed."""
        return {path.name for path in (self.root / "removed").iterdir()}

    def calls(self) -> list[list[str]]:
        """Every call's arguments, in order."""
        path = self.root / "calls.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def created(self) -> list[str]:
        """The names of the containers ``docker run`` created, in order."""
        return [call[call.index("--name") + 1] for call in self.calls() if call[0] == "run"]

    def peak(self) -> int:
        """The most calls that were running at once."""
        return int((self.root / "peak").read_text())

    def file(self, name: str, path: str) -> Path:
        """Where the fake keeps ``path`` of container ``name``, removed or not."""
        folder = self.root / "containers" / name
        if not folder.exists():
            folder = self.root / "removed" / name
        return folder / "fs" / path.lstrip("/")

    async def read(self, name: str, path: str) -> str:
        """``path`` of container ``name`` once something wrote it."""
        async with asyncio.timeout(10):
            while not self.file(name, path).exists():
                await asyncio.sleep(0.01)
        return self.file(name, path).read_text()

    async def wait_for(self, name: str, marker: str) -> None:
        """Wait until the fake writes ``marker`` in container ``name``'s folder."""
        async with asyncio.timeout(10):
            while not (self.root / "containers" / name / marker).exists():
                await asyncio.sleep(0.01)

    def shutdown(self) -> None:
        """End what a test left running: stalled calls, and the commands of containers it did not remove."""
        (self.root / "shutdown").touch()
        for container in (self.root / "containers").iterdir():
            sessions = container / "sessions"
            for pgid in sessions.read_text().split() if sessions.exists() else []:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(int(pgid), signal.SIGKILL)


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "docker"
    for folder in ("bin", "containers", "removed", "active"):
        (root / folder).mkdir(parents=True)
    (root / "fake_docker.py").write_text(FAKE_DOCKER)
    cli = root / "bin" / "docker"
    cli.write_text(
        f'#!/bin/sh\nexec {shlex.quote(sys.executable)} -I {shlex.quote(str(root / "fake_docker.py"))} "$@"\n'
    )
    cli.chmod(0o755)
    monkeypatch.setenv("PATH", f"{root / 'bin'}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}")
    fake = FakeDocker(root)
    yield fake
    fake.shutdown()


def running(pid: int) -> bool:
    """Whether process ``pid`` exists and has not exited (a zombie, killed but not yet reaped, has)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    with contextlib.suppress(OSError):
        return stat.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    return True


async def stopped(pid: int) -> bool:
    """Whether process ``pid`` ends within two seconds: a kill takes effect once the process is scheduled."""
    for _ in range(200):
        if not running(pid):
            return True
        await asyncio.sleep(0.01)
    return False


def collection(provider: DockerProvider, document: dict, *, timeout_s: float) -> AsyncSandboxCompose:
    group = AsyncSandboxCompose(provider, "compose.yaml", timeout_s=timeout_s, poll_interval_s=0.01)
    group._load = Mock(return_value=document)
    return group


# --------------------------------------------------------------------------- #
# A create cancelled after its container exists
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_startup_deadline_during_the_readiness_probe_removes_the_new_container(fake_docker):
    """``docker run`` succeeded, then the collection's deadline expired while the readiness probe waited. No one holds
    the new container's handle yet, so the create removes the container itself before the timeout propagates."""
    provider = DockerProvider(exec={"exec_shell": "sh"})
    group = collection(
        provider, {"services": {"main": {"image": "stalled-probe", "command": ["sleep", "60"]}}}, timeout_s=1
    )
    with pytest.raises(TimeoutError):
        await group.start()
    (created,) = fake_docker.created()
    assert ["rm", "-f", created] in fake_docker.calls()
    assert fake_docker.removed() == {created}
    assert fake_docker.containers() == set()


@pytest.mark.asyncio
async def test_a_cancelled_create_removes_its_container_even_when_cancelled_again(fake_docker):
    """Cancelling ``create()`` during the readiness probe removes the container, and a second cancellation that
    arrives while the removal runs waits for it rather than leaving the container behind."""
    provider = DockerProvider(exec={"exec_shell": "sh"})
    create = asyncio.create_task(provider.create(SandboxSpec(image="stalled-probe-slow-rm")))
    async with asyncio.timeout(10):
        while not fake_docker.created():
            await asyncio.sleep(0.01)
    (created,) = fake_docker.created()
    await fake_docker.wait_for(created, "probe-waiting")
    create.cancel()
    await fake_docker.wait_for(created, "rm-started")
    create.cancel()
    with pytest.raises(asyncio.CancelledError):
        await create
    assert fake_docker.removed() == {created}
    assert fake_docker.containers() == set()


# --------------------------------------------------------------------------- #
# Compose services run detached and hold no exec slot
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_a_detached_command_runs_on_after_the_call_with_the_exec_overrides(fake_docker):
    provider = DockerProvider(exec={"exec_shell": "sh"})
    assert isinstance(provider, SupportsSandboxDetachedExec)
    handle = await provider.create(SandboxSpec(image="image", env={"BASE": "1"}))
    name = handle.sandbox_id
    try:
        result = await provider.exec_detached(
            handle, 'echo $$ > pid; printf %s "$K" > value; exec sleep 60', cwd="/work", env={"K": "V"}, user=0
        )
        assert (result.return_code, result.error_type) == (0, None)
        assert fake_docker.calls()[-1] == [
            "exec",
            "-d",
            "-w",
            "/work",
            "--env",
            "BASE=1",
            "--env",
            "K=V",
            "--user",
            "0",
            name,
            "sh",
            "-c",
            'echo $$ > pid; printf %s "$K" > value; exec sleep 60',
        ]
        assert await fake_docker.read(name, "/work/value") == "V"
        pid = int(await fake_docker.read(name, "/work/pid"))
        assert running(pid)
        # A command Docker cannot start fails the result.
        refused = await provider.exec_detached(handle, "true", user="no-such-user")
        assert refused.return_code == 126
        assert "no-such-user" in refused.stderr
    finally:
        await provider.close(handle)
    assert fake_docker.removed() == {name}
    assert await stopped(pid)


@pytest.mark.asyncio
async def test_a_running_service_leaves_the_exec_capacity_to_other_commands(fake_docker):
    """With ``exec.concurrency: 1`` a service that runs for a minute does not hold the one slot: startup (whose
    started-marker probe needs it) finishes within the five-second deadline, and other commands run while the service
    does, one at a time and each within its own timeout. Stopping the collection removes the container and so ends the
    service. The service's user, working directory and environment apply to its command."""
    provider = DockerProvider(exec={"exec_shell": "sh", "concurrency": 1})
    command = 'echo $$ > /tmp/service.pid; printf %s "$GREETING" > greeting; exec sleep 60'
    document = {
        "services": {
            "main": {
                "image": "image",
                "user": "1000",
                "working_dir": "/srv",
                "environment": {"GREETING": "hello"},
                "command": ["sh", "-c", command],
            }
        }
    }
    group = collection(provider, document, timeout_s=5)
    async with group:
        (name,) = fake_docker.containers()
        runtime = f"/tmp/{group.project}/main"
        (launch,) = [call for call in fake_docker.calls() if call[:2] == ["exec", "-d"]]
        assert launch == [
            "exec",
            "-d",
            "-w",
            "/srv",
            "--env",
            "GREETING=hello",
            "--user",
            "1000",
            name,
            "sh",
            "-c",
            f"sh {runtime}/run.sh >{runtime}/log 2>&1 </dev/null",
        ]
        pid = int(await fake_docker.read(name, "/tmp/service.pid"))
        assert await fake_docker.read(name, "/srv/greeting") == "hello"
        main = group.services["main"]
        result = await main.exec("echo ok", timeout_s=5)
        assert (result.return_code, result.stdout) == (0, "ok\n")
        loop = asyncio.get_running_loop()
        began = loop.time()
        results = await asyncio.gather(*(main.exec("sleep 0.2", timeout_s=5) for _ in range(3)))
        assert [result.return_code for result in results] == [0, 0, 0]
        assert loop.time() - began >= 0.6
        timed_out = await main.exec("sleep 30", timeout_s=0.5)
        assert timed_out.error_type == "timeout"
        assert fake_docker.peak() == 1
        assert running(pid)
        assert not fake_docker.file(name, f"{runtime}/exit").exists()
    assert fake_docker.containers() == set()
    assert fake_docker.removed() == {name}
    assert await stopped(pid)


@pytest.mark.asyncio
async def test_a_later_startup_failure_removes_every_container(fake_docker):
    """A failure after a service started (here the next service's ``post_start`` hook) stops the collection: every
    container is removed, in reverse order of creation, and the running service ends with its container."""
    provider = DockerProvider(exec={"exec_shell": "sh", "concurrency": 1})
    document = {
        "services": {
            "db": {"image": "image", "command": ["sh", "-c", "echo $$ > /tmp/service.pid; exec sleep 60"]},
            "app": {
                "image": "image",
                "command": ["sleep", "60"],
                "depends_on": {"db": {"condition": "service_started"}},
                "post_start": [{"command": "exit 3"}],
            },
        }
    }
    group = collection(provider, document, timeout_s=5)
    with pytest.raises(RuntimeError, match=r"post_start hook 0 exited \(3\)"):
        await group.start()
    db, app = fake_docker.created()
    pid = int(await fake_docker.read(db, "/tmp/service.pid"))
    assert fake_docker.containers() == set()
    assert [call[-1] for call in fake_docker.calls() if call[0] == "rm"] == [app, db]
    assert await stopped(pid)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [0, 7])
async def test_a_detached_one_shot_service_reports_its_exit_to_its_dependents(fake_docker, code):
    """The run script records a detached service's exit status as it does an attached one's:
    ``service_completed_successfully`` starts the dependent after exit 0 and refuses it after any other."""
    provider = DockerProvider(exec={"exec_shell": "sh", "concurrency": 1})
    document = {
        "services": {
            "seed": {"image": "image", "command": ["sh", "-c", f"exit {code}"]},
            "app": {
                "image": "image",
                "command": ["sh", "-c", 'printf %s "$VALUE" > /tmp/value; exec sleep 60'],
                "environment": {"VALUE": "compose"},
                "depends_on": {"seed": {"condition": "service_completed_successfully"}},
            },
        }
    }
    group = collection(provider, document, timeout_s=5)
    if code:
        with pytest.raises(RuntimeError, match=r"'seed' exited \(7\)"):
            await group.start()
        seed, app = fake_docker.created()
        assert not fake_docker.file(app, "/tmp/value").exists()
    else:
        async with group:
            seed, app = fake_docker.created()
            assert await fake_docker.read(app, "/tmp/value") == "compose"
    assert fake_docker.containers() == set()


@pytest.mark.asyncio
async def test_a_service_docker_cannot_start_fails_the_collection(fake_docker):
    provider = DockerProvider(exec={"exec_shell": "sh", "concurrency": 1})
    document = {"services": {"main": {"image": "image", "user": "no-such-user", "command": ["sleep", "60"]}}}
    group = collection(provider, document, timeout_s=5)
    with pytest.raises(RuntimeError, match="'main' did not start: unable to find user no-such-user"):
        await group.start()
    assert fake_docker.containers() == set()
