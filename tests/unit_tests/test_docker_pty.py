# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the docker provider's PTY sessions (no docker daemon needed)."""

import base64
import json
from typing import Any, Callable

import pytest

from nemo_gym.sandbox.providers.base import (
    SandboxHandle,
    SandboxPtyError,
    SandboxPtySpec,
    SupportsSandboxPty,
    SupportsSandboxPtyAttach,
)
from nemo_gym.sandbox.providers.docker import provider as docker_provider
from nemo_gym.sandbox.providers.docker import pty as docker_pty


pytestmark = pytest.mark.sandbox


FAKE_BINARY = "/usr/bin/docker"


class RunRecorder:
    """Stand-in for DockerProvider._run that records argv and returns canned output."""

    def __init__(self, responder: Callable[[list[str]], tuple[int, str, str]]) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responder = responder

    async def __call__(
        self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None
    ) -> tuple[int, str, str]:
        self.calls.append({"argv": list(argv), "timeout_s": timeout_s, "stdin": stdin})
        return self._responder(list(argv))


def _contains_seq(haystack: list[str], needle: list[str]) -> bool:
    return any(haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1))


def _make_handle(*, name: str = "nemo-gym-x", shell: str = "sh", env: dict[str, str] | None = None) -> SandboxHandle:
    inst = docker_provider._DockerContainer(name=name, image="img", shell=shell, env=env or {})
    return SandboxHandle(sandbox_id=name, provider_name="docker", raw=inst)


def _make_provider(
    monkeypatch: pytest.MonkeyPatch, responder: Callable[[list[str]], tuple[int, str, str]], **kwargs: Any
) -> tuple[Any, RunRecorder]:
    monkeypatch.setattr(docker_provider, "_require_docker", lambda: FAKE_BINARY)
    kwargs.setdefault("exec", {"exec_shell": "sh"})
    provider = docker_provider.DockerProvider(**kwargs)
    rec = RunRecorder(responder)
    monkeypatch.setattr(provider, "_run", rec)
    return provider, rec


def _poll_payload(
    *,
    out: bytes = b"",
    out_size: int = 0,
    err: bytes = b"",
    err_size: int = 0,
    exit_code: int | None = None,
    alive: bool = True,
) -> str:
    return json.dumps(
        {
            "out": base64.b64encode(out).decode(),
            "out_size": out_size,
            "err": base64.b64encode(err).decode(),
            "err_size": err_size,
            "exit": exit_code,
            "alive": alive,
        }
    )


class SessionResponder:
    """Scripted per-op responder for helper calls (`ngpty.py <op> ...`)."""

    def __init__(self) -> None:
        self.poll_results: list[tuple[int, str, str]] = []
        self.write_result: tuple[int, str, str] = (0, "", "")
        self.default: tuple[int, str, str] = (0, "", "")

    def __call__(self, argv: list[str]) -> tuple[int, str, str]:
        op = self._op(argv)
        if op == "poll":
            if not self.poll_results:
                raise AssertionError("unexpected poll")
            return self.poll_results.pop(0)
        if op in ("write", "ctl"):
            return self.write_result
        return self.default

    @staticmethod
    def _op(argv: list[str]) -> str | None:
        for i, tok in enumerate(argv):
            if tok.endswith(docker_pty.HELPER_FILENAME) and i + 1 < len(argv):
                return argv[i + 1]
        return None


def _make_session(
    monkeypatch: pytest.MonkeyPatch, responder: Callable[[list[str]], tuple[int, str, str]], **kwargs: Any
) -> tuple[docker_pty.DockerPtySession, RunRecorder]:
    provider, rec = _make_provider(monkeypatch, responder)
    session = docker_pty.DockerPtySession(
        provider=provider,
        container_name="nemo-gym-x",
        session_id="sid1",
        token="tok1",
        mode=kwargs.pop("mode", "pty"),
        owned=kwargs.pop("owned", True),
        **kwargs,
    )
    return session, rec


# --------------------------------------------------------------------------- #
# Protocol conformance
# --------------------------------------------------------------------------- #
def test_provider_satisfies_pty_protocols(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    assert isinstance(provider, SupportsSandboxPty)
    assert isinstance(provider, SupportsSandboxPtyAttach)


def test_helper_source_is_valid_python() -> None:
    import ast

    ast.parse(docker_pty.HELPER_SOURCE)


# --------------------------------------------------------------------------- #
# create_pty
# --------------------------------------------------------------------------- #
async def test_create_pty_stages_and_starts_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    handle = _make_handle(shell="bash", env={"BASE": "1"})
    spec = SandboxPtySpec(command="htop", cwd="/work", env={"EXTRA": "2"}, rows=40, cols=120, user="root")

    session = await provider.create_pty(handle, spec)

    assert session.session_id
    assert session.mode == "pty"
    assert session.closed is False

    stage = rec.calls[0]
    assert stage["argv"][:4] == [FAKE_BINARY, "exec", "-i", "nemo-gym-x"]
    assert stage["argv"][4:6] == ["sh", "-c"]
    assert f"exit {docker_pty.EXIT_NO_PYTHON3}" in stage["argv"][6]
    assert docker_pty._session_dir(session.session_id) in stage["argv"][6]
    assert stage["stdin"] == docker_pty.HELPER_SOURCE.encode()

    broker = rec.calls[1]["argv"]
    assert broker[:3] == [FAKE_BINARY, "exec", "-d"]
    assert _contains_seq(broker, ["-w", "/work"])
    assert _contains_seq(broker, ["--env", "BASE=1"])
    assert _contains_seq(broker, ["--env", "EXTRA=2"])
    assert _contains_seq(broker, ["--user", "0"])
    helper = docker_pty._helper_path(session.session_id)
    assert _contains_seq(broker, ["python3", helper, "broker"])
    assert broker[-5:] == ["bash", "40", "120", "pty", "htop"]

    ready = rec.calls[2]["argv"]
    assert ready[-1].endswith("/ready")


async def test_create_pty_shell_session_and_pipe_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, "", ""))
    session = await provider.create_pty(_make_handle(), SandboxPtySpec(pty=False))
    assert session.mode == "pipe"
    broker = rec.calls[1]["argv"]
    # No command -> the shell itself is the session process.
    assert broker[-4:] == ["sh", "24", "80", "pipe"]


async def test_create_pty_requires_python3(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(monkeypatch, lambda argv: (docker_pty.EXIT_NO_PYTHON3, "", ""))
    with pytest.raises(SandboxPtyError, match="python3"):
        await provider.create_pty(_make_handle(), SandboxPtySpec())


async def test_create_pty_broker_start_failure_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if "-d" in argv:
            return (1, "", "boom")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    with pytest.raises(SandboxPtyError, match="broker"):
        await provider.create_pty(_make_handle(), SandboxPtySpec())
    assert any("rm -rf" in tok for c in rec.calls for tok in c["argv"])


async def test_create_pty_ready_timeout_reports_broker_log(monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        if any("test -f" in tok for tok in argv):
            return (1, "", "")
        if any("broker.log" in tok for tok in argv):
            return (0, "Traceback: kaput", "")
        return (0, "", "")

    provider, rec = _make_provider(monkeypatch, responder)
    monkeypatch.setattr(docker_pty, "READY_DEADLINE_S", 0.0)
    with pytest.raises(SandboxPtyError, match="kaput"):
        await provider.create_pty(_make_handle(), SandboxPtySpec())
    assert any("rm -rf" in tok for c in rec.calls for tok in c["argv"])


# --------------------------------------------------------------------------- #
# attach_pty
# --------------------------------------------------------------------------- #
async def test_attach_pty_defaults_to_live_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = json.dumps({"mode": "pty", "out_size": 500, "err_size": 0, "exit": None, "alive": True})
    provider, rec = _make_provider(monkeypatch, lambda argv: (0, meta, ""))
    session = await provider.attach_pty(_make_handle(), "sid9")

    argv = rec.calls[0]["argv"]
    assert _contains_seq(argv, ["python3", docker_pty._helper_path("sid9"), "attach"])
    assert argv[-1] == "1"  # takeover default
    assert session.session_id == "sid9"
    assert session.mode == "pty"
    assert session._out_offset == 500  # live tail: replay nothing
    assert session._owned is False


async def test_attach_pty_since_replays_from_offset(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = json.dumps({"mode": "pty", "out_size": 500, "err_size": 0, "exit": None, "alive": True})
    provider, _rec = _make_provider(monkeypatch, lambda argv: (0, meta, ""))
    session = await provider.attach_pty(_make_handle(), "sid9", since=0)
    assert session._out_offset == 0


async def test_attach_pty_without_takeover_on_held_session(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, rec = _make_provider(monkeypatch, lambda argv: (docker_pty.EXIT_HELD, "", ""))
    with pytest.raises(SandboxPtyError, match="takeover=True"):
        await provider.attach_pty(_make_handle(), "sid9", takeover=False)
    assert rec.calls[0]["argv"][-1] == "0"


async def test_attach_pty_missing_session(monkeypatch: pytest.MonkeyPatch) -> None:
    provider, _rec = _make_provider(
        monkeypatch, lambda argv: (2, "", "python3: can't open file '/tmp/.nemo-gym-pty/sid9/ngpty.py'")
    )
    with pytest.raises(SandboxPtyError, match="does not exist"):
        await provider.attach_pty(_make_handle(), "sid9")


# --------------------------------------------------------------------------- #
# session read / offset bookkeeping
# --------------------------------------------------------------------------- #
async def test_read_advances_offset_and_scans_from_it(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [
        (0, _poll_payload(out=b"hello ", out_size=6), ""),
        (0, _poll_payload(out=b"world", out_size=11), ""),
    ]
    session, rec = _make_session(monkeypatch, responder)

    assert await session.read() == b"hello "
    assert await session.read() == b"world"
    assert session._out_offset == 11

    first_poll = rec.calls[0]["argv"]
    second_poll = rec.calls[1]["argv"]
    assert _contains_seq(first_poll, ["poll", docker_pty._session_dir("sid1"), "tok1", "0", "0"])
    assert _contains_seq(second_poll, ["poll", docker_pty._session_dir("sid1"), "tok1", "6", "0"])


async def test_read_returns_eof_after_exit_and_drain(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [
        (0, _poll_payload(out=b"bye", out_size=3, exit_code=0), ""),
        (0, _poll_payload(out_size=3, exit_code=0), ""),
    ]
    session, _rec = _make_session(monkeypatch, responder)
    assert await session.read() == b"bye"
    assert await session.read() == b""
    assert session.closed is True  # exited and drained
    assert await session.wait_exit() == 0


async def test_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [(0, _poll_payload(), "")] * 10
    session, _rec = _make_session(monkeypatch, responder)
    with pytest.raises(TimeoutError):
        await session.read(timeout_s=0.15)


async def test_read_raises_when_broker_died_without_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [(0, _poll_payload(alive=False), "")]
    session, _rec = _make_session(monkeypatch, responder)
    with pytest.raises(SandboxPtyError, match="died without exiting"):
        await session.read()
    assert session.closed is True


async def test_read_stderr_pipe_mode_and_pty_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [
        (0, _poll_payload(err=b"oops", err_size=4), ""),
        (0, _poll_payload(out_size=0, err_size=4, exit_code=3), ""),
    ]
    session, _rec = _make_session(monkeypatch, responder, mode="pipe")
    assert await session.read_stderr() == b"oops"
    assert await session.read_stderr() == b""  # exited, drained


async def test_aiter_yields_until_eof(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [
        (0, _poll_payload(out=b"a", out_size=1), ""),
        (0, _poll_payload(out=b"b", out_size=2, exit_code=0), ""),
        (0, _poll_payload(out_size=2, exit_code=0), ""),
    ]
    session, _rec = _make_session(monkeypatch, responder)
    chunks = [chunk async for chunk in session]
    assert chunks == [b"a", b"b"]


async def test_poll_malformed_json_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [(0, "not json", "")]
    session, _rec = _make_session(monkeypatch, responder)
    with pytest.raises(SandboxPtyError, match="malformed"):
        await session.read()


# --------------------------------------------------------------------------- #
# takeover / eviction
# --------------------------------------------------------------------------- #
async def test_evicted_session_raises_on_read_and_write(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [(docker_pty.EXIT_EVICTED, "", "")]
    session, _rec = _make_session(monkeypatch, responder)
    with pytest.raises(SandboxPtyError, match="taken over"):
        await session.read()
    assert session.closed is True
    with pytest.raises(SandboxPtyError):
        await session.write(b"ls\n")


async def test_dead_session_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [(docker_pty.EXIT_DEAD, "", "")]
    session, _rec = _make_session(monkeypatch, responder)
    with pytest.raises(SandboxPtyError, match="no longer exists"):
        await session.read()


# --------------------------------------------------------------------------- #
# write / resize / signal
# --------------------------------------------------------------------------- #
async def test_write_sends_stdin_to_fifo(monkeypatch: pytest.MonkeyPatch) -> None:
    session, rec = _make_session(monkeypatch, SessionResponder())
    await session.write(b"echo hi\n")
    call = rec.calls[0]
    assert call["stdin"] == b"echo hi\n"
    assert "-i" in call["argv"]
    assert _contains_seq(call["argv"], ["write", docker_pty._session_dir("sid1"), "tok1"])


async def test_resize_and_signal_go_through_control_fifo(monkeypatch: pytest.MonkeyPatch) -> None:
    session, rec = _make_session(monkeypatch, SessionResponder())
    await session.resize(50, 132)
    await session.send_signal("SIGTERM")

    resize_call, signal_call = rec.calls
    assert _contains_seq(resize_call["argv"], ["ctl", docker_pty._session_dir("sid1"), "tok1"])
    assert json.loads(resize_call["stdin"]) == {"type": "resize", "rows": 50, "cols": 132}
    assert json.loads(signal_call["stdin"]) == {"type": "signal", "signal": "SIGTERM"}


async def test_send_signal_rejects_unknown_names(monkeypatch: pytest.MonkeyPatch) -> None:
    session, rec = _make_session(monkeypatch, SessionResponder())
    with pytest.raises(ValueError, match="unknown signal"):
        await session.send_signal("FROBNICATE")
    assert rec.calls == []


async def test_write_on_closed_session_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _rec = _make_session(monkeypatch, SessionResponder())
    await session.close()
    with pytest.raises(SandboxPtyError, match="closed"):
        await session.write(b"x")


# --------------------------------------------------------------------------- #
# wait_exit
# --------------------------------------------------------------------------- #
async def test_wait_exit_polls_until_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [
        (0, _poll_payload(), ""),
        (0, _poll_payload(exit_code=7), ""),
    ]
    session, rec = _make_session(monkeypatch, responder)
    assert await session.wait_exit() == 7
    # wait_exit polls with max_bytes=0: it must not consume output.
    assert all(c["argv"][-1] == "0" for c in rec.calls)
    assert session._out_offset == 0


async def test_wait_exit_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    responder = SessionResponder()
    responder.poll_results = [(0, _poll_payload(), "")] * 10
    session, _rec = _make_session(monkeypatch, responder)
    with pytest.raises(TimeoutError):
        await session.wait_exit(timeout_s=0.15)


# --------------------------------------------------------------------------- #
# run_detached
# --------------------------------------------------------------------------- #
async def test_run_detached_parses_marker_and_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    session, rec = _make_session(monkeypatch, SessionResponder())

    written: list[bytes] = []

    async def fake_write(data: bytes) -> None:
        written.append(data)

    reads: list[bytes] = []

    async def fake_read(*, timeout_s: float | None = None) -> bytes:
        if reads:
            return reads.pop(0)
        raise TimeoutError

    async def fake_read_stderr(*, timeout_s: float | None = None) -> bytes:
        raise TimeoutError

    monkeypatch.setattr(session, "write", fake_write)
    monkeypatch.setattr(session, "read", fake_read)
    monkeypatch.setattr(session, "read_stderr", fake_read_stderr)

    async def run() -> tuple[bytes, int | None]:
        return await session.run_detached("echo done", poll_interval_s=0.01)

    # Feed output that includes the marker assembled from the written command.
    task = __import__("asyncio").get_running_loop().create_task(run())
    while not written:
        await __import__("asyncio").sleep(0.001)
    marker = written[0].decode().rsplit("'", 4)  # printf '%s%s:%s\n' 'NGPTY' 'rest' "$?"
    token = marker[1] + marker[3]
    reads.extend([b"done\n", f"{token}:0\n".encode()])
    output, exit_code = await task
    assert output == b"done\n"
    assert exit_code == 0
    assert rec.calls == []  # everything went through the patched methods


# --------------------------------------------------------------------------- #
# close semantics
# --------------------------------------------------------------------------- #
async def test_close_owned_ends_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session, rec = _make_session(monkeypatch, SessionResponder(), owned=True)
    await session.close()
    assert session.closed is True
    assert _contains_seq(rec.calls[0]["argv"], ["end", docker_pty._session_dir("sid1")])
    await session.close()  # idempotent
    assert len(rec.calls) == 1


async def test_close_attached_releases_session(monkeypatch: pytest.MonkeyPatch) -> None:
    session, rec = _make_session(monkeypatch, SessionResponder(), owned=False)
    await session.close()
    assert _contains_seq(rec.calls[0]["argv"], ["release", docker_pty._session_dir("sid1"), "tok1"])


async def test_close_suppresses_backend_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def responder(argv: list[str]) -> tuple[int, str, str]:
        return (1, "", "Error: No such container: nemo-gym-x")

    session, _rec = _make_session(monkeypatch, responder)
    await session.close()  # does not raise
    assert session.closed is True


async def test_context_manager_closes(monkeypatch: pytest.MonkeyPatch) -> None:
    session, rec = _make_session(monkeypatch, SessionResponder())
    async with session as s:
        assert s is session
    assert session.closed is True
    assert _contains_seq(rec.calls[0]["argv"], ["end", docker_pty._session_dir("sid1")])


# --------------------------------------------------------------------------- #
# staging script shape
# --------------------------------------------------------------------------- #
def test_staging_script_contents() -> None:
    script = docker_pty.staging_script("sid1", "tok1", "pty")
    d = docker_pty._session_dir("sid1")
    assert f"mkdir -p {d}" in script
    assert f"cat > {d}/{docker_pty.HELPER_FILENAME}" in script
    assert "tok1" in script
    assert '"mode": "pty"' in script
    assert f"chmod -R 777 {d}" in script
