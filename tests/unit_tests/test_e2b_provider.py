# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the e2b sandbox provider (SDK faked; no network)."""

import types
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers import e2b as e2b_pkg
from nemo_gym.sandbox.providers.base import SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.e2b import provider as e2b_provider
from nemo_gym.sandbox.providers.e2b.provider import _API_PARAM_KEYS, E2BCreateError, E2BProvider
from nemo_gym.sandbox.providers.registry import get_provider_class


# --------------------------------------------------------------------------
# Fake e2b SDK
# --------------------------------------------------------------------------


class FakeSandboxNotFound(Exception):
    pass


class FakeTimeout(Exception):
    pass


class FakeCommandExit(Exception):
    def __init__(self, exit_code: int, stdout: str = "", stderr: str = "") -> None:
        super().__init__(f"exit {exit_code}")
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class FakeCommandResult:
    def __init__(self, stdout: str = "", stderr: str = "", exit_code: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code


# The real SDK's sandbox-scoped methods take `request_timeout` only -- they are
# bound to an already-connected sandbox. Mirroring that here (rather than a
# permissive **kwargs) is what catches connection params leaking onto them:
# the SDK raises `TypeError: unexpected keyword argument 'api_key'`.
_SANDBOX_SCOPED_ALLOWED = {"request_timeout"}


def _reject_connection_params(method: str, kwargs: dict) -> None:
    leaked = sorted((set(kwargs) & set(_API_PARAM_KEYS)) - _SANDBOX_SCOPED_ALLOWED)
    if leaked:
        raise TypeError(f"{method}() got an unexpected keyword argument {leaked[0]!r}")


class FakeCommandHandle:
    """Background handle: `wait()` replays a scripted sequence of outcomes.

    Falls back to the sandbox's `exec_behaviour` so a test can script an
    outcome once and have it apply in either exec mode.
    """

    def __init__(self, pid: int, outcomes: list, fallback=None) -> None:
        self.pid = pid
        self._outcomes = outcomes
        self._fallback = fallback
        self.waits = 0

    async def wait(self):
        self.waits += 1
        if self._outcomes:
            outcome = self._outcomes.pop(0)
        else:
            outcome = self._fallback or FakeCommandResult(stdout="ok")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeCommands:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox

    async def run(self, **kwargs):
        _reject_connection_params("Commands.run", kwargs)
        self._sandbox.exec_calls.append(kwargs)
        behaviour = self._sandbox.exec_behaviour
        if kwargs.get("background"):
            return FakeCommandHandle(self._sandbox.pid, self._sandbox.wait_outcomes, behaviour)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour or FakeCommandResult(stdout="ok")

    async def connect(self, pid, timeout=None, **kwargs):
        self._sandbox.connect_calls.append({"pid": pid, "timeout": timeout})
        if self._sandbox.connect_error is not None:
            raise self._sandbox.connect_error
        return FakeCommandHandle(pid, self._sandbox.wait_outcomes, self._sandbox.exec_behaviour)


class FakeFiles:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox

    async def write(self, path, data, **kwargs):
        _reject_connection_params("Filesystem.write", kwargs)
        self._sandbox.files_written[path] = data
        return None

    async def read(self, path, **kwargs):
        _reject_connection_params("Filesystem.read", kwargs)
        if path not in self._sandbox.files_written:
            raise FakeSandboxNotFound(path)
        data = self._sandbox.files_written[path]
        return data if isinstance(data, bytes) else str(data).encode()


class FakeSandbox:
    instances: list["FakeSandbox"] = []

    def __init__(self, sandbox_id: str = "sbx-1", **create_kwargs) -> None:
        self.sandbox_id = sandbox_id
        self.create_kwargs = create_kwargs
        self.exec_calls: list[dict] = []
        self.files_written: dict[str, object] = {}
        self.killed = False
        self.running = True
        self.exec_behaviour = None
        self.pid = 4242
        self.wait_outcomes: list = []
        self.connect_calls: list[dict] = []
        self.connect_error = None
        self.commands = FakeCommands(self)
        self.files = FakeFiles(self)
        FakeSandbox.instances.append(self)

    @classmethod
    async def create(cls, **kwargs):
        return cls(**kwargs)

    @classmethod
    async def connect(cls, sandbox_id, **kwargs):
        return cls(sandbox_id=sandbox_id, **kwargs)

    async def is_running(self, **kwargs):
        _reject_connection_params("Sandbox.is_running", kwargs)
        return self.running

    async def kill(self, **kwargs):
        self.killed = True


def _fake_sdk() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        AsyncSandbox=FakeSandbox,
        SandboxNotFoundException=FakeSandboxNotFound,
        NotFoundException=FakeSandboxNotFound,
        TimeoutException=FakeTimeout,
        CommandExitException=FakeCommandExit,
        AuthenticationException=type("FakeAuth", (Exception,), {}),
        InvalidArgumentException=type("FakeInvalid", (Exception,), {}),
    )


@pytest.fixture(autouse=True)
def fake_e2b_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.instances.clear()
    monkeypatch.setattr(e2b_provider, "_require_e2b_sdk", _fake_sdk)


def _spec(**kwargs) -> SandboxSpec:
    return SandboxSpec(**kwargs)


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_provider_is_registered_as_builtin() -> None:
    assert get_provider_class("e2b") is E2BProvider
    assert E2BProvider.name == "e2b"
    assert e2b_pkg.E2BProvider is E2BProvider


# --------------------------------------------------------------------------
# Template resolution -- E2B starts from an alias, not a registry reference
# --------------------------------------------------------------------------


class TestTemplateResolution:
    async def test_provider_options_template_wins(self) -> None:
        provider = E2BProvider(create={"template": "fallback"})
        handle = await provider.create(_spec(image="also-valid", provider_options={"template": "chosen"}))
        assert handle.raw.create_kwargs["template"] == "chosen"

    async def test_template_map_translates_registry_reference(self) -> None:
        provider = E2BProvider(create={"template_map": {"ghcr.io/acme/task:1.0": "acme-task"}})
        handle = await provider.create(_spec(image="ghcr.io/acme/task:1.0"))
        assert handle.raw.create_kwargs["template"] == "acme-task"

    async def test_image_used_directly_when_already_an_alias(self) -> None:
        provider = E2BProvider()
        handle = await provider.create(_spec(image="build-cython-ext__c9fba49d4bd3"))
        assert handle.raw.create_kwargs["template"] == "build-cython-ext__c9fba49d4bd3"

    async def test_falls_back_to_configured_template(self) -> None:
        provider = E2BProvider(create={"template": "base-template"})
        handle = await provider.create(_spec())
        assert handle.raw.create_kwargs["template"] == "base-template"

    async def test_registry_reference_without_mapping_raises_actionable_error(self) -> None:
        # Silently starting the wrong template would corrupt a benchmark run,
        # so an unmappable image must fail loudly.
        provider = E2BProvider()
        with pytest.raises(E2BCreateError, match="template_map"):
            await provider.create(_spec(image="ghcr.io/acme/task:1.0"))

    async def test_no_template_at_all_raises(self) -> None:
        provider = E2BProvider()
        with pytest.raises(E2BCreateError, match="No E2B template"):
            await provider.create(_spec())


# --------------------------------------------------------------------------
# Resources -- fixed at template build time, must not be dropped silently
# --------------------------------------------------------------------------


class TestResourceHandling:
    async def test_resource_request_warns_once_per_template(self, caplog) -> None:
        provider = E2BProvider(create={"template": "base"})
        with caplog.at_level("WARNING"):
            await provider.create(_spec(resources={"cpu": 8, "memory_mib": 16384}))
            await provider.create(_spec(resources={"cpu": 8, "memory_mib": 16384}))
        warnings = [r for r in caplog.records if "fixes sandbox resources" in r.message]
        assert len(warnings) == 1
        assert "cpu=8" in warnings[0].message and "memory_mib=16384" in warnings[0].message

    async def test_strict_resources_raises(self) -> None:
        provider = E2BProvider(create={"template": "base", "strict_resources": True})
        with pytest.raises(E2BCreateError, match="fixes sandbox resources"):
            await provider.create(_spec(resources={"cpu": 8}))

    async def test_no_warning_without_resource_request(self, caplog) -> None:
        provider = E2BProvider(create={"template": "base"})
        with caplog.at_level("WARNING"):
            await provider.create(_spec())
        assert not [r for r in caplog.records if "fixes sandbox resources" in r.message]


# --------------------------------------------------------------------------
# Connection params are connection-scoped, not per-call
# --------------------------------------------------------------------------


class TestConnectionParamScoping:
    """Only create/connect/kill open a connection and accept ``ApiParams``.

    ``commands.run``, ``files.*`` and ``is_running`` run against an
    already-connected sandbox and take ``request_timeout`` only. Passing them
    the full set raises ``TypeError: Commands.run() got an unexpected keyword
    argument 'api_key'`` -- which aborted every trial of a benchmark run at the
    first exec, since the sandbox had already been created successfully.
    """

    @staticmethod
    def _provider() -> E2BProvider:
        return E2BProvider(
            connection={"api_key": "k", "api_url": "http://gw:8080", "request_timeout_s": 30.0},
            create={"template": "base"},
        )

    async def test_exec_passes_only_request_timeout(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        await provider.exec(handle, "echo hi")
        kwargs = handle.raw.exec_calls[0]
        assert not set(kwargs) & set(_API_PARAM_KEYS), "connection params must not reach commands.run"
        # request_timeout is derived from the command timeout, not inherited
        # from the connection -- see TestExecRequestTimeout.
        assert kwargs["request_timeout"] > 0

    async def test_file_and_status_calls_do_not_leak_connection_params(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        # Each of these would raise TypeError from the fake (as the real SDK does).
        await provider.write_file(handle, "/tmp/f", b"data")
        assert await provider.read_file(handle, "/tmp/f") == b"data"
        assert await provider.status(handle) is SandboxStatus.RUNNING

    async def test_long_command_is_not_bounded_by_connection_timeout(self) -> None:
        # The SDK applies request_timeout to the streaming RPC carrying the
        # command's output. Inheriting the 30s connection timeout would tear
        # down any command running longer than that.
        provider = E2BProvider(
            connection={"api_key": "k", "request_timeout_s": 30.0},
            create={"template": "base"},
        )
        handle = await provider.create(_spec())
        await provider.exec(handle, "make -j8", timeout_s=1800)
        kwargs = handle.raw.exec_calls[0]
        assert kwargs["timeout"] == 1800.0
        assert kwargs["request_timeout"] > 1800.0, "request timeout must outlast the command"

    async def test_exec_request_timeout_override_is_honoured(self) -> None:
        provider = E2BProvider(
            connection={"request_timeout_s": 30.0},
            create={"template": "base"},
            exec={"request_timeout_s": 900.0},
        )
        handle = await provider.create(_spec())
        await provider.exec(handle, "sleep 1", timeout_s=60)
        assert handle.raw.exec_calls[0]["request_timeout"] == 900.0

    async def test_untimed_command_disables_the_request_timeout(self) -> None:
        provider = E2BProvider(
            connection={"request_timeout_s": 30.0},
            create={"template": "base"},
            exec={"default_timeout_s": None},
        )
        handle = await provider.create(_spec())
        await provider.exec(handle, "sleep forever")
        kwargs = handle.raw.exec_calls[0]
        assert kwargs["timeout"] is None
        # 0 is the SDK's "no request timeout"; None would inherit the 30s.
        assert kwargs["request_timeout"] == 0.0

    async def test_create_still_receives_full_connection_params(self) -> None:
        # The narrowing must not strip params from the call that needs them.
        provider = self._provider()
        handle = await provider.create(_spec())
        assert handle.raw.create_kwargs["api_key"] == "k"
        assert handle.raw.create_kwargs["api_url"] == "http://gw:8080"


# --------------------------------------------------------------------------
# Background exec -- survive a lost output stream
# --------------------------------------------------------------------------


class TestBackgroundExec:
    """A dropped stream must not destroy a command that is still running.

    `run(background=True)` returns once the process has started, so the command
    outlives the stream carrying its output and can be reattached by pid.
    """

    @staticmethod
    def _provider(**exec_opts) -> E2BProvider:
        opts = {"background": True, **exec_opts}
        return E2BProvider(create={"template": "base"}, exec=opts)

    async def test_on_by_default(self) -> None:
        # Matches Harbor's own e2b environment, which always dispatches with
        # background=True.
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        await provider.exec(handle, "echo hi")
        assert handle.raw.exec_calls[0]["background"] is True

    async def test_can_be_turned_off(self) -> None:
        provider = E2BProvider(create={"template": "base"}, exec={"background": False})
        handle = await provider.create(_spec())
        await provider.exec(handle, "echo hi")
        assert "background" not in handle.raw.exec_calls[0]

    async def test_background_flag_is_sent(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        result = await provider.exec(handle, "echo hi")
        assert handle.raw.exec_calls[0]["background"] is True
        assert result.return_code == 0

    async def test_lost_stream_is_reattached_by_pid(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [
            ConnectionError("peer closed connection without sending complete message body"),
            FakeCommandResult(stdout="finished", exit_code=0),
        ]
        result = await provider.exec(handle, "make -j8")
        assert handle.raw.connect_calls == [{"pid": 4242, "timeout": 1800.0}]
        assert result.return_code == 0
        assert result.stdout == "finished"

    async def test_reattach_recovers_the_real_exit_code(self) -> None:
        # The exit code is what a benchmark scores on, and it survives even
        # though the output emitted before the reattach does not.
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [ConnectionError("stream lost"), FakeCommandExit(7, stdout="", stderr="boom")]
        result = await provider.exec(handle, "false")
        assert result.return_code == 7

    async def test_reattach_attempts_are_bounded(self) -> None:
        provider = self._provider(reconnect_attempts=2)
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [ConnectionError("a"), ConnectionError("b"), ConnectionError("c")]
        with pytest.raises(ConnectionError):
            await provider.exec(handle, "sleep 1")
        assert len(handle.raw.connect_calls) == 2

    async def test_non_zero_exit_is_not_treated_as_a_lost_stream(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [FakeCommandExit(3, stdout="out", stderr="err")]
        result = await provider.exec(handle, "false")
        assert result.return_code == 3
        assert handle.raw.connect_calls == [], "a real exit must not trigger a reattach"

    async def test_timeout_is_not_treated_as_a_lost_stream(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [FakeTimeout("timed out")]
        with pytest.raises(TimeoutError):
            await provider.exec(handle, "sleep 999")
        assert handle.raw.connect_calls == []

    async def test_process_gone_reports_the_original_failure(self) -> None:
        # connect() raises not-found once the command has exited, so a command
        # that finishes during the gap cannot be recovered; the transport
        # failure is the useful error to surface.
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [ConnectionError("stream lost")]
        handle.raw.connect_error = FakeSandboxNotFound("process with pid 4242 not found")
        with pytest.raises(ConnectionError, match="stream lost"):
            await provider.exec(handle, "quick")


# --------------------------------------------------------------------------
# Create / lifecycle
# --------------------------------------------------------------------------


class TestCreateAndLifecycle:
    async def test_spec_fields_map_onto_sdk_kwargs(self) -> None:
        provider = E2BProvider(
            connection={"api_key": "k", "api_url": "http://gw:8080", "request_timeout_s": 30.0},
            create={"template": "base", "allow_internet_access": False},
        )
        handle = await provider.create(
            _spec(ttl_s=120, env={"FOO": "bar"}, metadata={"run": "1"}),
        )
        kwargs = handle.raw.create_kwargs
        assert kwargs["timeout"] == 120
        assert kwargs["envs"] == {"FOO": "bar"}
        assert kwargs["metadata"] == {"run": "1"}
        assert kwargs["allow_internet_access"] is False
        assert kwargs["api_key"] == "k"
        assert kwargs["api_url"] == "http://gw:8080"
        assert kwargs["request_timeout"] == 30.0
        assert handle.provider_name == "e2b"
        assert handle.sandbox_id == "sbx-1"

    async def test_spec_files_are_written_after_create(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec(files={"/app/seed.txt": "hello"}))
        assert handle.raw.files_written["/app/seed.txt"] == "hello"

    async def test_create_failure_is_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(**kwargs):
            raise RuntimeError("gateway exploded")

        monkeypatch.setattr(FakeSandbox, "create", boom)
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 0})
        with pytest.raises(E2BCreateError, match="gateway exploded"):
            await provider.create(_spec())

    async def test_status_and_close(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        assert await provider.status(handle) == SandboxStatus.RUNNING
        handle.raw.running = False
        assert await provider.status(handle) == SandboxStatus.STOPPED

        handle.raw.running = True
        sandbox = handle.raw
        await provider.close(handle)
        assert sandbox.killed is True
        assert handle.raw is None
        # Closing twice must not raise.
        await provider.close(handle)

    async def test_close_tolerates_already_gone_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        async def gone(**kwargs):
            raise FakeSandboxNotFound("expired")

        monkeypatch.setattr(handle.raw, "kill", gone)
        await provider.close(handle)
        assert handle.raw is None

    async def test_connect_returns_handle(self) -> None:
        provider = E2BProvider()
        handle = await provider.connect("sbx-existing")
        assert handle.sandbox_id == "sbx-existing"
        assert handle.provider_name == "e2b"

    async def test_aclose_is_a_noop(self) -> None:
        assert await E2BProvider().aclose() is None


# --------------------------------------------------------------------------
# exec
# --------------------------------------------------------------------------


class TestExec:
    async def test_exec_maps_arguments_and_result(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.exec_behaviour = FakeCommandResult(stdout="out", stderr="err", exit_code=0)

        result = await provider.exec(handle, "echo hi", cwd="/app", env={"A": "1"}, timeout_s=42, user="root")
        assert (result.stdout, result.stderr, result.return_code) == ("out", "err", 0)
        call = handle.raw.exec_calls[-1]
        assert call["cmd"] == "echo hi"
        assert call["cwd"] == "/app"
        assert call["envs"] == {"A": "1"}
        assert call["user"] == "root"
        assert call["timeout"] == 42.0

    async def test_default_exec_timeout_is_generous(self) -> None:
        # The SDK default is 60s, which silently truncates builds and verifier
        # suites; the provider default must be far larger.
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        await provider.exec(handle, "true")
        assert handle.raw.exec_calls[-1]["timeout"] >= 900

    async def test_nonzero_exit_is_a_result_not_an_exception(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.exec_behaviour = FakeCommandExit(exit_code=7, stdout="partial", stderr="bad")

        result = await provider.exec(handle, "false")
        assert result.return_code == 7
        assert result.stdout == "partial"
        assert result.stderr == "bad"

    async def test_timeout_is_raised_as_timeout_error(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        handle.raw.exec_behaviour = FakeTimeout("deadline exceeded")

        with pytest.raises(TimeoutError):
            await provider.exec(handle, "sleep 999", timeout_s=1)


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


class TestFiles:
    async def test_upload_then_download_round_trip(self, tmp_path: Path) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        source = tmp_path / "in.bin"
        source.write_bytes(b"payload\x00binary")
        await provider.upload_file(handle, source, "/remote/in.bin")

        target = tmp_path / "nested" / "out.bin"
        await provider.download_file(handle, "/remote/in.bin", target)
        assert target.read_bytes() == b"payload\x00binary"

    async def test_upload_missing_file_raises(self, tmp_path: Path) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        with pytest.raises(FileNotFoundError):
            await provider.upload_file(handle, tmp_path / "nope.txt", "/remote/nope.txt")


# --------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------


class TestRetries:
    async def test_transient_failure_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}
        original = FakeSandbox.create.__func__

        async def flaky(cls, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("502 bad gateway")
            return await original(cls, **kwargs)

        monkeypatch.setattr(FakeSandbox, "create", classmethod(flaky))
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 3, "retry_delay_s": 0})
        handle = await provider.create(_spec())
        assert calls["n"] == 3
        assert handle.sandbox_id == "sbx-1"

    async def test_deterministic_errors_are_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        async def not_found(cls, **kwargs):
            calls["n"] += 1
            raise FakeSandboxNotFound("template missing")

        monkeypatch.setattr(FakeSandbox, "create", classmethod(not_found))
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 5, "retry_delay_s": 0})
        with pytest.raises(E2BCreateError):
            await provider.create(_spec())
        assert calls["n"] == 1


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


def test_unknown_config_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown E2BCreateConfig keys"):
        E2BProvider(create={"template": "base", "nope": 1})
