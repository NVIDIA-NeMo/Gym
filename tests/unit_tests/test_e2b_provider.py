# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import inspect
import re
import sys
import types
from importlib.metadata import version
from pathlib import Path

import pytest

from nemo_gym.package_info import __version__ as nemo_gym_version
from nemo_gym.sandbox import AsyncSandbox, ConnectableProvider
from nemo_gym.sandbox.providers import e2b as e2b_pkg
from nemo_gym.sandbox.providers.base import SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.e2b import _sdk as e2b_sdk
from nemo_gym.sandbox.providers.e2b import provider as e2b_provider
from nemo_gym.sandbox.providers.e2b.provider import _API_PARAM_KEYS, E2BCreateError, E2BProvider
from nemo_gym.sandbox.providers.registry import get_provider_class


pytestmark = pytest.mark.sandbox

_REAL_PROVIDER_REQUIRE_E2B_SDK = e2b_provider._require_e2b_sdk


# --------------------------------------------------------------------------
# Fake e2b SDK
# --------------------------------------------------------------------------


class FakeSandboxNotFound(Exception):
    pass


class FakeTimeout(Exception):
    pass


class FakeRateLimit(Exception):
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


class FakeConnectionConfig:
    integrations: list[str] = []

    @classmethod
    def set_integration(cls, integration: str) -> None:
        cls.integrations.append(integration)


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

    async def connect(self, pid, timeout=None, request_timeout=None):
        self._sandbox.connect_calls.append({"pid": pid, "timeout": timeout, "request_timeout": request_timeout})
        if self._sandbox.connect_error is not None:
            raise self._sandbox.connect_error
        return FakeCommandHandle(pid, self._sandbox.wait_outcomes, self._sandbox.exec_behaviour)


class FakeFiles:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox

    async def write(self, path, data, **kwargs):
        _reject_connection_params("Filesystem.write", kwargs)
        self._sandbox.file_write_calls.append({"path": path, "data": data, **kwargs})
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
        self.file_write_calls: list[dict] = []
        self.files_written: dict[str, object] = {}
        self.killed = False
        self.kill_calls: list[dict] = []
        self.kill_outcomes: list[object] = []
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
        self.kill_calls.append(kwargs)
        if self.kill_outcomes:
            outcome = self.kill_outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            if outcome is False:
                return False
        self.killed = True
        return True


def _fake_sdk() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        AsyncSandbox=FakeSandbox,
        ConnectionConfig=FakeConnectionConfig,
        SandboxNotFoundException=FakeSandboxNotFound,
        NotFoundException=FakeSandboxNotFound,
        TimeoutException=FakeTimeout,
        RateLimitException=FakeRateLimit,
        CommandExitException=FakeCommandExit,
        AuthenticationException=type("FakeAuth", (Exception,), {}),
        InvalidArgumentException=type("FakeInvalid", (Exception,), {}),
    )


def _fake_sdk_module() -> types.ModuleType:
    module = types.ModuleType("e2b")
    module.__dict__.update(vars(_fake_sdk()))
    return module


@pytest.fixture(autouse=True)
def fake_e2b_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.instances.clear()
    FakeConnectionConfig.integrations.clear()
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


async def test_runtime_loader_sets_integration_once_per_sdk_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk_module = _fake_sdk_module()
    monkeypatch.setitem(sys.modules, "e2b", sdk_module)
    monkeypatch.setattr(e2b_sdk, "_CONFIGURED_SDK_MODULES", {})
    monkeypatch.setattr(e2b_provider, "_require_e2b_sdk", _REAL_PROVIDER_REQUIRE_E2B_SDK)

    provider = E2BProvider(create={"template": "base"})
    handle = await provider.create(_spec())
    assert await provider.status(handle) is SandboxStatus.RUNNING

    assert FakeConnectionConfig.integrations == [f"nemo-gym/{nemo_gym_version}"]


def test_loader_reports_missing_optional_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "e2b", None)

    with pytest.raises(ImportError, match=r"pip install 'e2b>=2\.36\.0,<3\.0\.0'"):
        e2b_sdk.require_e2b_sdk("Testing the e2b provider")


def test_real_sdk_user_agent_and_call_shapes() -> None:
    e2b = pytest.importorskip("e2b", reason="e2b optional sandbox dependency is not installed")
    installed_match = re.match(r"^(\d+)\.(\d+)", version("e2b"))
    assert installed_match is not None
    assert (int(installed_match[1]), int(installed_match[2])) >= (2, 36)
    assert int(installed_match[1]) < 3
    assert set(_API_PARAM_KEYS) <= set(e2b.ApiParams.__annotations__)

    e2b_sdk.require_e2b_sdk("Testing the e2b provider")
    products = e2b.ConnectionConfig().headers["User-Agent"].split()
    assert f"nemo-gym/{nemo_gym_version}" in products

    inspect.signature(e2b.AsyncSandbox.create).bind(
        template="base",
        timeout=60,
        request_timeout=30,
        envs={},
        metadata={},
        secure=True,
        allow_internet_access=True,
    )
    inspect.signature(e2b.AsyncSandbox.connect).bind("sbx-existing", request_timeout=30)
    inspect.signature(e2b.AsyncTemplate.exists).bind("base")
    inspect.signature(e2b.AsyncTemplate.build).bind(
        object(),
        name="base",
        cpu_count=2,
        memory_mb=1024,
    )


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

    async def test_registry_reference_without_mapping_does_not_use_fallback(self) -> None:
        # Silently starting the wrong template would corrupt a benchmark run,
        # so an unmappable image must fail loudly.
        provider = E2BProvider(create={"template": "fallback"})
        with pytest.raises(E2BCreateError, match="template_map"):
            await provider.create(_spec(image="ghcr.io/acme/task:1.0"))
        assert FakeSandbox.instances == []

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
            await provider.create(
                _spec(
                    resources={
                        "cpu": 8,
                        "memory_mib": 16384,
                        "disk_gib": 100,
                        "gpu": 1,
                        "gpu_type": "H100",
                    }
                )
            )
            await provider.create(_spec(resources={"cpu": 8}))
        warnings = [r for r in caplog.records if "fixes sandbox resources" in r.message]
        assert len(warnings) == 1
        for detail in ("cpu=8", "memory_mib=16384", "disk_gib=100", "gpu=1", "gpu_type=H100"):
            assert detail in warnings[0].message

    @pytest.mark.parametrize(
        "resources",
        [
            {"cpu": 1},
            {"memory_mib": 1024},
            {"disk_gib": 10},
            {"gpu": 1},
            {"gpu_type": "H100"},
        ],
    )
    async def test_strict_resources_rejects_every_resource(self, resources: dict[str, object]) -> None:
        provider = E2BProvider(create={"template": "base", "strict_resources": True})
        with pytest.raises(E2BCreateError, match="fixes sandbox resources"):
            await provider.create(_spec(resources=resources))
        assert FakeSandbox.instances == []

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
        assert kwargs["request_timeout"] == 30.0

    async def test_file_and_status_calls_do_not_leak_connection_params(self) -> None:
        provider = self._provider()
        handle = await provider.create(_spec())
        # Each of these would raise TypeError from the fake (as the real SDK does).
        await provider.write_file(handle, "/tmp/f", b"data")
        assert await provider.read_file(handle, "/tmp/f") == b"data"
        assert await provider.status(handle) is SandboxStatus.RUNNING

    async def test_command_and_request_timeouts_are_independent(self) -> None:
        # In E2B 2.36 request_timeout bounds opening the command stream, while
        # timeout bounds the running output stream/wait, not the remote process.
        provider = E2BProvider(
            connection={"api_key": "k", "request_timeout_s": 30.0},
            create={"template": "base"},
        )
        handle = await provider.create(_spec())
        await provider.exec(handle, "make -j8", timeout_s=1800)
        kwargs = handle.raw.exec_calls[0]
        assert kwargs["timeout"] == 1800.0
        assert kwargs["request_timeout"] == 30.0

    async def test_exec_request_timeout_override_is_honoured(self) -> None:
        provider = E2BProvider(
            connection={"request_timeout_s": 30.0},
            create={"template": "base"},
            exec={"request_timeout_s": 900.0},
        )
        handle = await provider.create(_spec())
        await provider.exec(handle, "sleep 1", timeout_s=60)
        assert handle.raw.exec_calls[0]["request_timeout"] == 900.0

    async def test_untimed_command_keeps_the_connection_request_timeout(self) -> None:
        provider = E2BProvider(
            connection={"request_timeout_s": 30.0},
            create={"template": "base"},
            exec={"default_timeout_s": None},
        )
        handle = await provider.create(_spec())
        await provider.exec(handle, "sleep forever")
        kwargs = handle.raw.exec_calls[0]
        assert kwargs["timeout"] is None
        assert kwargs["request_timeout"] == 30.0

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

    async def test_lost_stream_is_reattached_by_pid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monotonic_values = iter([100.0, 105.0])
        monkeypatch.setattr(e2b_provider, "monotonic", monotonic_values.__next__)
        provider = self._provider(request_timeout_s=75.0)
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [
            ConnectionError("peer closed connection without sending complete message body"),
            FakeCommandResult(stdout="finished", exit_code=0),
        ]
        result = await provider.exec(handle, "make -j8", timeout_s=60)
        assert handle.raw.connect_calls == [{"pid": 4242, "timeout": 55.0, "request_timeout": 55.0}]
        assert result.return_code == 0
        assert result.stdout == "finished"

    async def test_lost_stream_does_not_reattach_after_deadline(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monotonic_values = iter([100.0, 161.0])
        monkeypatch.setattr(e2b_provider, "monotonic", monotonic_values.__next__)
        provider = self._provider()
        handle = await provider.create(_spec())
        handle.raw.wait_outcomes = [ConnectionError("stream lost")]

        with pytest.raises(TimeoutError, match="timed out after 60"):
            await provider.exec(handle, "make -j8", timeout_s=60)

        assert handle.raw.connect_calls == []

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

    async def test_ready_timeout_overrides_connection_request_timeout(self) -> None:
        provider = E2BProvider(
            connection={"request_timeout_s": 30.0},
            create={"template": "base"},
        )
        handle = await provider.create(_spec(ready_timeout_s=90))
        assert handle.raw.create_kwargs["request_timeout"] == 90.0

    @pytest.mark.parametrize("ready_timeout_s", [0, -1, float("nan"), float("inf"), True, "60"])
    async def test_invalid_ready_timeout_is_rejected_before_create(self, ready_timeout_s: object) -> None:
        provider = E2BProvider(create={"template": "base"})
        with pytest.raises(E2BCreateError, match="ready_timeout_s"):
            await provider.create(_spec(ready_timeout_s=ready_timeout_s))
        assert FakeSandbox.instances == []

    @pytest.mark.parametrize("ttl_s", [0, -1, float("nan"), float("inf"), True, "60"])
    async def test_invalid_ttl_is_rejected_before_create(self, ttl_s: object) -> None:
        provider = E2BProvider(create={"template": "base"})
        with pytest.raises(E2BCreateError, match="ttl_s"):
            await provider.create(_spec(ttl_s=ttl_s))
        assert FakeSandbox.instances == []

    async def test_entrypoint_is_rejected_before_create(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        with pytest.raises(E2BCreateError, match="entrypoint"):
            await provider.create(_spec(entrypoint=["python", "app.py"]))
        assert FakeSandbox.instances == []

    async def test_unknown_provider_options_are_rejected_before_create(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        with pytest.raises(ValueError, match="provider option"):
            await provider.create(_spec(provider_options={"template": "base", "unknown": True}))
        assert FakeSandbox.instances == []

    async def test_provider_create_leaves_spec_files_to_the_facade(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec(files={"/app/seed.txt": "hello"}))
        assert handle.raw.file_write_calls == []

    async def test_async_sandbox_uploads_spec_files_exactly_once(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        sandbox = await AsyncSandbox(
            provider,
            _spec(files={"/app/seed.txt": "hello"}),
        ).start()
        raw = FakeSandbox.instances[0]
        assert raw.file_write_calls == [{"path": "/app/seed.txt", "data": b"hello"}]
        assert raw.files_written == {"/app/seed.txt": b"hello"}
        await sandbox.stop()

    async def test_async_sandbox_cleans_up_when_spec_file_upload_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        provider = E2BProvider(create={"template": "base"})

        async def fail_upload(handle, source_path, target_path):
            raise RuntimeError("upload failed")

        monkeypatch.setattr(provider, "upload_file", fail_upload)

        with pytest.raises(RuntimeError, match="upload failed"):
            await AsyncSandbox(
                provider,
                _spec(files={"/app/seed.txt": "hello"}),
            ).start()

        raw = FakeSandbox.instances[0]
        assert raw.killed is True
        assert len(raw.kill_calls) == 1

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
        assert len(sandbox.kill_calls) == 1

    @pytest.mark.parametrize(
        "transient_error",
        [ConnectionError("connection reset"), FakeTimeout("request timed out")],
    )
    async def test_close_retries_transient_kill_errors(self, transient_error: Exception) -> None:
        provider = E2BProvider(
            create={"template": "base"},
            operations={"retries": 1, "retry_delay_s": 0},
        )
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [transient_error, True]

        await provider.close(handle)

        assert len(sandbox.kill_calls) == 2
        assert sandbox.killed is True
        assert handle.raw is None

    async def test_close_exhausted_failure_preserves_raw_handle(self) -> None:
        provider = E2BProvider(
            create={"template": "base"},
            operations={"retries": 1, "retry_delay_s": 0},
        )
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [ConnectionError("temporary"), ConnectionError("still unavailable")]

        with pytest.raises(ConnectionError, match="still unavailable"):
            await provider.close(handle)

        assert len(sandbox.kill_calls) == 2
        assert handle.raw is sandbox

    async def test_rate_limit_is_not_retried(self) -> None:
        provider = E2BProvider(
            create={"template": "base"},
            operations={"retries": 5, "retry_delay_s": 0},
        )
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [FakeRateLimit("429 resource exhausted"), True]

        with pytest.raises(FakeRateLimit, match="resource exhausted"):
            await provider.close(handle)

        assert len(sandbox.kill_calls) == 1
        assert handle.raw is sandbox

    async def test_close_false_result_clears_handle(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())
        sandbox = handle.raw
        sandbox.kill_outcomes = [False]

        await provider.close(handle)

        assert sandbox.killed is False
        assert handle.raw is None

    async def test_close_tolerates_already_gone_sandbox(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        async def gone(**kwargs):
            raise FakeSandboxNotFound("expired")

        monkeypatch.setattr(handle.raw, "kill", gone)
        await provider.close(handle)
        assert handle.raw is None

    async def test_connectable_provider_serializes_and_connects_mapping(self) -> None:
        provider = E2BProvider()
        assert isinstance(provider, ConnectableProvider)
        descriptor = await provider.serialize_handle(
            SandboxHandle(sandbox_id="sbx-existing", provider_name="e2b", raw=object())
        )
        assert descriptor == {"sandbox_id": "sbx-existing"}

        handle = await provider.connect({**descriptor, "workdir": "/repo"})
        assert handle.sandbox_id == "sbx-existing"
        assert handle.provider_name == "e2b"

    async def test_async_sandbox_serialize_connect_round_trip_preserves_workdir(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        sandbox = await AsyncSandbox(provider, _spec(workdir="/repo")).start()
        descriptor = await sandbox.serialize()
        assert descriptor == {"sandbox_id": "sbx-1", "workdir": "/repo"}

        connected_provider = E2BProvider()
        connected = await AsyncSandbox.connect(descriptor, provider=connected_provider)
        await connected.exec("pwd")
        assert FakeSandbox.instances[-1].exec_calls[-1]["cwd"] == "/repo"

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

    async def test_default_exec_timeout_matches_the_public_facade(self) -> None:
        provider = E2BProvider(create={"template": "base"})
        sandbox = await AsyncSandbox(provider, _spec()).start()
        await sandbox.exec("true")
        assert FakeSandbox.instances[0].exec_calls[-1]["timeout"] == 180.0

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

        with pytest.raises(TimeoutError, match=r"deadline exceeded.*wait/stream budget=1s"):
            await provider.exec(handle, "sleep 999", timeout_s=1)

    @pytest.mark.parametrize("timeout_s", [-1, float("nan"), float("inf"), True, "1"])
    async def test_invalid_timeout_is_rejected_before_command_start(self, timeout_s: object) -> None:
        provider = E2BProvider(create={"template": "base"})
        handle = await provider.create(_spec())

        with pytest.raises(ValueError, match="timeout_s"):
            await provider.exec(handle, "true", timeout_s=timeout_s)

        assert handle.raw.exec_calls == []


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
    async def test_transient_create_failure_is_not_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        async def flaky(cls, **kwargs):
            calls["n"] += 1
            raise RuntimeError("502 bad gateway")

        monkeypatch.setattr(FakeSandbox, "create", classmethod(flaky))
        provider = E2BProvider(create={"template": "base"}, operations={"retries": 3, "retry_delay_s": 0})
        with pytest.raises(E2BCreateError, match="502 bad gateway"):
            await provider.create(_spec())
        assert calls["n"] == 1

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


@pytest.mark.parametrize(
    "create",
    [
        {"template": ""},
        {"template_map": {"": "template"}},
        {"template_map": {"image": ""}},
        {"template_map": []},
    ],
)
def test_invalid_template_config_is_rejected(create: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError), match=r"create\.template"):
        E2BProvider(create=create)


@pytest.mark.parametrize(
    ("section", "config", "message"),
    [
        ("connection", {"request_timeout_s": -1}, "connection.request_timeout_s must be >= 0"),
        ("connection", {"request_timeout_s": float("nan")}, "connection.request_timeout_s must be >= 0"),
        ("create", {"timeout_s": 0}, "create.timeout_s must be > 0"),
        ("create", {"timeout_s": float("inf")}, "create.timeout_s must be > 0"),
        ("exec", {"default_timeout_s": -1}, "exec.default_timeout_s must be >= 0"),
        ("exec", {"request_timeout_s": -1}, "exec.request_timeout_s must be >= 0"),
        ("exec", {"reconnect_attempts": -1}, "exec.reconnect_attempts must be >= 0"),
        ("exec", {"reconnect_attempts": 1.5}, "exec.reconnect_attempts must be >= 0"),
        ("operations", {"retries": -1}, "operations.retries must be >= 0"),
        ("operations", {"retries": True}, "operations.retries must be >= 0"),
        ("operations", {"retry_delay_s": -1}, "operations.retry_delay_s must be >= 0"),
        ("operations", {"retry_delay_s": None}, "operations.retry_delay_s must be >= 0"),
        ("operations", {"retry_max_delay_s": -1}, "operations.retry_max_delay_s must be >= 0"),
        ("operations", {"retry_max_delay_s": float("nan")}, "operations.retry_max_delay_s must be >= 0"),
    ],
)
def test_invalid_config_values_are_rejected(section: str, config: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        E2BProvider(**{section: config})
