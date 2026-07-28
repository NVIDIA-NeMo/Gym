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

import asyncio
import re
import types
from pathlib import Path

import pytest

from nemo_gym.sandbox.providers import e2b as e2b_pkg
from nemo_gym.sandbox.providers.base import SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.e2b import provider as e2b_provider
from nemo_gym.sandbox.providers.e2b.provider import E2BCreateError, E2BProvider
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


class FakeCommands:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox

    async def run(self, **kwargs):
        self._sandbox.exec_calls.append(kwargs)
        behaviour = self._sandbox.exec_behaviour
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour or FakeCommandResult(stdout="ok")


class FakeFiles:
    def __init__(self, sandbox: "FakeSandbox") -> None:
        self._sandbox = sandbox

    async def write(self, path, data, **kwargs):
        self._sandbox.files_written[path] = data
        return None

    async def read(self, path, **kwargs):
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
        return self.running

    async def kill(self, **kwargs):
        self.killed = True


class FakeTemplateBuilder:
    def __init__(self, image: str, username=None, password=None) -> None:
        self.image = image
        self.username = username
        self.password = password


class FakeTemplate:
    """Records build calls; ``existing_aliases`` fakes server-side state."""

    builds: list[dict] = []
    existing_aliases: set[str] = set()
    build_error: Exception | None = None
    build_delay_s: float = 0.0

    def from_image(self, image, username=None, password=None):
        return FakeTemplateBuilder(image, username, password)

    @staticmethod
    async def alias_exists(alias, **kwargs):
        return alias in FakeTemplate.existing_aliases

    @staticmethod
    async def build(builder, *, alias=None, cpu_count=None, memory_mb=None, **kwargs):
        if FakeTemplate.build_delay_s:
            await asyncio.sleep(FakeTemplate.build_delay_s)
        if FakeTemplate.build_error is not None:
            raise FakeTemplate.build_error
        FakeTemplate.builds.append(
            {"image": builder.image, "alias": alias, "cpu_count": cpu_count, "memory_mb": memory_mb, **kwargs}
        )
        FakeTemplate.existing_aliases.add(alias)
        return types.SimpleNamespace(alias=alias)


def _fake_sdk() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        AsyncSandbox=FakeSandbox,
        AsyncTemplate=FakeTemplate,
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
    FakeTemplate.builds.clear()
    FakeTemplate.existing_aliases.clear()
    FakeTemplate.build_error = None
    FakeTemplate.build_delay_s = 0.0
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
        with pytest.raises(E2BCreateError, match="auto_build_from_image"):
            await provider.create(_spec(image="ghcr.io/acme/task:1.0"))

    async def test_no_template_at_all_raises(self) -> None:
        provider = E2BProvider()
        with pytest.raises(E2BCreateError, match="No E2B template"):
            await provider.create(_spec())


# --------------------------------------------------------------------------
# On-demand template building from an OCI image
# --------------------------------------------------------------------------


class TestAutoBuildFromImage:
    async def test_builds_template_then_starts_sandbox_from_it(self) -> None:
        provider = E2BProvider(create={"auto_build_from_image": True})
        handle = await provider.create(_spec(image="ghcr.io/acme/task:1.0"))

        assert len(FakeTemplate.builds) == 1
        build = FakeTemplate.builds[0]
        assert build["image"] == "ghcr.io/acme/task:1.0"
        # Alias must be charset-safe for E2B and traceable back to the image.
        alias = build["alias"]
        assert re.fullmatch(r"[A-Za-z0-9_-]+", alias)
        assert alias.startswith("task-1-0__")
        assert handle.raw.create_kwargs["template"] == alias

    async def test_spec_resources_become_build_arguments(self) -> None:
        # The whole point of building ourselves: E2B fixes cpu/memory at build
        # time, so a spec's resources CAN be honoured on this path.
        provider = E2BProvider(create={"auto_build_from_image": True})
        await provider.create(_spec(image="ghcr.io/acme/task:1.0", resources={"cpu": 8, "memory_mib": 16384}))
        build = FakeTemplate.builds[0]
        assert build["cpu_count"] == 8
        assert build["memory_mb"] == 16384

    async def test_no_resource_warning_when_applied_at_build(self, caplog) -> None:
        provider = E2BProvider(create={"auto_build_from_image": True})
        with caplog.at_level("WARNING"):
            await provider.create(_spec(image="ghcr.io/acme/task:1.0", resources={"cpu": 8}))
        assert not [r for r in caplog.records if "fixes sandbox resources" in r.message]

    async def test_different_resources_get_different_templates(self) -> None:
        # Sharing one template across differing resource requests would
        # silently give the second sandbox the first one's sizing.
        provider = E2BProvider(create={"auto_build_from_image": True})
        a = await provider.create(_spec(image="ghcr.io/acme/task:1.0", resources={"cpu": 2}))
        b = await provider.create(_spec(image="ghcr.io/acme/task:1.0", resources={"cpu": 8}))
        assert a.raw.create_kwargs["template"] != b.raw.create_kwargs["template"]
        assert len(FakeTemplate.builds) == 2

    async def test_template_is_built_once_and_reused(self) -> None:
        provider = E2BProvider(create={"auto_build_from_image": True})
        await provider.create(_spec(image="ghcr.io/acme/task:1.0"))
        await provider.create(_spec(image="ghcr.io/acme/task:1.0"))
        assert len(FakeTemplate.builds) == 1

    async def test_concurrent_creates_build_only_once(self) -> None:
        # N sandboxes on one image must not trigger N builds.
        FakeTemplate.build_delay_s = 0.02
        provider = E2BProvider(create={"auto_build_from_image": True})
        handles = await asyncio.gather(*(provider.create(_spec(image="ghcr.io/acme/task:1.0")) for _ in range(6)))
        assert len(FakeTemplate.builds) == 1
        assert len({h.raw.create_kwargs["template"] for h in handles}) == 1

    async def test_preexisting_alias_is_not_rebuilt(self) -> None:
        provider = E2BProvider(create={"auto_build_from_image": True})
        alias = provider._derive_alias("ghcr.io/acme/task:1.0", 2, 1024)
        FakeTemplate.existing_aliases.add(alias)
        handle = await provider.create(_spec(image="ghcr.io/acme/task:1.0"))
        assert FakeTemplate.builds == []
        assert handle.raw.create_kwargs["template"] == alias

    async def test_registry_credentials_are_forwarded(self) -> None:
        provider = E2BProvider(
            create={"auto_build_from_image": True, "registry_username": "u", "registry_password": "p"}
        )
        captured: dict = {}
        original = FakeTemplate.from_image

        def spy(self, image, username=None, password=None):
            captured.update({"username": username, "password": password})
            return original(self, image, username, password)

        FakeTemplate.from_image = spy
        try:
            await provider.create(_spec(image="ghcr.io/acme/task:1.0"))
        finally:
            FakeTemplate.from_image = original
        assert captured == {"username": "u", "password": "p"}

    async def test_build_failure_is_wrapped(self) -> None:
        FakeTemplate.build_error = RuntimeError("builder offline")
        provider = E2BProvider(create={"auto_build_from_image": True})
        with pytest.raises(E2BCreateError, match="builder offline"):
            await provider.create(_spec(image="ghcr.io/acme/task:1.0"))

    async def test_build_timeout_is_reported(self) -> None:
        FakeTemplate.build_delay_s = 0.2
        provider = E2BProvider(create={"auto_build_from_image": True, "build_timeout_s": 0.01})
        with pytest.raises(E2BCreateError, match="Timed out"):
            await provider.create(_spec(image="ghcr.io/acme/task:1.0"))

    async def test_explicit_mapping_still_wins_over_building(self) -> None:
        provider = E2BProvider(
            create={"auto_build_from_image": True, "template_map": {"ghcr.io/acme/task:1.0": "preset"}}
        )
        handle = await provider.create(_spec(image="ghcr.io/acme/task:1.0"))
        assert handle.raw.create_kwargs["template"] == "preset"
        assert FakeTemplate.builds == []


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
