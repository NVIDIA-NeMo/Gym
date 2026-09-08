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

"""Unit tests for the Modal sandbox provider (SDK faked; no network or account).

The fake exercises lifecycle, binary exec streams, native file copies, and
tunnels. ``test_modal_sdk_surface`` separately checks the installed SDK contract.
"""

from __future__ import annotations

import asyncio
import inspect
import types
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox import AsyncSandbox, ConnectableProvider, SupportsSandboxEndpoint
from nemo_gym.sandbox.providers.base import (
    SandboxCreateVerificationError,
    SandboxProvider,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.modal import provider as modal_provider
from nemo_gym.sandbox.providers.modal.provider import (
    ModalCreateError,
    ModalCreateVerificationError,
    ModalProvider,
)
from nemo_gym.sandbox.providers.registry import create_provider, get_provider_class, list_providers


pytestmark = pytest.mark.sandbox


# --------------------------------------------------------------------------
# Fake Modal SDK
# --------------------------------------------------------------------------


class _Aio:
    """A sync-looking callable that carries an ``.aio`` coroutine, like synchronicity does."""

    def __init__(self, fn):
        self._fn = fn

    async def aio(self, *args, **kwargs):
        result = self._fn(*args, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    def __call__(self, *args, **kwargs):  # pragma: no cover - provider always uses .aio
        raise AssertionError("provider must use the .aio surface")


class FakeStream:
    def __init__(self, data):
        self._data = data
        self.read = _Aio(self._read)

    def _read(self):
        return self._data


class FakeStdin:
    def __init__(self):
        self.chunks: list[bytes] = []
        self.eof = False
        self.drains = 0
        self.drain = _Aio(self._drain)

    def write(self, data):
        assert not self.eof, "write after EOF"
        self.chunks.append(bytes(data))

    def write_eof(self):
        self.eof = True

    def _drain(self):
        self.drains += 1


class FakeProcess:
    def __init__(self, stdout="", stderr="", returncode=0, *, raise_on_wait=None, wait_delay=0.0):
        self.stdout = FakeStream(stdout)
        self.stderr = FakeStream(stderr)
        self.stdin = FakeStdin()
        self._rc = returncode
        self._raise = raise_on_wait
        self._delay = wait_delay
        self.wait = _Aio(self._wait)

    async def _wait(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raise is not None:
            raise self._raise
        return self._rc


class FakeFilesystem:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self.calls = []
        self.error = None
        self.write_bytes = _Aio(self._write)
        self.read_bytes = _Aio(self._read)
        self.copy_from_local = _Aio(self._upload)
        self.copy_to_local = _Aio(self._download)

    def _write(self, data, path):
        if self.error:
            raise self.error
        self.calls.append(("write", path))
        self.sandbox.files[path] = bytes(data)

    def _read(self, path):
        if self.error:
            raise self.error
        self.calls.append(("read", path))
        return self.sandbox.files[path]

    def _upload(self, source, target):
        self.calls.append(("upload", source, target))
        self._write(Path(source).read_bytes(), target)

    def _download(self, source, target):
        self.calls.append(("download", source, target))
        data = self._read(source)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(data)


class FakeSandbox:
    """One Modal sandbox object. ``exec_script`` maps shell-command substrings to outcomes."""

    _counter = 0

    def __init__(self, entrypoint, kwargs, *, exec_script=None):
        FakeSandbox._counter += 1
        self.object_id = f"sb-{FakeSandbox._counter:04d}"
        self.entrypoint = tuple(entrypoint)
        self.create_kwargs = kwargs
        self.exec_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        self.files: dict[str, bytes] = {}
        self.mkdirs: list[tuple[str, bool]] = []
        self.filesystem = FakeFilesystem(self)
        self.detached = 0
        self.terminate_wait = None
        self.detach = _Aio(self._detach)
        self.terminated = 0
        self.poll_result: Any = None
        self.poll_exc: BaseException | None = None
        self.terminate_exc: BaseException | None = None
        self.tunnel_map: dict[int, Any] = {}
        self.too_large_exc: BaseException | None = None
        self.exec_script = exec_script or {}
        self.last_process: FakeProcess | None = None
        self.exec = _Aio(self._exec)
        self.mkdir = _Aio(self._mkdir)
        self.poll = _Aio(self._poll)
        self.terminate = _Aio(self._terminate)
        self.tunnels = _Aio(self._tunnels)

    def _exec(self, *argv, **kwargs):
        self.exec_calls.append((tuple(argv), kwargs))
        command = argv[-1] if argv else ""
        outcome = None
        for needle, scripted in self.exec_script.items():
            if needle in " ".join(argv):
                outcome = scripted
                break
        if callable(outcome):
            outcome = outcome(argv, kwargs)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, FakeProcess):
            self.last_process = outcome
            return outcome
        # Default: a probe-friendly success that echoes printf.
        if "printf ok" in command:
            proc = FakeProcess(stdout="ok")
        elif argv[:1] == ("cat",):
            path = argv[1]
            if path not in self.files:
                proc = FakeProcess(stdout=b"", stderr=b"cat: No such file or directory", returncode=1)
            else:
                proc = FakeProcess(stdout=self.files[path], stderr=b"", returncode=0)
        else:
            proc = FakeProcess(stdout="", stderr="", returncode=0)
        self.last_process = proc
        return proc

    def _mkdir(self, path, parents=False):
        self.mkdirs.append((path, parents))

    def _poll(self):
        if self.poll_exc is not None:
            raise self.poll_exc
        return self.poll_result

    def _detach(self):
        self.detached += 1

    def _terminate(self, *, wait=False):
        self.terminate_wait = wait
        self.terminated += 1
        if self.terminate_exc is not None:
            raise self.terminate_exc

    def _tunnels(self, timeout=50):
        return self.tunnel_map


class FakeTunnel:
    def __init__(self, url=None, tls_socket=None, tcp_socket=None):
        self.url = url
        self.tls_socket = tls_socket
        self.tcp_socket = tcp_socket


def _build_fake_modal(*, exec_script=None, create_exc=None):
    """Return a fake ``modal`` module plus the list of sandboxes it creates."""
    created: list[FakeSandbox] = []
    by_id: dict[str, FakeSandbox] = {}

    exc = types.SimpleNamespace()
    for name in (
        "Error",
        "NotFoundError",
        "AuthError",
        "InvalidError",
        "PermissionDeniedError",
        "ResourceExhaustedError",
        "ExecTimeoutError",
        "SandboxTimeoutError",
        "SandboxTerminatedError",
        "SandboxFilesystemError",
        "SandboxFilesystemFileTooLargeError",
        "SandboxFilesystemNotFoundError",
        "ConnectionError",
        "InternalError",
        "ServiceError",
    ):
        setattr(exc, name, type(name, (Exception,), {}))

    class Sandbox:
        @staticmethod
        def _create(*entrypoint, **kwargs):
            if create_exc is not None:
                raise create_exc
            sb = FakeSandbox(entrypoint, kwargs, exec_script=exec_script)
            created.append(sb)
            by_id[sb.object_id] = sb
            return sb

        @staticmethod
        def _from_id(sandbox_id):
            if sandbox_id not in by_id:
                raise exc.NotFoundError(sandbox_id)
            return by_id[sandbox_id]

    Sandbox.create = _Aio(Sandbox._create)
    Sandbox.from_id = _Aio(Sandbox._from_id)

    class App:
        lookups: list[tuple[str, dict[str, Any]]] = []

        @staticmethod
        def _lookup(name, **kwargs):
            App.lookups.append((name, kwargs))
            return types.SimpleNamespace(name=name)

    App.lookup = _Aio(App._lookup)

    class Image:
        calls: list[tuple[str, dict[str, Any]]] = []

        @staticmethod
        def from_registry(tag, **kwargs):
            Image.calls.append((tag, kwargs))
            return types.SimpleNamespace(tag=tag, kwargs=kwargs)

    class Secret:
        @staticmethod
        def from_name(name, **kwargs):
            return types.SimpleNamespace(kind="secret", name=name, kwargs=kwargs)

    class Volume:
        @staticmethod
        def from_name(name, **kwargs):
            return types.SimpleNamespace(kind="volume", name=name, kwargs=kwargs)

    modal = types.SimpleNamespace(
        Sandbox=Sandbox,
        App=App,
        Image=Image,
        Secret=Secret,
        Volume=Volume,
        exception=exc,
        __version__="1.5.5-fake",
    )
    return modal, created


@pytest.fixture
def fake_modal(monkeypatch):
    """Install a fake SDK behind the provider's ``_require_modal`` seam."""

    def _install(**kwargs):
        modal, created = _build_fake_modal(**kwargs)
        monkeypatch.setattr(modal_provider, "_require_modal", lambda: modal)
        return modal, created

    return _install


def _spec(**overrides) -> SandboxSpec:
    base: dict[str, Any] = {"image": "ghcr.io/acme/task:1.0"}
    base.update(overrides)
    return SandboxSpec(**base)


def _provider(**kwargs) -> ModalProvider:
    # Fast probes in tests.
    probe = {"stable_delay_s": 0.0, "deadline_s": 2.0, "timeout_s": 1.0}
    probe.update(kwargs.pop("probe", {}))
    ops = {"retries": 0, "retry_delay_s": 0.0}
    ops.update(kwargs.pop("operations", {}))
    return ModalProvider(probe=probe, operations=ops, **kwargs)


# --------------------------------------------------------------------------
# protocol + registry
# --------------------------------------------------------------------------


class TestProtocolConformance:
    def test_implements_every_sandbox_provider_method_with_matching_signature(self):
        for name, proto_member in inspect.getmembers(SandboxProvider, inspect.isfunction):
            if name.startswith("_"):
                continue
            impl = getattr(ModalProvider, name, None)
            assert impl is not None, f"ModalProvider is missing {name}()"
            assert inspect.iscoroutinefunction(impl), f"{name}() must be async"
            proto_params = list(inspect.signature(proto_member).parameters)
            impl_params = list(inspect.signature(impl).parameters)
            assert impl_params == proto_params, f"{name}() signature drift: {impl_params} != {proto_params}"
        assert ModalProvider.name == "modal"

    def test_optional_capabilities_are_detected_by_runtime_checks(self):
        provider = ModalProvider()
        assert isinstance(provider, SupportsSandboxEndpoint)
        assert isinstance(provider, ConnectableProvider)


class TestRegistry:
    def test_builtin_registry_makes_modal_available_by_name(self):
        assert "modal" in list_providers()
        assert get_provider_class("modal") is ModalProvider

    def test_single_key_config_instantiates_provider(self):
        provider = create_provider({"modal": {"connection": {"app_name": "eval-sandboxes"}}})
        assert isinstance(provider, ModalProvider)
        assert provider._connection.app_name == "eval-sandboxes"

    def test_shipped_yaml_parses_into_the_constructor(self):
        import yaml

        cfg_path = Path(modal_provider.__file__).parent / "configs" / "modal.yaml"
        raw = yaml.safe_load(cfg_path.read_text())
        provider_cfg = raw["sandbox"]["modal"]
        # OmegaConf interpolations are resolved by Gym, not by us; substitute for this test.
        provider_cfg["connection"]["app_name"] = "nemo-gym-sandboxes"
        provider_cfg["connection"]["environment_name"] = None
        provider = ModalProvider(**provider_cfg)
        assert provider._create.timeout_s == 3600.0
        assert provider._probe.command == "printf ok"
        assert raw["sandbox"]["default_metadata"] == {"sandbox-api": "modal"}


class TestConfigValidation:
    def test_unknown_keys_are_rejected_per_section(self):
        with pytest.raises(ValueError, match="Unknown ModalCreateConfig keys: nope"):
            ModalProvider(create={"nope": 1})
        with pytest.raises(ValueError, match="Unknown ModalExecConfig keys"):
            ModalProvider(exec={"timeout": 1})

    def test_port_mode_and_numbers_are_validated(self):
        with pytest.raises(ValueError, match="port_mode"):
            ModalProvider(create={"port_mode": "plaintext"})
        with pytest.raises(ValueError, match="timeout_s must be > 0"):
            ModalProvider(create={"timeout_s": 0})
        with pytest.raises(ValueError, match="retries must be >= 0"):
            ModalProvider(operations={"retries": -1})


# --------------------------------------------------------------------------
# create
# --------------------------------------------------------------------------


class TestCreate:
    async def test_requires_an_image(self, fake_modal):
        fake_modal()
        with pytest.raises(ModalCreateError, match="image is required"):
            await _provider().create(SandboxSpec())

    async def test_maps_spec_onto_sandbox_create_kwargs(self, fake_modal):
        modal, created = fake_modal()
        provider = _provider(create={"secrets": ["shared-env"], "default_tags": {"team": "eval"}})
        spec = _spec(
            ttl_s=90.5,
            workdir="/work",
            env={"A": "1"},
            metadata={"run": "r1"},
            resources={"cpu": 2, "memory_mib": 4096, "gpu": 2, "gpu_type": "A100"},
            ports=[8080],
            provider_options={
                "secrets": ["hf-token"],
                "volumes": {"/data": "eval-data"},
                "tags": {"extra": "x"},
                "region": ["us-east"],
            },
        )
        handle = await provider.create(spec)

        assert handle.provider_name == "modal"
        assert handle.sandbox_id == created[0].object_id
        sb = created[0]
        kw = sb.create_kwargs
        assert kw["image"].tag == "ghcr.io/acme/task:1.0"
        assert modal.Image.calls[-1] == ("ghcr.io/acme/task:1.0", {})
        assert kw["timeout"] == 91  # ceil(ttl_s)
        assert kw["workdir"] == "/work"
        assert kw["env"] == {"A": "1"}
        assert kw["cpu"] == 2.0
        assert kw["memory"] == 4096
        assert kw["gpu"] == "A100:2"
        assert kw["region"] == ["us-east"]
        assert kw["block_network"] is False
        assert kw["encrypted_ports"] == [8080]
        assert [s.name for s in kw["secrets"]] == ["shared-env", "hf-token"]
        assert kw["volumes"]["/data"].name == "eval-data"
        assert kw["tags"] == {"team": "eval", "run": "r1", "extra": "x"}
        assert kw["app"].name == "nemo-gym-sandboxes"
        # No entrypoint in spec -> POSIX keep-alive loop, not Modal's default.
        assert sb.entrypoint == ("/bin/sh", "-c", "while :; do sleep 3600; done")
        # Probe ran through exec and passed.
        assert any("printf ok" in " ".join(argv) for argv, _ in sb.exec_calls)

    async def test_entrypoint_overrides_keepalive_and_port_mode_is_honoured(self, fake_modal):
        _, created = fake_modal()
        provider = _provider(create={"port_mode": "encrypted"}, probe={"command": None})
        await provider.create(_spec(entrypoint=["python", "-m", "http.server", "8000"], ports=[8000]))
        sb = created[0]
        assert sb.entrypoint == ("python", "-m", "http.server", "8000")
        assert sb.create_kwargs["encrypted_ports"] == [8000]
        assert sb.exec_calls == []  # probe disabled

    async def test_gpu_without_type_uses_default_and_explicit_option_wins(self, fake_modal):
        _, created = fake_modal()
        provider = _provider(probe={"command": None})
        await provider.create(_spec(resources={"gpu": 1}))
        assert created[0].create_kwargs["gpu"] == "any"
        await provider.create(_spec(resources={"gpu": 1}, provider_options={"gpu": "H100:4"}))
        assert created[1].create_kwargs["gpu"] == "H100:4"

    async def test_image_secret_resolves_registry_credentials(self, fake_modal):
        modal, created = fake_modal()
        provider = _provider(probe={"command": None})
        await provider.create(_spec(provider_options={"image_secret": "ghcr-creds"}))
        tag, kwargs = modal.Image.calls[-1]
        assert tag == "ghcr.io/acme/task:1.0"
        assert kwargs["secret"].name == "ghcr-creds"

    async def test_unknown_provider_option_is_rejected_before_any_sandbox_exists(self, fake_modal):
        _, created = fake_modal()
        with pytest.raises(ValueError, match="Unknown modal provider option"):
            await _provider().create(_spec(provider_options={"template": "x"}))
        assert created == []

    async def test_disk_gib_warns_unless_strict(self, fake_modal, caplog):
        _, created = fake_modal()
        with caplog.at_level("WARNING"):
            await _provider(probe={"command": None}).create(_spec(resources={"disk_gib": 50}))
        assert "disk_gib=50" in caplog.text
        assert len(created) == 1
        with pytest.raises(ModalCreateError, match="disk_gib"):
            await _provider(create={"strict_resources": True}).create(_spec(resources={"disk_gib": 50}))

    async def test_create_failure_is_wrapped_and_never_retried(self, fake_modal):
        modal, created = fake_modal()
        boom = modal.exception.ServiceError("region full")
        modal, created = fake_modal(create_exc=boom)
        provider = _provider(operations={"retries": 3})
        with pytest.raises(ModalCreateError, match="region full"):
            await provider.create(_spec())
        assert created == []

    async def test_failed_probe_terminates_sandbox_and_raises_verification_error(self, fake_modal):
        _, created = fake_modal(exec_script={"printf ok": FakeProcess(stdout="", stderr="not yet", returncode=1)})
        provider = _provider(probe={"deadline_s": 0.05, "stable_delay_s": 0.01})
        with pytest.raises(ModalCreateVerificationError, match="did not pass readiness probe"):
            await provider.create(_spec())
        assert created[0].terminated == 1

    async def test_verification_error_is_the_public_subclass(self):
        assert issubclass(ModalCreateVerificationError, SandboxCreateVerificationError)
        assert issubclass(ModalCreateVerificationError, ModalCreateError)

    async def test_ready_timeout_overrides_probe_deadline(self, fake_modal):
        _, created = fake_modal(exec_script={"printf ok": FakeProcess(stdout="", returncode=1)})
        provider = _provider(probe={"deadline_s": 60.0, "stable_delay_s": 0.01})
        with pytest.raises(ModalCreateVerificationError, match="within 0.05s"):
            await provider.create(_spec(ready_timeout_s=0.05))

    async def test_app_is_looked_up_once_per_provider(self, fake_modal):
        modal, _ = fake_modal()
        provider = _provider(probe={"command": None}, connection={"app_name": "x", "environment_name": "dev"})
        modal.App.lookups.clear()
        await provider.create(_spec())
        await provider.create(_spec())
        assert modal.App.lookups == [("x", {"environment_name": "dev", "create_if_missing": False})]

    async def test_missing_app_is_created_only_after_lookup_reports_not_found(self, fake_modal):
        modal, _ = fake_modal()
        original_lookup = modal.App.lookup

        async def lookup(name, **kwargs):
            if not kwargs["create_if_missing"]:
                modal.App.lookups.append((name, kwargs))
                raise modal.exception.NotFoundError(name)
            return await original_lookup.aio(name, **kwargs)

        modal.App.lookup = types.SimpleNamespace(aio=lookup)
        provider = _provider(probe={"command": None}, connection={"app_name": "x", "environment_name": "dev"})
        modal.App.lookups.clear()

        await provider.create(_spec())

        assert modal.App.lookups == [
            ("x", {"environment_name": "dev", "create_if_missing": False}),
            ("x", {"environment_name": "dev", "create_if_missing": True}),
        ]


# --------------------------------------------------------------------------
# exec
# --------------------------------------------------------------------------


class TestExec:
    async def _started(self, fake_modal, **provider_kwargs):
        modal, created = fake_modal()
        provider = _provider(probe={"command": None}, **provider_kwargs)
        handle = await provider.create(_spec())
        return provider, handle, created[0], modal

    async def test_wraps_command_in_shell_and_forwards_cwd_env_timeout(self, fake_modal):
        provider, handle, sb, _ = await self._started(fake_modal)
        sb.exec_script = {"echo hi": FakeProcess(stdout="hi\n", stderr="", returncode=0)}
        result = await provider.exec(handle, "echo hi", cwd="/tmp", env={"X": "1"}, timeout_s=12.2)
        argv, kwargs = sb.exec_calls[-1]
        assert argv == ("/bin/sh", "-c", "echo hi")
        assert kwargs == {"workdir": "/tmp", "env": {"X": "1"}, "timeout": 13, "text": False}
        assert result.stdout == "hi\n" and result.return_code == 0 and result.error_type is None

    async def test_default_timeout_matches_the_gym_facade(self, fake_modal):
        provider, handle, sb, _ = await self._started(fake_modal)
        await provider.exec(handle, "true")
        assert sb.exec_calls[-1][1]["timeout"] == 180

    async def test_nonzero_exit_is_a_result_not_an_error(self, fake_modal):
        provider, handle, sb, _ = await self._started(fake_modal)
        sb.exec_script = {"false": FakeProcess(stdout="", stderr="bad", returncode=3)}
        result = await provider.exec(handle, "false")
        assert result.return_code == 3 and result.stderr == "bad"

    async def test_exec_timeout_error_becomes_timeout_error(self, fake_modal):
        provider, handle, sb, modal = await self._started(fake_modal)
        sb.exec_script = {"sleep": FakeProcess(raise_on_wait=modal.exception.ExecTimeoutError("deadline"))}
        with pytest.raises(TimeoutError, match="timed out after 1s"):
            await provider.exec(handle, "sleep 5", timeout_s=1)

    async def test_minus_one_after_deadline_is_a_timeout_but_early_minus_one_is_unexpected_exit(self, fake_modal):
        provider, handle, sb, _ = await self._started(fake_modal)
        # Modal reports -1 instead of raising on timeout; elapsed ~= deadline -> TimeoutError.
        sb.exec_script = {"slow": FakeProcess(returncode=-1, wait_delay=1.0)}
        with pytest.raises(TimeoutError):
            await provider.exec(handle, "slow", timeout_s=1)
        # The pinned SDK sentinel is a timeout regardless of observed client time.
        sb.exec_script = {"weird": FakeProcess(returncode=-1)}
        with pytest.raises(TimeoutError):
            await provider.exec(handle, "weird", timeout_s=30)

    async def test_sandbox_terminated_mid_command_is_reported_without_exit_code(self, fake_modal):
        provider, handle, sb, modal = await self._started(fake_modal)
        sb.exec_script = {"work": FakeProcess(raise_on_wait=modal.exception.SandboxTerminatedError("gone"))}
        result = await provider.exec(handle, "work")
        assert result.error_type == "sandbox_terminated" and result.return_code == -1

    async def test_user_requires_opt_in_rewrite(self, fake_modal):
        provider, handle, sb, _ = await self._started(fake_modal)
        with pytest.raises(ValueError, match="allow_user_rewrite"):
            await provider.exec(handle, "id", user="agent")
        provider2, handle2, sb2, _ = await self._started(fake_modal, exec={"allow_user_rewrite": True})
        await provider2.exec(handle2, "id -u", user="agent")
        assert sb2.exec_calls[-1][0] == ("su", "-s", "/bin/sh", "-c", "id -u", "--", "agent")

    async def test_relative_cwd_is_rejected_like_modal_would(self, fake_modal):
        provider, handle, _, _ = await self._started(fake_modal)
        with pytest.raises(ValueError, match="absolute"):
            await provider.exec(handle, "pwd", cwd="relative/dir")


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


class TestFiles:
    async def _started(self, fake_modal, **provider_kwargs):
        _, created = fake_modal()
        provider = _provider(probe={"command": None}, **provider_kwargs)
        handle = await provider.create(_spec())
        return provider, handle, created[0]

    async def test_file_copies_use_the_streaming_sdk_api(self, fake_modal, tmp_path):
        provider, handle, sb = await self._started(fake_modal)
        src = tmp_path / "in.bin"
        src.write_bytes(b"\x00\xffpayload")
        await provider.upload_file(handle, src, "/work/data/in.bin")
        dest = tmp_path / "nested" / "out.bin"
        await provider.download_file(handle, "/work/data/in.bin", dest)
        assert dest.read_bytes() == src.read_bytes()
        assert ("upload", src, "/work/data/in.bin") in sb.filesystem.calls
        assert ("download", "/work/data/in.bin", dest) in sb.filesystem.calls
        assert not sb.exec_calls

    async def test_remote_missing_file_maps_to_file_not_found(self, fake_modal, tmp_path):
        modal, created = fake_modal()
        provider = _provider(probe={"command": None})
        handle = await provider.create(_spec())
        created[0].filesystem.error = modal.exception.SandboxFilesystemNotFoundError("missing")
        with pytest.raises(FileNotFoundError):
            await provider.download_file(handle, "/missing", tmp_path / "out")

    async def test_permission_error_is_not_retried_or_converted_to_shell_exec(self, fake_modal):
        modal, created = fake_modal()
        provider = _provider(probe={"command": None})
        handle = await provider.create(_spec())
        created[0].filesystem.error = modal.exception.SandboxFilesystemError("permission denied")
        with pytest.raises(modal.exception.SandboxFilesystemError, match="permission denied"):
            await provider.write_file(handle, "/protected", "hello")
        assert not created[0].exec_calls


# --------------------------------------------------------------------------
# status / close / endpoint / connect
# --------------------------------------------------------------------------


class TestLifecycle:
    async def test_status_maps_poll_results(self, fake_modal):
        modal, created = fake_modal()
        provider = _provider(probe={"command": None})
        handle = await provider.create(_spec())
        sb = created[0]
        assert await provider.status(handle) is SandboxStatus.RUNNING
        sb.poll_result = 0
        assert await provider.status(handle) is SandboxStatus.STOPPED
        sb.poll_result = None
        sb.poll_exc = modal.exception.NotFoundError("gone")
        assert await provider.status(handle) is SandboxStatus.STOPPED
        sb.poll_exc = modal.exception.ConnectionError("blip")
        assert await provider.status(handle) is SandboxStatus.UNKNOWN

    async def test_close_terminates_once_and_is_idempotent(self, fake_modal):
        modal, created = fake_modal()
        provider = _provider(probe={"command": None})
        handle = await provider.create(_spec())
        sb = created[0]
        await provider.close(handle)
        await provider.close(handle)
        assert sb.terminated == 1 and handle.raw is None
        assert sb.terminate_wait is True and sb.detached == 1
        assert await provider.status(handle) is SandboxStatus.STOPPED

    async def test_close_swallows_already_gone(self, fake_modal):
        modal, created = fake_modal()
        provider = _provider(probe={"command": None})
        handle = await provider.create(_spec())
        created[0].terminate_exc = modal.exception.SandboxTimeoutError("expired")
        await provider.close(handle)
        assert handle.raw is None

    async def test_endpoint_requires_declared_port_and_returns_tunnel_url(self, fake_modal):
        _, created = fake_modal()
        provider = _provider(probe={"command": None})
        handle = await provider.create(_spec(ports=[8080, 9090]))
        sb = created[0]
        sb.tunnel_map = {
            8080: FakeTunnel(url="https://sb-abc-8080.modal.host"),
            9090: FakeTunnel(url="https://sb-abc.modal.host:44311"),
        }
        assert (await provider.endpoint(handle, 8080)).endpoint == "https://sb-abc-8080.modal.host"
        assert (await provider.endpoint(handle, 9090)).endpoint == "https://sb-abc.modal.host:44311"
        with pytest.raises(ValueError, match="not declared"):
            await provider.endpoint(handle, 7000)

    async def test_serialize_and_connect_round_trip(self, fake_modal):
        _, created = fake_modal()
        provider = _provider(probe={"command": None})
        handle = await provider.create(_spec(ports=[8080]))
        descriptor = await provider.serialize_handle(handle)
        assert descriptor == {
            "sandbox_id": created[0].object_id,
            "ports": [8080],
            "port_mode": "encrypted",
            "image": "ghcr.io/acme/task:1.0",
        }
        other = ModalProvider()
        rebuilt = await other.connect(descriptor)
        assert rebuilt.sandbox_id == handle.sandbox_id
        assert rebuilt.raw.sandbox is created[0]
        assert (await other.status(rebuilt)) is SandboxStatus.RUNNING


# --------------------------------------------------------------------------
# through the real NeMo Gym facade
# --------------------------------------------------------------------------


class TestFacadeIntegration:
    async def test_async_sandbox_start_uploads_spec_files_then_exec_then_stop(self, fake_modal):
        _, created = fake_modal()
        provider = _provider(probe={"command": None})
        spec = _spec(files={"/work/task.md": "# task\n", "/work/run.sh": "echo go\n"}, workdir="/work")
        async with await AsyncSandbox(provider, spec).start() as sandbox:
            sb = created[0]
            assert sb.files["/work/task.md"] == b"# task\n"
            assert sb.files["/work/run.sh"] == b"echo go\n"
            sb.exec_script = {"echo go": FakeProcess(stdout="go\n")}
            result = await sandbox.exec("echo go")
            assert result.stdout == "go\n"
            assert await sandbox.status() is SandboxStatus.RUNNING
        assert created[0].terminated == 1

    async def test_facade_cleans_up_when_a_spec_file_upload_fails(self, fake_modal):
        modal, created = fake_modal()
        provider = _provider(probe={"command": None})

        sandbox = AsyncSandbox(provider, _spec(files={"/x": "y"}))
        original_write = provider.upload_file

        async def failing_write(handle, target_path, data):
            raise modal.exception.SandboxFilesystemError("disk on fire")

        provider.upload_file = failing_write  # type: ignore[method-assign]
        with pytest.raises(modal.exception.SandboxFilesystemError):
            await sandbox.start()
        provider.upload_file = original_write  # type: ignore[method-assign]
        assert created[0].terminated == 1

    async def test_facade_resolves_provider_from_single_key_mapping(self, fake_modal):
        _, created = fake_modal()
        sandbox = AsyncSandbox({"modal": {"probe": {"command": None}, "operations": {"retries": 0}}}, _spec())
        await sandbox.start()
        assert isinstance(sandbox._provider, ModalProvider)
        await sandbox.stop()
        assert created[0].terminated == 1
