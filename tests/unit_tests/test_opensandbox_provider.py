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

import asyncio
import builtins
import logging
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec, SandboxStatus


pytestmark = pytest.mark.sandbox


pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")

from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


TEST_REGISTRY_PASSWORD = "secret"  # pragma: allowlist secret


@dataclass(frozen=True)
class FakePlatformSpec:
    os: str
    arch: str


class FakeConnectionConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@dataclass(frozen=True)
class FakeVolume:
    name: str


class FakeSandbox:
    created_kwargs: dict[str, Any] = {}
    connected_args: tuple[Any, ...] = ()
    connected_kwargs: dict[str, Any] = {}

    def __init__(self, sandbox_id: str = "sandbox-1") -> None:
        self.id = sandbox_id

    @classmethod
    async def create(cls, *_args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.created_kwargs = kwargs
        return cls()

    @classmethod
    async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.connected_args = args
        cls.connected_kwargs = kwargs
        return cls()


@pytest.fixture
def fake_opensandbox_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    def require_sdk() -> tuple[Any, Any, Any, Any, Any]:
        return FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object

    monkeypatch.setattr(opensandbox_provider, "_require_opensandbox_sdk", require_sdk)


def test_sdk_info_logs_are_silenced(caplog: pytest.LogCaptureFixture) -> None:
    sdk_logger = logging.getLogger("opensandbox.sandbox")

    with caplog.at_level(logging.INFO):
        sdk_logger.info("SDK info")
        sdk_logger.warning("SDK warning")

    sdk_messages = [record.message for record in caplog.records if record.name == sdk_logger.name]
    assert sdk_messages == ["SDK warning"]


def test_sdk_import_helpers_and_retry_classification() -> None:
    assert len(opensandbox_provider._require_opensandbox_sdk()) == 5
    assert len(opensandbox_provider._require_tenacity()) == 4

    class StatusCodeError(Exception):
        status_code = 429

    assert opensandbox_provider._exception_status_code(StatusCodeError("rate limited")) == 429
    assert opensandbox_provider._is_retryable_create_error(
        opensandbox_provider.OpenSandboxCreateError("create failed")
    )

    from opensandbox.exceptions import (  # noqa: PLC0415
        InvalidArgumentException,
        SandboxApiException,
        SandboxException,
        SandboxInternalException,
    )

    assert opensandbox_provider._is_retryable_create_error(InvalidArgumentException("bad input")) is False
    assert opensandbox_provider._is_retryable_create_error(SandboxInternalException("server failed")) is True

    retryable_api_error = SandboxApiException("busy")
    retryable_api_error.status_code = 503
    assert opensandbox_provider._is_retryable_create_error(retryable_api_error) is True

    nonretryable_api_error = SandboxApiException("not found")
    nonretryable_api_error.status_code = 404
    assert opensandbox_provider._is_retryable_create_error(nonretryable_api_error) is False
    assert opensandbox_provider._is_retryable_create_error(SandboxException("gateway timeout")) is True

    retry_state = SimpleNamespace(
        outcome=SimpleNamespace(exception=lambda: RuntimeError("temporary")),
        next_action=SimpleNamespace(sleep=0.5),
        attempt_number=2,
    )
    opensandbox_provider._log_create_retry(retry_state)


def test_missing_optional_dependency_import_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def block_imports(*blocked_names: str) -> None:
        def fake_import(
            name: str,
            globals_: dict[str, Any] | None = None,
            locals_: dict[str, Any] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> Any:
            if any(name == blocked or name.startswith(f"{blocked}.") for blocked in blocked_names):
                raise ModuleNotFoundError(name)
            return real_import(name, globals_, locals_, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)

    block_imports("opensandbox")
    with pytest.raises(ModuleNotFoundError, match="OpenSandbox SDK is required"):
        opensandbox_provider._require_opensandbox_sdk()

    block_imports("tenacity")
    with pytest.raises(ModuleNotFoundError, match="tenacity is required"):
        opensandbox_provider._require_tenacity()

    block_imports("opensandbox.exceptions")
    assert opensandbox_provider._is_retryable_create_error(RuntimeError("gateway timeout")) is True


async def test_provider_conversion_helpers(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection_config = opensandbox_provider.OpenSandboxConnectionConfig(domain="sandbox.example")
    assert (
        opensandbox_provider._coerce_config(connection_config, opensandbox_provider.OpenSandboxConnectionConfig)
        is connection_config
    )

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, object, FakePlatformSpec, FakeVolume),
    )
    assert opensandbox_provider._to_volumes([{"name": "workspace"}]) == [FakeVolume(name="workspace")]


async def test_direct_create_passes_platform_to_sdk_create(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
    )

    handle = await provider.create(
        SandboxSpec(
            image="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
            provider_options={"platform": {"os": "linux", "arch": "amd64"}},
        ),
    )

    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["platform"] == FakePlatformSpec(
        os="linux",
        arch="amd64",
    )


async def test_direct_create_passes_resource_requests_to_sdk_create(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    await provider.create(
        SandboxSpec(
            image="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
            resources={"cpu": 1, "memory_mib": 8192, "disk_gib": 30},
            provider_options={"resource_requests": {"cpu": 0.5, "memory_mib": 2048, "disk_gib": 30}},
        ),
    )

    assert FakeSandbox.created_kwargs["resource"] == {"cpu": "1", "memory": "8192Mi", "ephemeral-storage": "30Gi"}
    assert FakeSandbox.created_kwargs["resource_requests"] == {
        "cpu": "0.5",
        "memory": "2048Mi",
        "ephemeral-storage": "30Gi",
    }

    with pytest.raises(TypeError, match="'resource_requests' must be a mapping"):
        opensandbox_provider.OpenSandboxProviderOptions.from_mapping({"resource_requests": "big"})

    with pytest.raises(ValueError, match="Unknown sandbox resource keys"):
        await provider.create(
            SandboxSpec(
                image="mirror.gcr.io/astral/uv:python3.12-bookworm-slim",
                provider_options={"resource_requests": {"memory_gib": 2}},
            ),
        )


async def test_direct_create_passes_image_auth_to_sdk_create(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    await provider.create(
        SandboxSpec(
            image="registry.example/repo:tag",
            provider_options={"image_auth": {"username": "user", "password": TEST_REGISTRY_PASSWORD}},
        )
    )

    image = FakeSandbox.created_kwargs["image"]
    assert image.image == "registry.example/repo:tag"
    assert image.auth.username == "user"
    assert image.auth.password == TEST_REGISTRY_PASSWORD


async def test_pool_create_uses_sdk_compatibility_image_and_proxy_auth(
    fake_opensandbox_sdk: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "http://sandbox.example/",
            "api_key": "pool-api-key",  # pragma: allowlist secret
            "request_timeout_s": 30,
            "use_server_proxy": True,
        },
        create={
            "request_timeout_s": 120,
            "timeout_s": 30,
        },
        probe={"command": None},
    )
    handle = await provider.create(
        SandboxSpec(
            image="busybox:1.36",
            ttl_s=1800,
            metadata={"purpose": "osworld"},
            provider_options={
                "skip_health_check": True,
                "extensions": {"poolRef": "osworld-kvm"},
            },
        )
    )

    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["image"] == "busybox:1.36"
    assert FakeSandbox.created_kwargs["timeout"] == timedelta(seconds=1800)
    assert FakeSandbox.created_kwargs["extensions"]["poolRef"] == "osworld-kvm"
    assert FakeSandbox.created_kwargs["metadata"]["purpose"] == "osworld"
    assert FakeSandbox.created_kwargs["skip_health_check"] is True
    create_connection = FakeSandbox.created_kwargs["connection_config"]
    assert create_connection.kwargs["domain"] == "http://sandbox.example"
    assert create_connection.kwargs["headers"] == {
        "OPEN-SANDBOX-API-KEY": "pool-api-key"  # pragma: allowlist secret
    }
    assert FakeSandbox.connected_args == ()


async def test_endpoint_normalizes_missing_scheme_and_merges_sdk_headers() -> None:
    class FakeRaw:
        connection_config = SimpleNamespace(
            get_base_url=lambda: "https://sandbox.example/v1",
            headers={
                "OPEN-SANDBOX-API-KEY": "pool-api-key",  # pragma: allowlist secret
                "X-Shared": "connection",
            },
        )

        async def get_endpoint(self, port: int) -> Any:
            assert port == 5000
            return SimpleNamespace(
                endpoint="10.0.0.22:5000",
                headers={"X-Route": "sandbox", "X-Shared": "endpoint"},
            )

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "https://sandbox.example/",
            "api_key": "pool-api-key",  # pragma: allowlist secret
            "use_server_proxy": True,
        },
        operations={"retries": 0},
        probe={"command": None},
    )
    resolved = await provider.endpoint(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=FakeRaw(),
        ),
        5000,
    )

    assert resolved.endpoint == "https://10.0.0.22:5000"
    assert resolved.headers == {
        "OPEN-SANDBOX-API-KEY": "pool-api-key",  # pragma: allowlist secret
        "X-Shared": "endpoint",
        "X-Route": "sandbox",
    }


async def test_endpoint_uses_effective_sdk_scheme_when_provider_input_is_unset() -> None:
    class FakeRaw:
        connection_config = SimpleNamespace(
            get_base_url=lambda: "https://gateway.example/v1",
            headers={},
        )

        async def get_endpoint(self, _port: int) -> Any:
            return SimpleNamespace(endpoint="sandbox.example:5000", headers={})

    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 0},
        probe={"command": None},
    )
    resolved = await provider.endpoint(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=FakeRaw(),
        ),
        5000,
    )

    assert resolved.endpoint == "https://sandbox.example:5000"


async def test_direct_endpoint_never_receives_management_api_key() -> None:
    class FakeRaw:
        connection_config = SimpleNamespace(headers={})

        async def get_endpoint(self, _port: int) -> Any:
            return SimpleNamespace(endpoint="http://10.0.0.22:5000", headers={})

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "api_key": "pool-api-key",  # pragma: allowlist secret
            "use_server_proxy": False,
        },
        operations={"retries": 0},
        probe={"command": None},
    )
    resolved = await provider.endpoint(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=FakeRaw(),
        ),
        5000,
    )

    assert resolved.headers == {}


def test_provider_validation_and_retry_helpers() -> None:
    with pytest.raises(ValueError, match="image_pull_policy"):
        opensandbox_provider.validate_image_pull_policy("Sometimes")
    with pytest.raises(TypeError, match="extensions"):
        opensandbox_provider.OpenSandboxProviderOptions.from_mapping({"extensions": ["not", "a", "mapping"]})
    with pytest.raises(TypeError, match="must be a bool"):
        opensandbox_provider.OpenSandboxProviderOptions.from_mapping({"skip_health_check": "true"})

    assert opensandbox_provider._resource_map(SandboxResources(cpu=2.0))["cpu"] == "2"
    assert opensandbox_provider._to_sandbox_status("starting") == SandboxStatus.STARTING
    assert opensandbox_provider._to_sandbox_status("terminated") == SandboxStatus.STOPPED
    assert opensandbox_provider._to_sandbox_status("failed") == SandboxStatus.ERROR
    assert opensandbox_provider._to_sandbox_status(None) == SandboxStatus.UNKNOWN

    invalid_kwargs = [
        {"create": {"timeout_s": 0}},
        {"probe": {"timeout_s": 0}},
        {"probe": {"deadline_s": 0}},
        {"probe": {"stable_count": 0}},
        {"probe": {"stable_delay_s": -1}},
        {"create": {"retries": -1}},
        {"create": {"retry_delay_s": -1}},
        {"create": {"retry_max_delay_s": -1}},
        {"operations": {"retries": -1}},
        {"operations": {"retry_delay_s": -1}},
        {"operations": {"retry_max_delay_s": -1}},
        {"operations": {"command_retries": -1}},
        {"operations": {"close_timeout_s": 0}},
        {"create": {"connect_attempt_timeout_s": 0}},
        {"create": {"connect_poll_s": 0}},
        {"create": {"image_pull_policy": "Sometimes"}},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            opensandbox_provider.OpenSandboxProvider(**kwargs)
    with pytest.raises(TypeError):
        opensandbox_provider.OpenSandboxProvider(**{"batch_" + "create_retries": 1})
    with pytest.raises(TypeError):
        opensandbox_provider.OpenSandboxProvider(connection=object())

    assert opensandbox_provider._exception_status_code(RuntimeError("HTTP status code: 503")) == 503
    assert opensandbox_provider._exception_status_code(RuntimeError("plain error")) is None
    attrs = opensandbox_provider._sdk_error_attributes(
        RuntimeError("HTTP 502 bad gateway"),
        operation="exec",
        sandbox_id="sandbox-1",
        attempt_number=2,
        max_attempts=3,
        sleep_s=0.5,
    )
    assert attrs["status_code"] == 502
    assert attrs["attempt_number"] == 2
    assert attrs["next_sleep_s"] == 0.5


def test_provider_options_from_mapping() -> None:
    options_cls = opensandbox_provider.OpenSandboxProviderOptions

    assert options_cls.from_mapping(None) == options_cls()

    parsed = options_cls.from_mapping(
        {
            "image_auth": {"username": "user", "password": TEST_REGISTRY_PASSWORD},
            "platform": {"os": "linux", "arch": "amd64"},
            "snapshot_id": "snap-1",
            "volumes": [{"name": "workspace"}],
            "skip_health_check": True,
            "extensions": {"imagePullPolicy": "Never"},
        }
    )
    assert parsed.image_auth == {"username": "user", "password": TEST_REGISTRY_PASSWORD}
    assert parsed.platform == {"os": "linux", "arch": "amd64"}
    assert parsed.snapshot_id == "snap-1"
    assert parsed.volumes == ({"name": "workspace"},)
    assert parsed.skip_health_check is True
    assert parsed.extensions == {"imagePullPolicy": "Never"}

    with pytest.raises(ValueError, match="Unknown OpenSandbox provider option"):
        options_cls.from_mapping({"bogus": 1})
    with pytest.raises(TypeError, match="provider_options must be a mapping"):
        options_cls.from_mapping(["not", "a", "mapping"])
    with pytest.raises(TypeError, match="'platform' must be a mapping"):
        options_cls.from_mapping({"platform": "linux/amd64"})
    with pytest.raises(TypeError, match="'image_auth' must be a mapping"):
        options_cls.from_mapping({"image_auth": "not-a-mapping"})
    with pytest.raises(TypeError, match="'snapshot_id' must be a string"):
        options_cls.from_mapping({"snapshot_id": 123})
    with pytest.raises(TypeError, match="'volumes' must be a list of mappings"):
        options_cls.from_mapping({"volumes": ["workspace"]})


def test_connection_config_and_image_policy(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "sandbox.example/",
            "api_key": "key",  # pragma: allowlist secret
            "protocol": "https",
            "request_timeout_s": 10,
            "use_server_proxy": True,
        }
    )

    config = provider._connection_config()
    transport = config.kwargs.pop("transport")
    assert isinstance(transport, httpx.AsyncBaseTransport)
    assert config.kwargs == {
        "domain": "sandbox.example",
        "api_key": "key",  # pragma: allowlist secret
        "protocol": "https",
        "request_timeout": timedelta(seconds=10),
        "use_server_proxy": True,
        # The API key must also travel as a header: the SDK's execd clients
        # (health ping, commands, files) send only ConnectionConfig.headers,
        # and proxied /proxy/* routes may enforce auth.
        "headers": {"OPEN-SANDBOX-API-KEY": "key"},  # pragma: allowlist secret
    }
    short_timeout_config = provider._connection_config(request_timeout_s=3)
    assert short_timeout_config.kwargs["request_timeout"] == timedelta(seconds=3)

    # Direct-endpoint mode must NOT carry the key: the sandbox runs untrusted
    # code and would be able to read it.
    direct = opensandbox_provider.OpenSandboxProvider(
        connection={"domain": "sandbox.example", "api_key": "key"}  # pragma: allowlist secret
    )
    assert "headers" not in direct._connection_config().kwargs


def test_connection_transport_backends(fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch) -> None:
    # Default backend is httpx, with the configured keepalive expiry on the pool.
    provider = opensandbox_provider.OpenSandboxProvider()
    transport = provider._build_transport()
    assert isinstance(transport, httpx.AsyncHTTPTransport)

    # Custom pool settings still produce an httpx transport.
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "transport_backend": "httpx",
            "keepalive_expiry_s": 2.5,
            "max_connections": 7,
            "max_keepalive_connections": 3,
            "connect_retries": 1,
        }
    )
    transport = provider._build_transport()
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    # connect_retries reaches the pool rather than silently falling back.
    assert transport._pool._retries == 1

    # aiohttp requested but httpx-aiohttp unavailable: falls back to httpx.
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(sys.modules, "httpx_aiohttp", None)
        provider = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "aiohttp"})
        transport = provider._build_transport()
        assert isinstance(transport, httpx.AsyncHTTPTransport)

    # keepalive_expiry_s=null disables transport injection entirely.
    provider = opensandbox_provider.OpenSandboxProvider(connection={"keepalive_expiry_s": None})
    config = provider._connection_config()
    assert "transport" not in config.kwargs

    # max_connections=null uncaps the pool; max_keepalive_connections=0 disables reuse.
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"max_connections": None, "max_keepalive_connections": 0}
    )
    transport = provider._build_transport()
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    assert transport._pool._max_connections > 2**32
    assert transport._pool._max_keepalive_connections == 0


async def test_connection_transport_is_shared_and_closed_by_provider(fake_opensandbox_sdk: None) -> None:
    # The SDK never closes a transport it did not create, so the provider owns
    # one: built on first use, reused by every ConnectionConfig rather than
    # leaking a pool per call, and closed in aclose().
    class FakeTransport:
        def __init__(self) -> None:
            self.aclosed = False

        async def aclose(self) -> None:
            self.aclosed = True

    provider = opensandbox_provider.OpenSandboxProvider()
    provider._build_transport = FakeTransport

    transport = provider._connection_config().kwargs["transport"]
    assert provider._connection_config().kwargs["transport"] is transport

    await provider.aclose()
    assert transport.aclosed
    assert provider._transport is None


def test_connection_transport_backend_aiohttp_opt_in(fake_opensandbox_sdk: None) -> None:
    # Opt-in aiohttp backend via the httpx-aiohttp bridge; the package is not a
    # declared dependency, so this coverage only runs where it is installed.
    httpx_aiohttp = pytest.importorskip("httpx_aiohttp", reason="optional httpx-aiohttp is not installed")
    provider = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "aiohttp"})
    transport = provider._build_transport()
    assert isinstance(transport, httpx_aiohttp.AiohttpTransport)
    assert transport.limits.keepalive_expiry == 3.0
    # Both backends honor connect_retries; the bridge default is 0, so this
    # would catch the option being dropped on the aiohttp path.
    assert transport.retries == 2

    extensions = provider._resolve_extensions({"imagePullPolicy": "Never"})
    assert extensions["imagePullPolicy"] == "Never"
    assert extensions["opensandbox.extensions.image-pull-policy"] == "Never"

    no_policy_provider = opensandbox_provider.OpenSandboxProvider(create={"image_pull_policy": None})
    assert no_policy_provider._resolve_extensions({"imagePullPolicy": "Never"}) == {"imagePullPolicy": "Never"}


def test_connection_config_disable_pooling_sets_fresh_transport(fake_opensandbox_sdk: None) -> None:
    import httpx

    # Default: a keepalive-bounded transport with connection reuse enabled.
    pooled = opensandbox_provider.OpenSandboxProvider(connection={"domain": "sandbox.example"})
    pooled_transport = pooled._connection_config().kwargs["transport"]
    assert isinstance(pooled_transport, httpx.AsyncHTTPTransport)
    assert pooled_transport._pool._max_keepalive_connections > 0

    # disable_connection_pooling -> same transport plumbing, but no reuse.
    fresh = opensandbox_provider.OpenSandboxProvider(
        connection={"domain": "sandbox.example", "disable_connection_pooling": True}
    )
    transport = fresh._connection_config().kwargs.get("transport")
    assert isinstance(transport, httpx.AsyncHTTPTransport)
    assert transport._pool._max_keepalive_connections == 0


async def test_exec_file_operations_and_reference_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeLog:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeCommands:
        def __init__(self) -> None:
            self.calls: list[tuple[str, FakeRunCommandOpts]] = []

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            self.calls.append((command, opts))
            if "fail" in command:
                return SimpleNamespace(
                    logs=SimpleNamespace(stdout=[], stderr=[FakeLog("stderr")]),
                    error=SimpleNamespace(name="CommandError", value="failed"),
                    exit_code=None,
                )
            return SimpleNamespace(
                logs=SimpleNamespace(stdout=[FakeLog("stdout")], stderr=[]),
                error=None,
                exit_code=None,
            )

    class FakeFiles:
        def __init__(self) -> None:
            self.writes: list[tuple[str, str | bytes]] = []

        async def write_file(self, target_path: str, data: str | bytes) -> None:
            self.writes.append((target_path, data))

        async def read_bytes(self, source_path: str) -> bytes:
            return f"bytes:{source_path}".encode()

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()
            self.files = FakeFiles()

        async def get_info(self) -> Any:
            return SimpleNamespace(status=SimpleNamespace(state="RUNNING"))

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
    )
    raw = FakeRaw()
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=raw)

    result = await provider.exec(
        handle,
        "echo hello",
        cwd="/repo",
        env={"A": "B"},
        timeout_s=2,
        user=1000,
    )
    assert result == opensandbox_provider.SandboxExecResult(stdout="stdout", stderr=None, return_code=0)
    command, opts = raw.commands.calls[0]
    assert command == "echo hello"
    assert opts.kwargs == {
        "working_directory": "/repo",
        "envs": {"A": "B"},
        "timeout": timedelta(seconds=2),
        "uid": 1000,
    }

    result = await provider.exec(handle, "fail", user="agent")
    assert result.return_code == 125
    assert result.error_type == "sandbox"
    assert result.stderr == "stderr\nCommandError: failed"
    assert raw.commands.calls[1][0] == "su -s /bin/sh -c fail agent"

    upload_path = tmp_path / "upload.txt"
    upload_path.write_text("upload", encoding="utf-8")
    await provider.upload_file(handle, upload_path, "/remote/upload.txt")
    download_path = tmp_path / "nested" / "download.txt"
    await provider.download_file(handle, "/remote/download.txt", download_path)
    assert raw.files.writes == [("/remote/upload.txt", b"upload")]
    assert download_path.read_bytes() == b"bytes:/remote/download.txt"
    assert await provider.status(handle) == SandboxStatus.RUNNING
    bare_handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-2", provider_name="opensandbox", raw=object())
    assert await provider.status(bare_handle) == SandboxStatus.UNKNOWN


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_exec_background_polls_status_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    """background_exec submits, polls status until finished, then reads logs."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        def __init__(self) -> None:
            self.run_calls: list[tuple[str, FakeRunCommandOpts]] = []
            self.status_calls: list[str] = []
            self.log_calls: list[str] = []
            self._status_sequence = [
                SimpleNamespace(running=True, exit_code=None, error=None),
                SimpleNamespace(running=False, exit_code=7, error=None),
            ]

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            self.run_calls.append((command, opts))
            return SimpleNamespace(id="exec-42")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls.append(execution_id)
            return self._status_sequence[min(len(self.status_calls) - 1, len(self._status_sequence) - 1)]

        async def get_background_command_logs(self, execution_id: str) -> Any:
            self.log_calls.append(execution_id)
            return SimpleNamespace(content="combined output", cursor=None)

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01},
    )
    raw = FakeRaw()
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-bg", provider_name="opensandbox", raw=raw)

    result = await provider.exec(handle, "make build", cwd="/repo", timeout_s=30)

    assert result == opensandbox_provider.SandboxExecResult(
        stdout="combined output", stderr=None, return_code=7, error_type=None
    )
    # Submitted once with background=True; polled twice (running -> finished); read logs once.
    assert len(raw.commands.run_calls) == 1
    assert raw.commands.run_calls[0][1].kwargs["background"] is True
    assert raw.commands.status_calls == ["exec-42", "exec-42"]
    assert raw.commands.log_calls == ["exec-42"]


@pytest.mark.parametrize(
    ("status", "missing"),
    [
        (SimpleNamespace(exit_code=0, error=None), "running"),
        (SimpleNamespace(running=False, error=None), "exit_code"),
    ],
)
@pytest.mark.asyncio
async def test_exec_background_rejects_status_missing_a_field(
    monkeypatch: pytest.MonkeyPatch, status: Any, missing: str
) -> None:
    """An SDK field rename must fail loudly, not score a failed command as success."""

    class FakeCommands:
        async def run(self, command: str, *, opts: Any) -> Any:
            return SimpleNamespace(id="exec-42")

        async def get_command_status(self, execution_id: str) -> Any:
            return status

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="combined output", cursor=None)

    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01},
    )
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-bg", provider_name="opensandbox", raw=SimpleNamespace(commands=FakeCommands())
    )

    with pytest.raises(RuntimeError, match=missing):
        await provider.exec(handle, "make build", timeout_s=30)


async def test_exec_hard_cap_converts_wait_for_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged exec (hard wall-clock cap tripping) surfaces as TimeoutError."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:  # pragma: no cover
            return SimpleNamespace(id="exec-wedged")

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    # Simulate the outer hard-cap wait_for timing out before _dispatch completes.
    async def fake_wait_for(awaitable: Any, timeout: float | None = None) -> Any:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True},
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-wedge", provider_name="opensandbox", raw=FakeRaw())

    with pytest.raises(TimeoutError, match="hard cap"):
        await provider.exec(handle, "sleep 999", timeout_s=30)


@pytest.mark.parametrize("request_timeout_s", [None, 5])
async def test_exec_background_without_timeout_skips_hard_cap(
    monkeypatch: pytest.MonkeyPatch, request_timeout_s: int | None
) -> None:
    """An uncapped background command stays uncapped; request_timeout_s bounds one poll."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        def __init__(self) -> None:
            self.status_calls: list[str] = []

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            return SimpleNamespace(id="exec-uncapped")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls.append(execution_id)
            running = len(self.status_calls) < 2
            return SimpleNamespace(running=running, exit_code=0, error=None)

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="ok", cursor=None)

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    wait_for_timeouts: list[float | None] = []
    real_wait_for = asyncio.wait_for

    async def recording_wait_for(awaitable: Any, timeout: float | None = None) -> Any:
        wait_for_timeouts.append(timeout)
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(asyncio, "wait_for", recording_wait_for)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": request_timeout_s},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01},
    )
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-uncapped", provider_name="opensandbox", raw=FakeRaw()
    )

    result = await provider.exec(handle, "echo hi")

    assert result == opensandbox_provider.SandboxExecResult(stdout="ok", stderr=None, return_code=0, error_type=None)
    if request_timeout_s is not None:
        # The per-poll timeout must not become the whole command's hard cap.
        assert 2.0 * request_timeout_s + 30.0 not in wait_for_timeouts


async def test_provider_create_probe_and_close_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_poll_s": 0.01},
        probe={
            "command": "probe",
            "expected_stdout": "ready",
            "timeout_s": 1,
            "deadline_s": 0.01,
        },
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=object())

    async def bad_probe(*_args: Any, **_kwargs: Any) -> opensandbox_provider.SandboxExecResult:
        return opensandbox_provider.SandboxExecResult(stdout="not ready", stderr="bad", return_code=1)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(provider, "_exec", bad_probe)
    with pytest.raises(opensandbox_provider.OpenSandboxCreateVerificationError):
        await provider._verify_created_handle(handle)

    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": "probe", "expected_stdout": None, "stable_count": 2, "stable_delay_s": 0.01},
    )
    sleep_calls: list[float] = []

    async def record_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    async def good_probe(*_args: Any, **_kwargs: Any) -> opensandbox_provider.SandboxExecResult:
        return opensandbox_provider.SandboxExecResult(stdout="ready", stderr=None, return_code=0)

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(provider, "_exec", good_probe)
    await provider._verify_created_handle(handle)
    assert sleep_calls == [0.01]

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": "probe"})

    async def cancelled_probe(*_args: Any, **_kwargs: Any) -> opensandbox_provider.SandboxExecResult:
        raise asyncio.CancelledError()

    monkeypatch.setattr(provider, "_exec", cancelled_probe)
    with pytest.raises(asyncio.CancelledError):
        await provider._verify_created_handle(handle)

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    async def close_raises(_handle: Any) -> None:
        raise RuntimeError("close failed")

    monkeypatch.setattr(provider, "close", close_raises)
    await provider._cleanup_failed_create_handle(handle)
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    class StopAlreadyGoneRaw:
        async def kill(self) -> None:
            raise RuntimeError("sandbox sandbox-1 not found")

        async def close(self) -> None:
            return None

    await provider.close(
        opensandbox_provider.SandboxHandle(
            sandbox_id="sandbox-1",
            provider_name="opensandbox",
            raw=StopAlreadyGoneRaw(),
        ),
    )

    class StopAndCloseFailRaw:
        async def kill(self) -> None:
            raise RuntimeError("stop failed")

        async def close(self) -> None:
            raise RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="Failed to stop and close"):
        await provider.close(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-2",
                provider_name="opensandbox",
                raw=StopAndCloseFailRaw(),
            ),
        )

    class StopFailsCloseSucceedsRaw:
        async def kill(self) -> None:
            raise RuntimeError("stop failed")

        async def close(self) -> None:
            return None

    with pytest.raises(RuntimeError, match="stop failed"):
        await provider.close(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-3",
                provider_name="opensandbox",
                raw=StopFailsCloseSucceedsRaw(),
            ),
        )


async def test_create_once_and_connect_after_create_error_paths(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 1, "skip_health_check": True},
        probe={"command": None},
    )
    monkeypatch.setattr(opensandbox_provider, "_to_volumes", lambda volumes: volumes)
    spec = SandboxSpec(
        image="image:tag",
        ttl_s=10,
        ready_timeout_s=20,
        resources=SandboxResources(cpu=2, memory_mib=8192, disk_gib=20, gpu=1, gpu_type="H100"),
        entrypoint=["/bin/sh"],
        provider_options={
            "snapshot_id": "snapshot-1",
            "platform": {"os": "linux", "arch": "amd64"},
            "volumes": [{"name": "workspace"}],
            "skip_health_check": False,
        },
    )
    handle = await provider._create_once(spec)
    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["snapshot_id"] == "snapshot-1"
    assert FakeSandbox.created_kwargs["timeout"] == timedelta(seconds=10)
    assert FakeSandbox.created_kwargs["ready_timeout"] == timedelta(seconds=20)
    assert FakeSandbox.created_kwargs["resource"] == {
        "cpu": "2",
        "memory": "8192Mi",
        "ephemeral-storage": "20Gi",
        "gpu": "1",
        "gpu_type": "H100",
    }
    assert FakeSandbox.created_kwargs["entrypoint"] == ["/bin/sh"]
    assert FakeSandbox.created_kwargs["platform"] == FakePlatformSpec(os="linux", arch="amd64")
    assert FakeSandbox.created_kwargs["volumes"] == [{"name": "workspace"}]
    assert FakeSandbox.created_kwargs["skip_health_check"] is True

    class FailingConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            raise ConnectionError("pod may still be starting")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FailingConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_attempt_timeout_s": 0.01, "connect_poll_s": 0.01},
        probe={"command": None},
    )

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", no_sleep)
    with pytest.raises(opensandbox_provider.OpenSandboxCreateTimeoutError):
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag"),
        )

    class CancelledConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            raise asyncio.CancelledError()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (CancelledConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(asyncio.CancelledError):
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag", ready_timeout_s=1),
        )

    class NonRetryableConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            raise ValueError("bad connection request")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (NonRetryableConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(ValueError, match="bad connection request"):
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sandbox-1", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag", ready_timeout_s=1),
        )

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 3},
        probe={"command": None},
    )
    handle = await provider._create_once(SandboxSpec(image="image:tag", provider_options={"skip_health_check": True}))
    assert handle.sandbox_id == "sandbox-1"
    assert FakeSandbox.created_kwargs["skip_health_check"] is True

    class TimeoutSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **_kwargs: Any) -> "FakeSandbox":
            await asyncio.get_running_loop().create_future()
            return cls()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (TimeoutSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"timeout_s": 0.01},
        probe={"command": None},
    )
    with pytest.raises(opensandbox_provider.OpenSandboxCreateTimeoutError):
        await provider._create_once(SandboxSpec(image="image:tag"))

    class EmptyCreateSandbox(FakeSandbox):
        @classmethod
        async def create(cls, **_kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (EmptyCreateSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(RuntimeError, match="returned no sandbox handle"):
        await provider._create_once(SandboxSpec(image="image:tag"))

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (FakeSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": "probe"})
    cleanup_calls: list[str] = []

    async def fail_verify(_handle: opensandbox_provider.SandboxHandle) -> None:
        raise RuntimeError("probe failed")

    async def cleanup(handle: opensandbox_provider.SandboxHandle) -> None:
        cleanup_calls.append(handle.sandbox_id)

    monkeypatch.setattr(provider, "_verify_created_handle", fail_verify)
    monkeypatch.setattr(provider, "_cleanup_failed_create_handle", cleanup)
    with pytest.raises(RuntimeError, match="probe failed"):
        await provider._create_once(SandboxSpec(image="image:tag"))
    assert cleanup_calls == ["sandbox-1"]


async def test_retry_classification_and_await_sdk_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        operations={"retries": 0},
        probe={"command": None},
    )
    assert await provider.aclose() is None
    assert await provider._await_sdk_call(_return_value("ok"), operation="op", sandbox_id="sandbox-1", timeout_s=None)
    assert opensandbox_provider._is_retryable_sdk_operation_error(TimeoutError("command timeout")) is False
    assert opensandbox_provider._is_retryable_sdk_operation_error(ConnectionError("connection failed")) is True
    wrapped = RuntimeError("wrapper")
    wrapped.__cause__ = ConnectionError("connection reset")
    assert opensandbox_provider._is_retryable_sdk_operation_error(wrapped) is True
    wrapped.__cause__ = wrapped
    assert opensandbox_provider._is_retryable_sdk_operation_error(wrapped) is False

    from opensandbox.exceptions import SandboxApiException  # noqa: PLC0415

    cyclic_api_error = SandboxApiException("proxy failed")
    cyclic_api_error.status_code = 500
    cyclic_api_error.__cause__ = cyclic_api_error
    assert opensandbox_provider._is_retryable_sdk_operation_error(cyclic_api_error) is True

    async def cancelled() -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await provider._await_sdk_operation(
            cancelled,
            operation="cancelled",
            sandbox_id="sandbox-1",
            timeout_s=None,
        )


async def test_retry_loop_empty_iterator_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    class EmptyAsyncRetrying:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __aiter__(self) -> "EmptyAsyncRetrying":
            return self

        async def __anext__(self) -> Any:
            raise StopAsyncIteration

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_tenacity",
        lambda: (EmptyAsyncRetrying, lambda predicate: predicate, lambda attempts: attempts, lambda **kwargs: kwargs),
    )

    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    with pytest.raises(RuntimeError, match="SDK operation retry loop did not run"):
        await provider._await_sdk_operation(
            lambda: _return_value("ok"),
            operation="noop",
            sandbox_id="sandbox-1",
            timeout_s=None,
        )
    with pytest.raises(opensandbox_provider.OpenSandboxCreateError, match="create retry loop did not run"):
        await provider._create_with_retries(SandboxSpec(image="image:tag"))


async def _return_value(value: Any) -> Any:
    return value


ATTRIBUTION_ENV_VARS = (
    "NEMO_GYM_TEAM",
    "NEMO_GYM_USER",
    "NEMO_GYM_WORKLOAD",
    "NEMO_GYM_RUN_ID",
    "NEMO_GYM_CONFIG_PATH",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_USER",
    "SLURM_JOB_NAME",
)


@pytest.fixture
def clean_attribution_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ATTRIBUTION_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


async def test_create_injects_attribution_metadata(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_GYM_TEAM", "gym team")  # sanitized to a valid label value below
    monkeypatch.setenv("NEMO_GYM_USER", "alice")
    monkeypatch.setenv("NEMO_GYM_WORKLOAD", "swe-gym")
    monkeypatch.setenv("NEMO_GYM_RUN_ID", "run-123")
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
    )

    await provider.create(SandboxSpec(image="image:tag", metadata={"purpose": "test"}))

    assert FakeSandbox.created_kwargs["metadata"] == {
        "nemo-gym.nvidia.com/team": "gym_team",
        "nemo-gym.nvidia.com/user": "alice",
        "nemo-gym.nvidia.com/workload": "swe-gym",
        "nemo-gym.nvidia.com/run": "run-123",
        "purpose": "test",
    }


async def test_create_spec_metadata_and_config_win_over_attribution_detection(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEMO_GYM_TEAM", "env-team")
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
        attribution={"team": "cfg-team", "user": "cfg-user", "workload": "cfg-workload", "run": "cfg-run"},
    )

    await provider.create(SandboxSpec(image="image:tag", metadata={"nemo-gym.nvidia.com/team": "explicit-team"}))

    assert FakeSandbox.created_kwargs["metadata"] == {
        "nemo-gym.nvidia.com/team": "explicit-team",
        "nemo-gym.nvidia.com/user": "cfg-user",
        "nemo-gym.nvidia.com/workload": "cfg-workload",
        "nemo-gym.nvidia.com/run": "cfg-run",
    }


async def test_create_attribution_disabled(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
        attribution={"enabled": False, "team": "cfg-team"},
    )

    await provider.create(SandboxSpec(image="image:tag"))

    assert FakeSandbox.created_kwargs["metadata"] == {}


@pytest.mark.parametrize(
    ("key_prefix", "expected_key"),
    [
        ("", "team"),
        ("acme.example.com/", "acme.example.com/team"),
        ("acme.example.com", "acme.example.com/team"),  # trailing slash is normalized in
        ("  ", "team"),
    ],
)
async def test_create_attribution_key_prefix(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
    key_prefix: str,
    expected_key: str,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
        attribution={"team": "cfg-team", "key_prefix": key_prefix},
    )

    await provider.create(SandboxSpec(image="image:tag"))

    metadata = FakeSandbox.created_kwargs["metadata"]
    assert metadata[expected_key] == "cfg-team"


async def test_create_attribution_run_id_generated(
    fake_opensandbox_sdk: None,
    clean_attribution_env: None,
) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 10},
        probe={"command": None},
    )

    await provider.create(SandboxSpec(image="image:tag"))

    assert FakeSandbox.created_kwargs["metadata"]["nemo-gym.nvidia.com/run"]  # generated per process


@pytest.mark.parametrize("key_prefix", ["Not_A_Valid_Prefix/", "-bad.example.com/", "bad..example.com/"])
def test_attribution_invalid_key_prefix_raises(key_prefix: str) -> None:
    with pytest.raises(ValueError, match="key_prefix"):
        opensandbox_provider.OpenSandboxAttributionConfig(key_prefix=key_prefix)


async def test_connect_health_checks_by_default(fake_opensandbox_sdk: None) -> None:
    """An unchecked handle would defer the exec-daemon startup gap to the first call."""
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    await provider.connect({"sandbox_id": "sandbox-9"})

    assert FakeSandbox.connected_kwargs["skip_health_check"] is False


async def test_connect_honours_skip_health_check_opt_out(fake_opensandbox_sdk: None) -> None:
    """Callers that explicitly opt out still get an unchecked handle."""
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"skip_health_check": True},
        probe={"command": None},
    )

    await provider.connect({"sandbox_id": "sandbox-9"})

    assert FakeSandbox.connected_kwargs["skip_health_check"] is True


@pytest.mark.asyncio
async def test_exec_retries_backend_connect_502_despite_zero_command_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A proxy 502 means the command never started; it retries even with command_retries=0."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Backend502Error(Exception):
        status_code = 502

    calls = {"n": 0}

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            calls["n"] += 1
            if calls["n"] < 3:
                raise Backend502Error("Failed to run command. Status code: 502")
            return SimpleNamespace(logs=SimpleNamespace(stdout=[], stderr=[]), error=None, exit_code=0)

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"retries": 3},
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sb-flap", provider_name="opensandbox", raw=FakeRaw())

    result = await provider.exec(handle, "echo ok", timeout_s=30)

    assert result.return_code == 0
    assert calls["n"] == 3  # two 502s absorbed, command never double-ran


@pytest.mark.asyncio
async def test_exec_persistent_502_raises_typed_backend_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """502s that outlive the budget mean a dead backend: fail fast and typed."""

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class Backend502Error(Exception):
        status_code = 502

    calls = {"n": 0}

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            calls["n"] += 1
            raise Backend502Error("Failed to run command. Status code: 502")

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"retries": 2},
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sb-dead", provider_name="opensandbox", raw=FakeRaw())

    with pytest.raises(opensandbox_provider.SandboxBackendUnreachableError, match="likely dead"):
        await provider.exec(handle, "echo ok", timeout_s=30)

    assert calls["n"] == 3  # operations.retries + 1 submissions, then typed failure
