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
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec, SandboxStatus


pytestmark = pytest.mark.sandbox


pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")

from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


TEST_REGISTRY_PASSWORD = "secret"  # pragma: allowlist secret


@pytest.mark.parametrize("recovers", [True, False])
@pytest.mark.parametrize("timeout_kind", ["client", "server", "readiness"])
async def test_tb4_create_timeout_retries_with_jitter(monkeypatch, recovers, timeout_kind):
    import tenacity
    import tenacity.wait
    import yaml
    from opensandbox.exceptions import SandboxApiException, SandboxReadyTimeoutException

    config = yaml.safe_load(Path("benchmarks/terminal_bench_4/resources.yaml").read_text())
    environment = config["terminal_bench_4"]["resources_servers"]["terminal_bench_4"]["environment"]
    provider = opensandbox_provider.OpenSandboxProvider(
        create=environment["sandbox_provider"]["opensandbox"]["create"]
    )
    errors = {
        "client": opensandbox_provider.OpenSandboxCreateTimeoutError("create timeout"),
        "server": SandboxApiException("POD_READY_TIMEOUT: BATCHSANDBOX_PENDING", status_code=504),
        "readiness": SandboxReadyTimeoutException("readiness timeout"),
    }
    handle = opensandbox_provider.SandboxHandle(sandbox_id="ready", provider_name="opensandbox", raw=None)
    provider._create_once = AsyncMock(
        side_effect=[errors[timeout_kind]] * 5 + [handle if recovers else errors[timeout_kind]]
    )
    sleep = AsyncMock()
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_tenacity",
        lambda: (
            lambda **kwargs: tenacity.AsyncRetrying(sleep=sleep, **kwargs),
            tenacity.retry_if_exception,
            tenacity.stop_after_attempt,
            tenacity.wait_random_exponential,
        ),
    )
    bounds = []

    def midpoint(low, high):
        bounds.append((low, high))
        return (low + high) / 2

    monkeypatch.setattr(tenacity.wait.random, "uniform", midpoint)
    if recovers:
        assert await provider.create(SandboxSpec(image="task")) is handle
    else:
        with pytest.raises(type(errors[timeout_kind])):
            await provider.create(SandboxSpec(image="task"))
    assert provider._create_once.await_count == 6
    assert [call.args[0] for call in sleep.await_args_list] == [2.5, 5, 10, 20, 30]
    assert bounds[:5] == [(0, 5), (0, 10), (0, 20), (0, 40), (0, 60)]


async def test_create_does_not_retry_rejected_credentials():
    from opensandbox.exceptions import SandboxApiException

    provider = opensandbox_provider.OpenSandboxProvider(create={"retries": 5})
    provider._create_once = AsyncMock(side_effect=SandboxApiException("unauthorized", status_code=401))
    with pytest.raises(SandboxApiException):
        await provider.create(SandboxSpec(image="task"))
    provider._create_once.assert_awaited_once()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_renewal_rejects_invalid_intervals(interval: float) -> None:
    with pytest.raises(ValueError, match="renew_interval_s"):
        opensandbox_provider.OpenSandboxCreateConfig(renew_interval_s=interval)


@pytest.mark.parametrize("ttl", [None, 1, 2])
async def test_renewal_requires_ttl_longer_than_interval(ttl: int | None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(create={"renew_interval_s": 2, "retries": 0})
    with pytest.raises(ValueError, match="longer, explicit"):
        await provider.create(SandboxSpec(image="example", ttl_s=ttl))


async def test_renewal_refreshes_ttl_and_stops_before_termination(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(create={"renew_interval_s": 0.01}, probe={"command": None})
    handle = await provider.create(SandboxSpec(image="example", ttl_s=28800))
    renewed = asyncio.Event()
    durations = []

    async def renew(duration: timedelta) -> None:
        durations.append(duration)
        renewed.set()

    async def kill() -> None:
        assert handle.sandbox_id not in provider._renewals

    handle.raw.renew = renew
    handle.raw.kill = AsyncMock(side_effect=kill)
    handle.raw.close = AsyncMock()
    await asyncio.wait_for(renewed.wait(), timeout=1)
    task = provider._renewals[handle.sandbox_id]
    await provider.close(handle)
    assert durations and all(duration == timedelta(seconds=28800) for duration in durations)
    assert task.cancelled()
    handle.raw.kill.assert_awaited_once()
    handle.raw.close.assert_awaited_once()
    await provider.aclose()


async def test_renewal_failure_surfaces_and_still_allows_cleanup(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"renew_interval_s": 0.01}, probe={"command": None}, operations={"retries": 0}
    )
    handle = await provider.create(SandboxSpec(image="example", ttl_s=28800))
    handle.raw.renew = AsyncMock(side_effect=RuntimeError("renewal rejected"))
    handle.raw.kill = AsyncMock()
    handle.raw.close = AsyncMock()
    task = provider._renewals[handle.sandbox_id]
    with pytest.raises(RuntimeError, match="renewal rejected"):
        await asyncio.wait_for(asyncio.shield(task), timeout=1)
    operation = AsyncMock()
    with pytest.raises(RuntimeError, match="lifetime renewal failed"):
        await provider._await_sdk_operation(operation, operation="get_info", sandbox_id=handle.sandbox_id, timeout_s=1)
    operation.assert_not_awaited()
    with pytest.raises(RuntimeError, match="lifetime renewal failed"):
        await provider.close(handle)
    handle.raw.kill.assert_awaited_once()
    handle.raw.close.assert_awaited_once()
    await provider.aclose()


async def test_provider_shutdown_cancels_renewal_without_terminating_sandbox(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(create={"renew_interval_s": 10}, probe={"command": None})
    handle = await provider.create(SandboxSpec(image="example", ttl_s=28800))
    handle.raw.kill = AsyncMock()
    task = provider._renewals[handle.sandbox_id]
    await provider.aclose()
    assert task.cancelled()
    assert not provider._renewals
    handle.raw.kill.assert_not_awaited()


@dataclass(frozen=True)
class FakePlatformSpec:
    os: str
    arch: str


class FakeConnectionConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def with_transport_if_missing(self) -> "FakeConnectionConfig":
        return self

    async def close_transport_if_owned(self) -> None:
        return None


@dataclass(frozen=True)
class FakeVolume:
    name: str


class FakeSandbox:
    created_kwargs: dict[str, Any] = {}
    connected_args: tuple[Any, ...] = ()
    connected_state = "RUNNING"
    ready_calls = 0
    resumed_args: tuple[Any, ...] = ()
    resumed_kwargs: dict[str, Any] = {}

    def __init__(self, sandbox_id: str = "sandbox-1") -> None:
        self.id = sandbox_id

    @classmethod
    async def create(cls, *_args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.created_kwargs = kwargs
        return cls()

    @classmethod
    async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.connected_args = args
        sandbox = cls(str(args[0]))
        if not kwargs.get("skip_health_check", False):
            await sandbox.check_ready(
                kwargs["connect_timeout"],
                kwargs.get("health_check_polling_interval", timedelta(milliseconds=200)),
            )
        return sandbox

    @classmethod
    async def resume(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
        cls.resumed_args = args
        cls.resumed_kwargs = kwargs
        return cls(str(args[0]))

    async def get_info(self) -> Any:
        return SimpleNamespace(status=SimpleNamespace(state=type(self).connected_state))

    async def check_ready(self, _timeout: timedelta, _polling_interval: timedelta) -> None:
        if type(self).connected_state == "PAUSED":
            raise RuntimeError("a paused sandbox has no exec daemon")
        type(self).ready_calls += 1

    async def close(self) -> None:
        return None


@pytest.fixture
def fake_opensandbox_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSandbox.connected_state = "RUNNING"
    FakeSandbox.ready_calls = 0
    FakeSandbox.resumed_args = ()
    FakeSandbox.resumed_kwargs = {}

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


def test_sdk_missing_sandbox_terminate_warning_is_silenced(caplog: pytest.LogCaptureFixture) -> None:
    sdk_logger = logging.getLogger("opensandbox.adapters.sandboxes_adapter")

    with caplog.at_level(logging.WARNING):
        sdk_logger.warning(
            "Failed to terminate sandbox sandbox-1: Kill sandbox sandbox-1 failed: [KUBERNETES::SANDBOX_NOT_FOUND]"
        )
        sdk_logger.warning("Failed to terminate sandbox sandbox-2: backend unavailable")
        sdk_logger.warning("Sandbox sandbox-3 not found | [KUBERNETES::SANDBOX_NOT_FOUND]")

    sdk_messages = [record.message for record in caplog.records if record.name == sdk_logger.name]
    assert sdk_messages == [
        "Failed to terminate sandbox sandbox-2: backend unavailable",
        "Sandbox sandbox-3 not found | [KUBERNETES::SANDBOX_NOT_FOUND]",
    ]


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

    status_only_not_found = SandboxApiException("gone")
    status_only_not_found.status_code = 404
    assert opensandbox_provider._is_missing_sandbox_delete_error(status_only_not_found) is True
    assert (
        opensandbox_provider._is_missing_sandbox_delete_error(RuntimeError("[KUBERNETES::SANDBOX_NOT_FOUND]")) is True
    )
    assert opensandbox_provider._is_missing_sandbox_delete_error(RuntimeError("sandbox sandbox-1 not found")) is True
    assert opensandbox_provider._is_missing_sandbox_delete_error(RuntimeError("http 500 boom")) is False

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
    assert "network_policy" not in FakeSandbox.created_kwargs


async def test_direct_create_passes_network_policy_to_sdk_create(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    policy = {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "pypi.org"}],
    }

    await provider.create(SandboxSpec(image="image:tag", provider_options={"network_policy": policy}))

    assert FakeSandbox.created_kwargs["network_policy"].model_dump(by_alias=True, exclude_none=True) == policy


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


@pytest.mark.parametrize(
    ("resources", "expected"),
    [
        (
            {"cpu": 4, "memory_mib": 4096, "disk_gib": 10},
            {"cpu": "4", "memory": "4096Mi", "ephemeral-storage": "10Gi"},
        ),
        ({"cpu": 0.5, "memory_mib": 512}, {"cpu": "0.5", "memory": "512Mi"}),
        (
            {"cpu": 4, "memory_mib": 16384, "gpu": 1, "gpu_type": "H100"},
            {"cpu": "4", "memory": "16384Mi", "gpu": "1", "gpu_type": "H100"},
        ),
        ({}, {"cpu": "1", "memory": "2Gi"}),
    ],
)
async def test_explicit_requests_match_each_sandbox_limits(fake_opensandbox_sdk, resources, expected):
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})
    await provider.create(
        SandboxSpec(image="image:tag", resources=resources, provider_options={"resource_requests": "limits"})
    )
    assert FakeSandbox.created_kwargs["resource"] == expected
    assert FakeSandbox.created_kwargs["resource_requests"] == expected
    assert FakeSandbox.created_kwargs["resource_requests"] is not FakeSandbox.created_kwargs["resource"]


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
    # A scheme in the configured domain is split into protocol + bare host before it reaches the SDK.
    assert create_connection.kwargs["domain"] == "sandbox.example"
    assert create_connection.kwargs["protocol"] == "http"
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
    assert opensandbox_provider._to_sandbox_status("pausing") == SandboxStatus.STARTING
    assert opensandbox_provider._to_sandbox_status("paused") == SandboxStatus.PAUSED
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
        {"operations": {"status_poll_timeout_s": 0}},
        {"operations": {"pause_resume_timeout_s": 0}},
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
            "network_policy": {"defaultAction": "allow", "egress": []},
            "platform": {"os": "linux", "arch": "amd64"},
            "snapshot_id": "snap-1",
            "volumes": [{"name": "workspace"}],
            "skip_health_check": True,
            "extensions": {"imagePullPolicy": "Never"},
        }
    )
    assert parsed.image_auth == {"username": "user", "password": TEST_REGISTRY_PASSWORD}
    assert parsed.network_policy == {"defaultAction": "allow", "egress": []}
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
    with pytest.raises(TypeError, match="'network_policy' must be a mapping"):
        options_cls.from_mapping({"network_policy": "allow"})
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


def test_connection_transport_backends(fake_opensandbox_sdk: None) -> None:
    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    # The default backend borrows Gym's global aiohttp client.
    provider = opensandbox_provider.OpenSandboxProvider()
    transport = provider._build_transport()
    assert isinstance(transport, GymAiohttpTransport)

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

    # SDK transport defaults are sufficient when certificate verification is enabled.
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "httpx", "keepalive_expiry_s": None, "tls_verify": True}
    )
    config = provider._connection_config()
    assert "transport" not in config.kwargs

    # max_connections=null uncaps the pool; max_keepalive_connections=0 disables reuse.
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "httpx", "max_connections": None, "max_keepalive_connections": 0}
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


@pytest.mark.parametrize("backend", ["httpx", "aiohttp"])
@pytest.mark.parametrize("verify", [True, False])
@pytest.mark.parametrize(
    "keepalive_expiry_s,disable_pooling", [(None, False), (3.0, False), (None, True), (3.0, True)]
)
async def test_connection_tls_is_independent_of_pool_settings(
    fake_opensandbox_sdk: None,
    backend: str,
    verify: bool,
    keepalive_expiry_s: float | None,
    disable_pooling: bool,
) -> None:
    import ssl

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "transport_backend": backend,
            "tls_verify": verify,
            "keepalive_expiry_s": keepalive_expiry_s,
            "disable_connection_pooling": disable_pooling,
        }
    )
    try:
        config = provider._connection_config()
        if backend == "httpx" and verify and keepalive_expiry_s is None and not disable_pooling:
            # The SDK verifies certificates by default; no custom transport is needed.
            assert "transport" not in config.kwargs
            assert provider._transport is None
        else:
            transport = config.kwargs["transport"]
            if backend == "aiohttp":
                assert transport.verify is verify
            else:
                context = transport._pool._ssl_context
                assert context.verify_mode == (ssl.CERT_REQUIRED if verify else ssl.CERT_NONE)
                assert context.check_hostname is verify
            assert provider._connection_config().kwargs["transport"] is transport
    finally:
        await provider.aclose()
    assert provider._transport is None


def test_connection_transport_backend_aiohttp_opt_in(fake_opensandbox_sdk: None) -> None:
    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    provider = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "aiohttp"})
    transport = provider._build_transport()
    assert isinstance(transport, GymAiohttpTransport)

    extensions = provider._resolve_extensions({"imagePullPolicy": "Never"})
    assert extensions["imagePullPolicy"] == "Never"
    assert extensions["opensandbox.extensions.image-pull-policy"] == "Never"

    no_policy_provider = opensandbox_provider.OpenSandboxProvider(create={"image_pull_policy": None})
    assert no_policy_provider._resolve_extensions({"imagePullPolicy": "Never"}) == {"imagePullPolicy": "Never"}


def test_connection_config_disable_pooling_sets_fresh_transport(fake_opensandbox_sdk: None) -> None:
    import httpx

    # Default: a keepalive-bounded transport with connection reuse enabled.
    pooled = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "httpx", "domain": "sandbox.example"}
    )
    pooled_transport = pooled._connection_config().kwargs["transport"]
    assert isinstance(pooled_transport, httpx.AsyncHTTPTransport)
    assert pooled_transport._pool._max_keepalive_connections > 0

    # disable_connection_pooling -> same transport plumbing, but no reuse.
    fresh = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "httpx", "domain": "sandbox.example", "disable_connection_pooling": True}
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


@pytest.mark.asyncio
async def test_exec_background_reports_oom_status_after_502(monkeypatch: pytest.MonkeyPatch) -> None:
    class Backend502Error(Exception):
        status_code = 502

    class FakeCommands:
        async def run(self, command: str, *, opts: Any) -> Any:
            return SimpleNamespace(id="exec-oom")

        async def get_command_status(self, execution_id: str) -> Any:
            raise Backend502Error("Get command status failed: HTTP 502")

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()
            self.statuses = [
                SimpleNamespace(state="Running", reason=None, message=None),
                SimpleNamespace(
                    state="Failed",
                    reason="FAILED",
                    message="container sandbox terminated with OOMKilled (exit code 137); " + "x" * 1000,
                ),
            ]
            self.info_calls = 0

        async def get_info(self) -> Any:
            status = self.statuses[min(self.info_calls, len(self.statuses) - 1)]
            self.info_calls += 1
            return SimpleNamespace(status=status)

    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "retries": 0},
    )
    raw = FakeRaw()
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-oom", provider_name="opensandbox", raw=raw)

    with pytest.raises(opensandbox_provider.SandboxBackendUnreachableError) as exc_info:
        await provider.exec(handle, "allocate memory", timeout_s=30)

    message = str(exc_info.value)
    assert "OOM-killed" in message
    assert "SandboxResources.memory_mib" not in message
    assert "reason='FAILED'" in message
    assert len(message) < 800
    assert isinstance(exc_info.value.__cause__, Backend502Error)
    assert raw.info_calls == 2

    raw.statuses = [SimpleNamespace(state="Failed", reason="FAILED", message="sandbox node was drained")]
    raw.info_calls = 0
    with pytest.raises(Backend502Error, match="Get command status failed"):
        await provider.exec(handle, "retry after non-OOM failure", timeout_s=30)


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


async def test_exec_hard_cap_labels_genuinely_wedged_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A wedged exec (hard wall-clock cap tripping) surfaces as the hard-cap TimeoutError.

    The cap formula floors the real duration at minutes, so the test shrinks it
    by patching asyncio.timeout to ignore the requested duration — the genuine
    cancellation-to-TimeoutError conversion path still runs.
    """

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            await asyncio.Event().wait()  # wedged: never returns

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )

    real_timeout = asyncio.timeout
    monkeypatch.setattr(asyncio, "timeout", lambda delay: real_timeout(0.05))

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sandbox-wedge", provider_name="opensandbox", raw=FakeRaw())

    with pytest.raises(TimeoutError, match="hard cap"):
        await provider.exec(handle, "sleep 999", timeout_s=30)


async def test_exec_hard_cap_does_not_relabel_inner_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    """A TimeoutError raised inside the dispatch keeps its own message.

    Since Python 3.11 asyncio.TimeoutError IS builtin TimeoutError, so a
    wait_for-based cap caught e.g. an exhausted status-poll budget (a
    minutes-scale failure) and relabeled it as a trip of the hours-scale hard
    cap, corrupting the failure taxonomy. Only the cap's own expiry may carry
    the wedged message.
    """

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        def __init__(self) -> None:
            self.status_calls = 0

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            return SimpleNamespace(id="exec-slowpolls")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls += 1
            raise TimeoutError("simulated status poll budget expiry")

        async def get_background_command_logs(self, execution_id: str) -> Any:  # pragma: no cover
            return SimpleNamespace(content="ok", cursor=None)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 120},
        probe={"command": None},
        operations={
            "background_exec": True,
            "retries": 1,
            "retry_delay_s": 0.001,
            "background_poll_interval_s": 0.01,
        },
    )
    commands = FakeCommands()
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-slowpolls", provider_name="opensandbox", raw=SimpleNamespace(commands=commands)
    )

    with pytest.raises(TimeoutError) as exc_info:
        await provider.exec(handle, "echo ok", timeout_s=30)

    # The poll-budget failure surfaced with its own message (after using its
    # retry budget), not relabeled as the wall-clock hard cap tripping.
    assert commands.status_calls == 2
    assert "command status" in str(exc_info.value)
    assert "hard cap" not in str(exc_info.value)


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


async def test_close_treats_missing_sandbox_as_terminated_without_retry(caplog: pytest.LogCaptureFixture) -> None:
    from opensandbox.exceptions import SandboxApiException  # noqa: PLC0415

    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": None},
        operations={"retries": 2, "retry_delay_s": 0, "retry_max_delay_s": 0},
    )
    not_found = SandboxApiException(
        "Kill sandbox sandbox-1 failed: Sandbox 'sandbox-1' not found | "
        "[KUBERNETES::SANDBOX_NOT_FOUND] Sandbox 'sandbox-1' not found"
    )
    not_found.status_code = 404
    kill_calls = 0

    class AlreadyGoneRaw:
        async def kill(self) -> None:
            nonlocal kill_calls
            kill_calls += 1
            raise not_found

        async def close(self) -> None:
            return None

    with caplog.at_level(logging.DEBUG):
        await provider.close(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-1",
                provider_name="opensandbox",
                raw=AlreadyGoneRaw(),
            ),
        )

    assert kill_calls == 1
    assert any("already gone; treating terminate as success" in record.message for record in caplog.records)


async def test_non_terminate_operation_still_raises_on_missing_sandbox() -> None:
    from opensandbox.exceptions import SandboxApiException  # noqa: PLC0415

    provider = opensandbox_provider.OpenSandboxProvider(
        probe={"command": None},
        operations={"retries": 2, "retry_delay_s": 0, "retry_max_delay_s": 0},
    )
    not_found = SandboxApiException("Get sandbox sandbox-1 failed: Sandbox 'sandbox-1' not found")
    not_found.status_code = 404
    get_info_calls = 0

    class MissingRaw:
        async def get_info(self) -> Any:
            nonlocal get_info_calls
            get_info_calls += 1
            raise not_found

    with pytest.raises(SandboxApiException, match="not found"):
        await provider.status(
            opensandbox_provider.SandboxHandle(
                sandbox_id="sandbox-1",
                provider_name="opensandbox",
                raw=MissingRaw(),
            ),
        )

    assert get_info_calls == 1


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

    handle = await provider.connect({"sandbox_id": "sandbox-9"})

    assert handle.sandbox_id == "sandbox-9"
    assert FakeSandbox.ready_calls == 1


async def test_connect_accepts_a_paused_sandbox_without_health_checking_execd(fake_opensandbox_sdk: None) -> None:
    FakeSandbox.connected_state = "PAUSED"
    provider = opensandbox_provider.OpenSandboxProvider(probe={"command": None})

    handle = await provider.connect({"sandbox_id": "sandbox-9"})

    assert handle.sandbox_id == "sandbox-9"
    assert FakeSandbox.ready_calls == 0


async def test_connect_honours_skip_health_check_opt_out(fake_opensandbox_sdk: None) -> None:
    """Callers that explicitly opt out still get an unchecked handle."""
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"skip_health_check": True},
        probe={"command": None},
    )

    await provider.connect({"sandbox_id": "sandbox-9"})

    assert FakeSandbox.ready_calls == 0


async def test_connect_preserves_failure_when_cleanup_stalls(
    fake_opensandbox_sdk: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(FakeSandbox, "check_ready", AsyncMock(side_effect=ConnectionError("readiness failed")))
    close = AsyncMock(side_effect=asyncio.Event().wait)
    monkeypatch.setattr(FakeSandbox, "close", close)
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_attempt_timeout_s": 0.05},
        operations={"close_timeout_s": 0.05},
        probe={"command": None},
    )

    with pytest.raises(ConnectionError, match="readiness failed"):
        async with asyncio.timeout(0.5):
            await provider.connect({"sandbox_id": "sandbox-9"})
    close.assert_awaited_once_with()


async def test_pause_waits_until_opensandbox_reports_paused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_attempt_timeout_s": 0.000001},
        operations={"pause_resume_timeout_s": 1},
    )
    raw = SimpleNamespace(
        pause=AsyncMock(),
        get_info=AsyncMock(
            side_effect=[
                SimpleNamespace(status=SimpleNamespace(state="PAUSING")),
                SimpleNamespace(status=SimpleNamespace(state="PAUSED")),
            ]
        ),
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    await provider.pause(handle)

    raw.pause.assert_awaited_once_with()
    assert raw.get_info.await_count == 2


async def test_pause_failure_is_not_retried() -> None:
    provider = opensandbox_provider.OpenSandboxProvider(operations={"retries": 5})
    raw = SimpleNamespace(
        pause=AsyncMock(side_effect=ConnectionError("pause request outcome is unknown")),
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(ConnectionError, match="outcome is unknown"):
        await provider.pause(handle)

    raw.pause.assert_awaited_once_with()


async def test_pause_timeout_bounds_the_full_transition() -> None:
    opensandbox_provider._require_tenacity()
    status_started = asyncio.Event()

    async def slow_pause() -> None:
        await asyncio.sleep(0.05)

    async def stalled_status() -> None:
        status_started.set()
        await asyncio.Event().wait()

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 1},
        operations={"pause_resume_timeout_s": 0.1, "retries": 0},
    )
    raw = SimpleNamespace(
        pause=slow_pause,
        get_info=stalled_status,
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(TimeoutError, match=r"waiting .* to pause after 0\.1s"):
        await asyncio.wait_for(provider.pause(handle), timeout=0.14)
    assert status_started.is_set()


async def test_pause_preserves_a_shorter_request_timeout() -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 0.001},
        operations={"pause_resume_timeout_s": 0.05},
    )
    raw = SimpleNamespace(pause=AsyncMock(side_effect=asyncio.Event().wait))
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(TimeoutError, match="OpenSandbox pause after 0.001s"):
        await provider.pause(handle)


async def test_resume_rebuilds_the_sdk_handle_after_readiness(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5, "keepalive_expiry_s": None},
        create={
            "connect_poll_s": 0.25,
            "skip_health_check": False,
        },
        operations={"pause_resume_timeout_s": 7},
    )
    raw = SimpleNamespace(close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    await provider.resume(handle)

    assert handle.raw is not raw
    raw.close.assert_awaited_once_with()
    assert FakeSandbox.resumed_args == ("sandbox-paused",)
    assert FakeSandbox.resumed_kwargs["resume_timeout"] == timedelta(seconds=7)
    assert FakeSandbox.resumed_kwargs["health_check_polling_interval"] == timedelta(seconds=0.25)
    assert FakeSandbox.resumed_kwargs["skip_health_check"] is False
    assert FakeSandbox.resumed_kwargs["connection_config"].kwargs["request_timeout"] == timedelta(seconds=5)


async def test_resume_failure_keeps_the_paused_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    resume = AsyncMock(side_effect=ConnectionError("resume request outcome is unknown"))
    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (SimpleNamespace(resume=resume), FakeConnectionConfig, object, FakePlatformSpec, FakeVolume),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"keepalive_expiry_s": None},
        operations={"retries": 5},
    )
    raw = SimpleNamespace(close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(ConnectionError, match="outcome is unknown"):
        await provider.resume(handle)

    assert resume.await_count == 1
    raw.close.assert_not_awaited()


async def test_resume_timeout_bounds_the_full_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK applies resume_timeout only to the readiness check; the request and endpoint rebuild must count too."""
    resume_started = asyncio.Event()

    async def stalled_resume(*_args: Any, **_kwargs: Any) -> Any:
        resume_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (SimpleNamespace(resume=stalled_resume), FakeConnectionConfig, object, FakePlatformSpec, FakeVolume),
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"keepalive_expiry_s": None},
        operations={"pause_resume_timeout_s": 0.05},
    )
    raw = SimpleNamespace(close=AsyncMock())
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(TimeoutError, match=r"to resume after 0\.05s; reconnect and check status\(\)"):
        await asyncio.wait_for(provider.resume(handle), timeout=1)

    assert resume_started.is_set()
    assert handle.raw is raw
    raw.close.assert_not_awaited()


async def test_resume_timeout_closes_an_sdk_owned_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """The timeout cancels the SDK call with CancelledError, which the SDK's own cleanup never sees."""
    pytest.importorskip("opensandbox", reason="opensandbox SDK is not installed")
    from opensandbox.config import ConnectionConfig

    transport_close = AsyncMock()

    async def stalled_resume(_sandbox_id: str, *, connection_config: Any, **_kwargs: Any) -> Any:
        assert connection_config.transport is not None and connection_config._owns_transport
        monkeypatch.setattr(connection_config.transport, "aclose", transport_close)
        await asyncio.Event().wait()

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (SimpleNamespace(resume=stalled_resume), ConnectionConfig, object, FakePlatformSpec, FakeVolume),
    )
    # The only configuration that leaves the SDK to build its own transport.
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "sandbox.example",
            "keepalive_expiry_s": None,
            "transport_backend": "httpx",
            "tls_verify": True,
        },
        operations={"pause_resume_timeout_s": 0.05},
    )
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", SimpleNamespace(close=AsyncMock()))

    with pytest.raises(TimeoutError, match=r"to resume after 0\.05s"):
        await asyncio.wait_for(provider.resume(handle), timeout=1)

    transport_close.assert_awaited_once_with()


async def test_resume_installs_the_new_handle_before_closing_the_old_one(fake_opensandbox_sdk: None) -> None:
    """A cancellation during the best-effort close of the old handle must not lose a successful resume."""
    provider = opensandbox_provider.OpenSandboxProvider(connection={"keepalive_expiry_s": None})
    raw = SimpleNamespace(close=AsyncMock(side_effect=asyncio.Event().wait))
    handle = opensandbox_provider.SandboxHandle("sandbox-paused", "opensandbox", raw)

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(provider.resume(handle), timeout=0.05)

    assert isinstance(handle.raw, FakeSandbox)
    assert handle.raw.id == "sandbox-paused"
    raw.close.assert_awaited_once_with()


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

        async def get_info(self) -> Any:
            raise ConnectionError("status API unavailable")

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


def _sandbox_ended_api_error(
    message: str = "Sandbox sb-1 has ended (OOMKilled): PodFailed: container sandbox was OOMKilled",
) -> Any:
    """Build the SDK exception the server's 410 SANDBOX::ENDED answer produces."""
    from opensandbox.exceptions import SandboxApiException  # noqa: PLC0415
    from opensandbox.exceptions.sandbox import SandboxError  # noqa: PLC0415

    error = SandboxApiException(
        message=f"Command run failed: {message}",
        status_code=410,
        error=SandboxError(code="SANDBOX::ENDED", message=message),
    )
    return error


def _error_carrier(error: Any) -> Exception:
    carrier = Exception("opaque")
    carrier.error = error  # type: ignore[attr-defined]
    return carrier


def test_sandbox_ended_errors_are_never_retryable() -> None:
    """410 SANDBOX::ENDED is terminal even though its message matches a retryable marker."""
    ended = _sandbox_ended_api_error()

    # The message says "PodFailed", which the marker heuristic reads as retryable;
    # the sandbox-ended guard has to win over it.
    assert opensandbox_provider._has_retryable_error_marker(ended) is True
    assert opensandbox_provider._is_sandbox_ended_error(ended) is True
    assert opensandbox_provider._is_retryable_create_error(ended) is False
    assert opensandbox_provider._is_retryable_sdk_operation_error(ended) is False

    wrapped = RuntimeError("command run failed")
    wrapped.__cause__ = ended
    assert opensandbox_provider._is_sandbox_ended_error(wrapped) is True
    assert opensandbox_provider._is_retryable_sdk_operation_error(wrapped) is False

    # Each signal recognises the answer on its own: status code, SDK error code, message.
    class StatusOnly(Exception):
        status_code = 410

    assert opensandbox_provider._is_sandbox_ended_error(StatusOnly("gone")) is True
    code_only = SimpleNamespace(code="sandbox::ended")
    assert opensandbox_provider._is_sandbox_ended_error(_error_carrier(code_only)) is True
    assert opensandbox_provider._is_sandbox_ended_error(RuntimeError("[SANDBOX::ENDED] gone")) is True

    # A dead backend is still a backend problem, not an ended sandbox.
    unreachable = opensandbox_provider.SandboxBackendUnreachableError("proxy 502: no TCP connection to execd")
    assert opensandbox_provider._is_sandbox_ended_error(unreachable) is False
    assert opensandbox_provider._is_retryable_sdk_operation_error(unreachable) is False

    class Backend502Error(Exception):
        status_code = 502

    assert opensandbox_provider._is_sandbox_ended_error(Backend502Error("HTTP 502")) is False
    assert opensandbox_provider._is_retryable_sdk_operation_error(Backend502Error("HTTP 502")) is True


def test_sandbox_ended_reason_prefers_the_server_header() -> None:
    """The reason comes from the response header when the SDK kept it, else from the message."""
    from_message = _sandbox_ended_api_error()
    assert opensandbox_provider._sandbox_ended_reason(from_message) == "OOMKilled"

    with_header = _sandbox_ended_api_error()
    with_header.response = SimpleNamespace(  # type: ignore[attr-defined]
        headers=httpx.Headers({"OpenSandbox-Sandbox-Ended-Reason": "Evicted"})
    )
    assert opensandbox_provider._sandbox_ended_reason(with_header) == "Evicted"

    assert opensandbox_provider._sandbox_ended_reason(RuntimeError("[SANDBOX::ENDED] no reason here")) is None

    restated = opensandbox_provider._as_sandbox_ended_error(from_message, operation="command run", sandbox_id="sb-1")
    assert restated.reason == "OOMKilled"
    assert restated.sandbox_id == "sb-1"
    assert restated.status_code == 410
    assert "OOMKilled" in str(restated)
    assert opensandbox_provider._as_sandbox_ended_error(restated, operation="op", sandbox_id="sb-1") is restated


@pytest.mark.asyncio
async def test_exec_raises_sandbox_ended_on_first_410(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 410 ends the command at once: no resubmission, no death probe, reason attached."""

    calls = {"run": 0, "info": 0}

    class FakeCommands:
        async def run(self, command: str, *, opts: Any) -> Any:
            calls["run"] += 1
            raise _sandbox_ended_api_error()

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

        async def get_info(self) -> Any:
            calls["info"] += 1
            return SimpleNamespace(status=SimpleNamespace(state="Failed", reason="FAILED", message="OOMKilled"))

    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"retries": 3, "command_retries": 2},
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sb-ended", provider_name="opensandbox", raw=FakeRaw())

    with pytest.raises(opensandbox_provider.SandboxEndedError) as exc_info:
        await provider.exec(handle, "echo ok", timeout_s=30)

    assert exc_info.value.reason == "OOMKilled"
    assert exc_info.value.sandbox_id == "sb-ended"
    assert exc_info.value.status_code == 410
    assert "OOMKilled" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, Exception)
    assert calls["run"] == 1  # neither command_retries nor the 502 submission loop re-ran it
    assert calls["info"] == 0  # the server already said why; no status poll


@pytest.mark.asyncio
async def test_exec_background_reports_sandbox_ended_mid_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sandbox that ends mid-command ends the poll with the terminal typed error."""

    calls = {"status": 0, "info": 0}

    class FakeCommands:
        async def run(self, command: str, *, opts: Any) -> Any:
            return SimpleNamespace(id="exec-ended")

        async def get_command_status(self, execution_id: str) -> Any:
            calls["status"] += 1
            if calls["status"] == 1:
                return SimpleNamespace(running=True, exit_code=None, error=None)
            ended = _sandbox_ended_api_error("Sandbox sb-bg has ended (Evicted): PodFailed: node pressure")
            raise ended

        async def get_background_command_logs(self, execution_id: str) -> Any:
            raise AssertionError("logs must not be read after the sandbox ended")

    class FakeRaw:
        def __init__(self) -> None:
            self.commands = FakeCommands()

        async def get_info(self) -> Any:
            calls["info"] += 1
            return SimpleNamespace(status=SimpleNamespace(state="Failed", reason="Evicted", message="PodFailed"))

    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": None},
        operations={"background_exec": True, "retries": 2},
    )
    handle = opensandbox_provider.SandboxHandle(sandbox_id="sb-bg", provider_name="opensandbox", raw=FakeRaw())

    with pytest.raises(opensandbox_provider.SandboxEndedError) as exc_info:
        await provider.exec(handle, "sleep 60", timeout_s=30)

    assert exc_info.value.reason == "Evicted"
    assert exc_info.value.sandbox_id == "sb-bg"
    assert calls["status"] == 2  # polled once while running, then stopped on the 410
    assert calls["info"] == 0


@pytest.mark.asyncio
async def test_create_probe_fails_fast_when_the_sandbox_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 410 during the create probe fails the create instead of polling out its deadline."""

    calls = {"run": 0}

    class FakeCommands:
        async def run(self, command: str, *, opts: Any) -> Any:
            calls["run"] += 1
            raise _sandbox_ended_api_error()

    monkeypatch.setattr(
        opensandbox_provider, "_require_opensandbox_sdk", lambda: (object, object, dict, object, object)
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 5},
        probe={"command": "printf ready", "expected_stdout": "ready", "timeout_s": 5, "deadline_s": 60},
        operations={"retries": 3},
    )
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sb-probe", provider_name="opensandbox", raw=SimpleNamespace(commands=FakeCommands())
    )

    with pytest.raises(opensandbox_provider.OpenSandboxCreateError) as exc_info:
        await provider._verify_created_handle(handle)

    cause = exc_info.value.__cause__
    assert isinstance(cause, opensandbox_provider.SandboxEndedError)
    assert cause.reason == "OOMKilled"
    assert calls["run"] == 1  # one probe attempt, not a deadline's worth
    # The buried sandbox is never probed again, but the create may allocate a fresh one.
    assert opensandbox_provider._is_retryable_create_error(cause) is False
    assert opensandbox_provider._is_retryable_create_error(exc_info.value) is True


@pytest.mark.asyncio
async def test_connect_after_create_fails_fast_when_the_sandbox_ends(
    fake_opensandbox_sdk: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 410 while reconnecting after create is terminal, not a connect-poll retry."""

    calls = {"connect": 0}

    class EndedConnectSandbox(FakeSandbox):
        @classmethod
        async def connect(cls, *args: Any, **kwargs: Any) -> "FakeSandbox":
            del args, kwargs
            calls["connect"] += 1
            raise _sandbox_ended_api_error("Sandbox sb-connect has ended (Error): PodFailed")

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (EndedConnectSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        create={"connect_attempt_timeout_s": 0.01, "connect_poll_s": 0.01},
        probe={"command": None},
    )

    with pytest.raises(opensandbox_provider.OpenSandboxCreateError) as exc_info:
        await provider._connect_after_create(
            opensandbox_provider.SandboxHandle(sandbox_id="sb-connect", provider_name="opensandbox", raw=None),
            SandboxSpec(image="image:tag"),
        )

    cause = exc_info.value.__cause__
    assert isinstance(cause, opensandbox_provider.SandboxEndedError)
    assert cause.reason == "Error"
    assert calls["connect"] == 1
    assert not isinstance(exc_info.value, opensandbox_provider.OpenSandboxCreateTimeoutError)
    assert opensandbox_provider._is_retryable_create_error(cause) is False
    assert opensandbox_provider._is_retryable_create_error(exc_info.value) is True


@pytest.mark.asyncio
async def test_create_allocates_a_fresh_sandbox_after_one_ends(
    clean_attribution_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sandbox that ends at boot costs one create attempt, not the whole create."""

    calls = {"create": 0, "connect": 0, "kill": 0, "close": 0}

    class BootDeathSandbox:
        def __init__(self, sandbox_id: str) -> None:
            self.id = sandbox_id

        @classmethod
        async def create(cls, **kwargs: Any) -> "BootDeathSandbox":
            del kwargs
            calls["create"] += 1
            return cls(f"sandbox-{calls['create']}")

        @classmethod
        async def connect(cls, sandbox_id: str, **kwargs: Any) -> "BootDeathSandbox":
            del kwargs
            calls["connect"] += 1
            if calls["connect"] == 1:
                raise _sandbox_ended_api_error("Sandbox sandbox-1 has ended (Evicted): PodFailed: node pressure")
            return cls(sandbox_id)

        async def kill(self) -> None:
            calls["kill"] += 1

        async def close(self) -> None:
            calls["close"] += 1

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (BootDeathSandbox, FakeConnectionConfig, object, FakePlatformSpec, object),
    )
    monkeypatch.setattr(opensandbox_provider.asyncio, "sleep", _no_sleep)
    provider = opensandbox_provider.OpenSandboxProvider(
        # A 60s connect deadline: a sandbox-ended answer that was treated as
        # transient would poll inside the attempt instead of starting a new one.
        create={
            "timeout_s": 60,
            "retries": 1,
            "retry_delay_s": 0.0,
            "retry_max_delay_s": 0.0,
            "skip_health_check": True,
            "connect_attempt_timeout_s": 5,
            "connect_poll_s": 0.01,
        },
        probe={"command": None},
    )

    handle = await provider.create(SandboxSpec(image="image:tag"))

    assert handle.sandbox_id == "sandbox-2"  # the replacement, not the sandbox that ended
    assert calls["create"] == 2
    assert calls["connect"] == 2  # one connect per attempt; the dead one is not polled
    assert calls["kill"] == 1  # the ended sandbox is cleaned up before the retry


@pytest.mark.parametrize(
    ("operations_overrides", "expected_status_timeout_s"),
    [
        ({}, 10.0),  # config default
        ({"status_poll_timeout_s": 5.0}, 5.0),
        ({"status_poll_timeout_s": None}, 120.0),  # falls back to the shared request budget
    ],
)
@pytest.mark.asyncio
async def test_exec_background_status_polls_use_dedicated_timeout(
    monkeypatch: pytest.MonkeyPatch, operations_overrides: dict[str, Any], expected_status_timeout_s: float
) -> None:
    """Status polls get their own short budget; the submit and logs calls keep theirs.

    Status polls are sub-second GETs, so inheriting the shared request budget
    (tuned for long submits) lets each poll against an unreachable sandbox hang
    for minutes. status_poll_timeout_s: None restores the shared budget.
    """

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            return SimpleNamespace(id="exec-budget")

        async def get_command_status(self, execution_id: str) -> Any:
            return SimpleNamespace(running=False, exit_code=0, error=None)

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="ok", cursor=None)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    recorded: dict[str, float | None] = {}
    real_await_sdk_call = opensandbox_provider.OpenSandboxProvider._await_sdk_call

    async def recording_await_sdk_call(
        self: Any, awaitable: Any, *, operation: str, sandbox_id: str, timeout_s: float | None
    ) -> Any:
        recorded[operation] = timeout_s
        return await real_await_sdk_call(
            self, awaitable, operation=operation, sandbox_id=sandbox_id, timeout_s=timeout_s
        )

    monkeypatch.setattr(opensandbox_provider.OpenSandboxProvider, "_await_sdk_call", recording_await_sdk_call)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 120},
        probe={"command": None},
        operations={"background_exec": True, "background_poll_interval_s": 0.01, **operations_overrides},
    )
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-budget", provider_name="opensandbox", raw=SimpleNamespace(commands=FakeCommands())
    )

    result = await provider.exec(handle, "echo ok", timeout_s=30)

    assert result.return_code == 0
    assert recorded["command status"] == expected_status_timeout_s
    # Everything else keeps its existing budget: submit gets timeout_s + 60
    # headroom, logs the shared request budget (payloads can be large).
    assert recorded["command run (background submit)"] == 90.0
    assert recorded["command logs"] == 120.0


@pytest.mark.asyncio
async def test_exec_background_retries_timed_out_status_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed-out status poll retries instead of killing the running command.

    Per-call timeouts are deliberately terminal for submits (a retry could
    double-run the command), so with the short poll budget a single slow poll
    would otherwise fail the whole command; re-polling a status is an
    idempotent GET and must retry within the normal budget.
    """

    class FakeRunCommandOpts:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

    class FakeCommands:
        def __init__(self) -> None:
            self.status_calls: list[str] = []

        async def run(self, command: str, *, opts: FakeRunCommandOpts) -> Any:
            return SimpleNamespace(id="exec-slowpoll")

        async def get_command_status(self, execution_id: str) -> Any:
            self.status_calls.append(execution_id)
            if len(self.status_calls) == 1:
                # Surfaces through _await_sdk_call the same way an expired
                # per-call budget does (asyncio.TimeoutError is TimeoutError).
                raise TimeoutError("simulated status poll budget expiry")
            return SimpleNamespace(running=False, exit_code=0, error=None)

        async def get_background_command_logs(self, execution_id: str) -> Any:
            return SimpleNamespace(content="ok", cursor=None)

    monkeypatch.setattr(
        opensandbox_provider,
        "_require_opensandbox_sdk",
        lambda: (object, object, FakeRunCommandOpts, object, object),
    )
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"request_timeout_s": 120},
        probe={"command": None},
        operations={
            "background_exec": True,
            "retries": 2,
            "retry_delay_s": 0.001,
            "background_poll_interval_s": 0.01,
        },
    )
    commands = FakeCommands()
    handle = opensandbox_provider.SandboxHandle(
        sandbox_id="sandbox-slowpoll", provider_name="opensandbox", raw=SimpleNamespace(commands=commands)
    )

    result = await provider.exec(handle, "echo ok", timeout_s=30)

    assert commands.status_calls == ["exec-slowpoll", "exec-slowpoll"]
    assert result.return_code == 0


def test_tls_verify_reaches_transports(fake_opensandbox_sdk: None) -> None:
    import ssl

    ConnectionConfig = opensandbox_provider.OpenSandboxConnectionConfig
    assert ConnectionConfig().tls_verify is False
    assert ConnectionConfig(tls_verify=True).tls_verify is True

    default = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "httpx"})
    assert default._build_transport()._pool._ssl_context.verify_mode == ssl.CERT_NONE
    verified = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "httpx", "tls_verify": True})
    assert verified._build_transport()._pool._ssl_context.verify_mode == ssl.CERT_REQUIRED

    from nemo_gym.sandbox.providers._http_transport import GymAiohttpTransport

    bridge = opensandbox_provider.OpenSandboxProvider(connection={"transport_backend": "aiohttp"})._build_transport()
    assert isinstance(bridge, GymAiohttpTransport)
    assert bridge.verify is False
    bridge_verified = opensandbox_provider.OpenSandboxProvider(
        connection={"transport_backend": "aiohttp", "tls_verify": True}
    )._build_transport()
    assert bridge_verified.verify is True


@pytest.mark.asyncio
async def test_pty_http_client_follows_tls_verify(fake_opensandbox_sdk: None) -> None:
    default = opensandbox_provider.OpenSandboxProvider()._pty_http_client()
    try:
        assert default.connector._ssl is False
    finally:
        await default.close()

    verified = opensandbox_provider.OpenSandboxProvider(connection={"tls_verify": True})._pty_http_client()
    try:
        assert verified.connector._ssl is True
    finally:
        await verified.close()


def test_split_domain_scheme() -> None:
    split = opensandbox_provider.split_domain_scheme
    assert split("sandbox.example") == ("sandbox.example", None)
    assert split(" sandbox.example\n") == ("sandbox.example", None)
    assert split("https://sandbox.example/") == ("sandbox.example", "https")
    assert split("HTTP://sandbox.example:8080/prefix/") == ("sandbox.example:8080/prefix", "http")
    for bad, message in (
        ("ftp://sandbox.example", "unsupported scheme"),
        ("https://", "no host"),
        ("https://sandbox.example/prefix?token=abc", "query string or fragment"),
        ("https://sandbox.example#staging", "query string or fragment"),
    ):
        with pytest.raises(ValueError, match=message):
            split(bad)


def test_connection_config_derives_protocol_from_domain_scheme() -> None:
    ConnectionConfig = opensandbox_provider.OpenSandboxConnectionConfig

    https = ConnectionConfig(domain="https://sandbox.example/")
    assert (https.domain, https.protocol) == ("sandbox.example", "https")

    # The shipped yaml always sets protocol (http), so a scheme in the domain must win over it.
    override = ConnectionConfig(domain="HTTP://sandbox.example:8080/prefix/", protocol="https")
    assert (override.domain, override.protocol) == ("sandbox.example:8080/prefix", "http")

    bare = ConnectionConfig(domain="sandbox.example", protocol="https")
    assert (bare.domain, bare.protocol) == ("sandbox.example", "https")
    assert ConnectionConfig().domain is None

    # Whitespace is normalized for every domain, not only scheme-carrying ones.
    padded = ConnectionConfig(domain=" sandbox.example\n", protocol="http")
    assert (padded.domain, padded.protocol) == ("sandbox.example", "http")

    with pytest.raises(ValueError, match="unsupported scheme"):
        ConnectionConfig(domain="ftp://sandbox.example")
    with pytest.raises(ValueError, match="no host"):
        ConnectionConfig(domain="https://")
    with pytest.raises(ValueError, match="query string or fragment"):
        ConnectionConfig(domain="https://sandbox.example/prefix?token=abc")
    with pytest.raises(ValueError, match="query string or fragment"):
        ConnectionConfig(domain="https://sandbox.example#staging")


def test_scheme_in_domain_reaches_sdk_and_pty_protocol(fake_opensandbox_sdk: None) -> None:
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={
            "domain": "https://sandbox.example/",
            "api_key": "key",  # pragma: allowlist secret
            "protocol": "http",
        }
    )
    kwargs = provider._connection_config().kwargs
    assert kwargs["domain"] == "sandbox.example"
    assert kwargs["protocol"] == "https"
    # The PTY WebSocket URL is built from the provider's own protocol, not the SDK's base URL.
    assert provider._connection.protocol == "https"


@pytest.mark.parametrize("size", ["true", "8Gi", "67108864"])
async def test_shared_memory_metadata_reaches_create_api(fake_opensandbox_sdk, size):
    from nemo_gym.sandbox.providers.opensandbox.provider import OpenSandboxProvider

    provider = OpenSandboxProvider(attribution={"enabled": False}, probe={"command": None})
    await provider.create(SandboxSpec(image="image:tag", metadata={"nemo.nvidia.com/shm": size}))
    assert FakeSandbox.created_kwargs["metadata"]["nemo.nvidia.com/shm"] == size
