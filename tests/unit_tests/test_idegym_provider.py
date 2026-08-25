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

"""Unit tests for the IdeGYM sandbox provider.

The provider is exercised against a fake :class:`IdeGymSession`, which is the
provider's whole view of IdeGYM. Two things keep that fake honest: a signature
test that binds the provider's session call shapes against the real class, and a
``bash``-backed session that runs the generated scripts for real, so the parts
most likely to be wrong -- the shell quoting, the ``cd``, the base64 chunking --
are tested by executing them rather than by comparing strings.
"""

import asyncio
import builtins
import inspect
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.idegym import naming as idegym_naming
from nemo_gym.sandbox.providers.idegym import provider as idegym_provider
from nemo_gym.sandbox.providers.idegym import session as idegym_session
from nemo_gym.sandbox.providers.idegym.config import (
    MAX_COMMAND_BYTES,
    MAX_UPLOAD_CHUNK_BYTES,
    IdeGymConnectionConfig,
    IdeGymExecConfig,
    IdeGymFilesConfig,
)
from nemo_gym.sandbox.providers.idegym.errors import (
    IdeGymCommandTooLongError,
    IdeGymOperationError,
    IdeGymUnknownServerError,
    is_command_timeout,
    is_retryable,
    is_sandbox_gone,
    orchestrator_status,
)
from nemo_gym.sandbox.providers.idegym.naming import (
    MAX_CLIENT_NAME_LENGTH,
    generate_server_name,
    sanitize_name,
)
from nemo_gym.sandbox.providers.idegym.provider import (
    NO_TIMEOUT_COMMAND_SECONDS,
    SANDBOX_RUNTIME_RETURN_CODE,
    IdeGymProvider,
    _IdeGymSandbox,
)
from nemo_gym.sandbox.providers.idegym.session import IdeGymBashResult, IdeGymServerRef
from nemo_gym.sandbox.providers.idegym.shell import BashScriptBuilder
from nemo_gym.sandbox.providers.idegym.spec import IdeGymProviderOptions, ServerRequestTranslator
from nemo_gym.sandbox.providers.idegym.transfer import Base64BashFileTransfer
from nemo_gym.sandbox.providers.registry import get_provider_class


pytestmark = pytest.mark.sandbox

MAX_SERVER_NAME_LENGTH = 63

requires_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")


def orchestrator_error(status: int, body: str = "boom") -> RuntimeError:
    """An error shaped like the IdeGYM SDK's request failures."""
    return RuntimeError(f"Request failed: url=/api/idegym-servers status={status} reason='x' data='{body}'")


def forward_error(status: int, body: str) -> RuntimeError:
    """An error shaped like the SDK's relayed orchestrator error responses."""
    return RuntimeError(f"Failed to forward request POST /api/x: {{'status_code': {status}, 'body': '{body}'}}")


class FakeSession:
    """Stands in for ``IdeGymSession``; records calls and replays queued results.

    A queued ``BaseException`` is raised instead of returned -- ``BaseException`` so a
    test can inject cancellation -- and the last queued entry repeats, so a test only
    has to queue the interesting prefix.
    """

    def __init__(self) -> None:
        self.client_name = "nemo-gym-test"
        self.start_calls: list[dict[str, Any]] = []
        self.bash_calls: list[dict[str, Any]] = []
        self.capability_calls: list[int] = []
        self.stop_calls: list[dict[str, Any]] = []
        self.health_calls = 0
        self.live_servers: set[int] = set()
        self.start_results: list[Any] = [
            IdeGymServerRef(server_id=7, server_name="nemo-gym-abcdef12", namespace="idegym")
        ]
        self.bash_results: list[Any] = [IdeGymBashResult(stdout="idegym-sandbox-ready", stderr="", exit_code=0)]
        self.capability_results: list[Any] = [["tools"]]
        self.stop_results: list[Any] = [None]
        self.health_results: list[Any] = ["healthy"]

    @staticmethod
    def _next(queue: list[Any]) -> Any:
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(result, BaseException):
            raise result
        return result

    async def health(self) -> str:
        self.health_calls += 1
        return self._next(self.health_results)

    async def start_server(self, request: Any, *, polling: Any, timeout_s: float) -> Any:
        self.start_calls.append({"request": dict(request), "polling": polling, "timeout_s": timeout_s})
        ref = self._next(self.start_results)
        self.live_servers.add(ref.server_id)
        return ref

    async def execute_bash(
        self,
        server_id: int,
        script: str,
        *,
        command_timeout_s: float,
        graceful_termination_timeout_s: float,
        request_timeout_s: float,
        polling: Any,
    ) -> Any:
        self.bash_calls.append(
            {
                "server_id": server_id,
                "script": script,
                "command_timeout_s": command_timeout_s,
                "graceful_termination_timeout_s": graceful_termination_timeout_s,
                "request_timeout_s": request_timeout_s,
            }
        )
        return self._next(self.bash_results)

    async def list_capabilities(self, server_id: int) -> Any:
        self.capability_calls.append(server_id)
        return self._next(self.capability_results)

    async def stop_server(self, server_id: int, *, polling: Any, timeout_s: float) -> Any:
        self.stop_calls.append({"server_id": server_id, "timeout_s": timeout_s})
        # Faithful to the real session's bookkeeping: a server is forgotten only once it
        # is actually stopped, so a retry after a failed stop still addresses it, and
        # only a genuinely unknown id raises IdeGymUnknownServerError.
        if server_id not in self.live_servers:
            raise IdeGymUnknownServerError(f"IdeGYM server {server_id} is not held by this session")
        result = self._next(self.stop_results)
        self.live_servers.discard(server_id)
        return result


class LocalBashSession(FakeSession):
    """A fake session that actually runs the generated scripts with local ``bash``.

    Mirrors the parts of IdeGYM's executor that the generated scripts depend on:
    a fresh ``bash -c`` per call, and stdout/stderr stripped on the way back.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[str] = []

    async def execute_bash(self, server_id: int, script: str, **kwargs: Any) -> Any:
        self.scripts.append(script)
        completed = await asyncio.to_thread(
            subprocess.run, ["bash", "-c", script], capture_output=True, text=True, check=False
        )
        return IdeGymBashResult(
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
            exit_code=completed.returncode,
        )


@pytest.fixture
def fake_session() -> FakeSession:
    return FakeSession()


@pytest.fixture
def make_provider(monkeypatch: pytest.MonkeyPatch, fake_session: FakeSession):
    """Build a provider whose session is the fake, with test-speed timings."""
    released: list[Any] = []

    def factory(session: Any = None, **overrides: Any) -> IdeGymProvider:
        target = session if session is not None else fake_session

        async def fake_acquire(connection: Any, attribution: Any) -> Any:
            return target

        async def fake_release(released_session: Any) -> None:
            released.append(released_session)

        monkeypatch.setattr(idegym_provider, "acquire_session", fake_acquire)
        monkeypatch.setattr(idegym_provider, "release_session", fake_release)
        kwargs: dict[str, Any] = {
            "create": {"retry_delay_s": 0.0, "retry_max_delay_s": 0.0, "ready_timeout_s": 30},
            "operations": {"retry_delay_s": 0.0, "retry_max_delay_s": 0.0, "close_timeout_s": 5.0},
        }
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(kwargs.get(key), dict):
                kwargs[key] = {**kwargs[key], **value}
            else:
                kwargs[key] = value
        provider = IdeGymProvider(**kwargs)
        provider.released = released  # type: ignore[attr-defined]
        return provider

    return factory


def make_handle(
    *,
    server_id: int = 7,
    workdir: str | None = None,
    env: dict[str, str] | None = None,
    stopped: bool = False,
) -> SandboxHandle:
    return SandboxHandle(
        sandbox_id=str(server_id),
        provider_name="idegym",
        raw=_IdeGymSandbox(
            server_id=server_id,
            server_name="nemo-gym-abcdef12",
            namespace="idegym",
            image="registry.example.com/idegym/env:1",
            workdir=workdir,
            env=env or {},
            stopped=stopped,
        ),
    )


def spec(**overrides: Any) -> SandboxSpec:
    values: dict[str, Any] = {"image": "registry.example.com/idegym/env:1"}
    values.update(overrides)
    return SandboxSpec(**values)


# --- registration and dependency ------------------------------------------


def test_registry_resolves_idegym() -> None:
    assert get_provider_class("idegym") is IdeGymProvider


def test_missing_idegym_dependency_message(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(
        name: str,
        globals_: dict[str, Any] | None = None,
        locals_: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "idegym" or name.startswith("idegym."):
            raise ModuleNotFoundError(name)
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ModuleNotFoundError, match=r"nemo-gym\[idegym\]"):
        idegym_session.require_idegym_client()


def test_provider_binds_the_session_api_it_calls() -> None:
    """The fake session is only trustworthy if it matches the real one's shape.

    Binding the provider's exact call shapes against ``IdeGymSession`` means a
    signature change there fails these tests instead of passing against a fake
    that has quietly drifted.
    """
    real = idegym_session.IdeGymSession
    inspect.signature(real.start_server).bind(None, {}, polling=None, timeout_s=1.0)
    inspect.signature(real.execute_bash).bind(
        None,
        1,
        "true",
        command_timeout_s=1.0,
        graceful_termination_timeout_s=2.0,
        request_timeout_s=3.0,
        polling=None,
    )
    inspect.signature(real.list_capabilities).bind(None, 1)
    inspect.signature(real.stop_server).bind(None, 1, polling=None, timeout_s=1.0)
    inspect.signature(real.health).bind(None)
    for name in ("start_server", "execute_bash", "list_capabilities", "stop_server", "health"):
        # Names and kinds only: the fake annotates loosely on purpose, but a
        # renamed, reordered, added or removed parameter has to fail here.
        def parameters(owner: type) -> list[tuple[str, Any]]:
            return [(p.name, p.kind) for p in inspect.signature(getattr(owner, name)).parameters.values()]

        assert parameters(real) == parameters(FakeSession), name


# --- config ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "kwargs"),
    [
        ("connection", {"orchestrator_url": ""}),
        ("connection", {"namespace": ""}),
        ("connection", {"client_name": "  "}),
        ("connection", {"nodes_count": -1}),
        ("connection", {"heartbeat_interval_s": 0}),
        ("connection", {"request_timeout_s": 0}),
        ("connection", {"transport_backend": "curl"}),
        ("connection", {"max_connections": 0}),
        ("connection", {"max_keepalive_connections": -1}),
        ("connection", {"keepalive_expiry_s": 0}),
        ("connection", {"connect_retries": -1}),
        ("connection", {"tracing_timeout_s": 0}),
        ("create", {"ready_timeout_s": 0}),
        ("create", {"retries": -1}),
        ("create", {"retry_delay_s": -1}),
        ("create", {"retry_max_delay_s": -1}),
        ("create", {"busy_retry_delay_s": -1}),
        ("create", {"server_name_prefix": ""}),
        ("create", {"service_port": 70000}),
        ("create", {"container_port": -1}),
        ("create", {"max_restarts": -1}),
        ("create", {"polling": {"initial_delay_s": 0}}),
        ("create", {"polling": {"interval_s": -1}}),
        ("create", {"polling": {"backoff_factor": 0.5}}),
        ("create", {"polling": {"max_delay_s": 0}}),
        ("exec", {"default_timeout_s": 0}),
        ("exec", {"graceful_termination_timeout_s": -1}),
        ("exec", {"request_overhead_s": -1}),
        ("exec", {"user_mode": "sudo"}),
        ("verify", {"timeout_s": 0}),
        ("files", {"upload_chunk_bytes": 0}),
        ("files", {"upload_chunk_bytes": MAX_UPLOAD_CHUNK_BYTES + 1}),
        ("files", {"download_chunk_bytes": 0}),
        ("files", {"max_download_bytes": 0}),
        ("files", {"timeout_s": 0}),
        ("operations", {"close_timeout_s": 0}),
        ("operations", {"status_timeout_s": 0}),
        ("operations", {"retries": -1}),
        ("operations", {"retry_delay_s": -1}),
        ("operations", {"retry_max_delay_s": -1}),
        ("attribution", {"client_name_prefix": ""}),
    ],
)
def test_config_validation_rejects_bad_values(group: str, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        IdeGymProvider(**{group: kwargs})


def test_config_coercion_and_defaults() -> None:
    provider = IdeGymProvider(connection={"orchestrator_url": "idegym.example", "namespace": "gym"})
    assert provider._connection == IdeGymConnectionConfig(orchestrator_url="idegym.example", namespace="gym")
    assert provider._connection.tracing_enabled is False
    assert provider._files == IdeGymFilesConfig()


def test_invalid_config_type_raises() -> None:
    with pytest.raises(TypeError, match="IdeGymConnectionConfig"):
        IdeGymProvider(connection=["not", "a", "mapping"])


def test_provider_options_validation() -> None:
    with pytest.raises(ValueError, match="Unknown idegym provider_options keys"):
        IdeGymProviderOptions.from_mapping({"nope": 1})
    with pytest.raises(TypeError, match="node_selector"):
        IdeGymProviderOptions.from_mapping({"node_selector": ["a"]})
    with pytest.raises(TypeError, match="pod_overrides"):
        IdeGymProviderOptions.from_mapping({"pod_overrides": []})
    with pytest.raises(TypeError, match="snapshot"):
        IdeGymProviderOptions.from_mapping({"snapshot": "latest"})
    with pytest.raises(TypeError, match="volumes.*list of mappings"):
        IdeGymProviderOptions.from_mapping({"volumes": {"name": "x"}})
    with pytest.raises(TypeError, match=r"volume_mounts'\]\[0\]"):
        IdeGymProviderOptions.from_mapping({"volume_mounts": ["x"]})
    assert IdeGymProviderOptions.from_mapping(None) == IdeGymProviderOptions()


# --- error classification --------------------------------------------------


@pytest.mark.parametrize(
    ("error", "status", "gone", "retryable"),
    [
        (orchestrator_error(404), 404, True, False),
        (orchestrator_error(410), 410, True, False),
        (orchestrator_error(429), 429, False, True),
        (orchestrator_error(503), 503, False, True),
        (orchestrator_error(400), 400, False, False),
        (forward_error(410, "gone"), 410, True, False),
        (RuntimeError("no status here"), None, False, False),
        (ConnectionError("refused"), None, False, True),
    ],
)
def test_error_classification(error: Exception, status: int | None, gone: bool, retryable: bool) -> None:
    assert orchestrator_status(error) == status
    assert is_sandbox_gone(error) is gone
    assert is_retryable(error) is retryable


def test_command_timeout_classification() -> None:
    assert is_command_timeout(TimeoutError("client gave up")) is True
    assert is_command_timeout(forward_error(500, "Command execution timed out after 5 seconds")) is True
    assert is_command_timeout(orchestrator_error(500, "kaboom")) is False


def test_transport_failures_are_retryable() -> None:
    """The SDK lets httpx's exceptions through, and none of them are builtins."""
    import httpx

    for error in (httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), httpx.PoolTimeout("full")):
        assert is_retryable(error) is True, error
    # A read timeout means we stopped waiting; failing to connect at all does not.
    assert is_command_timeout(httpx.ReadTimeout("slow")) is True
    assert is_command_timeout(httpx.ConnectTimeout("unreachable")) is False


# --- naming ----------------------------------------------------------------


def test_sanitize_and_clamp_names() -> None:
    assert sanitize_name("  My Team/Foo  ") == "my-team-foo"
    assert sanitize_name("///") == ""
    long_name = idegym_session.clamp_client_name("a" * 200)
    assert len(long_name) == MAX_CLIENT_NAME_LENGTH


def test_generated_server_names_leave_room_for_the_appended_server_id() -> None:
    name = generate_server_name("nemo-gym", ["django__django-11099"])
    assert name == "nemo-gym-django-django-11099"
    # Room must be left for the `-<server_id>` suffix the orchestrator appends.
    assert len(name) + len("-9999999") <= MAX_SERVER_NAME_LENGTH
    # A hint too long for that budget is truncated rather than overflowing it.
    assert len(generate_server_name("nemo-gym", ["x" * 200])) + len("-9999999") <= MAX_SERVER_NAME_LENGTH


def test_generated_server_names_satisfy_rfc1035() -> None:
    import re

    pattern = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
    for prefix, hints in (
        ("nemo-gym", ["A_VERY::Odd/Instance ID"]),
        ("nemo-gym", ["x" * 200]),
        ("9bad", ["also-bad"]),
        ("---", []),
    ):
        assert pattern.match(generate_server_name(prefix, hints)), (prefix, hints)


def test_server_id_reserve_must_leave_room_for_a_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raising the reserve to the 63-character cap must fail loudly.

    The budget is fixed by the length constants, so the alternative to raising is an
    empty name that Kubernetes then rejects.
    """
    monkeypatch.setattr(idegym_naming, "SERVER_ID_SUFFIX_RESERVE", MAX_SERVER_NAME_LENGTH)
    with pytest.raises(ValueError, match="leaves no room for a name"):
        generate_server_name("nemo-gym", [])


# --- spec translation ------------------------------------------------------


def test_translate_maps_resources_to_limits_and_requests() -> None:
    translator = ServerRequestTranslator(idegym_provider.IdeGymCreateConfig())
    request = translator.translate(
        spec(resources={"cpu": 2, "memory_mib": 8192, "disk_gib": 30}, metadata={"instance_id": "task-9"}),
        IdeGymProviderOptions.from_mapping({"resource_requests": {"cpu": 0.5, "memory_mib": 2048}}),
    )
    assert request["resources"] == {
        "requests": {"cpu": "500m", "memory": "2048Mi"},
        "limits": {"cpu": "2", "memory": "8192Mi", "ephemeral-storage": "30Gi"},
    }
    assert request["server_name"] == "nemo-gym-task-9"
    assert request["reuse_strategy"] == "NONE"


def test_translate_passes_pod_shape_options_through() -> None:
    translator = ServerRequestTranslator(idegym_provider.IdeGymCreateConfig())
    request = translator.translate(
        spec(),
        IdeGymProviderOptions.from_mapping(
            {
                "runtime_class_name": "gvisor",
                "node_selector": {"pool": "sandbox"},
                "volumes": [{"name": "creds", "secret": {"secretName": "creds"}}],  # pragma: allowlist secret
                "volume_mounts": [{"name": "creds", "mountPath": "/etc/creds"}],
                "env_from": [{"secretRef": {"name": "creds"}}],
                "service_account_name": "runner",
                "pod_overrides": {"tolerations": [{"key": "dedicated", "operator": "Exists"}]},
                "server_kind": "idegym",
                "snapshot": {"id": "17"},
                "max_restarts": 3,
                "reuse_strategy": "RESET",
                "server_name": "pinned-name",
                "service_port": 8080,
                "container_port": 9000,
                "run_as_root": True,
            }
        ),
    )
    assert request["server_name"] == "pinned-name"
    assert request["runtime_class_name"] == "gvisor"
    assert request["node_selector"] == {"pool": "sandbox"}
    assert request["volumes"] == [{"name": "creds", "secret": {"secretName": "creds"}}]  # pragma: allowlist secret
    assert request["env_from"] == [{"secretRef": {"name": "creds"}}]
    assert request["pod_overrides"]["tolerations"][0]["key"] == "dedicated"
    assert request["snapshot"] == {"id": "17"}
    assert (request["max_restarts"], request["service_port"], request["container_port"]) == (3, 8080, 9000)
    assert request["run_as_root"] is True
    assert "resources" not in request


def test_translate_normalizes_and_validates_the_image() -> None:
    translator = ServerRequestTranslator(idegym_provider.IdeGymCreateConfig())
    request = translator.translate(spec(image="docker://reg.example/env:1"), IdeGymProviderOptions())
    assert request["image_tag"] == "reg.example/env:1"
    with pytest.raises(SandboxCreateError, match="not a valid OCI image reference"):
        translator.translate(spec(image="Reg.Example/Env:1"), IdeGymProviderOptions())
    with pytest.raises(SandboxCreateError, match="spec.image is required"):
        translator.translate(spec(image=None), IdeGymProviderOptions())


def test_translate_rejects_entrypoint_and_warns_about_unsupported_fields(caplog: pytest.LogCaptureFixture) -> None:
    translator = ServerRequestTranslator(idegym_provider.IdeGymCreateConfig())
    with pytest.raises(SandboxCreateError, match="spec.entrypoint is not supported"):
        translator.translate(spec(entrypoint=["/bin/sh"]), IdeGymProviderOptions())
    unsupported = spec(ttl_s=60, ports=[8080], resources={"gpu": 1})
    with caplog.at_level("WARNING"):
        translator.translate(unsupported, IdeGymProviderOptions())
        translator.translate(unsupported, IdeGymProviderOptions())
    # Once each, not once per sandbox: a benchmark translates one spec per task, and
    # the agent config that ships with mini_swe_agent_2 sets ttl_s.
    assert caplog.text.count("spec.ttl_s is not enforced") == 1
    assert caplog.text.count("sandbox.endpoint() is unavailable") == 1
    assert caplog.text.count("cannot map GPU resource requests") == 1


@pytest.mark.parametrize("value", [0, -1])
def test_translate_rejects_non_positive_resources(value: int) -> None:
    translator = ServerRequestTranslator(idegym_provider.IdeGymCreateConfig())
    for key in ("cpu", "memory_mib", "disk_gib"):
        with pytest.raises(SandboxCreateError, match="greater than zero"):
            translator.translate(spec(resources={key: value}), IdeGymProviderOptions())


# --- create ----------------------------------------------------------------


async def test_create_verifies_the_workdir_and_returns_a_ready_handle(
    make_provider, fake_session: FakeSession
) -> None:
    provider = make_provider()
    handle = await provider.create(
        spec(workdir="/testbed", env={"FOO": "bar"}, ready_timeout_s=99, metadata={"instance_id": "task-1"})
    )
    assert handle.provider_name == "idegym"
    assert handle.sandbox_id == "7"
    assert handle.raw.workdir == "/testbed"
    assert handle.raw.env == {"FOO": "bar"}
    assert fake_session.start_calls[0]["timeout_s"] == pytest.approx(99, abs=1)
    # IdeGYM already waited for pod readiness, so the workdir check is the only
    # command create runs -- and it must not `cd` into the directory it is verifying.
    assert len(fake_session.bash_calls) == 1
    assert "[ -d /testbed ]" in fake_session.bash_calls[0]["script"]
    assert "cd -- /testbed" not in fake_session.bash_calls[0]["script"]


async def test_create_falls_back_to_configured_ready_timeout(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(create={"ready_timeout_s": 45})
    await provider.create(spec())
    assert fake_session.start_calls[0]["timeout_s"] == pytest.approx(45, abs=1)


async def test_create_fails_when_the_workdir_is_unusable(
    make_provider, fake_session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    fake_session.bash_results = [IdeGymBashResult(stdout="", stderr="", exit_code=1)]
    provider = make_provider()
    with caplog.at_level("WARNING"), pytest.raises(SandboxCreateVerificationError, match="is not usable"):
        await provider.create(spec(workdir="/nope"))
    assert len(fake_session.bash_calls) == 1
    assert fake_session.stop_calls == [{"server_id": 7, "timeout_s": 5.0}]
    # Nothing else records that a pod was started and thrown away.
    assert "Tearing down the IdeGYM sandbox" in caplog.text


async def test_create_logs_the_teardown_when_it_is_cancelled(
    make_provider, fake_session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    """A cancelled create still tears down a live pod, and CancelledError says nothing."""
    fake_session.bash_results = [asyncio.CancelledError()]
    provider = make_provider()
    with caplog.at_level("WARNING"), pytest.raises(asyncio.CancelledError):
        await provider.create(spec(workdir="/testbed"))
    assert "Tearing down the IdeGYM sandbox" in caplog.text
    assert "CancelledError" in caplog.text
    assert fake_session.stop_calls


async def test_create_runs_no_command_when_there_is_no_workdir_to_check(
    make_provider, fake_session: FakeSession
) -> None:
    provider = make_provider(verify={"check_workdir": False})
    await provider.create(spec(workdir="/testbed"))
    assert fake_session.bash_calls == []
    await provider.create(spec())
    assert fake_session.bash_calls == []


async def test_create_retries_transient_failures_keeping_the_generated_name(
    make_provider, fake_session: FakeSession
) -> None:
    """Renaming would defeat a RESTART/RESET reuse strategy, which matches on the name."""
    fake_session.start_results = [
        orchestrator_error(429),
        IdeGymServerRef(server_id=9, server_name="nemo-gym-second", namespace="idegym"),
    ]
    provider = make_provider()
    handle = await provider.create(spec())
    assert handle.sandbox_id == "9"
    first, second = (call["request"]["server_name"] for call in fake_session.start_calls)
    assert first == second


async def test_create_keeps_a_pinned_name_across_retries(make_provider, fake_session: FakeSession) -> None:
    fake_session.start_results = [
        orchestrator_error(503),
        IdeGymServerRef(server_id=9, server_name="pinned", namespace="idegym"),
    ]
    provider = make_provider()
    await provider.create(spec(provider_options={"server_name": "pinned"}))
    assert [call["request"]["server_name"] for call in fake_session.start_calls] == ["pinned", "pinned"]


async def test_create_exhausted_retries_are_wrapped(make_provider, fake_session: FakeSession) -> None:
    fake_session.start_results = [orchestrator_error(429)]
    provider = make_provider(create={"retries": 1})
    with pytest.raises(SandboxCreateError, match="could not start a server"):
        await provider.create(spec())
    assert len(fake_session.start_calls) == 2


async def test_create_does_not_retry_an_exhausted_readiness_budget(make_provider, fake_session: FakeSession) -> None:
    """Retrying would silently multiply ready_timeout_s by the retry count."""
    fake_session.start_results = [TimeoutError("Server start timed out after 1200 seconds")]
    provider = make_provider(create={"retries": 3})
    with pytest.raises(SandboxCreateError, match="could not start a server"):
        await provider.create(spec())
    assert len(fake_session.start_calls) == 1


async def test_create_retries_a_transport_timeout(make_provider, fake_session: FakeSession) -> None:
    """A transport timeout is one lost request, not the whole budget."""
    import httpx

    fake_session.start_results = [
        httpx.ReadTimeout("orchestrator slow"),
        IdeGymServerRef(server_id=9, server_name="nemo-gym-second", namespace="idegym"),
    ]
    provider = make_provider()
    handle = await provider.create(spec())
    assert handle.sandbox_id == "9"
    assert len(fake_session.start_calls) == 2


async def test_create_non_retryable_failure_is_not_retried(make_provider, fake_session: FakeSession) -> None:
    fake_session.start_results = [orchestrator_error(400, "bad image")]
    provider = make_provider()
    with pytest.raises(SandboxCreateError, match="could not start a server"):
        await provider.create(spec())
    assert len(fake_session.start_calls) == 1


async def test_create_teardown_failure_only_warns(
    make_provider, fake_session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    fake_session.bash_results = [IdeGymBashResult(stdout="", stderr="broken", exit_code=1)]
    fake_session.stop_results = [orchestrator_error(500)]
    provider = make_provider()
    with caplog.at_level("WARNING"), pytest.raises(SandboxCreateVerificationError):
        await provider.create(spec(workdir="/nope"))
    assert "may be left running on the cluster" in caplog.text


# --- exec ------------------------------------------------------------------


async def test_exec_shapes_the_script_and_the_timeouts(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(verify={"check_workdir": False})
    handle = await provider.create(spec(workdir="/testbed", env={"CONDA_ENV": "testbed"}))
    fake_session.bash_results = [IdeGymBashResult(stdout="out", stderr="err", exit_code=3)]
    # A value needing quoting proves the export is shell-safe; the call's env is
    # merged on top of the sandbox's.
    result = await provider.exec(handle, "pytest -q", env={"EXTRA": "a b'c"}, timeout_s=120)
    assert result == SandboxExecResult(stdout="out", stderr="err", return_code=3)
    call = fake_session.bash_calls[-1]
    assert call["script"] == (
        "{ :\ncd -- /testbed || exit 1\nexport CONDA_ENV=testbed\nexport EXTRA='a b'\"'\"'c'\npytest -q\n}"
    )
    assert call["command_timeout_s"] == 120
    # The client waits longer than the sandbox, so the sandbox's own timeout wins
    # and the caller gets output instead of a transport error.
    assert call["request_timeout_s"] == 120 + 60


async def test_exec_cwd_argument_overrides_the_sandbox_workdir(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(verify={"check_workdir": False})
    handle = await provider.create(spec(workdir="/testbed"))
    await provider.exec(handle, "true", cwd="/other")
    assert "cd -- /other" in fake_session.bash_calls[-1]["script"]


async def test_exec_without_timeout_uses_a_finite_ceiling(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(exec={"default_timeout_s": None})
    handle = await provider.create(spec())
    await provider.exec(handle, "true")
    assert fake_session.bash_calls[-1]["command_timeout_s"] == NO_TIMEOUT_COMMAND_SECONDS


async def test_exec_reports_a_sandbox_side_timeout(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [forward_error(500, "Command execution timed out after 5 seconds")]
    result = await provider.exec(handle, "sleep 100", timeout_s=5)
    assert result.error_type == "timeout"
    assert result.return_code == SANDBOX_RUNTIME_RETURN_CODE


async def test_exec_on_a_deleted_sandbox_marks_it_stopped(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [orchestrator_error(410, "gone")]
    result = await provider.exec(handle, "true")
    assert result.error_type == "sandbox"
    assert handle.raw.stopped is True
    assert await provider.status(handle) is SandboxStatus.STOPPED
    # A stopped sandbox short-circuits rather than making another round trip.
    calls_before = len(fake_session.bash_calls)
    assert (await provider.exec(handle, "true")).error_type == "sandbox"
    assert len(fake_session.bash_calls) == calls_before


async def test_exec_other_failures_are_reported_not_raised(
    make_provider, fake_session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    """An unclassified failure becomes a return value, so the log is the only traceback."""
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [orchestrator_error(500, "kaboom")]
    with caplog.at_level("WARNING"):
        result = await provider.exec(handle, "true")
    assert (result.error_type, result.return_code) == ("sandbox", SANDBOX_RUNTIME_RETURN_CODE)
    assert "failed to run the command" in caplog.text
    assert "Traceback" in caplog.text


async def test_exec_reports_an_undeliverable_command_instead_of_raising(make_provider) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    too_long = await provider.exec(handle, "echo " + "x" * MAX_COMMAND_BYTES)
    assert too_long.error_type == "command_too_long"
    assert "byte limit" in (too_long.stderr or "")
    bad_env = await provider.exec(handle, "true", env={"NOT AN IDENT": "v"})
    assert bad_env.error_type == "invalid_request"


async def test_exec_user_request_warns_once(
    make_provider, fake_session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    with caplog.at_level("WARNING"):
        await provider.exec(handle, "true", user="root")
        await provider.exec(handle, "true", user="root")
    assert caplog.text.count("cannot run commands as user") == 1
    assert "runuser" not in fake_session.bash_calls[-1]["script"]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("runuser", "exec runuser -u tester -- bash -c"), ("su", "exec su -s /bin/bash -c")],
)
def test_user_switch_modes_wrap_the_script(mode: str, expected: str) -> None:
    """Both modes pin bash: a service account's login shell may be /sbin/nologin."""
    builder = BashScriptBuilder(IdeGymExecConfig(user_mode=mode))
    script = builder.build("true", cwd="/w", user="tester")
    assert expected in script
    assert "tester" in script
    assert "cd -- /w" in script


@requires_bash
@pytest.mark.parametrize("mode", ["runuser", "su"])
def test_a_user_switched_script_is_valid_bash(mode: str) -> None:
    """The wrap nests two levels of quoting, which a substring assertion cannot check."""
    script = BashScriptBuilder(IdeGymExecConfig(user_mode=mode)).build(
        "echo hi && ls 'a b'", cwd="/testbed", env={"FOO": "a b"}, user="tester"
    )
    completed = subprocess.run(["bash", "-n", "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("mode", ["runuser", "su"])
def test_user_switch_modes_reject_numeric_ids(mode: str) -> None:
    builder = BashScriptBuilder(IdeGymExecConfig(user_mode=mode))
    with pytest.raises(ValueError, match="needs a user name"):
        builder.build("true", user=0)


def test_script_size_limit_is_enforced() -> None:
    builder = BashScriptBuilder(IdeGymExecConfig())
    with pytest.raises(IdeGymCommandTooLongError, match="one argument"):
        builder.build("x" * (MAX_COMMAND_BYTES + 1))


@requires_bash
@pytest.mark.parametrize("command", ["", "   ", "\n", "# just a comment"])
def test_a_command_with_nothing_to_run_is_not_a_syntax_error(command: str) -> None:
    """An empty brace group is a bash syntax error, not a no-op.

    A model that emits a blank or comment-only action would otherwise be told its
    command had a shell syntax error.
    """
    script = BashScriptBuilder(IdeGymExecConfig()).build(command)
    completed = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "syntax error" not in completed.stderr


@requires_bash
def test_the_largest_permitted_upload_chunk_still_fits_in_one_command() -> None:
    """The validator's ceiling has to leave room for the rest of the script."""
    files = IdeGymFilesConfig(upload_chunk_bytes=MAX_UPLOAD_CHUNK_BYTES)
    transfer = Base64BashFileTransfer(files, run=None)  # type: ignore[arg-type]
    chunk_script = transfer._write_chunk_script(
        "/testbed/nested/blob.bin", b"x" * files.upload_chunk_bytes, first=True
    )
    script = BashScriptBuilder(IdeGymExecConfig()).build(chunk_script, cwd="/testbed")
    assert len(script.encode()) <= MAX_COMMAND_BYTES


@pytest.mark.parametrize("name", ["FOO\n", "FO O", "1FOO", ""])
def test_env_names_that_export_cannot_carry_are_rejected(name: str) -> None:
    """A `$`-anchored check would accept a trailing newline and corrupt the script."""
    builder = BashScriptBuilder(IdeGymExecConfig())
    with pytest.raises(ValueError, match="not a valid shell identifier"):
        builder.build("true", env={name: "v"})


def test_an_image_with_a_trailing_newline_is_rejected() -> None:
    translator = ServerRequestTranslator(idegym_provider.IdeGymCreateConfig())
    with pytest.raises(SandboxCreateError, match="not a valid OCI image reference"):
        translator.translate(spec(image="reg.example/env:1\n"), IdeGymProviderOptions())


def test_a_relayed_sandbox_body_is_not_read_as_the_orchestrator_status() -> None:
    """The body carries the sandbox's own output; scraping it would kill live sandboxes."""
    relayed = forward_error(500, "curl failed: status=404 not found")
    assert orchestrator_status(relayed) == 500
    assert is_sandbox_gone(relayed) is False
    # Likewise for the timeout marker: ordinary tool output must not read as a timeout.
    assert is_command_timeout(forward_error(500, "pytest: test_foo timed out after 30 seconds")) is False
    assert is_command_timeout(forward_error(500, "Command execution timed out after 5 seconds")) is True


# --- files -----------------------------------------------------------------


@requires_bash
async def test_file_transfer_round_trips_binary_content(make_provider, tmp_path: Path) -> None:
    session = LocalBashSession()
    # Chunk sizes far below the defaults, so the chunk-boundary handling is what is
    # under test rather than a single-chunk shortcut.
    provider = make_provider(
        session,
        verify={"check_workdir": False},
        files={"upload_chunk_bytes": 100, "download_chunk_bytes": 60},
    )
    remote = tmp_path / "sandbox"
    remote.mkdir()
    handle = await provider.create(spec(workdir=str(remote)))
    payload = bytes(range(256)) * 3
    source = tmp_path / "blob.bin"
    source.write_bytes(payload)

    await provider.upload_file(handle, source, str(remote / "nested/dir/blob.bin"))
    assert (remote / "nested/dir/blob.bin").read_bytes() == payload

    out = tmp_path / "back.bin"
    await provider.download_file(handle, str(remote / "nested/dir/blob.bin"), out)
    assert out.read_bytes() == payload


@requires_bash
async def test_file_transfer_handles_empty_files_and_odd_names(make_provider, tmp_path: Path) -> None:
    provider = make_provider(LocalBashSession(), verify={"check_workdir": False})
    handle = await provider.create(spec())
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    target = tmp_path / "-dash name'with quote.bin"
    await provider.upload_file(handle, empty, str(target))
    assert target.exists() and target.stat().st_size == 0
    out = tmp_path / "empty-back"
    await provider.download_file(handle, str(target), out)
    assert out.read_bytes() == b""


@requires_bash
async def test_upload_resolves_relative_paths_against_the_workdir(make_provider, tmp_path: Path) -> None:
    provider = make_provider(LocalBashSession(), verify={"check_workdir": False})
    handle = await provider.create(spec(workdir=str(tmp_path)))
    source = tmp_path / "src.txt"
    source.write_text("hello\n")
    await provider.upload_file(handle, source, "relative/target.txt")
    assert (tmp_path / "relative/target.txt").read_text() == "hello\n"


@requires_bash
async def test_download_of_a_missing_file_raises(make_provider, tmp_path: Path) -> None:
    provider = make_provider(LocalBashSession(), verify={"check_workdir": False})
    handle = await provider.create(spec())
    with pytest.raises(idegym_provider.IdeGymOperationError, match="Cannot read"):
        await provider.download_file(handle, str(tmp_path / "nope"), tmp_path / "out")


@requires_bash
async def test_download_refuses_files_over_the_configured_cap(make_provider, tmp_path: Path) -> None:
    provider = make_provider(LocalBashSession(), verify={"check_workdir": False}, files={"max_download_bytes": 8})
    handle = await provider.create(spec())
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * 64)
    with pytest.raises(idegym_provider.IdeGymOperationError, match="max_download_bytes"):
        await provider.download_file(handle, str(big), tmp_path / "out")


async def test_upload_failure_reports_the_failing_chunk(
    make_provider, fake_session: FakeSession, tmp_path: Path
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    source = tmp_path / "src.bin"
    source.write_bytes(b"abcdefghij")
    fake_session.bash_results = [IdeGymBashResult(stdout="", stderr="disk full", exit_code=1)]
    with pytest.raises(idegym_provider.IdeGymOperationError, match="failed on chunk 1/"):
        await provider.upload_file(handle, source, "/tmp/x.bin")


async def test_upload_failure_names_a_later_chunk(make_provider, fake_session: FakeSession, tmp_path: Path) -> None:
    """The reported index counts chunks, and a later chunk must still be attempted."""
    provider = make_provider(files={"upload_chunk_bytes": 4})
    handle = await provider.create(spec())
    source = tmp_path / "src.bin"
    source.write_bytes(b"abcdefghij")
    fake_session.bash_results = [
        IdeGymBashResult(stdout="", stderr="", exit_code=0),
        IdeGymBashResult(stdout="", stderr="disk full", exit_code=1),
    ]
    with pytest.raises(idegym_provider.IdeGymOperationError, match="failed on chunk 2/3"):
        await provider.upload_file(handle, source, "/tmp/x.bin")


@requires_bash
async def test_download_without_a_cap_is_uncapped(make_provider, tmp_path: Path) -> None:
    """`max_download_bytes: null` disables the check rather than comparing against None."""
    provider = make_provider(
        LocalBashSession(),
        verify={"check_workdir": False},
        files={"max_download_bytes": None, "download_chunk_bytes": 16},
    )
    handle = await provider.create(spec())
    payload = bytes(range(64))
    source = tmp_path / "blob.bin"
    source.write_bytes(payload)
    out = tmp_path / "back.bin"
    await provider.download_file(handle, str(source), out)
    assert out.read_bytes() == payload


async def test_download_rejects_unparsable_size_and_bad_base64(
    make_provider, fake_session: FakeSession, tmp_path: Path
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [IdeGymBashResult(stdout="not-a-number", stderr="", exit_code=0)]
    with pytest.raises(idegym_provider.IdeGymOperationError, match="unparsable size"):
        await provider.download_file(handle, "/tmp/x", tmp_path / "out")

    fake_session.bash_results = [
        IdeGymBashResult(stdout="4", stderr="", exit_code=0),
        IdeGymBashResult(stdout="!!!not base64!!!", stderr="", exit_code=0),
    ]
    with pytest.raises(idegym_provider.IdeGymOperationError, match="invalid base64"):
        await provider.download_file(handle, "/tmp/x", tmp_path / "out")


async def test_download_chunk_failure_names_the_offset(
    make_provider, fake_session: FakeSession, tmp_path: Path
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [
        IdeGymBashResult(stdout="16", stderr="", exit_code=0),
        IdeGymBashResult(stdout="", stderr="I/O error", exit_code=1),
    ]
    with pytest.raises(idegym_provider.IdeGymOperationError, match="at offset 0"):
        await provider.download_file(handle, "/tmp/x", tmp_path / "out")


async def test_download_short_read_is_detected(make_provider, fake_session: FakeSession, tmp_path: Path) -> None:
    import base64

    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [
        IdeGymBashResult(stdout="8", stderr="", exit_code=0),
        IdeGymBashResult(stdout=base64.b64encode(b"abc").decode(), stderr="", exit_code=0),
    ]
    with pytest.raises(idegym_provider.IdeGymOperationError, match="Expected 8 bytes"):
        await provider.download_file(handle, "/tmp/x", tmp_path / "out")


# --- status ----------------------------------------------------------------


async def test_status_running_stopped_and_unknown(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    assert await provider.status(handle) is SandboxStatus.RUNNING

    fake_session.capability_results = [orchestrator_error(404)]
    assert await provider.status(handle) is SandboxStatus.STOPPED

    fake_session.capability_results = [orchestrator_error(500)]
    assert await provider.status(handle) is SandboxStatus.UNKNOWN


async def test_status_of_a_closed_sandbox_is_stopped(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    await provider.close(handle)
    assert await provider.status(handle) is SandboxStatus.STOPPED
    assert fake_session.capability_calls == []


# --- teardown --------------------------------------------------------------


async def test_close_is_idempotent(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    await provider.close(handle)
    await provider.close(handle)
    assert len(fake_session.stop_calls) == 1


async def test_close_treats_an_already_gone_server_as_success(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.stop_results = [orchestrator_error(404)]
    await provider.close(handle)


async def test_close_treats_a_server_the_session_forgot_as_success(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.live_servers.clear()
    await provider.close(handle)
    assert handle.raw.stopped is True


async def test_a_failed_stop_can_be_retried_against_the_same_server(make_provider, fake_session: FakeSession) -> None:
    """A failed stop must not make the pod unaddressable.

    The session forgets a server only once it is really stopped, so a caller retrying
    after a failure reaches the same pod rather than an "already stopped" answer that
    would leave it running.
    """
    provider = make_provider(operations={"retries": 0})
    handle = await provider.create(spec())
    fake_session.stop_results = [orchestrator_error(500), None]
    with pytest.raises(IdeGymOperationError):
        await provider.close(handle)
    assert handle.raw.stopped is False
    await provider.close(handle)
    assert handle.raw.stopped is True
    assert [call["server_id"] for call in fake_session.stop_calls] == [7, 7]


async def test_close_retries_then_raises_and_keeps_the_sandbox_live(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(operations={"retries": 2})
    handle = await provider.create(spec())
    fake_session.stop_results = [orchestrator_error(500)]
    with pytest.raises(IdeGymOperationError, match="Failed to stop IdeGYM sandbox"):
        await provider.close(handle)
    assert len(fake_session.stop_calls) == 3
    # A stop that failed for real must not leave the sandbox looking stopped: the
    # pod may well still be running, and status() has to keep saying so.
    assert handle.raw.stopped is False
    assert await provider.status(handle) is SandboxStatus.RUNNING


async def test_close_succeeds_after_a_transient_failure(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.stop_results = [orchestrator_error(503), None]
    await provider.close(handle)
    assert len(fake_session.stop_calls) == 2


async def test_aclose_releases_the_session_once(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    await provider.create(spec())
    await provider.aclose()
    await provider.aclose()
    assert provider.released == [fake_session]
    with pytest.raises(IdeGymOperationError, match="has been closed"):
        await provider.session()


async def test_aclose_without_a_session_is_a_no_op(make_provider) -> None:
    provider = make_provider()
    await provider.aclose()
    assert provider.released == []


async def test_health_reports_the_orchestrator_status(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    assert await provider.health() == "healthy"
    assert fake_session.health_calls == 1


async def test_the_session_is_acquired_only_once(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    first, second = await asyncio.gather(provider.session(), provider.session())
    assert first is second is fake_session


async def test_a_racing_acquire_hands_back_its_surplus_reference(
    make_provider, fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a lock held across the await, two callers can both acquire.

    ``acquire_session`` refcounts, so the loser must release or the shared client is
    never unregistered and its pods outlive the run.
    """
    provider = make_provider()
    acquired = 0

    async def slow_acquire(connection: Any, attribution: Any) -> Any:
        nonlocal acquired
        acquired += 1
        await asyncio.sleep(0)  # let the other caller past the "already set?" check
        return fake_session

    monkeypatch.setattr(idegym_provider, "acquire_session", slow_acquire)
    first, second = await asyncio.gather(provider.session(), provider.session())
    assert first is second is fake_session
    assert acquired == 2
    assert provider.released == [fake_session]


def test_the_session_survives_being_reached_from_two_event_loops(
    make_provider, fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sync Sandbox facade runs each sandbox on its own loop.

    Two of them sharing one provider is legal -- ``AsyncSandbox`` takes a provider
    instance -- and an ``asyncio.Lock`` here deadlocks the second loop: the release runs
    on the first loop while the waiter's future belongs to the second. The slow acquire
    is what forces that overlap; without it the lock is never contended.
    """
    provider = make_provider()
    started = threading.Event()

    async def slow_acquire(connection: Any, attribution: Any) -> Any:
        started.set()
        await asyncio.sleep(0.5)
        return fake_session

    # After make_provider(), which patches acquire_session itself.
    monkeypatch.setattr(idegym_provider, "acquire_session", slow_acquire)
    seen: list[Any] = []

    def use() -> None:
        seen.append(asyncio.run(provider.session()))

    first = threading.Thread(target=use, daemon=True)
    first.start()
    assert started.wait(5), "the first loop never reached acquire_session"
    second = threading.Thread(target=use, daemon=True)
    second.start()
    for thread in (first, second):
        thread.join(10)
    assert [first.is_alive(), second.is_alive()] == [False, False], "a loop deadlocked on the session lock"
    assert seen == [fake_session, fake_session]
