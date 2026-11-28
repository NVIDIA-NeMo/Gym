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
test that holds it to the real class's parameters, and a ``bash``-backed session
that runs commands for real in the requested ``cwd`` with the requested ``env``, so
what the provider sends is tested by executing it rather than by comparing strings.
Failures are the SDK's own typed exceptions, which is what :mod:`.errors` classifies.
"""

import asyncio
import builtins
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.idegym import naming as idegym_naming
from nemo_gym.sandbox.providers.idegym import provider as idegym_provider
from nemo_gym.sandbox.providers.idegym import session as idegym_session
from nemo_gym.sandbox.providers.idegym.config import (
    MAX_FILE_CHUNK_BYTES,
    IdeGymAttributionConfig,
    IdeGymConnectionConfig,
    IdeGymCreateConfig,
    IdeGymExecConfig,
    IdeGymFilesConfig,
    IdeGymOperationsConfig,
    IdeGymVerifyConfig,
)
from nemo_gym.sandbox.providers.idegym.errors import (
    IdeGymCreateVerificationError,
    IdeGymError,
    IdeGymInvalidRequestError,
    IdeGymOperationError,
    IdeGymTransferError,
    IdeGymUnknownServerError,
    is_command_timeout,
    is_deadline_exceeded,
    is_invalid_request,
    is_retryable,
    is_sandbox_gone,
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
)
from nemo_gym.sandbox.providers.idegym.session import IdeGymBashResult, IdeGymServerRef, IdeGymServerStatus
from nemo_gym.sandbox.providers.idegym.spec import IdeGymProviderOptions, ServerRequestTranslator
from nemo_gym.sandbox.providers.registry import get_provider_class


pytestmark = pytest.mark.sandbox

pytest.importorskip("idegym.client", reason="idegym-client optional dependency is not installed")

from idegym.client.exceptions import (  # noqa: E402 - after the importorskip guard
    IdeGYMBusyError,
    IdeGYMConnectionError,
    IdeGYMSandboxError,
    IdeGYMTimeoutError,
    http_error,
)


MAX_SERVER_NAME_LENGTH = 63
ROOT = Path(__file__).resolve().parents[3]
SHIPPED_CONFIG = ROOT / "nemo_gym/sandbox/providers/idegym/configs/idegym.yaml"

requires_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is not available")


def orchestrator_error(status: int, body: str = "boom") -> Exception:
    """The SDK's exception for a status the orchestrator itself returned."""
    return http_error(f"Request failed: url=/api/idegym-servers status={status}", status_code=status, body=body)


def sandbox_error(status: int, message: str) -> IdeGYMSandboxError:
    """The SDK's exception for a status the sandbox returned through a forward, worded as the SDK words it."""
    body = json.dumps({"message": message, "status": status})
    return IdeGYMSandboxError(
        f"Failed to forward request POST /api/tools/x: {{'status_code': {status}, 'body': {body!r}}}",
        status_code=status,
        body=body,
    )


def bash_timeout(seconds: int = 3) -> IdeGYMSandboxError:
    return sandbox_error(500, f"Command execution timed out after {seconds} seconds")


ALIVE = IdeGymServerStatus(usable=True, availability="ALIVE", pod_ready=True, details=None)


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
        self.status_calls: list[int] = []
        self.keepalive_calls: list[tuple[int, float]] = []
        self.upload_calls: list[dict[str, Any]] = []
        self.download_calls: list[dict[str, Any]] = []
        self.stop_calls: list[dict[str, Any]] = []
        self.health_calls = 0
        self.live_servers: set[int] = set()
        self.start_results: list[Any] = [
            IdeGymServerRef(server_id=7, server_name="nemo-gym-abcdef12", namespace="idegym")
        ]
        self.bash_results: list[Any] = [IdeGymBashResult(stdout="", stderr="", exit_code=0)]
        self.status_results: list[Any] = [ALIVE]
        # None answers with a hold of `minutes` from now, as the orchestrator does.
        self.keepalive_results: list[Any] = [None]
        self.upload_results: list[Any] = [0]
        self.download_results: list[Any] = [0]
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
        cwd: Any,
        env: Any,
        user: Any,
        max_output_bytes: Any,
        command_timeout_s: float,
        graceful_termination_timeout_s: float,
        request_timeout_s: float,
        polling: Any,
    ) -> Any:
        self.bash_calls.append(
            {
                "server_id": server_id,
                "script": script,
                "cwd": cwd,
                "env": env,
                "user": user,
                "max_output_bytes": max_output_bytes,
                "command_timeout_s": command_timeout_s,
                "graceful_termination_timeout_s": graceful_termination_timeout_s,
                "request_timeout_s": request_timeout_s,
            }
        )
        return self._next(self.bash_results)

    async def server_status(self, server_id: int) -> Any:
        self.status_calls.append(server_id)
        return self._next(self.status_results)

    async def keepalive(self, server_id: int, minutes: float) -> int:
        self.keepalive_calls.append((server_id, minutes))
        result = self._next(self.keepalive_results)
        return int(time.time() * 1000 + minutes * 60_000) if result is None else result

    async def upload_file(
        self,
        server_id: int,
        local_path: Path,
        remote_path: str,
        *,
        chunk_bytes: int,
        request_timeout_s: float,
        polling: Any,
    ) -> int:
        self.upload_calls.append(
            {
                "server_id": server_id,
                "local_path": local_path,
                "remote_path": remote_path,
                "chunk_bytes": chunk_bytes,
                "request_timeout_s": request_timeout_s,
            }
        )
        return self._next(self.upload_results)

    async def download_file(
        self,
        server_id: int,
        remote_path: str,
        local_path: Path,
        *,
        chunk_bytes: int,
        request_timeout_s: float,
        polling: Any,
    ) -> int:
        self.download_calls.append(
            {
                "server_id": server_id,
                "remote_path": remote_path,
                "local_path": local_path,
                "chunk_bytes": chunk_bytes,
                "request_timeout_s": request_timeout_s,
            }
        )
        return self._next(self.download_results)

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
    """A fake session that runs commands with local ``bash`` and moves files on local disk.

    Mirrors what the IdeGYM 0.13.1 server does with a request: a fresh ``bash -c`` in
    ``cwd`` with ``env`` on top of a minimal environment, output returned unstripped, a
    missing ``cwd`` answered with a 400, and the file endpoints' 404 for a missing path
    and 400 for a directory.
    """

    async def execute_bash(self, server_id: int, script: str, **kwargs: Any) -> Any:
        await super().execute_bash(server_id, script, **kwargs)
        cwd = kwargs["cwd"]
        if cwd is not None and not Path(cwd).is_dir():
            raise sandbox_error(400, f"Working directory {cwd} does not exist")
        completed = await asyncio.to_thread(
            subprocess.run,
            ["bash", "-c", script],
            cwd=cwd,
            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **(kwargs["env"] or {})},
            capture_output=True,
            check=False,
        )
        return IdeGymBashResult(
            stdout=completed.stdout.decode(errors="replace"),
            stderr=completed.stderr.decode(errors="replace"),
            exit_code=completed.returncode,
        )

    @staticmethod
    def _readable(remote_path: str) -> Path:
        path = Path(remote_path)
        if not path.exists():
            raise sandbox_error(404, f"Path not found: {remote_path}")
        if path.is_dir():
            raise sandbox_error(400, f"Not a file: {remote_path}")
        return path

    async def upload_file(self, server_id: int, local_path: Path, remote_path: str, **kwargs: Any) -> int:
        await super().upload_file(server_id, local_path, remote_path, **kwargs)
        target = Path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)
        return target.stat().st_size

    async def download_file(self, server_id: int, remote_path: str, local_path: Path, **kwargs: Any) -> int:
        await super().download_file(server_id, remote_path, local_path, **kwargs)
        source = self._readable(remote_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local_path)
        return Path(local_path).stat().st_size


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


def spec(**overrides: Any) -> SandboxSpec:
    values: dict[str, Any] = {"image": "registry.example.com/idegym/env:1"}
    values.update(overrides)
    return SandboxSpec(**values)


def translator() -> ServerRequestTranslator:
    return ServerRequestTranslator(IdeGymCreateConfig())


# --- registration, dependency and package surface --------------------------


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


SESSION_API = (
    "health",
    "start_server",
    "execute_bash",
    "server_status",
    "keepalive",
    "upload_file",
    "download_file",
    "stop_server",
)


@pytest.mark.parametrize("name", SESSION_API)
def test_provider_binds_the_session_api_it_calls(name: str) -> None:
    """The fake session is only trustworthy if it matches the real one's shape.

    Names and kinds only: the fake annotates loosely on purpose, but a renamed,
    reordered, added or removed parameter on ``IdeGymSession`` has to fail here
    instead of passing against a fake that has quietly drifted.
    """

    def parameters(owner: type) -> list[tuple[str, Any]]:
        return [(p.name, p.kind) for p in inspect.signature(getattr(owner, name)).parameters.values()]

    assert parameters(idegym_session.IdeGymSession) == parameters(FakeSession)


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
        ("connection", {"poll_retries": -1}),
        ("connection", {"retry_delay_s": -1}),
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
        ("create", {"keepalive_minutes": 0}),
        ("create", {"keepalive_minutes": 24 * 60 + 1}),
        ("create", {"polling": {"initial_delay_s": 0}}),
        ("create", {"polling": {"interval_s": -1}}),
        ("create", {"polling": {"backoff_factor": 0.5}}),
        ("create", {"polling": {"max_delay_s": 0}}),
        ("exec", {"default_timeout_s": 0}),
        ("exec", {"graceful_termination_timeout_s": -1}),
        ("exec", {"request_overhead_s": -1}),
        ("exec", {"max_output_bytes": 0}),
        ("verify", {"timeout_s": 0}),
        ("files", {"chunk_bytes": 0}),
        ("files", {"chunk_bytes": MAX_FILE_CHUNK_BYTES + 1}),
        ("files", {"timeout_s": 0}),
        ("operations", {"close_timeout_s": 0}),
        ("operations", {"status_timeout_s": 0}),
        ("operations", {"retries": -1}),
        ("operations", {"retry_delay_s": -1}),
        ("operations", {"retry_max_delay_s": -1}),
        ("operations", {"keepalive_minutes_max": 0}),
        ("attribution", {"client_name_prefix": ""}),
        ("attribution", {"key_prefix": "Not A Prefix"}),
    ],
)
def test_config_validation_rejects_bad_values(group: str, kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        IdeGymProvider(**{group: kwargs})


def test_an_unknown_config_key_fails_at_construction() -> None:
    """A YAML still setting a pre-0.13 key, such as exec.user_mode, must not be silently half-applied."""
    with pytest.raises(TypeError, match="user_mode"):
        IdeGymProvider(exec={"user_mode": "sudo"})


def test_config_sections_are_coerced_from_mappings() -> None:
    provider = IdeGymProvider(
        connection={"orchestrator_url": "idegym.example", "namespace": "gym"},
        create={"server_name_metadata_keys": ["instance_id", "repo"], "polling": {"max_delay_s": 5.0}},
    )
    assert provider._connection == IdeGymConnectionConfig(orchestrator_url="idegym.example", namespace="gym")
    # Hydra hands over lists; the frozen config keeps a hashable tuple and a nested dataclass.
    assert provider._create.server_name_metadata_keys == ("instance_id", "repo")
    assert provider._create.polling.max_delay_s == 5.0
    with pytest.raises(TypeError, match="IdeGymConnectionConfig"):
        IdeGymProvider(connection=["not", "a", "mapping"])


def test_the_shipped_config_block_constructs_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    from omegaconf import OmegaConf

    for name in list(os.environ):
        if name.startswith("IDEGYM_"):
            monkeypatch.delenv(name)
    shipped = OmegaConf.to_container(OmegaConf.load(SHIPPED_CONFIG), resolve=True)
    provider = IdeGymProvider(**shipped["sandbox"]["idegym"])
    assert provider._connection.orchestrator_url == "idegym.test"
    assert provider._connection.username is None
    assert provider._connection.tracing_endpoint is None
    assert provider._create.keepalive_minutes is None
    assert provider._exec.max_output_bytes == 1024 * 1024
    assert provider._files.chunk_bytes == 1024 * 1024
    assert provider._attribution.key_prefix == "nemo-gym.nvidia.com/"


def test_the_shipped_config_only_overrides_the_code_defaults_on_purpose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building the provider in code must behave like the documented setup, save for the listed overrides."""
    from omegaconf import OmegaConf

    for name in list(os.environ):
        if name.startswith("IDEGYM_"):
            monkeypatch.delenv(name)
    shipped = OmegaConf.to_container(OmegaConf.load(SHIPPED_CONFIG), resolve=True)["sandbox"]["idegym"]
    provider = IdeGymProvider(**shipped)
    # run_as_root is off in code, as in the SDK; the shipped config opts SWE-bench style images in.
    assert provider._create == IdeGymCreateConfig(run_as_root=True)
    assert provider._connection == IdeGymConnectionConfig()
    assert provider._exec == IdeGymExecConfig()
    assert provider._verify == IdeGymVerifyConfig()
    assert provider._files == IdeGymFilesConfig()
    assert provider._operations == IdeGymOperationsConfig()
    assert provider._attribution == IdeGymAttributionConfig()
    # Every key the code reads is in the shipped file, so each one has operator guidance there.
    for section, config in (("connection", IdeGymConnectionConfig), ("operations", IdeGymOperationsConfig)):
        assert set(shipped[section]) == set(config.__dataclass_fields__), section


def test_provider_options_validation() -> None:
    with pytest.raises(ValueError, match="Unknown idegym provider_options keys"):
        IdeGymProviderOptions.from_mapping({"nope": 1})
    with pytest.raises(TypeError, match="node_selector"):
        IdeGymProviderOptions.from_mapping({"node_selector": ["a"]})
    with pytest.raises(TypeError, match="labels"):
        IdeGymProviderOptions.from_mapping({"labels": ["a"]})
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


def connect_timeout() -> IdeGYMTimeoutError:
    import httpx

    try:
        raise IdeGYMTimeoutError("Request timed out") from httpx.ConnectTimeout("unreachable")
    except IdeGYMTimeoutError as e:
        return e


def malformed_json() -> json.JSONDecodeError:
    """What the SDK re-raises when the orchestrator answers with a body that is not JSON."""
    try:
        json.loads("<html>bad gateway</html>")
    except json.JSONDecodeError as e:
        return e
    raise AssertionError("unreachable")


def malformed_bash_response() -> Exception:
    """What the SDK raises when a bash response is missing fields."""
    from idegym.api.tools.bash import BashCommandResponse
    from pydantic import ValidationError

    try:
        BashCommandResponse.model_validate({"stdout": "x"})
    except ValidationError as e:
        return e
    raise AssertionError("unreachable")


@pytest.mark.parametrize(
    ("error", "gone", "retryable", "timeout", "invalid"),
    [
        pytest.param(orchestrator_error(404), True, False, False, False, id="orchestrator-404"),
        pytest.param(orchestrator_error(410), True, False, False, False, id="orchestrator-410"),
        pytest.param(sandbox_error(404, "Path not found"), False, False, False, False, id="sandbox-404"),
        pytest.param(orchestrator_error(429), False, True, False, False, id="busy-429"),
        pytest.param(orchestrator_error(503), False, True, False, False, id="busy-503"),
        pytest.param(orchestrator_error(502), False, True, False, False, id="bad-gateway"),
        # A status means the orchestrator or a gateway gave up on one request, not that the command ran out.
        pytest.param(orchestrator_error(504), False, True, False, False, id="gateway-timeout"),
        pytest.param(orchestrator_error(408), False, True, False, False, id="request-timeout"),
        pytest.param(orchestrator_error(500), False, False, False, False, id="orchestrator-500"),
        pytest.param(orchestrator_error(400), False, False, False, False, id="orchestrator-400"),
        pytest.param(IdeGYMConnectionError("reset"), False, True, False, False, id="connection"),
        pytest.param(IdeGYMTimeoutError("Server start timed out"), False, True, True, False, id="sdk-deadline"),
        pytest.param(connect_timeout(), False, True, False, False, id="connect-timeout"),
        pytest.param(TimeoutError(), False, True, True, False, id="builtin-timeout"),
        pytest.param(ConnectionError("refused"), False, True, False, False, id="builtin-connection"),
        pytest.param(IdeGymUnknownServerError("not held"), True, False, False, False, id="unknown-server"),
        pytest.param(sandbox_error(400, "Unknown user"), False, False, False, True, id="sandbox-400"),
        pytest.param(IdeGymInvalidRequestError("bad env name"), False, False, False, True, id="request-validation"),
        pytest.param(ValueError("not an identifier"), False, False, False, False, id="bare-value-error"),
        pytest.param(malformed_json(), False, False, False, False, id="malformed-json"),
        pytest.param(malformed_bash_response(), False, False, False, False, id="malformed-response"),
        pytest.param(bash_timeout(), False, False, True, False, id="bash-timeout"),
        pytest.param(sandbox_error(500, "kaboom"), False, False, False, False, id="sandbox-500"),
        pytest.param(sandbox_error(503, "busy"), False, False, False, False, id="sandbox-503"),
        pytest.param(RuntimeError("no status here"), False, False, False, False, id="unrelated"),
    ],
)
def test_error_classification(error: Exception, gone: bool, retryable: bool, timeout: bool, invalid: bool) -> None:
    assert is_sandbox_gone(error) is gone
    assert is_retryable(error) is retryable
    assert is_command_timeout(error) is timeout
    assert is_invalid_request(error) is invalid


@pytest.mark.parametrize(
    ("error", "spent"),
    [
        pytest.param(IdeGYMTimeoutError("Server start timed out"), True, id="sdk-deadline"),
        pytest.param(TimeoutError(), True, id="builtin"),
        pytest.param(connect_timeout(), False, id="connect-timeout"),
        pytest.param(orchestrator_error(504), False, id="gateway-timeout"),
        pytest.param(orchestrator_error(408), False, id="request-timeout"),
        pytest.param(IdeGYMConnectionError("reset"), False, id="connection"),
    ],
)
def test_only_a_client_side_deadline_counts_as_exceeded(error: Exception, spent: bool) -> None:
    assert is_deadline_exceeded(error) is spent


@pytest.mark.parametrize(
    ("body", "timeout"),
    [
        ("Command execution timed out after 5 seconds", True),
        (json.dumps({"message": "pytest: test_foo timed out after 30 seconds"}), False),
        # Tool output that mentions the marker is not the marker.
        (json.dumps({"message": "error: Command execution timed out after 5 seconds"}), False),
        (json.dumps({"message": 5}), False),
        (json.dumps(["Command execution timed out after 5 seconds"]), False),
        (None, False),
    ],
)
def test_the_bash_timeout_is_read_only_from_the_start_of_the_sandbox_message(body: str | None, timeout: bool) -> None:
    error = IdeGYMSandboxError("Sandbox returned 500", status_code=500, body=body)
    assert is_command_timeout(error) is timeout


# --- naming ----------------------------------------------------------------


def test_sanitize_and_clamp_names() -> None:
    assert sanitize_name("  My Team/Foo  ") == "my-team-foo"
    assert sanitize_name("///") == ""
    long_name = idegym_session.clamp_client_name("a" * 200)
    assert len(long_name) == MAX_CLIENT_NAME_LENGTH


def test_generated_server_names_leave_room_for_the_server_id_and_satisfy_rfc1035() -> None:
    name = generate_server_name("nemo-gym", ["django__django-11099"])
    assert name == "nemo-gym-django-django-11099"
    assert len(name) + len("-9999999") <= MAX_SERVER_NAME_LENGTH
    assert len(generate_server_name("nemo-gym", ["x" * 200])) + len("-9999999") <= MAX_SERVER_NAME_LENGTH
    pattern = re.compile(r"^[a-z]([-a-z0-9]*[a-z0-9])?$")
    for prefix, hints in (
        ("nemo-gym", ["A_VERY::Odd/Instance ID"]),
        ("nemo-gym", ["x" * 200]),
        ("9bad", ["also-bad"]),
        ("---", []),
    ):
        assert pattern.match(generate_server_name(prefix, hints)), (prefix, hints)


def test_server_id_reserve_must_leave_room_for_a_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(idegym_naming, "SERVER_ID_SUFFIX_RESERVE", MAX_SERVER_NAME_LENGTH)
    with pytest.raises(ValueError, match="leaves no room for a name"):
        generate_server_name("nemo-gym", [])


# --- spec translation ------------------------------------------------------


def test_translate_maps_resources_to_limits_and_requests() -> None:
    request = translator().translate(
        spec(resources={"cpu": 2, "memory_mib": 8192, "disk_gib": 30}, metadata={"instance_id": "task-9"}),
        IdeGymProviderOptions.from_mapping({"resource_requests": {"cpu": 0.5, "memory_mib": 2048}}),
        {},
    )
    assert request["resources"] == {
        "requests": {"cpu": "500m", "memory": "2048Mi"},
        "limits": {"cpu": "2", "memory": "8192Mi", "ephemeral-storage": "30Gi"},
    }
    assert request["server_name"] == "nemo-gym-task-9"
    assert request["reuse_strategy"] == "NONE"


def test_translate_passes_pod_shape_options_through() -> None:
    request = translator().translate(
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
        {},
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
    assert "annotations" not in request


def test_translate_merges_attribution_metadata_and_option_labels() -> None:
    """Later sources win: attribution, then spec.metadata, then provider_options.labels."""
    request = translator().translate(
        spec(metadata={"instance_id": "django__django-11099", "nemo-gym.nvidia.com/team": "from-spec", "x": "1"}),
        IdeGymProviderOptions.from_mapping({"labels": {"x": "2"}, "annotations": {"owner": "me"}}),
        {"nemo-gym.nvidia.com/team": "t", "nemo-gym.nvidia.com/run": "r"},
    )
    assert request["labels"] == {
        "nemo-gym.nvidia.com/team": "from-spec",
        "nemo-gym.nvidia.com/run": "r",
        "instance_id": "django__django-11099",
        "x": "2",
    }
    assert request["annotations"] == {"owner": "me"}


def test_label_values_are_sanitized_rather_than_rejected() -> None:
    request = translator().translate(spec(metadata={"long": "v" * 100, "colon": "a:b"}), IdeGymProviderOptions(), {})
    assert len(request["labels"]["long"]) == 63
    assert request["labels"]["colon"] == "a_b"


@pytest.mark.parametrize(
    ("metadata", "options"),
    [
        ({"app": "x"}, {}),
        ({"app.kubernetes.io/name": "x"}, {}),
        ({"idegym.jetbrains.com/server": "x"}, {}),
        ({}, {"labels": {"idegym.jetbrains.com/x": "y"}}),
        ({}, {"annotations": {"cluster-autoscaler.kubernetes.io/safe-to-evict": "true"}}),
        ({}, {"annotations": {"prometheus.io/scrape": "true"}}),
        ({}, {"annotations": {"podsnapshot.gke.io/policy": "x"}}),
    ],
)
def test_keys_idegym_manages_are_rejected(metadata: dict[str, str], options: dict[str, Any]) -> None:
    with pytest.raises(SandboxCreateError, match="managed by IdeGYM"):
        translator().translate(spec(metadata=metadata), IdeGymProviderOptions.from_mapping(options), {})


def test_translate_normalizes_and_validates_the_image() -> None:
    request = translator().translate(spec(image="docker://reg.example/env:1"), IdeGymProviderOptions(), {})
    assert request["image_tag"] == "reg.example/env:1"
    for image in ("Reg.Example/Env:1", "reg.example/env:1\n"):
        with pytest.raises(SandboxCreateError, match="not a valid OCI image reference"):
            translator().translate(spec(image=image), IdeGymProviderOptions(), {})
    with pytest.raises(SandboxCreateError, match="spec.image is required"):
        translator().translate(spec(image=None), IdeGymProviderOptions(), {})


def test_translate_rejects_entrypoint_and_warns_about_unsupported_fields(caplog: pytest.LogCaptureFixture) -> None:
    translate = translator()
    with pytest.raises(SandboxCreateError, match="spec.entrypoint is not supported"):
        translate.translate(spec(entrypoint=["/bin/sh"]), IdeGymProviderOptions(), {})
    unsupported = spec(ttl_s=60, ports=[8080], resources={"gpu": 1})
    with caplog.at_level("WARNING"):
        translate.translate(unsupported, IdeGymProviderOptions(), {})
        translate.translate(unsupported, IdeGymProviderOptions(), {})
    # Once each, not once per sandbox: a benchmark translates one spec per task.
    assert caplog.text.count("spec.ttl_s is not enforced") == 1
    assert caplog.text.count("sandbox.endpoint() is unavailable") == 1
    assert caplog.text.count("cannot map GPU resource requests") == 1


@pytest.mark.parametrize("value", [0, -1])
def test_translate_rejects_non_positive_resources(value: int) -> None:
    for key in ("cpu", "memory_mib", "disk_gib"):
        with pytest.raises(SandboxCreateError, match="greater than zero"):
            translator().translate(spec(resources={key: value}), IdeGymProviderOptions(), {})


# --- create ----------------------------------------------------------------


async def test_create_checks_the_workdir_with_one_command(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(
        spec(workdir="/testbed", env={"FOO": "bar"}, ready_timeout_s=99, metadata={"instance_id": "task-1"})
    )
    assert handle.provider_name == "idegym"
    assert handle.sandbox_id == "7"
    assert (handle.raw.workdir, handle.raw.env) == ("/testbed", {"FOO": "bar"})
    assert fake_session.start_calls[0]["timeout_s"] == pytest.approx(99, abs=1)
    # IdeGYM already waited for pod readiness, so one command in the workdir is all that is left to check.
    assert [(call["script"], call["cwd"], call["env"]) for call in fake_session.bash_calls] == [
        ("true", "/testbed", {})
    ]
    assert fake_session.status_calls == []
    assert fake_session.keepalive_calls == []


@pytest.mark.parametrize(
    ("attribution", "expected"),
    [
        (
            {"team": "t", "user": "u", "workload": "w", "run": "r"},
            {
                "nemo-gym.nvidia.com/team": "t",
                "nemo-gym.nvidia.com/user": "u",
                "nemo-gym.nvidia.com/workload": "w",
                "nemo-gym.nvidia.com/run": "r",
            },
        ),
        ({"enabled": False}, {}),
    ],
    ids=["attributed", "disabled"],
)
async def test_create_sends_labels_from_metadata_and_attribution(
    make_provider, fake_session: FakeSession, attribution: dict[str, Any], expected: dict[str, str]
) -> None:
    provider = make_provider(attribution=attribution)
    await provider.create(spec(metadata={"instance_id": "task-1"}))
    assert fake_session.start_calls[0]["request"]["labels"] == {**expected, "instance_id": "task-1"}


async def test_create_falls_back_to_configured_ready_timeout(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(create={"ready_timeout_s": 45})
    await provider.create(spec())
    assert fake_session.start_calls[0]["timeout_s"] == pytest.approx(45, abs=1)


async def test_create_fails_when_the_workdir_is_unusable(
    make_provider, fake_session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    fake_session.bash_results = [sandbox_error(400, "Working directory /nope does not exist")]
    provider = make_provider()
    with caplog.at_level("WARNING"), pytest.raises(SandboxCreateVerificationError, match="is not usable"):
        await provider.create(spec(workdir="/nope"))
    assert fake_session.stop_calls == [{"server_id": 7, "timeout_s": 5.0}]
    # Nothing else records that a pod was started and thrown away.
    assert "Tearing down the IdeGYM sandbox" in caplog.text


@requires_bash
async def test_create_against_a_local_server_checks_the_real_directory(make_provider, tmp_path: Path) -> None:
    session = LocalBashSession()
    provider = make_provider(session)
    await provider.create(spec(workdir=str(tmp_path)))
    with pytest.raises(SandboxCreateVerificationError, match="is not usable"):
        await provider.create(spec(workdir=str(tmp_path / "missing")))


async def test_create_fails_when_the_check_command_cannot_run(make_provider, fake_session: FakeSession) -> None:
    fake_session.bash_results = [orchestrator_error(500, "kaboom")]
    provider = make_provider()
    with pytest.raises(IdeGymCreateVerificationError, match="could not run its verification command"):
        await provider.create(spec(workdir="/testbed"))
    assert fake_session.stop_calls


@pytest.mark.parametrize(
    ("verify", "workdir"), [({"check_workdir": False}, "/testbed"), ({}, None)], ids=["check-off", "no-workdir"]
)
async def test_create_runs_no_command_without_a_workdir_to_check(
    make_provider, fake_session: FakeSession, verify: dict[str, Any], workdir: str | None
) -> None:
    provider = make_provider(verify=verify)
    await provider.create(spec(workdir=workdir))
    assert fake_session.bash_calls == []
    assert fake_session.status_calls == []


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


@pytest.mark.parametrize(
    "transient",
    [
        IdeGYMBusyError("429", status_code=429),
        IdeGYMConnectionError("reset"),
        orchestrator_error(502),
        # A 408 or 504 is one lost request, not the SDK's own deadline.
        orchestrator_error(504),
        orchestrator_error(408),
        # The request never reached the orchestrator, so no budget was spent on it.
        connect_timeout(),
    ],
    ids=["busy", "connection", "bad-gateway", "gateway-timeout", "request-timeout", "connect-timeout"],
)
async def test_create_retries_transient_failures_keeping_the_generated_name(
    make_provider, fake_session: FakeSession, transient: Exception
) -> None:
    """IdeGYM makes the pod name unique with its own server id, so a lost start that landed leaves an orphan."""
    fake_session.start_results = [
        transient,
        IdeGymServerRef(server_id=9, server_name="nemo-gym-second", namespace="idegym"),
    ]
    provider = make_provider()
    handle = await provider.create(spec())
    assert handle.sandbox_id == "9"
    first, second = (call["request"]["server_name"] for call in fake_session.start_calls)
    assert first == second
    # The second attempt gets what is left of the one budget, not a fresh one.
    assert fake_session.start_calls[1]["timeout_s"] <= fake_session.start_calls[0]["timeout_s"]


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


@pytest.mark.parametrize(
    "spent",
    [IdeGYMTimeoutError("Server start timed out after 1200 seconds"), TimeoutError("deadline")],
    ids=["sdk", "builtin"],
)
async def test_create_does_not_retry_an_exhausted_readiness_budget(
    make_provider, fake_session: FakeSession, spent: Exception
) -> None:
    """Retrying would silently multiply ready_timeout_s by the retry count."""
    fake_session.start_results = [spent]
    provider = make_provider(create={"retries": 3})
    with pytest.raises(SandboxCreateError, match="could not start a server"):
        await provider.create(spec())
    assert len(fake_session.start_calls) == 1


async def test_create_does_not_retry_past_the_deadline(make_provider, fake_session: FakeSession) -> None:
    fake_session.start_results = [orchestrator_error(429)]
    provider = make_provider(create={"retries": 3, "retry_delay_s": 60.0, "retry_max_delay_s": 60.0})
    with pytest.raises(SandboxCreateError, match="could not start a server"):
        await provider.create(spec())
    assert len(fake_session.start_calls) == 1


async def test_create_non_retryable_failure_is_not_retried(make_provider, fake_session: FakeSession) -> None:
    fake_session.start_results = [orchestrator_error(400, "bad image")]
    provider = make_provider()
    with pytest.raises(SandboxCreateError, match="could not start a server"):
        await provider.create(spec())
    assert len(fake_session.start_calls) == 1


async def test_create_teardown_failure_only_warns(
    make_provider, fake_session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    fake_session.bash_results = [sandbox_error(400, "Working directory /nope does not exist")]
    fake_session.stop_results = [orchestrator_error(500)]
    provider = make_provider()
    with caplog.at_level("WARNING"), pytest.raises(SandboxCreateVerificationError):
        await provider.create(spec(workdir="/nope"))
    assert "may be left running on the cluster" in caplog.text


# --- keepalive ---------------------------------------------------------------


async def test_create_takes_the_configured_hold(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(create={"keepalive_minutes": 5})
    handle = await provider.create(spec())
    assert fake_session.keepalive_calls == [(7, 5)]
    assert handle.raw.held_until_ms == pytest.approx(time.time() * 1000 + 5 * 60_000, abs=5_000)


async def test_provider_options_override_the_hold_per_sandbox(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(create={"keepalive_minutes": 5})
    await provider.create(spec(provider_options={"keepalive_minutes": 30}))
    await provider.create(spec(provider_options={"keepalive_minutes": None}))
    assert fake_session.keepalive_calls == [(7, 30.0)]
    opted_in = make_provider()
    await opted_in.create(spec(provider_options={"keepalive_minutes": 2}))
    assert fake_session.keepalive_calls[-1] == (7, 2.0)


@pytest.mark.parametrize("minutes", [0, -1, 24 * 60 + 1])
async def test_an_invalid_per_sandbox_hold_fails_before_a_pod_is_started(
    make_provider, fake_session: FakeSession, minutes: float
) -> None:
    provider = make_provider()
    with pytest.raises(ValueError, match="provider_options.keepalive_minutes"):
        await provider.create(spec(provider_options={"keepalive_minutes": minutes}))
    assert fake_session.start_calls == []


async def test_a_failed_hold_at_create_is_a_create_failure(make_provider, fake_session: FakeSession) -> None:
    fake_session.keepalive_results = [orchestrator_error(500)]
    provider = make_provider(create={"keepalive_minutes": 5})
    with pytest.raises(SandboxCreateError, match="inactivity reaper"):
        await provider.create(spec())
    assert [call["server_id"] for call in fake_session.stop_calls] == [7]
    assert fake_session.bash_calls == []


async def test_a_long_exec_extends_the_hold_once(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(create={"keepalive_minutes": 5})
    handle = await provider.create(spec())
    fake_session.keepalive_calls.clear()

    await provider.exec(handle, "true", timeout_s=60)
    assert fake_session.keepalive_calls == []

    await provider.exec(handle, "make test", timeout_s=600)
    # 600 s of command plus 60 s of request overhead, rounded up, plus a minute.
    assert fake_session.keepalive_calls == [(7, math.ceil(660 / 60) + 1)]

    await provider.exec(handle, "make test", timeout_s=600)
    assert len(fake_session.keepalive_calls) == 1

    # A sandbox that opted out of the hold never calls keepalive.
    unheld = make_provider()
    handle = await unheld.create(spec())
    await unheld.exec(handle, "make test", timeout_s=600)
    assert len(fake_session.keepalive_calls) == 1


async def test_a_command_without_a_timeout_takes_the_longest_hold(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider(create={"keepalive_minutes": 5}, exec={"default_timeout_s": None})
    handle = await provider.create(spec())
    await provider.exec(handle, "sleep infinity")
    assert fake_session.keepalive_calls[-1] == (7, 24 * 60.0)
    assert fake_session.bash_calls[-1]["command_timeout_s"] == NO_TIMEOUT_COMMAND_SECONDS


async def test_a_hold_that_finds_the_sandbox_gone_short_circuits_the_exec(
    make_provider, fake_session: FakeSession
) -> None:
    provider = make_provider(create={"keepalive_minutes": 5})
    handle = await provider.create(spec())
    calls = len(fake_session.bash_calls)
    fake_session.keepalive_results = [orchestrator_error(410, "gone")]
    result = await provider.exec(handle, "make test", timeout_s=600)
    assert (result.error_type, result.return_code) == ("sandbox", SANDBOX_RUNTIME_RETURN_CODE)
    assert handle.raw.stopped is True
    assert len(fake_session.bash_calls) == calls


async def test_a_failed_hold_does_not_stop_the_command(
    make_provider, fake_session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    provider = make_provider(create={"keepalive_minutes": 5})
    handle = await provider.create(spec())
    fake_session.keepalive_results = [orchestrator_error(500)]
    with caplog.at_level("WARNING"):
        result = await provider.exec(handle, "make test", timeout_s=600)
    assert result.return_code == 0
    assert "Could not extend the keepalive hold" in caplog.text


# --- exec ------------------------------------------------------------------


async def test_exec_sends_the_command_context_as_request_fields(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec(workdir="/testbed", env={"CONDA_ENV": "testbed"}))
    fake_session.bash_results = [IdeGymBashResult(stdout="out", stderr="err", exit_code=3)]
    result = await provider.exec(handle, "pytest -q", env={"EXTRA": "a b'c"}, timeout_s=120, user="tester")
    assert result == SandboxExecResult(stdout="out", stderr="err", return_code=3)
    call = fake_session.bash_calls[-1]
    # The command goes out untouched: no `cd`, `export`, brace group or user switch.
    assert call["script"] == "pytest -q"
    assert (call["cwd"], call["env"], call["user"]) == (
        "/testbed",
        {"CONDA_ENV": "testbed", "EXTRA": "a b'c"},
        "tester",
    )
    assert call["max_output_bytes"] == 1024 * 1024
    assert call["command_timeout_s"] == 120
    # The client waits longer than the sandbox, so the sandbox's own timeout wins
    # and the caller gets output instead of a transport error.
    assert call["request_timeout_s"] == 120 + 60
    await provider.exec(handle, "true", cwd="/other")
    assert fake_session.bash_calls[-1]["cwd"] == "/other"


async def test_exec_sends_a_large_script_unchanged(make_provider, fake_session: FakeSession) -> None:
    """The pre-0.13 128 KiB argv cap is gone: the server reads the script from a file."""
    provider = make_provider(exec={"max_output_bytes": None})
    handle = await provider.create(spec())
    script = "echo " + "x" * (200 * 1024)
    result = await provider.exec(handle, script)
    assert result.error_type is None
    assert fake_session.bash_calls[-1]["script"] == script
    assert fake_session.bash_calls[-1]["max_output_bytes"] is None


@pytest.mark.parametrize("user", [0, 1000])
async def test_exec_as_a_numeric_user_is_an_invalid_request(
    make_provider, fake_session: FakeSession, user: int
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    calls = len(fake_session.bash_calls)
    result = await provider.exec(handle, "true", user=user)
    assert (result.error_type, result.return_code) == ("invalid_request", SANDBOX_RUNTIME_RETURN_CODE)
    assert len(fake_session.bash_calls) == calls


@pytest.mark.parametrize(
    "rejection",
    [sandbox_error(400, "User nobody2 does not exist"), IdeGymInvalidRequestError("env name is not an identifier")],
    ids=["sandbox-400", "validation"],
)
async def test_exec_reports_a_rejected_request_instead_of_raising(
    make_provider, fake_session: FakeSession, rejection: Exception
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [rejection]
    result = await provider.exec(handle, "true", env={"NOT AN IDENT": "v"})
    assert (result.error_type, result.return_code) == ("invalid_request", SANDBOX_RUNTIME_RETURN_CODE)
    assert handle.raw.stopped is False


@pytest.mark.parametrize(
    "timeout", [bash_timeout(3), IdeGYMTimeoutError("Async operation 1 did not finish")], ids=["sandbox", "client"]
)
async def test_exec_reports_a_timeout(make_provider, fake_session: FakeSession, timeout: Exception) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [timeout]
    result = await provider.exec(handle, "sleep 100", timeout_s=3)
    assert (result.error_type, result.return_code) == ("timeout", SANDBOX_RUNTIME_RETURN_CODE)
    assert handle.raw.stopped is False


@pytest.mark.parametrize(
    "gone",
    [orchestrator_error(404), orchestrator_error(410), IdeGymUnknownServerError("not held")],
    ids=["404", "410", "unknown"],
)
async def test_exec_on_a_deleted_sandbox_marks_it_stopped(
    make_provider, fake_session: FakeSession, gone: Exception
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [gone]
    result = await provider.exec(handle, "true")
    assert result.error_type == "sandbox"
    assert handle.raw.stopped is True
    assert await provider.status(handle) is SandboxStatus.STOPPED
    # A stopped sandbox short-circuits rather than making another round trip.
    calls_before = len(fake_session.bash_calls)
    assert (await provider.exec(handle, "true")).error_type == "sandbox"
    assert len(fake_session.bash_calls) == calls_before


@pytest.mark.parametrize(
    "garbled", [malformed_json(), malformed_bash_response()], ids=["malformed-json", "malformed-response"]
)
async def test_a_garbled_response_is_not_blamed_on_the_request(
    make_provider, fake_session: FakeSession, garbled: Exception
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [garbled]
    result = await provider.exec(handle, "true")
    assert (result.error_type, result.return_code) == ("sandbox", SANDBOX_RUNTIME_RETURN_CODE)
    assert "rejected the command request" not in result.stderr

    with pytest.raises(IdeGymCreateVerificationError, match="could not run its verification command"):
        await provider.create(spec(workdir="/testbed"))


@pytest.mark.parametrize("status", [408, 504])
async def test_an_orchestrator_timeout_status_is_not_a_command_timeout(
    make_provider, fake_session: FakeSession, status: int
) -> None:
    """The gateway gave up on the request; the command may never have run."""
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [orchestrator_error(status)]
    result = await provider.exec(handle, "true", timeout_s=3)
    assert result.error_type == "sandbox"
    assert "timed out after" not in result.stderr


async def test_exec_other_failures_are_reported_not_raised(
    make_provider, fake_session: FakeSession, caplog: pytest.LogCaptureFixture
) -> None:
    """An unclassified failure becomes a return value, so the log is the only traceback."""
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.bash_results = [sandbox_error(500, "kaboom")]
    with caplog.at_level("WARNING"):
        result = await provider.exec(handle, "true")
    assert (result.error_type, result.return_code) == ("sandbox", SANDBOX_RUNTIME_RETURN_CODE)
    assert handle.raw.stopped is False
    assert "failed to run the command" in caplog.text
    assert "Traceback" in caplog.text


@requires_bash
async def test_exec_runs_in_the_cwd_with_the_env(make_provider, tmp_path: Path) -> None:
    other = tmp_path / "other dir"
    other.mkdir()
    provider = make_provider(LocalBashSession())
    handle = await provider.create(spec(workdir=str(tmp_path), env={"FROM_SPEC": "1"}))
    result = await provider.exec(handle, 'pwd; printf "%s|%s\\n" "$FROM_SPEC" "$EXTRA"', env={"EXTRA": "a b'c"})
    assert result.stdout == f"{tmp_path}\n1|a b'c\n"
    moved = await provider.exec(handle, "pwd", cwd=str(other))
    assert moved.stdout == f"{other}\n"
    missing = await provider.exec(handle, "pwd", cwd=str(tmp_path / "missing"))
    assert missing.error_type == "invalid_request"
    verbatim = await provider.exec(handle, "printf '  x  \\n'; printf ' e \\n' >&2")
    assert (verbatim.stdout, verbatim.stderr, verbatim.return_code) == ("  x  \n", " e \n", 0)


# --- files -----------------------------------------------------------------


@requires_bash
async def test_file_transfer_round_trips_binary_content(make_provider, tmp_path: Path) -> None:
    session = LocalBashSession()
    provider = make_provider(session, files={"chunk_bytes": 4096, "timeout_s": 12})
    remote = tmp_path / "sandbox"
    remote.mkdir()
    handle = await provider.create(spec(workdir=str(remote)))
    payload = bytes(range(256)) * 4096 + b"\xff"
    source = tmp_path / "blob.bin"
    source.write_bytes(payload)

    await provider.upload_file(handle, source, str(remote / "nested/dir/blob.bin"))
    assert (remote / "nested/dir/blob.bin").read_bytes() == payload

    out = tmp_path / "back" / "blob.bin"
    await provider.download_file(handle, str(remote / "nested/dir/blob.bin"), out)
    assert out.read_bytes() == payload
    assert {call["chunk_bytes"] for call in session.upload_calls + session.download_calls} == {4096}
    assert {call["request_timeout_s"] for call in session.upload_calls + session.download_calls} == {12}
    # Nothing but the file itself is left next to it.
    assert [path.name for path in out.parent.iterdir()] == ["blob.bin"]


@requires_bash
async def test_relative_paths_resolve_against_the_workdir(make_provider, tmp_path: Path) -> None:
    session = LocalBashSession()
    provider = make_provider(session)
    handle = await provider.create(spec(workdir=str(tmp_path)))
    source = tmp_path / "src.txt"
    source.write_text("hello\n")
    await provider.upload_file(handle, source, "relative/target.txt")
    assert (tmp_path / "relative/target.txt").read_text() == "hello\n"
    out = tmp_path / "out.txt"
    await provider.download_file(handle, "relative/target.txt", out)
    assert out.read_text() == "hello\n"
    assert session.download_calls[-1]["remote_path"] == f"{tmp_path}/relative/target.txt"


async def test_paths_are_passed_through_without_a_workdir(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    await provider.upload_file(handle, Path("/local/a"), "relative/a")
    await provider.upload_file(handle, Path("/local/a"), "/abs/a")
    assert [call["remote_path"] for call in fake_session.upload_calls] == ["relative/a", "/abs/a"]


@requires_bash
async def test_a_missing_remote_file_fails_the_transfer_not_the_sandbox(make_provider, tmp_path: Path) -> None:
    provider = make_provider(LocalBashSession())
    handle = await provider.create(spec())
    with pytest.raises(IdeGymTransferError, match="Path not found"):
        await provider.download_file(handle, str(tmp_path / "nope"), tmp_path / "out")
    with pytest.raises(IdeGymTransferError, match="Not a file"):
        await provider.download_file(handle, str(tmp_path), tmp_path / "out")
    assert handle.raw.stopped is False
    assert await provider.status(handle) is SandboxStatus.RUNNING


async def test_a_failed_download_leaves_the_existing_target_untouched(
    make_provider, fake_session: FakeSession, tmp_path: Path
) -> None:
    """The SDK writes in place, so without a staging file a failure would truncate the target."""
    target = tmp_path / "result.json"
    target.write_text("previous")
    provider = make_provider()
    handle = await provider.create(spec())

    async def half_written(server_id: int, remote_path: str, local_path: Path, **kwargs: Any) -> int:
        Path(local_path).write_bytes(b"{")
        raise IdeGYMConnectionError("reset mid-transfer")

    fake_session.download_file = half_written  # type: ignore[method-assign]
    with pytest.raises(IdeGymTransferError, match="reset mid-transfer"):
        await provider.download_file(handle, "/tmp/result.json", target)
    assert target.read_text() == "previous"
    assert [path.name for path in tmp_path.iterdir()] == ["result.json"]


@pytest.mark.parametrize("direction", ["upload", "download"])
async def test_a_transfer_to_a_gone_sandbox_marks_it_stopped(
    make_provider, fake_session: FakeSession, tmp_path: Path, direction: str
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.upload_results = [orchestrator_error(410)]
    fake_session.download_results = [orchestrator_error(410)]
    with pytest.raises(IdeGymTransferError, match="no longer available"):
        if direction == "upload":
            await provider.upload_file(handle, tmp_path / "x", "/tmp/x")
        else:
            await provider.download_file(handle, "/tmp/x", tmp_path / "x")
    assert handle.raw.stopped is True
    # Later transfers fail fast instead of making another round trip.
    with pytest.raises(IdeGymTransferError, match="has been stopped"):
        await provider.upload_file(handle, tmp_path / "x", "/tmp/x")
    assert len(fake_session.upload_calls) == (1 if direction == "upload" else 0)


async def test_an_upload_failure_names_the_paths(make_provider, fake_session: FakeSession, tmp_path: Path) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.upload_results = [sandbox_error(507, "No space left on device")]
    with pytest.raises(IdeGymTransferError, match=r"Uploading .*src\.bin to '/tmp/x\.bin'.*No space left") as raised:
        await provider.upload_file(handle, tmp_path / "src.bin", "/tmp/x.bin")
    assert isinstance(raised.value, IdeGymOperationError)
    assert isinstance(raised.value.__cause__, IdeGYMSandboxError)
    assert handle.raw.stopped is False


# --- status ----------------------------------------------------------------


async def test_status_running_stopped_and_unknown(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    assert await provider.status(handle) is SandboxStatus.RUNNING

    fake_session.status_results = [orchestrator_error(404)]
    assert await provider.status(handle) is SandboxStatus.STOPPED

    fake_session.status_results = [orchestrator_error(500)]
    assert await provider.status(handle) is SandboxStatus.UNKNOWN
    assert handle.raw.stopped is False

    fake_session.status_results = [
        IdeGymServerStatus(usable=False, availability="CRASHED", pod_ready=None, details="OOMKilled")
    ]
    assert await provider.status(handle) is SandboxStatus.STOPPED
    assert handle.raw.stopped is True
    calls = len(fake_session.status_calls)
    assert await provider.status(handle) is SandboxStatus.STOPPED
    assert len(fake_session.status_calls) == calls


async def test_status_of_a_closed_sandbox_is_stopped(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    await provider.close(handle)
    assert await provider.status(handle) is SandboxStatus.STOPPED
    assert fake_session.status_calls == []


# --- teardown --------------------------------------------------------------


async def test_close_is_idempotent(make_provider, fake_session: FakeSession) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    await provider.close(handle)
    await provider.close(handle)
    assert len(fake_session.stop_calls) == 1


@pytest.mark.parametrize("gone", [404, 410, "forgotten"])
async def test_close_treats_an_already_gone_server_as_success(
    make_provider, fake_session: FakeSession, gone: int | str
) -> None:
    provider = make_provider()
    handle = await provider.create(spec())
    if gone == "forgotten":
        fake_session.live_servers.clear()
    else:
        fake_session.stop_results = [orchestrator_error(gone)]
    await provider.close(handle)
    assert handle.raw.stopped is True
    assert len(fake_session.stop_calls) == 1


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
    provider = make_provider()
    handle = await provider.create(spec())
    fake_session.stop_results = [orchestrator_error(502)]
    with pytest.raises(IdeGymOperationError, match="Failed to stop IdeGYM sandbox"):
        await provider.close(handle)
    # The default operations.retries is 1: one round on top of the first.
    assert len(fake_session.stop_calls) == 2
    # A stop that failed for real must not leave the sandbox looking stopped: the
    # pod may well still be running, and status() has to keep saying so.
    assert handle.raw.stopped is False
    assert await provider.status(handle) is SandboxStatus.RUNNING
    # The next attempt reaches the same pod and succeeds after a transient failure.
    fake_session.stop_calls.clear()
    fake_session.stop_results = [orchestrator_error(503), None]
    await provider.close(handle)
    assert handle.raw.stopped is True
    assert len(fake_session.stop_calls) == 2


@pytest.mark.parametrize(
    "permanent",
    [orchestrator_error(500), orchestrator_error(400), orchestrator_error(401), IdeGymError("session stopped")],
    ids=["orchestrator-500", "bad-request", "unauthorized", "session-stopped"],
)
async def test_close_does_not_retry_a_failure_that_would_repeat(
    make_provider, fake_session: FakeSession, permanent: Exception
) -> None:
    provider = make_provider(operations={"retries": 3})
    handle = await provider.create(spec())
    fake_session.stop_results = [permanent]
    with pytest.raises(IdeGymOperationError, match="Failed to stop IdeGYM sandbox"):
        await provider.close(handle)
    assert len(fake_session.stop_calls) == 1
    assert handle.raw.stopped is False


async def test_close_retries_a_round_that_timed_out(
    make_provider, fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = make_provider(operations={"retries": 1, "close_timeout_s": 0.05})
    handle = await provider.create(spec())
    real_stop = fake_session.stop_server
    calls = 0

    async def slow_then_fast(server_id: int, *, polling: Any, timeout_s: float) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(1)
        return await real_stop(server_id, polling=polling, timeout_s=timeout_s)

    monkeypatch.setattr(fake_session, "stop_server", slow_then_fast)
    await provider.close(handle)
    assert calls == 2
    assert handle.raw.stopped is True


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


async def test_an_acquire_that_loses_to_aclose_is_handed_back(
    make_provider, fake_session: FakeSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = make_provider()
    gate = asyncio.Event()

    async def slow_acquire(connection: Any, attribution: Any) -> Any:
        await gate.wait()
        return fake_session

    monkeypatch.setattr(idegym_provider, "acquire_session", slow_acquire)
    pending = asyncio.ensure_future(provider.session())
    await asyncio.sleep(0)
    await provider.aclose()
    gate.set()
    with pytest.raises(IdeGymOperationError, match="has been closed"):
        await pending
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
