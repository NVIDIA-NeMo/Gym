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

"""Unit tests for the ECS Fargate sandbox *engine* internals.

Every AWS (boto3), SSH (subprocess), socket, and HTTP (aiohttp/urllib) seam is
mocked — no test touches a real network, process, or AWS account. The tests
drive the REAL method/function under test and assert on its observable
behavior (return values, raised errors, boto3/SSH call arguments, retry and
parsing logic). They complement ``test_ecs_fargate_provider.py`` (which covers
the Gym-facing adapter) by exercising the engine module directly.
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import threading
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlparse

import aiohttp
import pytest

from nemo_gym.sandbox.providers.ecs_fargate import engine


_ENG = "nemo_gym.sandbox.providers.ecs_fargate.engine"


class ClientError(Exception):
    """Local stand-in for ``botocore.exceptions.ClientError``.

    CI's base venv has no boto3/botocore (it's the ``sandbox`` extra), so we
    don't import it. engine.py only references ``ClientError`` via
    ``_require_aws_sdks()`` — which ``_patch_aws`` patches to return this class —
    so engine's ``except ClientError`` catches these instances unchanged.
    """

    def __init__(self, error_response, operation_name="Operation"):
        self.response = error_response
        self.operation_name = operation_name
        super().__init__(f"{operation_name}: {error_response}")


# ── Shared fixtures / helpers ─────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_engine_module_state(request):
    """Reset process-global engine caches and stub the AWS SDK accessor.

    CI's base venv has no boto3/botocore (it's the sandbox extra), so patch
    ``engine._require_aws_sdks`` to hand back mocks + the local fake ``ClientError``
    for every test — so engine's ``except ClientError`` catches the fakes the tests
    raise without real boto3. Tests needing specific clients override it via
    ``_patch_aws`` (nested, wins); the two tests that exercise ``_require_aws_sdks``
    itself opt out by name.
    """
    caches = (
        engine._ssm_config_cache,
        engine._exec_server_url_cache,
        engine._task_def_cache,
        engine._task_def_inflight,
        engine._active_sandboxes,
        engine.ImageBuilder._inflight_builds,
    )
    for cache in caches:
        cache.clear()
    engine.ImageBuilder._build_semaphore = None
    engine.ImageBuilder._build_semaphore_size = 0
    if request.node.name in {"test_require_aws_sdks_returns_triple", "test_require_aws_sdks_raises_when_missing"}:
        yield
    else:
        with patch(
            f"{_ENG}._require_aws_sdks",
            return_value=(MagicMock(name="boto3"), MagicMock(name="Config"), ClientError),
        ):
            yield
    for cache in caches:
        cache.clear()


def _client_error(code: str, op: str = "Operation", msg: str = "boom") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": msg}}, op)


@contextlib.contextmanager
def _patch_aws(**clients):
    """Patch ``engine._require_aws_sdks`` so ``boto3.client(name)`` returns mocks.

    Returns the fake ``boto3`` module. ``Config`` is a callable mock and
    ``ClientError`` is the *real* class so ``except ClientError`` works.
    """
    boto3 = MagicMock(name="boto3")

    def _client(name, **_kw):
        if name not in clients:
            raise AssertionError(f"unexpected boto3.client({name!r})")
        return clients[name]

    boto3.client.side_effect = _client
    with patch(f"{_ENG}._require_aws_sdks", return_value=(boto3, MagicMock(name="Config"), ClientError)):
        yield boto3


def _exec_sidecar(**overrides) -> engine.SshSidecarConfig:
    base = dict(
        sshd_port=2222,
        public_key_secret_arn="arn:aws:secretsmanager:us-east-1:1234:secret:pub",  # pragma: allowlist secret
        private_key_secret_arn="arn:aws:secretsmanager:us-east-1:1234:secret:priv",  # pragma: allowlist secret
        exec_server_port=5000,
    )
    base.update(overrides)
    return engine.SshSidecarConfig(**base)


def _agent_sidecar(**overrides) -> engine.SshSidecarConfig:
    base = dict(
        sshd_port=2222,
        public_key_secret_arn="arn:aws:secretsmanager:us-east-1:1234:secret:pub",  # pragma: allowlist secret
        private_key_secret_arn="arn:aws:secretsmanager:us-east-1:1234:secret:priv",  # pragma: allowlist secret
        exec_server_port=None,
    )
    base.update(overrides)
    return engine.SshSidecarConfig(**base)


def _make_sandbox(spec: engine.SandboxSpec | None = None, **cfg_overrides) -> engine.EcsFargateSandbox:
    cfg_kwargs = dict(
        region="us-west-2",
        cluster="test-cluster",
        subnets=["subnet-a"],
        security_groups=["sg-a"],
    )
    cfg_kwargs.update(cfg_overrides)
    cfg = engine.EcsFargateConfig(**cfg_kwargs)
    spec = spec or engine.SandboxSpec(image="python:3.12")
    return engine.EcsFargateSandbox(spec, ecs_config=cfg)


def _attach_exec_client(sb: engine.EcsFargateSandbox, result: engine.ExecResult | None = None):
    ec = MagicMock(name="exec_client")
    ec.exec = AsyncMock(return_value=result or engine.ExecResult("out", "err", 0))
    ec.upload = AsyncMock()
    ec.download = AsyncMock(return_value=b"payload")
    ec.close = AsyncMock()
    sb._exec_client = ec
    return ec


# ── _require_aws_sdks ─────────────────────────────────────────────────


def test_require_aws_sdks_returns_triple():
    pytest.importorskip("boto3")  # real boto3/botocore only with the sandbox extra
    from botocore.exceptions import ClientError as RealClientError

    boto3, config_cls, client_error = engine._require_aws_sdks()
    assert hasattr(boto3, "client")
    assert client_error is RealClientError
    assert callable(config_cls)


def test_require_aws_sdks_raises_when_missing():
    with patch("importlib.import_module", side_effect=ModuleNotFoundError("no boto3")):
        with pytest.raises(RuntimeError, match="requires boto3/botocore"):
            engine._require_aws_sdks()


# ── resolve_ecs_config_from_ssm ───────────────────────────────────────


def test_resolve_ecs_config_from_ssm_parses_and_caches():
    blob = {"cluster": "c1", "subnets": ["s1", "s2"]}
    ssm = MagicMock()
    ssm.get_parameter.return_value = {"Parameter": {"Value": json.dumps(blob)}}
    with _patch_aws(ssm=ssm):
        out = engine.resolve_ecs_config_from_ssm("us-west-2", project="harbor")
        # Second call is served from the per-(region, project) cache.
        out2 = engine.resolve_ecs_config_from_ssm("us-west-2", project="harbor")
    assert out == blob
    assert out2 is out
    ssm.get_parameter.assert_called_once_with(Name="/harbor/ecs-sandbox/config")


def test_resolve_ecs_config_from_ssm_parameter_not_found():
    ssm = MagicMock()
    ssm.get_parameter.side_effect = _client_error("ParameterNotFound", "GetParameter")
    with _patch_aws(ssm=ssm):
        with pytest.raises(RuntimeError, match="not found in us-west-2"):
            engine.resolve_ecs_config_from_ssm("us-west-2")


def test_resolve_ecs_config_from_ssm_other_client_error_reraised():
    ssm = MagicMock()
    ssm.get_parameter.side_effect = _client_error("AccessDenied", "GetParameter")
    with _patch_aws(ssm=ssm):
        with pytest.raises(ClientError):
            engine.resolve_ecs_config_from_ssm("us-west-2")


def test_resolve_ecs_config_from_ssm_invalid_json():
    ssm = MagicMock()
    ssm.get_parameter.return_value = {"Parameter": {"Value": "{not-json"}}
    with _patch_aws(ssm=ssm):
        with pytest.raises(RuntimeError, match="invalid JSON"):
            engine.resolve_ecs_config_from_ssm("us-east-1")


# ── _sanitize_id / _is_ecr_image_ref / _port_from_url ─────────────────


def test_sanitize_id_replaces_and_truncates():
    assert engine._sanitize_id("docker.io/foo/bar:latest") == "docker-io-foo-bar-latest"
    assert engine._sanitize_id("---") == "task"  # empty after strip -> fallback
    assert len(engine._sanitize_id("a" * 200, max_len=10)) == 10


def test_is_ecr_image_ref():
    assert engine._is_ecr_image_ref("463701203462.dkr.ecr.us-east-1.amazonaws.com/repo:tag")
    assert not engine._is_ecr_image_ref("docker.io/library/ubuntu:24.04")
    assert not engine._is_ecr_image_ref("ubuntu:24.04")


def test_port_from_url_defaults():
    assert engine._port_from_url(urlparse("http://h/x")) == 80
    assert engine._port_from_url(urlparse("https://h/x")) == 443
    assert engine._port_from_url(urlparse("http://h:8123/x")) == 8123


# ── _OutsideEndpointRoute / _OutsideEndpointRouting ───────────────────


def test_route_for_endpoint_and_rewrite():
    ep = engine.OutsideEndpoint(url="https://api.host:9000/v1", env_var="MODEL_BASE_URL")
    route = engine._OutsideEndpointRoute.for_endpoint(ep, remote_port=20001)
    assert route.host == "api.host"
    assert route.target_port == 9000
    assert route.remote_port == 20001
    assert route.scheme == "https"
    assert route.resolved_endpoint_url() == "https://127.0.0.1:20001/v1"
    assert route.resolve_url("https://api.host:9000/other") == "https://127.0.0.1:20001/other"


def test_route_for_endpoint_rejects_hostless_url():
    ep = engine.OutsideEndpoint(url="/relative/only", env_var="X")
    with pytest.raises(ValueError, match="Cannot resolve hostname"):
        engine._OutsideEndpointRoute.for_endpoint(ep)


def test_routing_for_exec_server_dedupes_and_allocates():
    endpoints = [
        engine.OutsideEndpoint(url="http://10.0.0.1:4000/v1", env_var="A"),
        engine.OutsideEndpoint(url="http://10.0.0.1:4000/v2", env_var="B"),
        engine.OutsideEndpoint(url="http://10.0.0.2:5000/v1", env_var="C"),
    ]
    sidecar = _exec_sidecar(sshd_port=2222, exec_server_port=5000)
    routing = engine._OutsideEndpointRouting.for_exec_server(endpoints, sidecar)
    specs = routing.reverse_specs
    # A and B share (host, port) so only two reverse specs are created.
    assert len(specs) == 2
    assert any(s.endswith(":10.0.0.1:4000") for s in specs)
    # 5000 is reserved by the exec server, so :5000 cannot be reused as a remote port.
    assert not any(s.startswith("5000:") for s in specs)
    overrides = routing.env_overrides()
    assert set(overrides) == {"A", "B", "C"}
    # A and B share the same reverse tunnel port (deduped) but keep their paths.
    assert urlparse(overrides["A"]).netloc == urlparse(overrides["B"]).netloc
    assert overrides["A"].endswith("/v1") and overrides["B"].endswith("/v2")
    assert all(v.startswith("http://127.0.0.1:") for v in overrides.values())


def test_routing_for_agent_server_validation_and_target():
    with pytest.raises(ValueError, match="only one OutsideEndpoint"):
        engine._OutsideEndpointRouting.for_agent_server(
            [engine.OutsideEndpoint("http://h:1/", "A"), engine.OutsideEndpoint("http://h:2/", "B")]
        )
    with pytest.raises(ValueError, match="requires OutsideEndpoint"):
        engine._OutsideEndpointRouting.for_agent_server([])

    routing = engine._OutsideEndpointRouting.for_agent_server(
        [engine.OutsideEndpoint("http://model.host:7000/v1", "MODEL")]
    )
    assert routing.agent_tunnel_port == 7000
    assert routing.agent_tunnel_target() == ("model.host", 7000)
    assert routing.resolved_endpoint_url("MODEL") == "http://127.0.0.1:7000/v1"
    assert routing.resolved_endpoint_url("MISSING") is None


def test_routing_agent_tunnel_target_requires_endpoints():
    routing = engine._OutsideEndpointRouting.empty()
    with pytest.raises(ValueError, match="requires OutsideEndpoint"):
        routing.agent_tunnel_target()


def test_routing_resolve_url_paths():
    # Reverse-tunnel match by netloc.
    endpoints = [engine.OutsideEndpoint("http://10.0.0.1:4000/v1", "A")]
    routing = engine._OutsideEndpointRouting.for_exec_server(endpoints, _exec_sidecar())
    assert routing.resolve_url("http://10.0.0.1:4000/chat").startswith("http://127.0.0.1:")

    # Agent-tunnel fallback rewrites netloc to the agent port.
    agent = engine._OutsideEndpointRouting.for_agent_server([engine.OutsideEndpoint("http://m:7000/v1", "M")])
    assert agent.resolve_url("http://anything:1/path") == "http://127.0.0.1:7000/path"

    # No routes and no agent tunnel -> error.
    with pytest.raises(RuntimeError, match="requires SSH reverse tunnel"):
        engine._OutsideEndpointRouting.empty().resolve_url("http://x:1/")


def test_allocate_reverse_port_prefers_then_scans_then_exhausts():
    used: set[int] = set()
    assert engine._OutsideEndpointRouting._allocate_reverse_port(4000, used) == 4000
    # 4000 now reserved -> a second request for it scans for a free port.
    second = engine._OutsideEndpointRouting._allocate_reverse_port(4000, used)
    assert second != 4000 and 20000 <= second < 61000
    # Invalid preferred (0) with the whole scan range occupied -> RuntimeError.
    full = set(range(20000, 61000))
    with pytest.raises(RuntimeError, match="No available local port"):
        engine._OutsideEndpointRouting._allocate_reverse_port(0, full)


# ── _is_retryable_error / _retry_with_backoff ─────────────────────────


def test_is_retryable_error_codes_and_messages():
    assert engine._is_retryable_error(_client_error("ThrottlingException"))
    assert engine._is_retryable_error(RuntimeError("Rate exceeded for op"))
    assert engine._is_retryable_error(RuntimeError("read timeout while connecting"))
    assert not engine._is_retryable_error(RuntimeError("totally fatal"))
    assert not engine._is_retryable_error(_client_error("AccessDenied", msg="nope"))


def test_retry_with_backoff_success_first_try():
    fn = MagicMock(return_value="ok")
    assert engine._retry_with_backoff(fn, operation_name="op") == "ok"
    fn.assert_called_once()


def test_retry_with_backoff_non_retryable_raises_immediately():
    fn = MagicMock(side_effect=ValueError("fatal"))
    with pytest.raises(ValueError):
        engine._retry_with_backoff(fn, operation_name="op")
    fn.assert_called_once()


def test_retry_with_backoff_retries_then_succeeds():
    fn = MagicMock(side_effect=[_client_error("ThrottlingException"), "done"])
    with patch(f"{_ENG}.time.sleep") as sleep:
        out = engine._retry_with_backoff(fn, operation_name="op", base_delay=0.01)
    assert out == "done"
    assert fn.call_count == 2
    sleep.assert_called_once()


def test_retry_with_backoff_exhausts_max_retries():
    fn = MagicMock(side_effect=_client_error("ThrottlingException"))
    with patch(f"{_ENG}.time.sleep"):
        with pytest.raises(ClientError):
            engine._retry_with_backoff(fn, operation_name="op", max_retries=2)
    # 1 initial + 2 retries.
    assert fn.call_count == 3


# ── _free_port / secrets ──────────────────────────────────────────────


def test_free_port_returns_bound_port():
    sock = MagicMock()
    sock.__enter__.return_value = sock
    sock.getsockname.return_value = ("127.0.0.1", 54321)
    with patch(f"{_ENG}.socket.socket", return_value=sock):
        assert engine._free_port() == 54321
    sock.bind.assert_called_once_with(("127.0.0.1", 0))


def test_download_secret_to_string():
    sm = MagicMock()
    sm.get_secret_value.return_value = {"SecretString": "the-key-material"}  # pragma: allowlist secret
    with _patch_aws(secretsmanager=sm):
        out = engine.download_secret_to_string("arn:secret:priv", region="us-east-1")  # pragma: allowlist secret
    assert out == "the-key-material"
    sm.get_secret_value.assert_called_once_with(SecretId="arn:secret:priv")  # pragma: allowlist secret


def test_download_secret_to_file_writes_0600():
    with patch(f"{_ENG}.download_secret_to_string", return_value="PRIVATE-KEY"):  # pragma: allowlist secret
        path = engine.download_secret_to_file("arn:secret:priv", region="us-east-1")  # pragma: allowlist secret
    p = Path(path)
    try:
        assert p.read_text() == "PRIVATE-KEY"
        assert (p.stat().st_mode & 0o777) == 0o600
    finally:
        p.unlink()


# ── SshTunnel ─────────────────────────────────────────────────────────


def _alive_proc(pid: int = 4242):
    proc = MagicMock(name="proc")
    proc.poll.return_value = None
    proc.pid = pid
    return proc


def _dead_proc(stderr: bytes):
    proc = MagicMock(name="proc")
    proc.poll.return_value = 1
    proc.stderr.read.return_value = stderr
    return proc


def test_ssh_tunnel_build_cmd_simple_forward_and_extra():
    t = engine.SshTunnel(
        host="1.2.3.4",
        port=2222,
        key_file="/tmp/k",
        forward_port=5000,
        forwards=["6000:localhost:6000"],
        reverses=["7000:10.0.0.1:7000"],
        local_port_override=18000,
    )
    cmd = t._build_ssh_cmd()
    assert cmd[0] == "ssh" and "-N" in cmd
    assert "/tmp/k" in cmd
    assert "-L" in cmd and "127.0.0.1:18000:127.0.0.1:5000" in cmd
    assert "6000:localhost:6000" in cmd
    assert "7000:10.0.0.1:7000" in cmd
    assert cmd[-1] == "root@1.2.3.4"


def test_ssh_tunnel_local_port_property_before_open():
    t = engine.SshTunnel(host="h", key_file="/k")
    with pytest.raises(RuntimeError, match="Tunnel not open yet"):
        _ = t.local_port


def test_ssh_tunnel_open_noop_when_already_open():
    t = engine.SshTunnel(host="h", key_file="/k", forward_port=5000)
    t._proc = _alive_proc()
    with patch(f"{_ENG}.subprocess.Popen") as popen:
        t.open()
    popen.assert_not_called()


def test_ssh_tunnel_open_success_simple_forward():
    t = engine.SshTunnel(host="h", key_file="/k", forward_port=5000)
    proc = _alive_proc(pid=99)
    with (
        patch(f"{_ENG}.subprocess.Popen", return_value=proc) as popen,
        patch(f"{_ENG}._free_port", return_value=18055),
        patch(f"{_ENG}.time.sleep"),
        patch.object(engine.SshTunnel, "_wait_for_local_port") as wait_port,
    ):
        t.open()
    assert t.is_open
    assert t.local_port == 18055
    wait_port.assert_called_once()
    popen.assert_called_once()


def test_ssh_tunnel_open_retries_when_forward_port_not_ready():
    t = engine.SshTunnel(host="h", key_file="/k", forward_port=5000)
    with (
        patch(f"{_ENG}.subprocess.Popen", side_effect=[_alive_proc(), _alive_proc()]),
        patch(f"{_ENG}._free_port", side_effect=[18001, 18002]),
        patch(f"{_ENG}.time.sleep"),
        patch.object(engine.SshTunnel, "_kill"),
        patch.object(engine.SshTunnel, "_wait_for_local_port", side_effect=[TimeoutError("nope"), None]),
    ):
        t.open(max_retries=3)
    assert t.local_port == 18002


def test_ssh_tunnel_open_retries_on_connection_refused_then_succeeds():
    t = engine.SshTunnel(host="h", key_file="/k", forward_port=5000)
    with (
        patch(
            f"{_ENG}.subprocess.Popen", side_effect=[_dead_proc(b"ssh: connect: Connection refused"), _alive_proc()]
        ),
        patch(f"{_ENG}._free_port", side_effect=[18001, 18002]),
        patch(f"{_ENG}.time.sleep"),
        patch.object(engine.SshTunnel, "_wait_for_local_port"),
    ):
        t.open(max_retries=3, initial_backoff=0.01)
    assert t.is_open


def test_ssh_tunnel_open_raises_on_fatal_stderr():
    t = engine.SshTunnel(host="h", key_file="/k", forward_port=5000)
    with (
        patch(f"{_ENG}.subprocess.Popen", return_value=_dead_proc(b"Permission denied (publickey)")),
        patch(f"{_ENG}._free_port", return_value=18001),
        patch(f"{_ENG}.time.sleep"),
    ):
        with pytest.raises(RuntimeError, match="exited immediately"):
            t.open(max_retries=3)


def test_ssh_tunnel_open_exhausts_retries():
    t = engine.SshTunnel(host="h", key_file="/k", forward_port=5000)
    with (
        patch(f"{_ENG}.subprocess.Popen", return_value=_dead_proc(b"Connection timed out")),
        patch(f"{_ENG}._free_port", return_value=18001),
        patch(f"{_ENG}.time.sleep"),
    ):
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            t.open(max_retries=2, initial_backoff=0.01)


def test_ssh_tunnel_kill_terminates_and_force_kills():
    t = engine.SshTunnel(host="h", key_file="/k")
    proc = MagicMock()
    proc.wait.side_effect = engine.subprocess.TimeoutExpired(cmd="ssh", timeout=5)
    t._proc = proc
    t.close()
    proc.terminate.assert_called_once()
    proc.kill.assert_called_once()
    assert t._proc is None


def test_ssh_tunnel_kill_swallows_process_lookup_error():
    t = engine.SshTunnel(host="h", key_file="/k")
    proc = MagicMock()
    proc.terminate.side_effect = ProcessLookupError()
    t._proc = proc
    t.close()  # must not raise
    assert t._proc is None


def test_ssh_tunnel_kill_noop_when_no_proc():
    t = engine.SshTunnel(host="h", key_file="/k")
    t.close()  # _proc is None -> early return, no error
    assert t._proc is None


def _connect_sock(recv: bytes | None = None, *, raise_oserror: bool = False):
    sock = MagicMock()
    sock.__enter__.return_value = sock
    if raise_oserror:
        sock.connect.side_effect = OSError("refused")
    if recv is not None:
        sock.recv.return_value = recv
    return sock


def test_ssh_tunnel_wait_for_local_port_success():
    t = engine.SshTunnel(host="h", key_file="/k")
    with patch(f"{_ENG}.socket.socket", return_value=_connect_sock()):
        t._wait_for_local_port(18000, timeout=5.0)  # connects immediately, returns


def test_ssh_tunnel_wait_for_local_port_proc_died():
    t = engine.SshTunnel(host="h", key_file="/k")
    t._proc = MagicMock()
    t._proc.poll.return_value = 1
    with patch(f"{_ENG}.time.monotonic", side_effect=[0.0, 1.0]):
        with pytest.raises(RuntimeError, match="exited while waiting"):
            t._wait_for_local_port(18000, timeout=30.0)


def test_ssh_tunnel_wait_for_local_port_timeout():
    t = engine.SshTunnel(host="h", key_file="/k")
    with (
        patch(f"{_ENG}.socket.socket", return_value=_connect_sock(raise_oserror=True)),
        patch(f"{_ENG}.time.monotonic", side_effect=[0.0, 0.1, 100.0]),
        patch(f"{_ENG}.time.sleep"),
    ):
        with pytest.raises(TimeoutError, match="not open after"):
            t._wait_for_local_port(18000, timeout=5.0)


class _UrlResp:
    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_ssh_tunnel_poll_health_success():
    t = engine.SshTunnel(host="h", key_file="/k")
    t._proc = _alive_proc()
    with (
        patch("urllib.request.urlopen", return_value=_UrlResp(200)),
        patch(f"{_ENG}.time.monotonic", side_effect=[0.0, 0.1]),
        patch(f"{_ENG}.time.sleep"),
    ):
        t._poll_health("http://127.0.0.1:18000/health", timeout=5.0)


def test_ssh_tunnel_poll_health_dies_midway():
    t = engine.SshTunnel(host="h", key_file="/k")
    t._proc = MagicMock()
    t._proc.poll.return_value = 1  # not open
    with patch(f"{_ENG}.time.monotonic", side_effect=[0.0, 0.1]):
        with pytest.raises(RuntimeError, match="died while waiting"):
            t._poll_health("http://127.0.0.1:18000/health", timeout=5.0)


def test_ssh_tunnel_poll_health_timeout():
    import urllib.error

    t = engine.SshTunnel(host="h", key_file="/k")
    t._proc = _alive_proc()
    with (
        patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")),
        patch(f"{_ENG}.time.monotonic", side_effect=[0.0, 0.1, 100.0]),
        patch(f"{_ENG}.time.sleep"),
    ):
        with pytest.raises(TimeoutError, match="not reachable"):
            t._poll_health("http://127.0.0.1:18000/health", timeout=5.0)


def test_ssh_tunnel_wait_ready_dispatch():
    t = engine.SshTunnel(host="h", key_file="/k")
    with patch.object(engine.SshTunnel, "_poll_health") as poll:
        t.wait_ready(health_url="http://x/health", timeout=10.0)
    poll.assert_called_once()

    t2 = engine.SshTunnel(host="h", key_file="/k", local_port_override=18000)
    with patch.object(engine.SshTunnel, "_wait_for_local_port") as wait_port:
        t2.wait_ready(timeout=10.0)
    wait_port.assert_called_once()

    # Neither health_url nor local_port -> no-op (no error).
    engine.SshTunnel(host="h", key_file="/k").wait_ready()


def test_ssh_tunnel_check_health_and_context_manager():
    t = engine.SshTunnel(host="h", key_file="/k", forward_port=5000)
    assert t.check_health() is False  # no proc
    proc = _alive_proc()
    with (
        patch(f"{_ENG}.subprocess.Popen", return_value=proc),
        patch(f"{_ENG}._free_port", return_value=18000),
        patch(f"{_ENG}.time.sleep"),
        patch.object(engine.SshTunnel, "_wait_for_local_port"),
        patch.object(engine.SshTunnel, "_kill") as kill,
    ):
        with t as ctx:
            assert ctx is t
            assert t.check_health() is True
    kill.assert_called_once()  # __exit__ -> close -> _kill


# ── build_ssh_sidecar_container ───────────────────────────────────────


def test_build_ssh_sidecar_container_defaults_and_watchdog():
    cfg = _exec_sidecar(sshd_port=52222, image=None)
    c = engine.build_ssh_sidecar_container(cfg, public_key_value="ssh-rsa KEY", max_lifetime_sec=3600)
    assert c["name"] == "ssh-tunnel"
    assert c["image"] == "alpine:latest"
    assert c["environment"] == [{"name": "SSH_PUBLIC_KEY", "value": "ssh-rsa KEY"}]
    assert "sleep 3600" in c["command"][0]  # watchdog present
    assert "Port 52222" in c["command"][0]
    assert c["healthCheck"]["command"] == ["CMD-SHELL", "nc -z localhost 52222 || exit 1"]
    assert "logConfiguration" not in c


def test_build_ssh_sidecar_container_no_watchdog_and_logs():
    cfg = _exec_sidecar(sshd_port=2222, image="myimg:1")
    c = engine.build_ssh_sidecar_container(
        cfg,
        public_key_value="K",
        max_lifetime_sec=0,
        log_group="/aws/ecs/sb",
        log_region="us-west-2",
        log_stream_prefix="pref",
    )
    assert c["image"] == "myimg:1"
    assert "sidecar watchdog" not in c["command"][0]
    assert c["logConfiguration"]["options"]["awslogs-group"] == "/aws/ecs/sb"
    assert c["logConfiguration"]["options"]["awslogs-stream-prefix"] == "pref-tunnel"


# ── ExecClient ────────────────────────────────────────────────────────


class _FakeAiohttpCtx:
    """Async context manager standing in for ``session.request(...)``."""

    def __init__(self, result):
        self._result = result  # _FakeResp or Exception

    async def __aenter__(self):
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    async def __aexit__(self, *exc):
        return False


class _FakeResp:
    def __init__(self, status: int, body: bytes):
        self.status = status
        self._body = body

    async def read(self) -> bytes:
        return self._body


def _fake_session(results):
    session = MagicMock(name="session")
    session.closed = False
    session.request = MagicMock(side_effect=[_FakeAiohttpCtx(r) for r in results])
    return session


async def test_exec_client_exec_maps_result():
    client = engine.ExecClient(port=18000)
    with patch.object(client, "_post", AsyncMock(return_value={"stdout": "o", "stderr": "e", "rc": 7})) as post:
        out = await client.exec("echo hi", timeout=11)
    assert (out.stdout, out.stderr, out.return_code) == ("o", "e", 7)
    post.assert_awaited_once_with("/exec", {"cmd": "echo hi", "timeout": 11})


async def test_exec_client_upload_bytes_and_mode():
    client = engine.ExecClient(port=18000)
    with patch.object(client, "_post", AsyncMock(return_value={"ok": True})) as post:
        await client.upload("/remote/f", b"hello", mode="755")
    _, body = post.await_args.args
    assert body["path"] == "/remote/f"
    assert base64.b64decode(body["content"]) == b"hello"
    assert body["mode"] == "755"


async def test_exec_client_upload_reads_path(tmp_path):
    f = tmp_path / "blob.bin"
    f.write_bytes(b"frompath")
    client = engine.ExecClient(port=18000)
    with patch.object(client, "_post", AsyncMock(return_value={"ok": True})) as post:
        await client.upload("/remote/blob", f)
    _, body = post.await_args.args
    assert base64.b64decode(body["content"]) == b"frompath"


async def test_exec_client_upload_retries_then_fails():
    client = engine.ExecClient(port=18000)
    with (
        patch.object(client, "_post", AsyncMock(side_effect=TimeoutError("slow"))),
        patch(f"{_ENG}.asyncio.sleep", AsyncMock()) as sleep,
    ):
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            await client.upload("/remote/f", b"x", max_retries=2)
    sleep.assert_awaited()  # backed off between attempts


async def test_exec_client_upload_raises_when_server_reports_not_ok():
    client = engine.ExecClient(port=18000)
    with (
        patch.object(client, "_post", AsyncMock(return_value={"ok": False, "error": "disk full"})),
        patch(f"{_ENG}.asyncio.sleep", AsyncMock()),
    ):
        with pytest.raises(RuntimeError, match="failed after 1 attempts"):
            await client.upload("/remote/f", b"x", max_retries=1)


async def test_exec_client_download_quotes_path():
    client = engine.ExecClient(port=18000)
    with patch.object(client, "_request", AsyncMock(return_value=b"DL")) as req:
        out = await client.download("/a b/c")
    assert out == b"DL"
    assert req.await_args.kwargs["url"].endswith("/download?path=/a%20b/c")


async def test_exec_client_health_true_and_false():
    client = engine.ExecClient(port=18000)
    with patch.object(client, "_request", AsyncMock(return_value=b"{}")):
        assert await client.health() is True
    with patch.object(client, "_request", AsyncMock(side_effect=ConnectionError("x"))):
        assert await client.health() is False


async def test_exec_client_post_computes_timeout_and_parses_json():
    client = engine.ExecClient(port=18000, connect_timeout=30.0)
    with patch.object(client, "_request", AsyncMock(return_value=b'{"rc": 0}')) as req:
        out = await client._post("/exec", {"cmd": "x", "timeout": 100})
    assert out == {"rc": 0}
    # cmd timeout (100) + 30 dominates the connect timeout (30).
    assert req.await_args.kwargs["timeout"] == 130


async def test_exec_client_post_honors_timeout_override():
    client = engine.ExecClient(port=18000, connect_timeout=30.0)
    with patch.object(client, "_request", AsyncMock(return_value=b"{}")) as req:
        await client._post("/upload", {"path": "x"}, timeout_override=222.0)
    assert req.await_args.kwargs["timeout"] == 222.0


async def test_exec_client_request_success():
    client = engine.ExecClient(port=18000)
    session = _fake_session([_FakeResp(200, b"BODY")])
    with patch.object(client, "_ensure_session", AsyncMock(return_value=session)):
        out = await client._request(label="t", url="http://x/y", method="GET", timeout=5, max_retries=2)
    assert out == b"BODY"


async def test_exec_client_request_http_error_not_retried():
    client = engine.ExecClient(port=18000)
    session = _fake_session([_FakeResp(500, b"oops")])
    with patch.object(client, "_ensure_session", AsyncMock(return_value=session)):
        with pytest.raises(RuntimeError, match="HTTP 500"):
            await client._request(label="t", url="http://x/y", method="GET", timeout=5, max_retries=3)
    # 4xx/5xx is terminal — exactly one request attempt.
    assert session.request.call_count == 1


async def test_exec_client_request_retries_transient_then_succeeds():
    client = engine.ExecClient(port=18000)
    session = _fake_session([aiohttp.ClientError("reset"), _FakeResp(200, b"OK")])
    with (
        patch.object(client, "_ensure_session", AsyncMock(return_value=session)),
        patch(f"{_ENG}.asyncio.sleep", AsyncMock()) as sleep,
    ):
        out = await client._request(label="t", url="http://x/y", method="GET", timeout=5, max_retries=3)
    assert out == b"OK"
    sleep.assert_awaited_once()


async def test_exec_client_request_exhausts_to_connection_error():
    client = engine.ExecClient(port=18000)
    session = _fake_session([aiohttp.ClientError("a"), aiohttp.ClientError("b")])
    with (
        patch.object(client, "_ensure_session", AsyncMock(return_value=session)),
        patch(f"{_ENG}.asyncio.sleep", AsyncMock()),
    ):
        with pytest.raises(ConnectionError, match="failed after 2 attempts"):
            await client._request(label="t", url="http://x/y", method="GET", timeout=5, max_retries=2)


async def test_exec_client_request_zero_retries_is_unreachable():
    # Defensive guard: a non-positive retry budget yields a clear ConnectionError
    # rather than silently returning None.
    client = engine.ExecClient(port=18000)
    session = _fake_session([])
    with patch.object(client, "_ensure_session", AsyncMock(return_value=session)):
        with pytest.raises(ConnectionError, match="unreachable"):
            await client._request(label="t", url="http://x/y", method="GET", timeout=5, max_retries=0)
    session.request.assert_not_called()


async def test_exec_client_ensure_session_creates_and_reuses():
    client = engine.ExecClient(port=18000)
    fake = MagicMock(closed=False)
    fake.close = AsyncMock()
    with patch(f"{_ENG}.aiohttp.ClientSession", return_value=fake) as ctor:
        s1 = await client._ensure_session()
        s2 = await client._ensure_session()
    assert s1 is fake and s2 is fake
    ctor.assert_called_once()  # reused, not recreated
    await client.close()
    fake.close.assert_awaited_once()
    assert client._session is None


# ── Embedded exec server (_H request handler) ─────────────────────────


@pytest.fixture(scope="module")
def exec_server_handler():
    """Compile the embedded exec-server script and return its ``_H`` handler."""
    ns: dict = {}
    exec(compile(engine.EXEC_SERVER_SCRIPT, "<exec_server>", "exec"), ns)
    return ns["_H"]


def _drive_handler(handler_cls, *, path: str, method: str, body: dict | None = None):
    """Invoke a handler method with stubbed HTTP plumbing; return (code, payload)."""
    h = handler_cls.__new__(handler_cls)
    raw = json.dumps(body).encode() if body is not None else b""
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw))}
    captured: dict = {}
    h.send_response = lambda code: captured.__setitem__("code", code)
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda: None
    if method == "GET":
        h.do_GET()
    else:
        h.do_POST()
    out = h.wfile.getvalue()
    parsed = json.loads(out) if out and path != "/download" else out
    return captured.get("code"), parsed


def test_exec_server_health(exec_server_handler):
    code, payload = _drive_handler(exec_server_handler, path="/health", method="GET")
    assert code == 200 and payload == {"ok": True}


def test_exec_server_exec_runs_command(exec_server_handler):
    fake = types.SimpleNamespace(stdout=b"hello\n", stderr=b"", returncode=0)
    with patch("subprocess.run", return_value=fake):
        code, payload = _drive_handler(exec_server_handler, path="/exec", method="POST", body={"cmd": "echo hello"})
    assert code == 200
    assert payload == {"stdout": "hello\n", "stderr": "", "rc": 0}


def test_exec_server_exec_missing_cmd(exec_server_handler):
    code, payload = _drive_handler(exec_server_handler, path="/exec", method="POST", body={"timeout": 5})
    assert code == 400
    assert "missing 'cmd'" in payload["error"]


def test_exec_server_exec_timeout(exec_server_handler):
    import subprocess as _sp

    with patch("subprocess.run", side_effect=_sp.TimeoutExpired(cmd="x", timeout=3)):
        code, payload = _drive_handler(
            exec_server_handler, path="/exec", method="POST", body={"cmd": "sleep 9", "timeout": 3}
        )
    assert code == 200
    assert payload["rc"] == 124 and "timed out" in payload["stderr"]


def test_exec_server_exec_generic_error(exec_server_handler):
    with patch("subprocess.run", side_effect=RuntimeError("boom")):
        code, payload = _drive_handler(exec_server_handler, path="/exec", method="POST", body={"cmd": "x"})
    assert code == 200
    assert payload["rc"] == -1 and "boom" in payload["stderr"]


def test_exec_server_upload_and_chmod(exec_server_handler, tmp_path):
    target = tmp_path / "sub" / "out.txt"
    content = base64.b64encode(b"written").decode()
    code, payload = _drive_handler(
        exec_server_handler,
        path="/upload",
        method="POST",
        body={"path": str(target), "content": content, "mode": "600"},
    )
    assert code == 200 and payload == {"ok": True}
    assert target.read_bytes() == b"written"
    assert (target.stat().st_mode & 0o777) == 0o600


def test_exec_server_upload_missing_fields(exec_server_handler):
    code, payload = _drive_handler(exec_server_handler, path="/upload", method="POST", body={"path": "/x"})
    assert code == 400 and "missing path/content" in payload["error"]


def test_exec_server_download_roundtrip(exec_server_handler, tmp_path):
    f = tmp_path / "dl.bin"
    f.write_bytes(b"DATA-BYTES")
    code, raw = _drive_handler(exec_server_handler, path="/download", method="GET", body=None)
    # No ?path -> 400 (driver appends none); test explicit path separately below.
    assert code == 400

    h = exec_server_handler.__new__(exec_server_handler)
    h.path = f"/download?path={f}"
    h.wfile = io.BytesIO()
    h.headers = {}
    captured: dict = {}
    h.send_response = lambda code: captured.__setitem__("code", code)
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda: None
    h.do_GET()
    assert captured["code"] == 200
    assert h.wfile.getvalue() == b"DATA-BYTES"


def test_exec_server_download_not_found(exec_server_handler, tmp_path):
    missing = tmp_path / "nope.bin"
    h = exec_server_handler.__new__(exec_server_handler)
    h.path = f"/download?path={missing}"
    h.wfile = io.BytesIO()
    h.headers = {}
    captured: dict = {}
    h.send_response = lambda code: captured.__setitem__("code", code)
    h.send_header = lambda *a, **k: None
    h.end_headers = lambda: None
    h.do_GET()
    assert captured["code"] == 404


def test_exec_server_unknown_routes(exec_server_handler):
    code, payload = _drive_handler(exec_server_handler, path="/bogus", method="GET")
    assert code == 404
    code, payload = _drive_handler(exec_server_handler, path="/bogus", method="POST", body={})
    assert code == 404


# ── ImageBuilder: ECR queries ─────────────────────────────────────────


def _ecr_repo(region: str = "us-west-2") -> str:
    return f"123456789012.dkr.ecr.{region}.amazonaws.com/sandbox"


def _zip_names(data: bytes) -> set[str]:
    import zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return set(zf.namelist())


def test_ecr_region_extraction_and_fallback():
    assert engine.ImageBuilder._ecr_region(_ecr_repo("eu-west-1")) == "eu-west-1"
    assert engine.ImageBuilder._ecr_region("plainrepo", fallback="us-east-2") == "us-east-2"


def test_image_exists_in_ecr_true():
    ecr = MagicMock()
    ecr.describe_images.return_value = {"imageDetails": [{}]}
    with _patch_aws(ecr=ecr) as boto3:
        assert engine.ImageBuilder.image_exists_in_ecr(_ecr_repo(), "tag1") is True
    # region parsed from the repo URL, repo_name stripped of the registry host.
    boto3.client.assert_called_once_with("ecr", region_name="us-west-2")
    ecr.describe_images.assert_called_once_with(repositoryName="sandbox", imageIds=[{"imageTag": "tag1"}])


def test_image_exists_in_ecr_false_on_image_not_found():
    ecr = MagicMock()
    ecr.describe_images.side_effect = _client_error("ImageNotFoundException", "DescribeImages")
    with _patch_aws(ecr=ecr):
        assert engine.ImageBuilder.image_exists_in_ecr(_ecr_repo(), "tag1") is False


def test_image_exists_in_ecr_fails_fast_on_missing_repo():
    # A missing repository fails fast with a clear error rather than reporting "image absent" and
    # then building/pushing into a non-existent repo (which would only fail later at the push).
    ecr = MagicMock()
    ecr.describe_images.side_effect = _client_error("RepositoryNotFoundException", "DescribeImages")
    with _patch_aws(ecr=ecr):
        with pytest.raises(RuntimeError, match="does not exist"):
            engine.ImageBuilder.image_exists_in_ecr(_ecr_repo(), "tag1")


def test_image_exists_in_ecr_reraises_other():
    ecr = MagicMock()
    ecr.describe_images.side_effect = _client_error("AccessDeniedException", "DescribeImages")
    with _patch_aws(ecr=ecr):
        with pytest.raises(ClientError):
            engine.ImageBuilder.image_exists_in_ecr(_ecr_repo(), "tag1")


def test_list_ecr_tags_paginates():
    ecr = MagicMock()
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {"imageIds": [{"imageTag": "a"}, {"imageDigest": "sha256:x"}]},
        {"imageIds": [{"imageTag": "b"}, {"imageTag": "a"}]},
    ]
    ecr.get_paginator.return_value = paginator
    with _patch_aws(ecr=ecr):
        tags = engine.ImageBuilder.list_ecr_tags(_ecr_repo())
    assert tags == {"a", "b"}
    paginator.paginate.assert_called_once_with(repositoryName="sandbox", filter={"tagStatus": "TAGGED"})


def test_list_ecr_tags_empty_on_repo_not_found():
    ecr = MagicMock()
    paginator = MagicMock()
    paginator.paginate.side_effect = _client_error("RepositoryNotFoundException", "ListImages")
    ecr.get_paginator.return_value = paginator
    with _patch_aws(ecr=ecr):
        assert engine.ImageBuilder.list_ecr_tags(_ecr_repo()) == set()


def test_list_ecr_tags_reraises_other():
    ecr = MagicMock()
    paginator = MagicMock()
    paginator.paginate.side_effect = _client_error("AccessDenied", "ListImages")
    ecr.get_paginator.return_value = paginator
    with _patch_aws(ecr=ecr):
        with pytest.raises(ClientError):
            engine.ImageBuilder.list_ecr_tags(_ecr_repo())


def test_ecr_docker_login_success_builds_command():
    result = types.SimpleNamespace(returncode=0, stderr="")
    with patch(f"{_ENG}.subprocess.run", return_value=result) as run:
        engine.ImageBuilder.ecr_docker_login(_ecr_repo("us-west-2"))
    cmd = run.call_args.args[0]
    assert "aws ecr get-login-password --region us-west-2" in cmd
    assert "docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com" in cmd


def test_ecr_docker_login_no_region_flag_for_plain_repo():
    result = types.SimpleNamespace(returncode=0, stderr="")
    with patch(f"{_ENG}.subprocess.run", return_value=result) as run:
        engine.ImageBuilder.ecr_docker_login("plainrepo")
    assert "--region" not in run.call_args.args[0]


def test_ecr_docker_login_failure_raises():
    result = types.SimpleNamespace(returncode=1, stderr="bad creds")
    with patch(f"{_ENG}.subprocess.run", return_value=result):
        with pytest.raises(RuntimeError, match="ECR docker login failed: bad creds"):
            engine.ImageBuilder.ecr_docker_login(_ecr_repo())


def test_docker_push_to_ecr_success():
    tag_res = types.SimpleNamespace(returncode=0, stderr="")
    push_res = types.SimpleNamespace(returncode=0, stderr="")
    with patch(f"{_ENG}.subprocess.run", side_effect=[tag_res, push_res]) as run:
        url = engine.ImageBuilder.docker_push_to_ecr("local:img", _ecr_repo(), "t1")
    assert url == f"{_ecr_repo()}:t1"
    assert run.call_args_list[0].args[0] == ["docker", "tag", "local:img", f"{_ecr_repo()}:t1"]
    assert run.call_args_list[1].args[0] == ["docker", "push", f"{_ecr_repo()}:t1"]


def test_docker_push_to_ecr_failure_raises():
    tag_res = types.SimpleNamespace(returncode=0, stderr="")
    push_res = types.SimpleNamespace(returncode=1, stderr="denied")
    with patch(f"{_ENG}.subprocess.run", side_effect=[tag_res, push_res]):
        with pytest.raises(RuntimeError, match="docker push .* failed: denied"):
            engine.ImageBuilder.docker_push_to_ecr("local:img", _ecr_repo(), "t1")


# ── ImageBuilder: build orchestration ─────────────────────────────────


def _build_cfg(**overrides) -> engine.EcsFargateConfig:
    base = dict(
        region="us-west-2",
        ecr_repository=_ecr_repo(),
        environment_dir="/env",
        s3_bucket="bkt",
        codebuild_service_role="arn:aws:iam::123:role/cb",
        build_parallelism=4,
    )
    base.update(overrides)
    return engine.EcsFargateConfig(**base)


def test_ensure_image_built_requires_repo_and_dir():
    with pytest.raises(ValueError, match="ecr_repository and environment_dir are required"):
        engine.ImageBuilder.ensure_image_built(cfg=engine.EcsFargateConfig(region="us-west-2"), environment_name="env")


def test_ensure_image_built_cache_hit_skips_build():
    cfg = _build_cfg()
    with (
        patch.object(engine.ImageBuilder, "get_ecr_image_tag", return_value="env__abcd1234"),
        patch.object(engine.ImageBuilder, "image_exists_in_ecr", return_value=True),
        patch.object(engine.ImageBuilder, "_build_and_push") as build,
    ):
        url = engine.ImageBuilder.ensure_image_built(cfg=cfg, environment_name="env")
    assert url == f"{cfg.ecr_repository}:env__abcd1234"
    build.assert_not_called()


def test_ensure_image_built_builds_on_miss():
    cfg = _build_cfg()
    with (
        patch.object(engine.ImageBuilder, "get_ecr_image_tag", return_value="env__abcd1234"),
        patch.object(engine.ImageBuilder, "image_exists_in_ecr", return_value=False),
        patch.object(engine.ImageBuilder, "_build_and_push") as build,
    ):
        url = engine.ImageBuilder.ensure_image_built(cfg=cfg, environment_name="env")
    assert url == f"{cfg.ecr_repository}:env__abcd1234"
    build.assert_called_once()
    assert build.call_args.kwargs["tag"] == "env__abcd1234"


def test_ensure_image_built_force_skips_cache_check():
    cfg = _build_cfg()
    with (
        patch.object(engine.ImageBuilder, "get_ecr_image_tag", return_value="env__abcd1234"),
        patch.object(engine.ImageBuilder, "image_exists_in_ecr") as exists,
        patch.object(engine.ImageBuilder, "_build_and_push") as build,
    ):
        engine.ImageBuilder.ensure_image_built(cfg=cfg, environment_name="env", force_build=True)
    exists.assert_not_called()
    build.assert_called_once()


def test_ensure_image_built_dedupes_on_inflight():
    # A peer build for this tag is in flight; once it finishes the image is in ECR, so the waiter
    # returns the URL without building. The waiter verifies via ECR (not the event) on wake.
    cfg = _build_cfg()
    tag = "env__abcd1234"
    done = threading.Event()
    done.set()
    engine.ImageBuilder._inflight_builds[tag] = done
    try:
        with (
            patch.object(engine.ImageBuilder, "get_ecr_image_tag", return_value=tag),
            patch.object(engine.ImageBuilder, "image_exists_in_ecr", return_value=True),
            patch.object(engine.ImageBuilder, "_build_and_push") as build,
        ):
            url = engine.ImageBuilder.ensure_image_built(cfg=cfg, environment_name="env")
        assert url == f"{cfg.ecr_repository}:{tag}"
        build.assert_not_called()
    finally:
        engine.ImageBuilder._inflight_builds.pop(tag, None)


def test_ensure_image_built_waiter_raises_when_peer_build_failed():
    # A peer build for this tag is in flight but fails (image never lands in ECR). The waiter must
    # raise rather than return a URL for an image that was never pushed (it would fail later at the
    # ECS image pull). It verifies via ECR on wake instead of trusting the in-flight event.
    cfg = _build_cfg()
    tag = "env__deadbeef"
    done = threading.Event()
    done.set()
    engine.ImageBuilder._inflight_builds[tag] = done
    try:
        with (
            patch.object(engine.ImageBuilder, "get_ecr_image_tag", return_value=tag),
            patch.object(engine.ImageBuilder, "image_exists_in_ecr", return_value=False),
            patch.object(engine.ImageBuilder, "_build_and_push") as build,
            pytest.raises(RuntimeError, match="not in ECR"),
        ):
            engine.ImageBuilder.ensure_image_built(cfg=cfg, environment_name="env")
        build.assert_not_called()
    finally:
        engine.ImageBuilder._inflight_builds.pop(tag, None)


def test_ensure_image_built_cache_filled_after_semaphore():
    # The image appears in ECR (built by a peer) by the time we acquire the
    # build semaphore -> skip the build.
    cfg = _build_cfg()
    with (
        patch.object(engine.ImageBuilder, "get_ecr_image_tag", return_value="env__cafe"),
        patch.object(engine.ImageBuilder, "image_exists_in_ecr", side_effect=[False, True]),
        patch.object(engine.ImageBuilder, "_build_and_push") as build,
    ):
        url = engine.ImageBuilder.ensure_image_built(cfg=cfg, environment_name="env")
    assert url == f"{cfg.ecr_repository}:env__cafe"
    build.assert_not_called()


def test_ensure_mirrored_requires_repo():
    with pytest.raises(ValueError, match="ecr_repository is required"):
        engine.ImageBuilder.ensure_mirrored(cfg=engine.EcsFargateConfig(region="us-west-2"), src_image="ubuntu:24.04")


def test_ensure_mirrored_dedupes_on_inflight():
    # Peer mirror in flight + image now in ECR -> waiter returns the URL without re-mirroring.
    cfg = _build_cfg()
    tag = engine._sanitize_id("ubuntu:24.04")
    done = threading.Event()
    done.set()
    engine.ImageBuilder._inflight_builds[tag] = done
    try:
        with (
            patch.object(engine.ImageBuilder, "image_exists_in_ecr", return_value=True),
            patch.object(engine.ImageBuilder, "run_buildspec_via_codebuild") as cb,
        ):
            url = engine.ImageBuilder.ensure_mirrored(cfg=cfg, src_image="ubuntu:24.04")
        assert url.endswith(tag)
        cb.assert_not_called()
    finally:
        engine.ImageBuilder._inflight_builds.pop(tag, None)


def test_ensure_mirrored_waiter_raises_when_peer_build_failed():
    # Peer mirror in flight but failed (image absent from ECR) -> waiter raises instead of returning
    # a URL for an image that was never pushed.
    cfg = _build_cfg()
    tag = engine._sanitize_id("ubuntu:24.04")
    done = threading.Event()
    done.set()
    engine.ImageBuilder._inflight_builds[tag] = done
    try:
        with (
            patch.object(engine.ImageBuilder, "image_exists_in_ecr", return_value=False),
            patch.object(engine.ImageBuilder, "run_buildspec_via_codebuild") as cb,
            pytest.raises(RuntimeError, match="not in ECR"),
        ):
            engine.ImageBuilder.ensure_mirrored(cfg=cfg, src_image="ubuntu:24.04")
        cb.assert_not_called()
    finally:
        engine.ImageBuilder._inflight_builds.pop(tag, None)


def test_ensure_mirrored_cache_filled_after_semaphore():
    cfg = _build_cfg()
    with (
        patch.object(engine.ImageBuilder, "image_exists_in_ecr", side_effect=[False, True]),
        patch.object(engine.ImageBuilder, "run_buildspec_via_codebuild") as cb,
    ):
        engine.ImageBuilder.ensure_mirrored(cfg=cfg, src_image="ubuntu:24.04")
    cb.assert_not_called()


def test_generate_buildspec_with_dockerhub_login():
    cfg = engine.EcsFargateConfig(
        region="us-west-2",
        ecr_repository=_ecr_repo(),
        dockerhub_secret_arn="arn:aws:secretsmanager:us-west-2:123:secret:dh",  # pragma: allowlist secret
    )
    spec = engine.ImageBuilder._generate_buildspec(cfg, "sandbox", "t1", f"{cfg.ecr_repository}:t1")
    assert "docker build -t sandbox:t1" in spec
    assert "secretsmanager get-secret-value" in spec  # Docker Hub auth branch
    assert f"docker push {cfg.ecr_repository}:t1" in spec


def test_upload_build_context_zips_and_uploads(tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f.txt").write_text("hi")
    cfg = _build_cfg(environment_dir=str(tmp_path), s3_prefix="pfx")
    s3 = MagicMock()
    with _patch_aws(s3=s3):
        key = engine.ImageBuilder._upload_build_context(cfg, "env", "nonce1")
    assert key == "pfx/codebuild/env-nonce1.zip"
    put = s3.put_object.call_args.kwargs
    assert put["Bucket"] == "bkt" and put["Key"] == key
    # The uploaded payload is a real zip containing the env files.
    names = _zip_names(put["Body"])
    assert "Dockerfile" in names and "sub/f.txt" in names


def test_resolve_codebuild_project_uses_explicit():
    cfg = _build_cfg(codebuild_project="my-project")
    with _patch_aws():
        # No client calls expected when project is explicit.
        assert engine.ImageBuilder._resolve_codebuild_project(cfg, MagicMock()) == "my-project"


def test_resolve_codebuild_project_requires_role():
    cfg = _build_cfg(codebuild_service_role=None, codebuild_project=None)
    with _patch_aws():
        with pytest.raises(RuntimeError, match="codebuild_project or codebuild_service_role"):
            engine.ImageBuilder._resolve_codebuild_project(cfg, MagicMock())


def test_resolve_codebuild_project_creates_project():
    cfg = _build_cfg()
    cb = MagicMock()
    with _patch_aws():
        name = engine.ImageBuilder._resolve_codebuild_project(cfg, cb)
    assert name == "ecs-sandbox-build"
    cb.create_project.assert_called_once()
    assert cb.create_project.call_args.kwargs["serviceRole"] == "arn:aws:iam::123:role/cb"


def test_resolve_codebuild_project_tolerates_already_exists():
    cfg = _build_cfg()
    cb = MagicMock()
    cb.create_project.side_effect = _client_error(
        "ResourceAlreadyExistsException", "CreateProject", msg="already exists"
    )
    with _patch_aws():
        name = engine.ImageBuilder._resolve_codebuild_project(cfg, cb)
    assert name == "ecs-sandbox-build"


def test_resolve_codebuild_project_reraises_other_error():
    cfg = _build_cfg()
    cb = MagicMock()
    cb.create_project.side_effect = _client_error("AccessDenied", "CreateProject", msg="forbidden")
    with _patch_aws():
        with pytest.raises(ClientError):
            engine.ImageBuilder._resolve_codebuild_project(cfg, cb)


def test_poll_codebuild_success():
    cb = MagicMock()
    cb.batch_get_builds.return_value = {"builds": [{"buildStatus": "SUCCEEDED"}]}
    with patch(f"{_ENG}.time.sleep"), patch(f"{_ENG}.random.uniform", return_value=0):
        engine.ImageBuilder._poll_codebuild(cb, "bid", "img:tag")  # returns without error


def test_poll_codebuild_in_progress_then_success():
    cb = MagicMock()
    cb.batch_get_builds.side_effect = [
        {"builds": [{"buildStatus": "IN_PROGRESS", "currentPhase": "BUILD"}]},
        {"builds": [{"buildStatus": "SUCCEEDED"}]},
    ]
    with patch(f"{_ENG}.time.sleep"), patch(f"{_ENG}.random.uniform", return_value=0):
        engine.ImageBuilder._poll_codebuild(cb, "bid", "img:tag")
    assert cb.batch_get_builds.call_count == 2


def test_poll_codebuild_failure_reports_phases():
    cb = MagicMock()
    cb.batch_get_builds.return_value = {
        "builds": [
            {
                "buildStatus": "FAILED",
                "phases": [
                    {"phaseType": "BUILD", "phaseStatus": "FAILED"},
                    {"phaseType": "DOWNLOAD_SOURCE", "phaseStatus": "SUCCEEDED"},
                ],
            }
        ]
    }
    with patch(f"{_ENG}.time.sleep"), patch(f"{_ENG}.random.uniform", return_value=0):
        with pytest.raises(RuntimeError, match="CodeBuild failed for img:tag: BUILD: FAILED"):
            engine.ImageBuilder._poll_codebuild(cb, "bid", "img:tag")


def test_build_and_push_orchestration():
    cfg = _build_cfg()
    cb = MagicMock()
    cb.start_build.return_value = {"build": {"id": "build-99"}}
    with (
        _patch_aws(codebuild=cb),
        patch.object(engine.ImageBuilder, "_upload_build_context", return_value="pfx/ctx.zip"),
        patch.object(engine.ImageBuilder, "_resolve_codebuild_project", return_value="proj"),
        patch.object(engine.ImageBuilder, "_generate_buildspec", return_value="version: 0.2"),
        patch.object(engine.ImageBuilder, "_poll_codebuild") as poll,
    ):
        engine.ImageBuilder._build_and_push(
            cfg=cfg, environment_name="env", tag="t1", image_url=f"{cfg.ecr_repository}:t1"
        )
    sb = cb.start_build.call_args.kwargs
    assert sb["projectName"] == "proj"
    assert sb["sourceTypeOverride"] == "S3"
    assert sb["sourceLocationOverride"] == "bkt/pfx/ctx.zip"
    poll.assert_called_once_with(cb, "build-99", f"{cfg.ecr_repository}:t1")


def test_run_buildspec_via_codebuild_no_source():
    cfg = _build_cfg()
    cb = MagicMock()
    cb.start_build.return_value = {"build": {"id": "build-7"}}
    with (
        _patch_aws(codebuild=cb),
        patch.object(engine.ImageBuilder, "_resolve_codebuild_project", return_value="proj"),
        patch.object(engine.ImageBuilder, "_poll_codebuild") as poll,
    ):
        engine.ImageBuilder.run_buildspec_via_codebuild(
            cfg=cfg, buildspec="version: 0.2", job_label="mirror::x", timeout_minutes=12
        )
    sb = cb.start_build.call_args.kwargs
    assert sb["sourceTypeOverride"] == "NO_SOURCE"
    assert sb["timeoutInMinutesOverride"] == 12
    poll.assert_called_once_with(cb, "build-7", "mirror::x")


# ── EcsFargateSandbox: client init + env/command builders ─────────────


def test_init_aws_clients_creates_three_clients():
    sb = _make_sandbox()
    with _patch_aws(ecs=MagicMock(), ec2=MagicMock(), ssm=MagicMock()) as boto3:
        sb._init_aws_clients()
    assert sb._ecs is not None and sb._ec2 is not None and sb._ssm is not None
    created = {c.args[0] for c in boto3.client.call_args_list}
    assert created == {"ecs", "ec2", "ssm"}
    for c in boto3.client.call_args_list:
        assert c.kwargs["region_name"] == "us-west-2"


def test_build_container_command_none_without_exec_server():
    sb = _make_sandbox()
    assert sb._build_container_command(_agent_sidecar()) is None


def test_build_container_command_bootstraps_exec_server():
    sb = _make_sandbox(spec=engine.SandboxSpec(image="my/img:1"))
    cmd = sb._build_container_command(_exec_sidecar(exec_server_port=5000))
    assert cmd[:2] == ["sh", "-lc"]
    setup = cmd[2]
    assert "base64 -d > /tmp/_exec_server.py" in setup
    assert "TB_EXEC_PORT=5000" in setup
    assert "exec python3 /tmp/_exec_server.py" in setup


def test_build_env_vars_merges_spec_extra_and_routing():
    spec = engine.SandboxSpec(image="img:1", env={"FROM_SPEC": "1"})
    sb = _make_sandbox(spec=spec, extra_env={"RENDERED": "ip={task_ip} img={image}"})
    sb._task_ip = "9.9.9.9"
    sb._outside_endpoint_routing = engine._OutsideEndpointRouting.for_exec_server(
        [engine.OutsideEndpoint("http://10.0.0.1:4000/v1", "MODEL_BASE_URL")], _exec_sidecar()
    )
    env = sb._build_env_vars()
    assert env["FROM_SPEC"] == "1"
    assert env["RENDERED"] == "ip=9.9.9.9 img=img:1"
    assert env["MODEL_BASE_URL"].startswith("http://127.0.0.1:")


def test_split_env_separates_runtime_keys():
    sb = _make_sandbox()
    sb._outside_endpoint_routing = engine._OutsideEndpointRouting.for_exec_server(
        [engine.OutsideEndpoint("http://10.0.0.1:4000/v1", "MODEL_BASE_URL")], _exec_sidecar()
    )
    env = {"STABLE": "a", "_NEL_EFS_SESSION": "sess", "MODEL_BASE_URL": "http://127.0.0.1:4000/v1"}
    stable, runtime = sb._split_env(env)
    assert stable == {"STABLE": "a"}
    assert runtime == {"_NEL_EFS_SESSION": "sess", "MODEL_BASE_URL": "http://127.0.0.1:4000/v1"}


def test_render_env_value_substitutes_placeholders():
    sb = _make_sandbox(spec=engine.SandboxSpec(image="img:2"))
    sb._ssh_tunnel_port = 7000
    sb._task_ip = "1.2.3.4"
    assert sb._render_env_value("p={ssh_tunnel_port} ip={task_ip} i={image}") == "p=7000 ip=1.2.3.4 i=img:2"


def test_make_family_name_sanitizes_and_prefixes():
    sb = _make_sandbox(spec=engine.SandboxSpec(image="weird/Name:v1"), task_definition_family_prefix="ecs-sandbox")
    fam = sb._make_family_name()
    assert fam.startswith("ecs-sandbox-")
    assert all(ch.isalnum() or ch in "_-" for ch in fam)


def test_make_family_name_prepends_ecs_when_non_alnum_start():
    sb = _make_sandbox(spec=engine.SandboxSpec(image=""), task_definition_family_prefix="")
    fam = sb._make_family_name()
    assert fam.startswith("ecs_")


# ── EcsFargateSandbox: _resolve_image branches ────────────────────────


def test_resolve_image_built_passthrough():
    sb = _make_sandbox()
    assert sb._resolve_image("built:img") == "built:img"


def test_resolve_image_template_key_error():
    sb = _make_sandbox(spec=engine.SandboxSpec(image="x"), image_template="{nonexistent_key}-img")
    with pytest.raises(ValueError, match="placeholder"):
        sb._resolve_image()


def test_resolve_image_bare_without_ecr_is_verbatim():
    sb = _make_sandbox(spec=engine.SandboxSpec(image="ubuntu:24.04"))
    assert sb._resolve_image() == "ubuntu:24.04"


def test_resolve_image_empty_with_task_definition():
    sb = _make_sandbox(spec=engine.SandboxSpec(image=""), task_definition="arn:td")
    assert sb._resolve_image() == ""


def test_resolve_image_no_source_raises():
    sb = _make_sandbox(spec=engine.SandboxSpec(image=""))
    with pytest.raises(ValueError, match="No image available"):
        sb._resolve_image()


# ── EcsFargateSandbox: exec-server upload ─────────────────────────────


def test_upload_exec_server_requires_bucket():
    sb = _make_sandbox()
    with pytest.raises(ValueError, match="s3_bucket is required"):
        sb._upload_exec_server()


def test_upload_exec_server_uploads_and_caches():
    sb = _make_sandbox(s3_bucket="bkt", s3_prefix="pfx")
    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://signed.example/exec"
    with _patch_aws(s3=s3):
        url1 = sb._upload_exec_server()
        url2 = sb._upload_exec_server()
    assert url1 == "https://signed.example/exec"
    assert url2 == url1
    s3.put_object.assert_called_once()  # cached on the second call
    assert s3.put_object.call_args.kwargs["Bucket"] == "bkt"


# ── EcsFargateSandbox: EFS volumes ────────────────────────────────────


def test_build_efs_volumes_access_point_and_root_dir():
    spec = engine.SandboxSpec(
        image="img:1",
        volumes=[
            engine.VolumeMount(
                container_path="/mnt/a", readonly=True, efs_filesystem_id="fs-1", efs_access_point_id="ap-1"
            ),
            engine.VolumeMount(container_path="/mnt/b", efs_filesystem_id="fs-2", efs_root_directory="/data"),
            engine.VolumeMount(host_path="/h", container_path="/mnt/c"),  # non-EFS -> skipped
        ],
    )
    sb = _make_sandbox(spec=spec)
    volumes, mounts = sb._build_efs_volumes()
    assert len(volumes) == 2 and len(mounts) == 2
    assert volumes[0]["efsVolumeConfiguration"]["authorizationConfig"]["accessPointId"] == "ap-1"
    assert volumes[1]["efsVolumeConfiguration"]["rootDirectory"] == "/data"
    assert mounts[0] == {"sourceVolume": "efs-0", "containerPath": "/mnt/a", "readOnly": True}


# ── EcsFargateSandbox: task definition registration ───────────────────


def test_register_task_definition_from_scratch_when_no_base():
    sb = _make_sandbox(execution_role_arn="arn:exec")
    sb._ecs = MagicMock()
    with patch.object(engine.EcsFargateSandbox, "_register_from_scratch", return_value="scratch-arn") as scratch:
        arn = sb._register_task_definition(image="img", command=["sh"], env={}, sidecar_def={"name": "ssh-tunnel"})
    assert arn == "scratch-arn"
    scratch.assert_called_once()


def test_register_task_definition_from_base_when_describable():
    sb = _make_sandbox(task_definition="arn:td", log_group="/g")
    sb._ecs = MagicMock()
    base = {"taskDefinition": {"containerDefinitions": [{"name": "main"}]}}
    sb._ecs.describe_task_definition.return_value = base
    with patch.object(engine.EcsFargateSandbox, "_register_from_base", return_value="base-arn") as from_base:
        arn = sb._register_task_definition(
            image="img", command=None, env={"K": "V"}, sidecar_def={"name": "ssh-tunnel"}
        )
    assert arn == "base-arn"
    assert from_base.call_args.kwargs["base"] == base["taskDefinition"]
    # log_group set -> a log configuration is threaded through.
    assert from_base.call_args.kwargs["log_cfg"]["options"]["awslogs-group"] == "/g"


def test_register_task_definition_missing_base_falls_back_to_scratch():
    sb = _make_sandbox(task_definition="arn:missing")
    sb._ecs = MagicMock()
    sb._ecs.describe_task_definition.side_effect = _client_error("ClientException", "DescribeTaskDefinition")
    with patch.object(engine.EcsFargateSandbox, "_register_from_scratch", return_value="scratch-arn") as scratch:
        arn = sb._register_task_definition(image="img", command=None, env={}, sidecar_def={"name": "ssh-tunnel"})
    assert arn == "scratch-arn"
    scratch.assert_called_once()


def test_register_task_definition_describe_other_error_raises():
    sb = _make_sandbox(task_definition="arn:td")
    sb._ecs = MagicMock()
    sb._ecs.describe_task_definition.side_effect = _client_error("AccessDenied", "DescribeTaskDefinition")
    with pytest.raises(ClientError):
        sb._register_task_definition(image="img", command=None, env={}, sidecar_def={"name": "ssh-tunnel"})


def test_register_from_base_builds_payload():
    sb = _make_sandbox(
        spec=engine.SandboxSpec(image="img:1"),
        execution_role_arn="arn:exec",
        task_role_arn="arn:taskrole",
        cpu="8192",
        memory="16384",
    )
    base = {
        "containerDefinitions": [
            {"name": "main", "environment": [{"name": "OLD", "value": "1"}]},
            {"name": "ssh-tunnel"},  # stale sidecar — must be dropped
        ],
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": "256",
        "memory": "512",
        "executionRoleArn": "arn:base-exec",
    }
    with patch.object(engine.EcsFargateSandbox, "_do_register", return_value="arn") as do_reg:
        arn = sb._register_from_base(
            base=base,
            image="newimg:2",
            command=["sh", "-lc", "x"],
            env={"NEW": "2"},
            sidecar_def={"name": "ssh-tunnel", "image": "alpine"},
            log_cfg={"logDriver": "awslogs"},
        )
    assert arn == "arn"
    payload = do_reg.call_args.args[0]
    names = [c["name"] for c in payload["containerDefinitions"]]
    assert names.count("ssh-tunnel") == 1  # old sidecar replaced
    main = next(c for c in payload["containerDefinitions"] if c["name"] == "main")
    assert main["image"] == "newimg:2"
    assert main["command"] == ["sh", "-lc", "x"]
    assert main["dependsOn"] == [{"containerName": "ssh-tunnel", "condition": "HEALTHY"}]
    assert {"name": "NEW", "value": "2"} in main["environment"]
    assert {"name": "OLD", "value": "1"} in main["environment"]
    # cpu/memory are the max of base and config.
    assert payload["cpu"] == "8192" and payload["memory"] == "16384"
    assert payload["executionRoleArn"] == "arn:exec"
    assert payload["taskRoleArn"] == "arn:taskrole"


def test_register_from_base_missing_container_raises():
    sb = _make_sandbox(container_name="main")
    base = {"containerDefinitions": [{"name": "other"}]}
    with pytest.raises(RuntimeError, match="no container 'main'"):
        sb._register_from_base(
            base=base, image="i", command=None, env={}, sidecar_def={"name": "ssh-tunnel"}, log_cfg=None
        )


def test_register_from_base_appends_efs_volumes_to_existing():
    spec = engine.SandboxSpec(
        image="img:1", volumes=[engine.VolumeMount(container_path="/mnt/efs", efs_filesystem_id="fs-9")]
    )
    sb = _make_sandbox(spec=spec)
    base = {
        "containerDefinitions": [{"name": "main", "mountPoints": [{"sourceVolume": "pre", "containerPath": "/pre"}]}],
        "volumes": [{"name": "pre"}],
    }
    with patch.object(engine.EcsFargateSandbox, "_do_register", return_value="arn") as do_reg:
        sb._register_from_base(
            base=base, image="i", command=None, env={}, sidecar_def={"name": "ssh-tunnel"}, log_cfg=None
        )
    payload = do_reg.call_args.args[0]
    main = next(c for c in payload["containerDefinitions"] if c["name"] == "main")
    # Existing mounts/volumes are preserved and the EFS ones are appended.
    assert {"sourceVolume": "pre", "containerPath": "/pre"} in main["mountPoints"]
    assert any(m["containerPath"] == "/mnt/efs" for m in main["mountPoints"])
    vol_names = {v["name"] for v in payload["volumes"]}
    assert vol_names == {"pre", "efs-0"}


def test_register_from_scratch_requires_execution_role():
    sb = _make_sandbox(execution_role_arn=None)
    with pytest.raises(RuntimeError, match="execution_role_arn required"):
        sb._register_from_scratch(image="i", command=None, env={}, sidecar_def={"name": "ssh-tunnel"}, log_cfg=None)


def test_register_from_scratch_builds_payload():
    spec = engine.SandboxSpec(
        image="img:1",
        volumes=[engine.VolumeMount(container_path="/mnt", efs_filesystem_id="fs-1")],
    )
    sb = _make_sandbox(
        spec=spec,
        execution_role_arn="arn:exec",
        task_role_arn="arn:taskrole",
        container_port=8080,
        ephemeral_storage_gib=40,
    )
    with patch.object(engine.EcsFargateSandbox, "_do_register", return_value="arn") as do_reg:
        sb._register_from_scratch(
            image="img:1",
            command=["sh"],
            env={"K": "V"},
            sidecar_def={"name": "ssh-tunnel"},
            log_cfg={"logDriver": "awslogs"},
        )
    payload = do_reg.call_args.args[0]
    main = payload["containerDefinitions"][0]
    assert main["name"] == "main"
    assert main["portMappings"] == [{"containerPort": 8080, "protocol": "tcp"}]
    assert main["environment"] == [{"name": "K", "value": "V"}]
    assert main["logConfiguration"] == {"logDriver": "awslogs"}
    assert main["mountPoints"][0]["containerPath"] == "/mnt"
    assert payload["executionRoleArn"] == "arn:exec"
    assert payload["taskRoleArn"] == "arn:taskrole"
    assert payload["ephemeralStorage"] == {"sizeInGiB": 40}
    assert payload["volumes"][0]["name"] == "efs-0"


# ── EcsFargateSandbox: SSM task-def cache + _do_register ───────────────


def test_ssm_lookup_task_def_active():
    sb = _make_sandbox()
    sb._ssm = MagicMock()
    sb._ssm.get_parameter.return_value = {"Parameter": {"Value": "arn:cached-td"}}
    sb._ecs = MagicMock()
    sb._ecs.describe_task_definition.return_value = {"taskDefinition": {"status": "ACTIVE"}}
    assert sb._ssm_lookup_task_def("hash1") == "arn:cached-td"


def test_ssm_lookup_task_def_not_found():
    sb = _make_sandbox()
    sb._ssm = MagicMock()
    sb._ssm.get_parameter.side_effect = _client_error("ParameterNotFound", "GetParameter")
    assert sb._ssm_lookup_task_def("hash1") is None


def test_ssm_lookup_task_def_other_param_error_returns_none():
    sb = _make_sandbox()
    sb._ssm = MagicMock()
    sb._ssm.get_parameter.side_effect = _client_error("ThrottlingException", "GetParameter")
    assert sb._ssm_lookup_task_def("hash1") is None


def test_ssm_lookup_task_def_describe_fails_returns_none():
    sb = _make_sandbox()
    sb._ssm = MagicMock()
    sb._ssm.get_parameter.return_value = {"Parameter": {"Value": "arn:gone"}}
    sb._ecs = MagicMock()
    sb._ecs.describe_task_definition.side_effect = _client_error("ClientException", "DescribeTaskDefinition")
    assert sb._ssm_lookup_task_def("hash1") is None


def test_ssm_lookup_task_def_inactive_returns_none():
    sb = _make_sandbox()
    sb._ssm = MagicMock()
    sb._ssm.get_parameter.return_value = {"Parameter": {"Value": "arn:old"}}
    sb._ecs = MagicMock()
    sb._ecs.describe_task_definition.return_value = {"taskDefinition": {"status": "INACTIVE"}}
    assert sb._ssm_lookup_task_def("hash1") is None


def test_ssm_write_task_def_ok_and_error_swallowed():
    sb = _make_sandbox()
    sb._ssm = MagicMock()
    sb._ssm_write_task_def("hash1", "arn:new")
    sb._ssm.put_parameter.assert_called_once()
    # A failing PutParameter must not raise (cache write is best-effort).
    sb._ssm.put_parameter.side_effect = _client_error("AccessDenied", "PutParameter")
    sb._ssm_write_task_def("hash2", "arn:new2")


def test_do_register_cache_hit():
    sb = _make_sandbox()
    payload = {"family": "f", "containerDefinitions": [{"name": "main"}]}
    h = engine._compute_task_def_hash(payload)
    engine._task_def_cache[h] = "arn:cached"
    assert sb._do_register(payload) == "arn:cached"


def test_do_register_fresh_path_writes_cache():
    sb = _make_sandbox()
    payload = {"family": "f", "containerDefinitions": [{"name": "main"}]}
    h = engine._compute_task_def_hash(payload)
    with (
        patch.object(engine.EcsFargateSandbox, "_ssm_lookup_task_def", return_value=None),
        patch.object(engine.EcsFargateSandbox, "_register_task_def_fresh", return_value="arn:fresh") as fresh,
    ):
        arn = sb._do_register(payload)
    assert arn == "arn:fresh"
    assert engine._task_def_cache[h] == "arn:fresh"
    fresh.assert_called_once()
    assert h not in engine._task_def_inflight  # inflight released


def test_do_register_uses_ssm_cache_entry():
    sb = _make_sandbox()
    payload = {"family": "f", "containerDefinitions": [{"name": "main"}]}
    with (
        patch.object(engine.EcsFargateSandbox, "_ssm_lookup_task_def", return_value="arn:ssm"),
        patch.object(engine.EcsFargateSandbox, "_register_task_def_fresh") as fresh,
    ):
        assert sb._do_register(payload) == "arn:ssm"
    fresh.assert_not_called()


def test_do_register_waits_for_inflight_then_reads_cache():
    sb = _make_sandbox()
    payload = {"family": "f", "containerDefinitions": [{"name": "main"}]}
    h = engine._compute_task_def_hash(payload)

    class _WaitThenCache(threading.Event):
        def wait(self, timeout=None):
            # Simulate the in-flight builder finishing and publishing the arn.
            engine._task_def_cache[h] = "arn:by-other"
            return True

    engine._task_def_inflight[h] = _WaitThenCache()
    with patch.object(engine.EcsFargateSandbox, "_register_task_def_fresh") as fresh:
        assert sb._do_register(payload) == "arn:by-other"
    fresh.assert_not_called()


def test_register_task_def_fresh_calls_ecs_and_writes_ssm():
    sb = _make_sandbox()
    sb._ecs = MagicMock()
    sb._ecs.register_task_definition.return_value = {"taskDefinition": {"taskDefinitionArn": "arn:reg"}}
    with patch.object(engine.EcsFargateSandbox, "_ssm_write_task_def") as write:
        arn = sb._register_task_def_fresh({"family": "f"}, "hash1")
    assert arn == "arn:reg"
    write.assert_called_once_with("hash1", "arn:reg")


# ── EcsFargateSandbox: _run_task ──────────────────────────────────────


def test_run_task_success_network_config():
    sb = _make_sandbox(assign_public_ip=True, container_name="main")
    sb._task_arn = None
    sb._ecs = MagicMock()
    sb._ecs.run_task.return_value = {"tasks": [{"taskArn": "arn:task-1"}]}
    arn = sb._run_task("arn:td")
    assert arn == "arn:task-1"
    kwargs = sb._ecs.run_task.call_args.kwargs
    vpc = kwargs["networkConfiguration"]["awsvpcConfiguration"]
    assert vpc["assignPublicIp"] == "ENABLED"
    assert vpc["subnets"] == ["subnet-a"] and vpc["securityGroups"] == ["sg-a"]
    assert "overrides" not in kwargs  # no per-invocation env


def test_run_task_includes_runtime_overrides_and_disabled_public_ip():
    sb = _make_sandbox(assign_public_ip=False, container_name="main", platform_version="1.5.0")
    sb._runtime_container_env = {"_NEL_EFS_SESSION": "sess-1"}
    sb._ecs = MagicMock()
    sb._ecs.run_task.return_value = {"tasks": [{"taskArn": "arn:task-2"}]}
    sb._run_task("arn:td")
    kwargs = sb._ecs.run_task.call_args.kwargs
    assert kwargs["networkConfiguration"]["awsvpcConfiguration"]["assignPublicIp"] == "DISABLED"
    assert kwargs["platformVersion"] == "1.5.0"
    overrides = kwargs["overrides"]["containerOverrides"][0]
    assert overrides["name"] == "main"
    assert overrides["environment"] == [{"name": "_NEL_EFS_SESSION", "value": "sess-1"}]


def test_run_task_efs_forces_platform_version():
    spec = engine.SandboxSpec(image="img", volumes=[engine.VolumeMount(container_path="/m", efs_filesystem_id="fs-1")])
    sb = _make_sandbox(spec=spec)
    sb._ecs = MagicMock()
    sb._ecs.run_task.return_value = {"tasks": [{"taskArn": "arn:task-3"}]}
    sb._run_task("arn:td")
    assert sb._ecs.run_task.call_args.kwargs["platformVersion"] == "1.4.0"


def test_run_task_no_tasks_raises():
    sb = _make_sandbox()
    sb._ecs = MagicMock()
    sb._ecs.run_task.return_value = {"tasks": []}
    with pytest.raises(RuntimeError, match="no tasks"):
        sb._run_task("arn:td")


def test_run_task_non_retryable_exception_reraised():
    sb = _make_sandbox()
    sb._ecs = MagicMock()
    sb._ecs.run_task.side_effect = ValueError("permanent")
    with patch(f"{_ENG}.time.sleep"):
        with pytest.raises(ValueError, match="permanent"):
            sb._run_task("arn:td")


def test_run_task_retryable_exception_then_success():
    sb = _make_sandbox(run_task_max_retries=5)
    sb._ecs = MagicMock()
    sb._ecs.run_task.side_effect = [
        _client_error("ThrottlingException", "RunTask"),
        {"tasks": [{"taskArn": "arn:task-ok"}]},
    ]
    # Isolate the outer retry loop from the inner per-call backoff helper.
    with (
        patch(f"{_ENG}._retry_with_backoff", side_effect=lambda func, **_kw: func()),
        patch(f"{_ENG}.time.sleep"),
    ):
        assert sb._run_task("arn:td") == "arn:task-ok"


def test_run_task_non_retryable_failures_raise():
    sb = _make_sandbox(run_task_max_retries=3)
    sb._ecs = MagicMock()
    sb._ecs.run_task.return_value = {"failures": [{"reason": "AccessDenied on subnet"}]}
    with patch(f"{_ENG}.time.sleep"):
        with pytest.raises(RuntimeError, match="run_task failures"):
            sb._run_task("arn:td")


def test_run_task_retryable_failures_exhaust():
    sb = _make_sandbox(run_task_max_retries=2)
    sb._ecs = MagicMock()
    sb._ecs.run_task.return_value = {"failures": [{"reason": "Capacity is unavailable right now"}]}
    with patch(f"{_ENG}.time.sleep"):
        with pytest.raises(RuntimeError, match="run_task failures"):
            sb._run_task("arn:td")
    assert sb._ecs.run_task.call_count == 2


def test_run_task_zero_retries_raises_final():
    sb = _make_sandbox(run_task_max_retries=0)
    sb._ecs = MagicMock()
    with pytest.raises(RuntimeError, match="failed after 0 retries"):
        sb._run_task("arn:td")
    sb._ecs.run_task.assert_not_called()


# ── EcsFargateSandbox: _wait_for_running ──────────────────────────────


def test_wait_for_running_returns_when_running():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.return_value = {"tasks": [{"lastStatus": "RUNNING"}]}
    with patch(f"{_ENG}.time.monotonic", return_value=0.0):
        sb._wait_for_running()


def test_wait_for_running_transitions_then_running():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.side_effect = [
        {"tasks": [{"lastStatus": "PROVISIONING"}]},
        {"tasks": [{"lastStatus": "RUNNING"}]},
    ]
    with (
        patch(f"{_ENG}.time.monotonic", return_value=0.0),
        patch(f"{_ENG}.time.sleep"),
        patch(f"{_ENG}.random.random", return_value=0.0),
    ):
        sb._wait_for_running()
    assert sb._ecs.describe_tasks.call_count == 2


def test_wait_for_running_stopped_raises():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.return_value = {"tasks": [{"lastStatus": "STOPPED", "stoppedReason": "OOM"}]}
    with patch(f"{_ENG}.time.monotonic", return_value=0.0):
        with pytest.raises(RuntimeError, match="ECS task stopped: OOM"):
            sb._wait_for_running()


def test_wait_for_running_no_tasks_raises():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.return_value = {"tasks": []}
    with patch(f"{_ENG}.time.monotonic", return_value=0.0):
        with pytest.raises(RuntimeError, match="disappeared"):
            sb._wait_for_running()


def test_wait_for_running_timeout():
    sb = _make_sandbox(startup_timeout_sec=300.0)
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    with patch(f"{_ENG}.time.monotonic", side_effect=[0.0, 999.0]):
        with pytest.raises(TimeoutError, match="not RUNNING"):
            sb._wait_for_running()


def test_wait_for_running_retryable_describe_then_running():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.side_effect = [
        _client_error("ThrottlingException", "DescribeTasks"),
        {"tasks": [{"lastStatus": "RUNNING"}]},
    ]
    with (
        patch(f"{_ENG}.time.monotonic", return_value=0.0),
        patch(f"{_ENG}.time.sleep"),
        patch(f"{_ENG}.random.random", return_value=0.0),
    ):
        sb._wait_for_running()


def test_wait_for_running_non_retryable_describe_raises():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.side_effect = ValueError("fatal")
    with patch(f"{_ENG}.time.monotonic", return_value=0.0):
        with pytest.raises(ValueError, match="fatal"):
            sb._wait_for_running()


# ── EcsFargateSandbox: _get_task_public_ip ────────────────────────────


def _eni_task(detail_name: str, value: str):
    return {
        "tasks": [
            {"attachments": [{"type": "ElasticNetworkInterface", "details": [{"name": detail_name, "value": value}]}]}
        ]
    }


def test_get_task_public_ip_returns_public():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.return_value = _eni_task("networkInterfaceId", "eni-1")
    sb._ec2 = MagicMock()
    sb._ec2.describe_network_interfaces.return_value = {
        "NetworkInterfaces": [{"Association": {"PublicIp": "1.2.3.4"}}]
    }
    assert sb._get_task_public_ip() == "1.2.3.4"


def test_get_task_public_ip_returns_private_when_no_public():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.return_value = _eni_task("networkInterfaceId", "eni-1")
    sb._ec2 = MagicMock()
    sb._ec2.describe_network_interfaces.return_value = {"NetworkInterfaces": [{"PrivateIpAddress": "10.0.0.5"}]}
    assert sb._get_task_public_ip() == "10.0.0.5"


def test_get_task_public_ip_uses_private_detail_without_eni():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.return_value = _eni_task("privateIPv4Address", "10.9.9.9")
    sb._ec2 = MagicMock()
    assert sb._get_task_public_ip() == "10.9.9.9"
    sb._ec2.describe_network_interfaces.assert_not_called()


def test_get_task_public_ip_no_ip_details_then_succeeds():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    # First poll: an attachment with neither ENI id nor a usable private IP.
    no_ip = {"tasks": [{"attachments": [{"type": "Other", "details": [{"name": "subnetId", "value": "sn"}]}]}]}
    sb._ecs.describe_tasks.side_effect = [no_ip, _eni_task("privateIPv4Address", "10.2.2.2")]
    sb._ec2 = MagicMock()
    with patch(f"{_ENG}.time.sleep"), patch(f"{_ENG}.random.random", return_value=0.0):
        assert sb._get_task_public_ip() == "10.2.2.2"


def test_get_task_public_ip_retries_on_missing_ip_then_succeeds():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.return_value = _eni_task("networkInterfaceId", "eni-1")
    sb._ec2 = MagicMock()
    sb._ec2.describe_network_interfaces.side_effect = [
        {"NetworkInterfaces": [{}]},  # no IP yet
        {"NetworkInterfaces": [{"Association": {"PublicIp": "5.6.7.8"}}]},
    ]
    with patch(f"{_ENG}.time.sleep"), patch(f"{_ENG}.random.random", return_value=0.0):
        assert sb._get_task_public_ip() == "5.6.7.8"


def test_get_task_public_ip_retryable_error_then_succeeds():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.side_effect = [
        _client_error("ThrottlingException", "DescribeTasks"),
        _eni_task("privateIPv4Address", "10.1.1.1"),
    ]
    sb._ec2 = MagicMock()
    with patch(f"{_ENG}.time.sleep"), patch(f"{_ENG}.random.random", return_value=0.0):
        assert sb._get_task_public_ip() == "10.1.1.1"


def test_get_task_public_ip_exhausts_retries():
    sb = _make_sandbox()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.describe_tasks.return_value = {"tasks": []}
    sb._ec2 = MagicMock()
    with patch(f"{_ENG}.time.sleep"), patch(f"{_ENG}.random.random", return_value=0.0):
        with pytest.raises(RuntimeError, match="Task not found"):
            sb._get_task_public_ip()


# ── EcsFargateSandbox: _wait_for_ssh_ready ────────────────────────────


def test_wait_for_ssh_ready_success():
    with (
        patch(f"{_ENG}.socket.socket", return_value=_connect_sock(recv=b"SSH-2.0-OpenSSH")),
        patch(f"{_ENG}.time.monotonic", side_effect=[0.0, 0.1]),
        patch(f"{_ENG}.time.sleep"),
    ):
        engine.EcsFargateSandbox._wait_for_ssh_ready("1.2.3.4", 2222, timeout=10.0)


def test_wait_for_ssh_ready_timeout():
    with (
        patch(f"{_ENG}.socket.socket", return_value=_connect_sock(raise_oserror=True)),
        patch(f"{_ENG}.time.monotonic", side_effect=[0.0, 0.1, 100.0]),
        patch(f"{_ENG}.time.sleep"),
    ):
        with pytest.raises(TimeoutError, match="SSH not ready"):
            engine.EcsFargateSandbox._wait_for_ssh_ready("1.2.3.4", 2222, timeout=10.0)


# ── EcsFargateSandbox: _open_tunnel ───────────────────────────────────


def test_open_tunnel_exec_server_mode():
    sb = _make_sandbox()
    sb._task_ip = "1.2.3.4"
    sb._ssh_key_file = "/tmp/key"
    sb._outside_endpoint_routing = engine._OutsideEndpointRouting.for_exec_server(
        [engine.OutsideEndpoint("http://10.0.0.1:4000/v1", "MODEL")], _exec_sidecar()
    )
    tunnel = MagicMock()
    with patch(f"{_ENG}.SshTunnel", return_value=tunnel) as ctor:
        sb._open_tunnel(_exec_sidecar(exec_server_port=5000))
    kwargs = ctor.call_args.kwargs
    assert kwargs["host"] == "1.2.3.4" and kwargs["port"] == 2222
    assert kwargs["forward_port"] == 5000
    assert any(s.endswith(":10.0.0.1:4000") for s in kwargs["reverses"])
    tunnel.open.assert_called_once()


def test_open_tunnel_agent_server_mode():
    sb = _make_sandbox(container_port=8080)
    sb._task_ip = "1.2.3.4"
    sb._ssh_key_file = "/tmp/key"
    sb._ssh_tunnel_port = 7000
    sb._outside_endpoint_routing = engine._OutsideEndpointRouting.for_agent_server(
        [engine.OutsideEndpoint("http://10.0.0.2:9000/v1", "MODEL")]
    )
    tunnel = MagicMock()
    with patch(f"{_ENG}.SshTunnel", return_value=tunnel) as ctor, patch(f"{_ENG}._free_port", return_value=15000):
        sb._open_tunnel(_agent_sidecar())
    kwargs = ctor.call_args.kwargs
    assert kwargs["forwards"] == ["15000:localhost:8080"]
    assert kwargs["reverses"] == ["7000:10.0.0.2:9000"]
    assert kwargs["local_port_override"] == 15000
    tunnel.open.assert_called_once()


def test_open_tunnel_agent_server_requires_container_port():
    sb = _make_sandbox(container_port=None)
    sb._task_ip = "1.2.3.4"
    sb._ssh_key_file = "/tmp/key"
    sb._ssh_tunnel_port = 7000
    sb._outside_endpoint_routing = engine._OutsideEndpointRouting.for_agent_server(
        [engine.OutsideEndpoint("http://10.0.0.2:9000/v1", "MODEL")]
    )
    with patch(f"{_ENG}._free_port", return_value=15000):
        with pytest.raises(ValueError, match="container_port is required"):
            sb._open_tunnel(_agent_sidecar())


# ── EcsFargateSandbox: cleanup ────────────────────────────────────────


def test_cleanup_closes_tunnel_stops_task_removes_key():
    sb = _make_sandbox()
    sb._ssh_tunnel = MagicMock()
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ssh_key_file = "/tmp/key"
    with patch(f"{_ENG}.os.remove") as rm:
        sb._cleanup()
    assert sb._ssh_tunnel is None
    sb._ecs.stop_task.assert_called_once()
    rm.assert_called_once_with("/tmp/key")
    assert sb._ssh_key_file is None


def test_cleanup_swallows_all_errors():
    sb = _make_sandbox()
    tunnel = MagicMock()
    tunnel.close.side_effect = RuntimeError("close boom")
    sb._ssh_tunnel = tunnel
    sb._task_arn = "arn:task"
    sb._ecs = MagicMock()
    sb._ecs.stop_task.side_effect = RuntimeError("stop boom")
    sb._ssh_key_file = "/tmp/key"
    with patch(f"{_ENG}.os.remove", side_effect=OSError("rm boom")):
        sb._cleanup()  # must not raise
    assert sb._ssh_tunnel is None


def test_sync_stop_idempotent():
    sb = _make_sandbox()
    with patch.object(engine.EcsFargateSandbox, "_cleanup") as cleanup:
        sb._sync_stop()
        assert sb._stopped is True
        sb._sync_stop()  # second call short-circuits
    cleanup.assert_called_once()


def test_require_exec_client_raises_in_agent_mode():
    sb = _make_sandbox()
    with pytest.raises(RuntimeError, match="require exec-server mode"):
        sb._require_exec_client()
    _attach_exec_client(sb)
    sb._require_exec_client()  # no error once set


def test_register_and_unregister_for_cleanup():
    sb = _make_sandbox()
    with patch(f"{_ENG}.atexit.register") as reg:
        sb._register_for_cleanup()
    assert id(sb) in engine._active_sandboxes
    reg.assert_called_once()
    sb._unregister_from_cleanup()
    assert id(sb) not in engine._active_sandboxes


def test_emergency_cleanup_invokes_sync_stop():
    good = MagicMock()
    bad = MagicMock()
    bad._sync_stop.side_effect = RuntimeError("boom")
    engine._active_sandboxes[1] = good
    engine._active_sandboxes[2] = bad
    engine._emergency_cleanup()  # bad's error is swallowed
    good._sync_stop.assert_called_once()
    bad._sync_stop.assert_called_once()


async def test_upload_via_s3_packs_and_extracts(tmp_path):
    f = tmp_path / "payload.txt"
    f.write_text("data")
    sb = _make_sandbox(s3_bucket="bkt", s3_prefix="pfx")
    ec = _attach_exec_client(sb, engine.ExecResult("ok\n", "", 0))
    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://signed/url"
    with _patch_aws(s3=s3):
        await sb._upload_via_s3([f], "/dest")
    s3.put_object.assert_called_once()
    dl_cmd = ec.exec.await_args.args[0]
    assert "tar xzf" in dl_cmd and "/dest" in dl_cmd


async def test_upload_via_s3_packs_a_directory(tmp_path):
    d = tmp_path / "dir"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "f.txt").write_text("nested")
    sb = _make_sandbox(s3_bucket="bkt")
    _attach_exec_client(sb, engine.ExecResult("ok\n", "", 0))
    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://signed/url"
    with _patch_aws(s3=s3):
        await sb._upload_via_s3([d], "/dest")
    s3.put_object.assert_called_once()
    # The directory branch tars children with paths relative to the directory.
    import tarfile

    body = s3.put_object.call_args.kwargs["Body"]
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        assert "sub/f.txt" in tar.getnames()


async def test_upload_via_s3_honors_arcname_for_single_file(tmp_path):
    # A large single file must land under the requested remote name, not the local temp basename.
    f = tmp_path / "local-temp-xyz.bin"
    f.write_text("data")
    sb = _make_sandbox(s3_bucket="bkt")
    _attach_exec_client(sb, engine.ExecResult("ok\n", "", 0))
    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://signed/url"
    import tarfile

    with _patch_aws(s3=s3):
        await sb._upload_via_s3([f], "/dest", arcnames={f: "final-name.bin"})
    body = s3.put_object.call_args.kwargs["Body"]
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
        assert tar.getnames() == ["final-name.bin"]


async def test_upload_via_s3_requires_bucket():
    sb = _make_sandbox(s3_bucket=None)
    _attach_exec_client(sb)
    with pytest.raises(ValueError, match="s3_bucket is required"):
        await sb._upload_via_s3([Path("/x")], "/dest")


def test_delete_s3_object_warns_once_then_silent(caplog):
    cfg = _make_sandbox(s3_bucket="bkt")._cfg
    s3 = MagicMock()
    s3.delete_object.side_effect = Exception("AccessDenied")
    engine._s3_delete_warned = False
    try:
        with _patch_aws(s3=s3):
            with caplog.at_level(logging.WARNING, logger="nemo_gym.sandbox.providers.ecs_fargate.engine"):
                engine._delete_s3_object(cfg, "k1")  # first failure -> one WARNING
                engine._delete_s3_object(cfg, "k2")  # subsequent failures silenced at WARNING
    finally:
        engine._s3_delete_warned = False
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "s3:DeleteObject" in r.message]
    assert len(warnings) == 1
    assert s3.delete_object.call_count == 2  # still attempted both times, never raised


async def test_upload_via_s3_raises_when_extraction_fails(tmp_path):
    f = tmp_path / "payload.txt"
    f.write_text("data")
    sb = _make_sandbox(s3_bucket="bkt")
    _attach_exec_client(sb, engine.ExecResult("nope", "tar error", 2))
    s3 = MagicMock()
    s3.generate_presigned_url.return_value = "https://signed/url"
    with _patch_aws(s3=s3):
        with pytest.raises(RuntimeError, match="S3 upload extraction failed"):
            await sb._upload_via_s3([f], "/dest")


# ── EcsFargateSandbox: properties ─────────────────────────────────────


def test_properties_reflect_state():
    spec = engine.SandboxSpec(image="img:1")
    sb = _make_sandbox(spec=spec)
    assert sb.spec is spec
    assert sb.is_running is False
    sb._started = True
    assert sb.is_running is True
    sb._stopped = True
    assert sb.is_running is False

    sb._task_arn = "arn:task"
    assert sb.task_arn == "arn:task"
    sb._task_ip = "1.2.3.4"
    assert sb.container_ip == "1.2.3.4"
    sb._ssh_tunnel_port = 7000
    assert sb.model_tunnel_port == 7000

    ec = _attach_exec_client(sb)
    assert sb.exec_client is ec


def test_local_port_property_paths():
    sb = _make_sandbox()
    assert sb.local_port is None  # no tunnel
    tunnel = MagicMock()
    tunnel.local_port = 18000
    sb._ssh_tunnel = tunnel
    assert sb.ssh_tunnel is tunnel
    assert sb.local_port == 18000
    # A not-yet-open tunnel raises RuntimeError, surfaced as None.
    type(tunnel).local_port = property(lambda self: (_ for _ in ()).throw(RuntimeError("not open")))
    assert sb.local_port is None


def test_resolved_endpoint_url_and_resolve_outside_endpoint():
    sb = _make_sandbox()
    sb._outside_endpoint_routing = engine._OutsideEndpointRouting.for_agent_server(
        [engine.OutsideEndpoint("http://model:7000/v1", "MODEL")]
    )
    assert sb.resolved_endpoint_url("MODEL") == "http://127.0.0.1:7000/v1"
    assert sb.resolve_outside_endpoint("http://anything/v1").startswith("http://127.0.0.1:7000")


# ── EcsFargateSandbox: async start/stop ───────────────────────────────


async def test_start_runs_do_start_and_marks_started():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    with patch.object(engine.EcsFargateSandbox, "_do_start") as do_start:
        await sb.start()
    do_start.assert_called_once()
    assert sb._started is True


async def test_start_is_idempotent():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    sb._started = True
    with patch.object(engine.EcsFargateSandbox, "_do_start") as do_start:
        await sb.start()
    do_start.assert_not_called()


async def test_start_agent_mode_validates_endpoints():
    sb = _make_sandbox(ssh_sidecar=_agent_sidecar())
    with patch.object(engine.EcsFargateSandbox, "_do_start") as do_start:
        with pytest.raises(ValueError, match="requires OutsideEndpoint"):
            await sb.start(outside_endpoints=[])
    do_start.assert_not_called()


async def test_start_failure_cleans_up_and_reraises():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb)
    with (
        patch.object(engine.EcsFargateSandbox, "_do_start", side_effect=RuntimeError("startup failed")),
        patch.object(engine.EcsFargateSandbox, "_cleanup") as cleanup,
    ):
        with pytest.raises(RuntimeError, match="startup failed"):
            await sb.start()
    ec.close.assert_awaited_once()
    cleanup.assert_called_once()
    assert sb._started is False


async def test_stop_closes_client_and_cleans_up():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb)
    with (
        patch.object(engine.EcsFargateSandbox, "_cleanup") as cleanup,
        patch.object(engine.EcsFargateSandbox, "_unregister_from_cleanup") as unreg,
    ):
        await sb.stop()
    assert sb._stopped is True
    ec.close.assert_awaited_once()
    cleanup.assert_called_once()
    unreg.assert_called_once()


async def test_stop_is_idempotent():
    sb = _make_sandbox()
    sb._stopped = True
    with patch.object(engine.EcsFargateSandbox, "_cleanup") as cleanup:
        await sb.stop()
    cleanup.assert_not_called()


async def test_aenter_aexit_delegate_to_start_stop():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    with (
        patch.object(engine.EcsFargateSandbox, "start", AsyncMock()) as start,
        patch.object(engine.EcsFargateSandbox, "stop", AsyncMock()) as stop,
    ):
        async with sb as ctx:
            assert ctx is sb
    start.assert_awaited_once()
    stop.assert_awaited_once()


# ── EcsFargateSandbox: async exec ─────────────────────────────────────


async def test_exec_passes_timeout_as_int():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb, engine.ExecResult("hi", "", 0))
    out = await sb.exec("echo hi", timeout_sec=42.7)
    assert out.return_code == 0
    assert ec.exec.await_args.args[0] == "echo hi"
    assert ec.exec.await_args.kwargs["timeout"] == 42


async def test_exec_wraps_env_cwd_and_user_string():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb)
    await sb.exec("run", env={"A": "1"}, cwd="/work", user="appuser")
    shell_cmd = ec.exec.await_args.args[0]
    assert shell_cmd.startswith("su -s /bin/bash appuser -c ")
    assert "cd /work &&" in shell_cmd
    assert "export A=1 &&" in shell_cmd


async def test_exec_user_int_uses_getent():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb)
    await sb.exec("run", user=1000)
    assert "getent passwd 1000" in ec.exec.await_args.args[0]


async def test_exec_connection_error_with_live_tunnel_reraises():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb)
    ec.exec = AsyncMock(side_effect=ConnectionError("dead"))
    sb._ssh_tunnel = MagicMock(is_open=True)
    with pytest.raises(ConnectionError):
        await sb.exec("run")


async def test_exec_reconnects_tunnel_then_succeeds():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb)
    ec.exec = AsyncMock(side_effect=ConnectionError("dead"))
    tunnel = MagicMock(is_open=False, local_port=18000)
    sb._ssh_tunnel = tunnel
    new_client = MagicMock()
    new_client.exec = AsyncMock(return_value=engine.ExecResult("recovered", "", 0))
    with (
        patch.object(engine.EcsFargateSandbox, "reconnect_tunnel", AsyncMock()) as reconnect,
        patch(f"{_ENG}.ExecClient", return_value=new_client),
    ):
        out = await sb.exec("run")
    assert out.stdout == "recovered"
    reconnect.assert_awaited_once()
    ec.close.assert_awaited_once()  # old client closed
    tunnel.wait_ready.assert_called_once()


async def test_exec_reconnect_failure_reraises_original():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb)
    ec.exec = AsyncMock(side_effect=ConnectionError("dead"))
    sb._ssh_tunnel = MagicMock(is_open=False)
    with patch.object(engine.EcsFargateSandbox, "reconnect_tunnel", AsyncMock(side_effect=RuntimeError("no luck"))):
        with pytest.raises(ConnectionError):
            await sb.exec("run")


async def test_exec_requires_exec_client():
    sb = _make_sandbox()
    with pytest.raises(RuntimeError, match="require exec-server mode"):
        await sb.exec("run")


# ── EcsFargateSandbox: async upload/download ──────────────────────────


async def test_upload_small_file_uses_exec_client(tmp_path):
    f = tmp_path / "small.txt"
    f.write_text("tiny")
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb)
    await sb.upload(f, "/remote/small.txt")
    ec.upload.assert_awaited_once()
    assert ec.upload.await_args.args[0] == "/remote/small.txt"


async def test_upload_directory_recurses(tmp_path):
    d = tmp_path / "dir"
    (d / "sub").mkdir(parents=True)
    (d / "a.txt").write_text("a")
    (d / "sub" / "b.txt").write_text("b")
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    ec = _attach_exec_client(sb)
    await sb.upload(d, "/remote")
    uploaded = {call.args[0] for call in ec.upload.await_args_list}
    assert uploaded == {"/remote/a.txt", "/remote/sub/b.txt"}


async def test_upload_large_file_uses_s3(tmp_path):
    # Local basename intentionally differs from the requested remote name to catch the case where
    # the S3 path drops the target filename.
    f = tmp_path / "local-temp-name.bin"
    f.write_bytes(b"x" * (512 * 1024 + 1))
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar(), s3_bucket="bkt")
    _attach_exec_client(sb)
    with patch.object(engine.EcsFargateSandbox, "_upload_via_s3", AsyncMock()) as via_s3:
        await sb.upload(f, "/remote/renamed.bin")
    via_s3.assert_awaited_once()
    assert via_s3.await_args.args[1] == "/remote"
    assert via_s3.await_args.kwargs["arcnames"] == {f: "renamed.bin"}


async def test_download_writes_bytes(tmp_path):
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    _attach_exec_client(sb)
    dest = tmp_path / "nested" / "out.bin"
    await sb.download("/remote/out.bin", dest)
    assert dest.read_bytes() == b"payload"


# ── EcsFargateSandbox: reconnect_tunnel ───────────────────────────────


async def test_reconnect_tunnel_rejects_stopped_or_unstarted():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    with pytest.raises(RuntimeError, match="stopped/unstarted"):
        await sb.reconnect_tunnel()  # not started
    sb._started = True
    sb._stopped = True
    with pytest.raises(RuntimeError, match="stopped/unstarted"):
        await sb.reconnect_tunnel()


async def test_reconnect_tunnel_noop_without_sidecar():
    sb = _make_sandbox(ssh_sidecar=None)
    sb._started = True
    with patch.object(engine.EcsFargateSandbox, "_open_tunnel") as open_tunnel:
        await sb.reconnect_tunnel()
    open_tunnel.assert_not_called()


async def test_reconnect_tunnel_reopens():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar())
    sb._started = True
    old_tunnel = MagicMock()
    sb._ssh_tunnel = old_tunnel
    with patch.object(engine.EcsFargateSandbox, "_open_tunnel") as open_tunnel:
        await sb.reconnect_tunnel()
    old_tunnel.close.assert_called_once()
    open_tunnel.assert_called_once()


# ── EcsFargateSandbox: _do_start orchestration ────────────────────────


@contextlib.contextmanager
def _do_start_seams(tunnel=None, exec_client=None):
    """Patch every AWS/SSH seam so the *real* ``_do_start`` runs offline."""
    tunnel = tunnel or MagicMock()
    tunnel.local_port = 19000
    ec = exec_client or MagicMock()
    with (
        patch.object(engine.EcsFargateSandbox, "_init_aws_clients"),
        patch.object(engine.EcsFargateSandbox, "_register_task_definition", return_value="td-arn"),
        patch.object(engine.EcsFargateSandbox, "_run_task", return_value="task-arn"),
        patch.object(engine.EcsFargateSandbox, "_register_for_cleanup"),
        patch.object(engine.EcsFargateSandbox, "_wait_for_running"),
        patch.object(engine.EcsFargateSandbox, "_get_task_public_ip", return_value="1.2.3.4"),
        patch.object(engine.EcsFargateSandbox, "_wait_for_ssh_ready"),
        patch(f"{_ENG}.download_secret_to_file", return_value="/tmp/key"),
        patch(f"{_ENG}.download_secret_to_string", return_value="ssh-rsa fake"),
        patch(f"{_ENG}.build_ssh_sidecar_container", return_value={"name": "ssh-tunnel"}),
        patch(f"{_ENG}._free_port", return_value=19001),
        patch(f"{_ENG}.SshTunnel", return_value=tunnel),
        patch(f"{_ENG}.ExecClient", return_value=ec),
    ):
        yield tunnel, ec


def test_do_start_requires_sidecar():
    sb = _make_sandbox(ssh_sidecar=None)
    with pytest.raises(ValueError, match="ssh_sidecar must be configured"):
        sb._do_start()


def test_do_start_requires_key_arns():
    sb = _make_sandbox(ssh_sidecar=_exec_sidecar(private_key_secret_arn=""))
    with _do_start_seams():
        with pytest.raises(ValueError, match="private_key_secret_arn and public_key_secret_arn"):
            sb._do_start()


def test_do_start_exec_server_full_path():
    sb = _make_sandbox(spec=engine.SandboxSpec(image="python:3.12"), ssh_sidecar=_exec_sidecar(exec_server_port=5000))
    with _do_start_seams() as (tunnel, ec):
        sb._do_start()
    assert sb._task_arn == "task-arn"
    assert sb._task_def_arn == "td-arn"
    assert sb._task_ip == "1.2.3.4"
    assert sb._ssh_tunnel is tunnel
    assert sb._exec_client is ec  # exec-server mode creates an ExecClient


def test_do_start_agent_server_full_path():
    sb = _make_sandbox(
        spec=engine.SandboxSpec(image="python:3.12"),
        ssh_sidecar=_agent_sidecar(),
        container_port=9000,
    )
    sb._outside_endpoints = [engine.OutsideEndpoint("http://10.0.0.9:9000/v1", "MODEL")]
    with _do_start_seams() as (tunnel, _ec):
        sb._do_start()
    assert sb._ssh_tunnel is tunnel
    assert sb._exec_client is None  # agent-server mode does not create an ExecClient
    assert sb._ssh_tunnel_port == 9000  # agent tunnel target port


def test_do_start_builds_image_when_ecr_and_env_dir():
    sb = _make_sandbox(
        spec=engine.SandboxSpec(image="python:3.12", environment_dir="/env"),
        ssh_sidecar=_exec_sidecar(exec_server_port=5000),
        ecr_repository=_ecr_repo(),
    )
    with (
        _do_start_seams(),
        patch.object(engine.ImageBuilder, "ensure_image_built", return_value=f"{_ecr_repo()}:built") as build,
    ):
        sb._do_start()
    build.assert_called_once()
    assert sb._task_def_arn == "td-arn"
