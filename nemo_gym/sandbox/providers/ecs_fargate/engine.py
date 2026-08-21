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

"""ECS Fargate sandbox engine.

Runs each sandbox as an ECS Fargate task behind an SSH sidecar. Defines the host-protocol
types (``ExecResult``, ``OutsideEndpoint``, ``SandboxSpec``) and the engine; the Gym-facing
provider adapter lives in ``provider.py``.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import shlex
import socket
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from dataclasses import replace as _dc_replace
from pathlib import Path
from typing import Any, Callable, Self, TypeVar
from urllib.parse import ParseResult, urlparse

import aiohttp


# Matches the exec-server script's TB_EXEC_PORT fallback and the
# exec_server_port written by the reference Terraform.
DEFAULT_EXEC_SERVER_PORT = 19542
# SSH sidecar port matching the reference Terraform ssh_tunnel_sshd_port.
DEFAULT_SSHD_PORT = 52222
DEFAULT_SSM_PROJECT = "harbor"


@dataclass
class ExecResult:
    """Result of executing a command inside a sandbox."""

    stdout: str
    stderr: str
    return_code: int


@dataclass
class OutsideEndpoint:
    """A host-side URL that must be reachable from inside the sandbox.

    The sandbox rewrites *url* for its network topology and exposes the
    resolved address as the environment variable *env_var* inside the
    container.
    """

    url: str
    env_var: str


@dataclass
class VolumeMount:
    """EFS mount (ECS Fargate). Host bind mounts are unused on Fargate."""

    host_path: str = ""
    container_path: str = ""
    readonly: bool = False
    efs: bool = False
    efs_filesystem_id: str | None = None
    efs_root_directory: str | None = None
    efs_access_point_id: str | None = None

    @property
    def is_efs(self) -> bool:
        # Either names its own filesystem, or opts into the provider-level EFS default via `efs: true`.
        return self.efs or self.efs_filesystem_id is not None


@dataclass
class SandboxSpec:
    """Per-problem sandbox requirements."""

    image: str
    workdir: str = "/workspace"
    env: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    entrypoint: str | None = None
    volumes: list[VolumeMount] = field(default_factory=list)
    environment_dir: str | None = None


logger = logging.getLogger(__name__)
T = TypeVar("T")


# ── Lazy AWS SDK import ──────────────────────────────────────────────


def _require_aws_sdks():
    try:
        import importlib

        boto3 = importlib.import_module("boto3")
        botocore_config = importlib.import_module("botocore.config")
        botocore_exceptions = importlib.import_module("botocore.exceptions")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "ECS Fargate sandbox requires boto3/botocore. "
            "Install them (`pip install boto3`) or use a different sandbox backend."
        ) from e
    return boto3, getattr(botocore_config, "Config"), getattr(botocore_exceptions, "ClientError")


# ── SSM auto-discovery ───────────────────────────────────────────────

_ssm_config_cache: dict[str, dict[str, Any]] = {}


def resolve_ecs_config_from_ssm(
    region: str,
    project: str = DEFAULT_SSM_PROJECT,
) -> dict[str, Any]:
    """Read ECS sandbox config from SSM Parameter Store.

    Returns a dict matching the JSON structure written by Terraform
    (cluster, subnets, security_groups, roles, SSH ARNs, EFS, etc.).
    Results are cached per (region, project) for the process lifetime.
    """
    cache_key = f"{region}:{project}"
    if cache_key in _ssm_config_cache:
        return _ssm_config_cache[cache_key]

    boto3, _, ClientError = _require_aws_sdks()
    ssm = boto3.client("ssm", region_name=region)
    param_name = f"/{project}/ecs-sandbox/config"
    try:
        resp = _retry_with_backoff(
            lambda: ssm.get_parameter(Name=param_name),
            operation_name="ssm.get_parameter",
            max_retries=5,
        )
    except ClientError as exc:
        code = (exc.response.get("Error") or {}).get("Code", "")
        if code == "ParameterNotFound":
            raise RuntimeError(
                f"SSM parameter '{param_name}' not found in {region}. "
                f"Run 'terraform apply' in the ecs-sandbox stack for this "
                f"region, or specify all ECS fields explicitly in your YAML."
            ) from exc
        raise

    raw = resp["Parameter"]["Value"]
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SSM parameter '{param_name}' in {region} contains invalid JSON: {exc}") from exc

    _ssm_config_cache[cache_key] = config
    logger.info(
        "Resolved ECS config from SSM %s in %s (cluster=%s, %d subnets)",
        param_name,
        region,
        config.get("cluster"),
        len(config.get("subnets", [])),
    )
    return config


# ── Config dataclasses ───────────────────────────────────────────────


def _sanitize_id(value: str, max_len: int = 100) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-")
    return cleaned[:max_len] or "task"


_ECR_IMAGE_REF_RE = re.compile(r"^[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/", re.IGNORECASE)


def _is_ecr_image_ref(image: str) -> bool:
    """True for references already pointing at an ECR registry host.

    Such references (e.g. an image resolved against the configured ECR mirror)
    must be used as-is; reapplying the ``ecr_repository`` + sanitize rewrite
    would corrupt the tag.
    """
    return bool(_ECR_IMAGE_REF_RE.match(image))


@dataclass(frozen=True)
class SshSidecarConfig:
    """SSH sidecar container configuration.

    exec_server_port set → exec-server mode (one-way tunnel).
    exec_server_port None → agent-server mode (two-way tunnel).
    """

    sshd_port: int = 2222
    ssh_ready_timeout_sec: float = 300.0
    public_key_secret_arn: str = ""
    private_key_secret_arn: str = ""
    image: str | None = None
    exec_server_port: int | None = None


@dataclass(frozen=True)
class EcsFargateConfig:
    """Configuration for the ECS Fargate sandbox."""

    region: str | None = None
    cluster: str = ""
    subnets: list[str] = field(default_factory=list)
    security_groups: list[str] = field(default_factory=list)
    assign_public_ip: bool = False
    task_definition: str | None = None
    task_definition_family_prefix: str = "ecs-sandbox"
    image_template: str | None = None
    container_name: str = "main"
    container_port: int | None = None
    cpu: str = "4096"
    memory: str = "8192"
    ephemeral_storage_gib: int | None = None
    platform_version: str | None = None
    execution_role_arn: str | None = None
    task_role_arn: str | None = None
    extra_env: dict[str, str] | None = None
    log_group: str | None = None
    log_stream_prefix: str | None = None
    max_task_lifetime_sec: int = 14400
    startup_timeout_sec: float = 300.0
    poll_interval_sec: float = 2.0
    run_task_max_retries: int = 30
    ssh_sidecar: SshSidecarConfig | None = None
    s3_bucket: str | None = None
    s3_prefix: str | None = None
    ecr_repository: str | None = None
    environment_dir: str | None = None
    codebuild_project: str | None = None
    codebuild_service_role: str | None = None
    codebuild_compute_type: str = "BUILD_GENERAL1_MEDIUM"
    codebuild_build_timeout: int = 60
    dockerhub_secret_arn: str | None = None
    build_parallelism: int = 50
    # Mirror a missing public/bare image into ECR on demand during create. Set False to require a
    # pre-populated mirror and fail fast on a miss.
    auto_mirror: bool = True
    efs_filesystem_id: str | None = None
    efs_access_point_id: str | None = None
    ssm_project: str = DEFAULT_SSM_PROJECT


@dataclass(frozen=True)
class _OutsideEndpointRoute:
    endpoint: OutsideEndpoint
    source_netloc: str
    host: str
    target_port: int
    remote_port: int
    scheme: str

    @classmethod
    def for_endpoint(cls, endpoint: OutsideEndpoint, *, remote_port: int | None = None) -> _OutsideEndpointRoute:
        parsed = urlparse(endpoint.url)
        host = parsed.hostname
        if not host:
            raise ValueError(f"Cannot resolve hostname from OutsideEndpoint: {endpoint.url}")
        target_port = _port_from_url(parsed)
        return cls(
            endpoint=endpoint,
            source_netloc=parsed.netloc,
            host=host,
            target_port=target_port,
            remote_port=remote_port or target_port,
            scheme=parsed.scheme or "http",
        )

    def resolved_endpoint_url(self) -> str:
        return self._rewrite(urlparse(self.endpoint.url))

    def resolve_url(self, url: str) -> str:
        return self._rewrite(urlparse(url))

    def _rewrite(self, parsed: ParseResult) -> str:
        return parsed._replace(
            scheme=self.scheme,
            netloc=f"127.0.0.1:{self.remote_port}",
        ).geturl()


def _port_from_url(parsed: ParseResult) -> int:
    return parsed.port or (443 if parsed.scheme == "https" else 80)


@dataclass(frozen=True)
class _OutsideEndpointRouting:
    endpoints: tuple[OutsideEndpoint, ...] = ()
    _routes_by_env: dict[str, _OutsideEndpointRoute] = field(default_factory=dict)
    _reverse_specs: tuple[str, ...] = ()
    _agent_tunnel_port: int | None = None

    @classmethod
    def empty(cls, endpoints: list[OutsideEndpoint] | None = None) -> _OutsideEndpointRouting:
        return cls(endpoints=tuple(endpoints or []))

    @classmethod
    def for_exec_server(
        cls,
        endpoints: list[OutsideEndpoint],
        sidecar: SshSidecarConfig,
    ) -> _OutsideEndpointRouting:
        reverse_specs: list[str] = []
        routes_by_env: dict[str, _OutsideEndpointRoute] = {}
        used_ports = {sidecar.sshd_port}
        if sidecar.exec_server_port is not None:
            used_ports.add(sidecar.exec_server_port)

        target_port_map: dict[tuple[str, int], int] = {}
        for ep in endpoints:
            target = _OutsideEndpointRoute.for_endpoint(ep)
            key = (target.host, target.target_port)
            remote_port = target_port_map.get(key)
            if remote_port is None:
                remote_port = cls._allocate_reverse_port(target.target_port, used_ports)
                target_port_map[key] = remote_port
                reverse_specs.append(f"{remote_port}:{target.host}:{target.target_port}")
            route = _OutsideEndpointRoute.for_endpoint(ep, remote_port=remote_port)
            logger.info(
                "Reverse tunnel: container :%d → host %s:%d (%s)",
                route.remote_port,
                route.host,
                route.target_port,
                ep.env_var,
            )
            routes_by_env[ep.env_var] = route

        return cls(endpoints=tuple(endpoints), _routes_by_env=routes_by_env, _reverse_specs=tuple(reverse_specs))

    @classmethod
    def for_agent_server(cls, endpoints: list[OutsideEndpoint]) -> _OutsideEndpointRouting:
        if len(endpoints) > 1:
            raise ValueError("Agent-server mode supports only one OutsideEndpoint")
        if not endpoints:
            raise ValueError("Agent-server mode requires OutsideEndpoint passed to start()")
        route = _OutsideEndpointRoute.for_endpoint(endpoints[0])
        return cls(
            endpoints=tuple(endpoints),
            _routes_by_env={route.endpoint.env_var: route},
            _agent_tunnel_port=route.target_port,
        )

    @property
    def reverse_specs(self) -> list[str]:
        return list(self._reverse_specs)

    @property
    def agent_tunnel_port(self) -> int | None:
        return self._agent_tunnel_port

    def agent_tunnel_target(self) -> tuple[str, int]:
        if not self.endpoints:
            raise ValueError("Agent-server mode requires OutsideEndpoint passed to start()")
        route = self._routes_by_env[self.endpoints[0].env_var]
        return route.host, route.target_port

    def env_overrides(self) -> dict[str, str]:
        return {env_var: route.resolved_endpoint_url() for env_var, route in self._routes_by_env.items()}

    def resolved_endpoint_url(self, env_var: str) -> str | None:
        route = self._routes_by_env.get(env_var)
        if route is None:
            return None
        return route.resolved_endpoint_url()

    def resolve_url(self, url: str) -> str:
        parsed = urlparse(url)
        for route in self._routes_by_env.values():
            if route.source_netloc == parsed.netloc:
                return route.resolve_url(url)
        if self._agent_tunnel_port is not None:
            return parsed._replace(netloc=f"127.0.0.1:{self._agent_tunnel_port}").geturl()
        raise RuntimeError("resolve_outside_endpoint() requires SSH reverse tunnel")

    @staticmethod
    def _allocate_reverse_port(preferred: int, used_ports: set[int]) -> int:
        if 0 < preferred <= 65535 and preferred not in used_ports:
            used_ports.add(preferred)
            return preferred
        for candidate in range(20000, 61000):
            if candidate not in used_ports:
                used_ports.add(candidate)
                return candidate
        raise RuntimeError("No available local port for ECS reverse tunnel")


# ── Retry utilities ──────────────────────────────────────────────────

_RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailable",
        "RequestLimitExceeded",
    }
)
_RETRYABLE_MESSAGES = (
    "capacity is unavailable",
    "rate exceeded",
    "too many concurrent",
    "throttl",
    "connect timeout",
    "read timeout",
    "connection reset",
    "endpointconnectionerror",
)


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    code = ""
    if hasattr(exc, "response"):
        code = (exc.response.get("Error") or {}).get("Code", "")  # type: ignore[union-attr]
    return code in _RETRYABLE_CODES or any(m in msg for m in _RETRYABLE_MESSAGES)


def _retry_with_backoff(
    func: Callable[[], T],
    *,
    operation_name: str,
    max_retries: int | None = None,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.5,
) -> T:
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:
            if not _is_retryable_error(exc):
                raise
            attempt += 1
            if max_retries is not None and attempt > max_retries:
                logger.error("%s failed after %d retries: %s", operation_name, attempt - 1, exc)
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 1 + random.uniform(-jitter, jitter)
            logger.warning("%s throttled (attempt %d), retrying in %.1fs: %s", operation_name, attempt, delay, exc)
            time.sleep(delay)


# ── SSH helpers ──────────────────────────────────────────────────────


def _free_port() -> int:
    # TOCTOU: the probed port can be claimed before ssh binds it; SshTunnel.open() retries on
    # port-not-open to mitigate this under concurrency.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ephemeral_storage_block(requested: Any) -> dict[str, int] | None:
    """Task-def ``ephemeralStorage`` block, or ``None`` for Fargate's implicit 20 GiB default.

    Fargate accepts only 21-200 GiB explicitly (20 is valid only as the omitted default).
    """
    if not requested:
        return None
    size = int(requested)
    if not 21 <= size <= 200:
        raise ValueError(f"ephemeral storage must be between 21 and 200 GiB (got {size})")
    return {"sizeInGiB": size}


# Valid Fargate task cpu (units) -> inclusive memory range (MiB). cpu/memory are not independent:
# https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-cpu-memory-error.html
_FARGATE_MEMORY_RANGE_MIB: dict[int, tuple[int, int]] = {
    256: (512, 2048),
    512: (1024, 4096),
    1024: (2048, 8192),
    2048: (4096, 16384),
    4096: (8192, 30720),
    8192: (16384, 61440),
    16384: (32768, 122880),
}


def _validate_fargate_cpu_memory(cpu: int, memory: int) -> None:
    """Validate a Fargate cpu (units) + memory (MiB) pair; Fargate rejects unsupported combinations."""
    memory_range = _FARGATE_MEMORY_RANGE_MIB.get(cpu)
    if memory_range is None:
        raise ValueError(f"Fargate task cpu must be one of {sorted(_FARGATE_MEMORY_RANGE_MIB)} units (got {cpu})")
    low, high = memory_range
    if not low <= memory <= high:
        raise ValueError(f"Fargate memory for cpu={cpu} must be {low}-{high} MiB (got {memory})")


_s3_delete_warned = False


def _delete_s3_object(cfg: EcsFargateConfig, key: str) -> None:
    """Best-effort delete of a transient S3 staging artifact; never raises.

    Warns once if the role lacks ``s3:DeleteObject`` (objects then rely on a bucket lifecycle policy).
    """
    global _s3_delete_warned
    if not cfg.s3_bucket or not key:
        return
    try:
        boto3, *_ = _require_aws_sdks()
        boto3.client("s3", region_name=cfg.region).delete_object(Bucket=cfg.s3_bucket, Key=key)
    except Exception:
        if not _s3_delete_warned:
            _s3_delete_warned = True
            logger.warning(
                "Could not delete S3 staging object s3://%s/%s (further failures silenced); grant "
                "s3:DeleteObject or set a bucket lifecycle policy to avoid leaking staging artifacts",
                cfg.s3_bucket,
                key,
                exc_info=True,
            )
        else:
            logger.debug("Failed to delete S3 object s3://%s/%s", cfg.s3_bucket, key, exc_info=True)


# Safe registry/repo/tag/digest characters. Gates dataset-controlled image refs before they are
# interpolated into shell commands inside a privileged CodeBuild job (injection guard).
_IMAGE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$")


def _validate_image_ref(ref: str) -> None:
    """Reject an image reference with characters outside the safe set (shell-injection guard)."""
    if not ref or not _IMAGE_REF_RE.match(ref):
        raise ValueError(f"Unsafe image reference {ref!r}: only [A-Za-z0-9._:/@-] characters are allowed")


def download_secret_to_file(secret_arn: str, region: str | None = None) -> str:
    """Fetch a Secrets Manager secret → temp file (mode 0600)."""
    key_material = download_secret_to_string(secret_arn, region=region)
    fd, path = tempfile.mkstemp(prefix="ecs-ssh-", suffix=".key")
    try:
        os.write(fd, key_material.encode())
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    return path


def download_secret_to_string(secret_arn: str, region: str | None = None) -> str:
    boto3, *_ = _require_aws_sdks()
    sm = boto3.client("secretsmanager", region_name=region)
    return _retry_with_backoff(
        lambda: sm.get_secret_value(SecretId=secret_arn)["SecretString"],
        operation_name="secretsmanager.get_secret_value",
        max_retries=5,
    )


# ── SSH tunnel ───────────────────────────────────────────────────────


class SshTunnel:
    """Manages an ``ssh -N`` subprocess with ``-L`` / ``-R`` tunnels."""

    def __init__(
        self,
        *,
        host: str,
        port: int = 2222,
        user: str = "root",
        key_file: str,
        forward_port: int | None = None,
        forwards: list[str] | None = None,
        reverses: list[str] | None = None,
        local_port_override: int | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._key_file = key_file
        self._simple_forward_port = forward_port
        self._forwards = list(forwards or [])
        self._reverses = list(reverses or [])
        self._local_port: int | None = local_port_override
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def local_port(self) -> int:
        if self._local_port is None:
            raise RuntimeError("Tunnel not open yet — call open() first")
        return self._local_port

    @property
    def is_open(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def open(self, *, max_retries: int = 15, initial_backoff: float = 5.0) -> None:
        if self.is_open:
            return
        use_simple = self._simple_forward_port is not None
        last_err = ""
        backoff = initial_backoff
        for attempt in range(1, max_retries + 1):
            if use_simple:
                self._local_port = _free_port()
            cmd = self._build_ssh_cmd()
            logger.info("SSH tunnel attempt %d/%d: %s", attempt, max_retries, " ".join(cmd))
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(3)
            if self._proc.poll() is None:
                if self._local_port:
                    try:
                        self._wait_for_local_port(self._local_port, timeout=15.0)
                    except Exception as port_exc:
                        logger.warning("SSH alive but forward port %d not open: %s", self._local_port, port_exc)
                        self._kill()
                        last_err = str(port_exc)
                        time.sleep(min(5.0, attempt * 1.5))
                        continue
                logger.info("SSH tunnel started (pid=%d, attempt %d/%d)", self._proc.pid, attempt, max_retries)
                return
            stderr = self._proc.stderr.read().decode(errors="replace") if self._proc.stderr else ""
            last_err = stderr.strip()
            self._proc = None
            if not any(
                m in last_err
                for m in (
                    "Connection refused",
                    "Connection timed out",
                    "No route to host",
                    "Connection reset",
                )
            ):
                raise RuntimeError(f"SSH tunnel exited immediately (attempt {attempt}): {last_err}")
            logger.warning(
                "SSH tunnel attempt %d/%d failed: %s — retrying in %.0fs", attempt, max_retries, last_err, backoff
            )
            time.sleep(backoff)
            backoff = min(30.0, backoff * 1.5)
        raise RuntimeError(f"SSH tunnel failed after {max_retries} attempts: {last_err}")

    def close(self) -> None:
        self._kill()

    def wait_ready(self, *, health_url: str | None = None, timeout: float = 300.0) -> None:
        if health_url:
            self._poll_health(health_url, timeout)
        elif self._local_port:
            self._wait_for_local_port(self._local_port, timeout)

    def check_health(self) -> bool:
        return self.is_open

    def __enter__(self) -> SshTunnel:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _build_ssh_cmd(self) -> list[str]:
        cmd = [
            "ssh",
            "-N",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=20",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "LogLevel=ERROR",
            "-i",
            self._key_file,
            "-p",
            str(self._port),
        ]
        if self._simple_forward_port is not None:
            cmd += ["-L", f"127.0.0.1:{self._local_port}:127.0.0.1:{self._simple_forward_port}"]
        for spec in self._forwards:
            cmd += ["-L", spec]
        for spec in self._reverses:
            cmd += ["-R", spec]
        cmd.append(f"{self._user}@{self._host}")
        return cmd

    def _kill(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            logger.info("SSH tunnel closed (pid=%d)", self._proc.pid)
        except ProcessLookupError:
            pass
        finally:
            self._proc = None

    def _wait_for_local_port(self, port: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError("SSH tunnel process exited while waiting for port")
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect(("127.0.0.1", port))
                    return
            except OSError:
                time.sleep(0.3)
        raise TimeoutError(f"Local port 127.0.0.1:{port} not open after {timeout:.0f}s")

    def _poll_health(self, url: str, timeout: float) -> None:
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            if not self.is_open:
                raise RuntimeError("SSH tunnel died while waiting for health endpoint")
            try:
                with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=5) as resp:
                    if resp.status == 200:
                        logger.info("Health endpoint ready (attempt %d): %s", attempt, url)
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(min(3.0, 1.0 + attempt * 0.5))
        raise TimeoutError(f"Health endpoint not reachable after {timeout:.0f}s: {url}")


# ── SSH sidecar container builder ────────────────────────────────────


def build_ssh_sidecar_container(
    sidecar_cfg: SshSidecarConfig,
    *,
    public_key_value: str,
    max_lifetime_sec: int,
    log_group: str | None = None,
    log_region: str = "us-east-1",
    log_stream_prefix: str = "ecs-sandbox",
) -> dict[str, Any]:
    port = sidecar_cfg.sshd_port
    image = sidecar_cfg.image or "alpine:latest"
    sshd_cfg = (
        f"Port {port}\\nPermitRootLogin prohibit-password\\n"
        "PasswordAuthentication no\\nAllowTcpForwarding yes\\n"
        "PermitListen any\\nGatewayPorts clientspecified\\n"
        "X11Forwarding no\\nPrintMotd no\\nLogLevel ERROR\\n"
        "ClientAliveInterval 30\\nClientAliveCountMax 20\\n"
        "TCPKeepAlive yes\\nUseDNS no\\nMaxSessions 50\\n"
    )
    watchdog = ""
    if max_lifetime_sec > 0:
        watchdog = (
            f"( sleep {max_lifetime_sec}; "
            f"echo 'sidecar watchdog: TTL ({max_lifetime_sec}s) reached'; "
            "kill 1 2>/dev/null; sleep 3; kill -9 1 2>/dev/null ) & "
        )
    sshd_cmd = (
        "set -e; apk add --no-cache openssh-server netcat-openbsd; "
        "mkdir -p /root/.ssh; chmod 700 /root/.ssh; "
        'printf "%s\\n" "$SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys; '
        "chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; "
        f"printf '{sshd_cfg}' > /etc/ssh/sshd_config; "
        f"{watchdog}exec /usr/sbin/sshd -D -e -p {port}"
    )
    container: dict[str, Any] = {
        "name": "ssh-tunnel",
        "image": image,
        "essential": True,
        "entryPoint": ["sh", "-c"],
        "command": [sshd_cmd],
        "environment": [{"name": "SSH_PUBLIC_KEY", "value": public_key_value}],
        "healthCheck": {
            "command": ["CMD-SHELL", f"nc -z localhost {port} || exit 1"],
            "interval": 5,
            "timeout": 3,
            "retries": 10,
            "startPeriod": 30,
        },
    }
    if log_group:
        container["logConfiguration"] = {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": log_group,
                "awslogs-region": log_region,
                "awslogs-stream-prefix": f"{log_stream_prefix}-tunnel",
                "awslogs-create-group": "true",
            },
        }
    return container


# ── Exec server — embedded script + HTTP client ─────────────────────

EXEC_SERVER_SCRIPT = r'''#!/usr/bin/env python3
"""Zero-dependency HTTP exec server for sandbox containers."""
from __future__ import annotations
import base64, json, os, shutil, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
_BASH = shutil.which("bash")
_PORT = int(os.environ.get("TB_EXEC_PORT", "19542"))
_BIND = os.environ.get("TB_EXEC_BIND", "127.0.0.1")
class _H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a): pass
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/health": self._ok({"ok": True})
        elif p.path == "/download":
            qs = parse_qs(p.query)
            paths = qs.get("path", [])
            if not paths: self._err(400, "missing ?path=")
            else: self._dl(paths[0])
        else: self._err(404, f"not found: {p.path}")
    def do_POST(self):
        p = urlparse(self.path)
        body = self._body()
        if p.path == "/exec": self._exec(body)
        elif p.path == "/upload": self._up(body)
        else: self._err(404, f"not found: {p.path}")
    def _exec(self, b):
        cmd = b.get("cmd")
        if not cmd: self._err(400, "missing 'cmd'"); return
        t = b.get("timeout", 300)
        try:
            cp = subprocess.run(cmd, shell=True, executable=_BASH, capture_output=True, timeout=t)
            self._ok({"stdout": cp.stdout.decode("utf-8", errors="replace"),
                       "stderr": cp.stderr.decode("utf-8", errors="replace"),
                       "rc": cp.returncode})
        except subprocess.TimeoutExpired:
            self._ok({"stdout":"","stderr":f"timed out after {t}s","rc":124})
        except Exception as e:
            self._ok({"stdout":"","stderr":str(e),"rc":-1})
    def _up(self, b):
        path, c = b.get("path"), b.get("content")
        if not path or c is None: self._err(400, "missing path/content"); return
        try:
            data = base64.b64decode(c)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f: f.write(data)
            m = b.get("mode")
            if m: os.chmod(path, int(m, 8))
            self._ok({"ok": True})
        except Exception as e: self._err(500, str(e))
    def _dl(self, path):
        if not os.path.isfile(path): self._err(404, f"not found: {path}"); return
        try:
            with open(path, "rb") as f: data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        except Exception as e: self._err(500, str(e))
    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n == 0: return {}
        try: return json.loads(self.rfile.read(n))
        except Exception: return {}
    def _ok(self, obj):
        p = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(p)))
        self.end_headers(); self.wfile.write(p)
    def _err(self, code, msg):
        p = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(p)))
        self.end_headers(); self.wfile.write(p)
if __name__ == "__main__":
    s = ThreadingHTTPServer((_BIND, _PORT), _H)
    print(f"exec_server on {_BIND}:{_PORT}", flush=True)
    try: s.serve_forever()
    except KeyboardInterrupt: pass
    finally: s.server_close()
'''

_EXEC_SERVER_B64 = base64.b64encode(EXEC_SERVER_SCRIPT.encode()).decode()

_TRANSIENT_ERRORS = (
    ConnectionResetError,
    ConnectionRefusedError,
    ConnectionAbortedError,
    BrokenPipeError,
    TimeoutError,
    OSError,
)


class ExecClient:
    """Async HTTP client for the exec server (through the SSH tunnel).

    Methods are coroutines so concurrent requests occupy event-loop slots rather than executor
    threads, scaling past asyncio's default thread-pool cap.
    """

    def __init__(self, *, port: int, connect_timeout: float = 30.0) -> None:
        self._base = f"http://127.0.0.1:{port}"
        self._timeout = connect_timeout
        # Lazy: aiohttp.ClientSession needs a running loop, but ExecClient is built inside a
        # to_thread; create it on the first request.
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    async def exec(self, cmd: str, *, timeout: int = 300) -> ExecResult:
        resp = await self._post("/exec", {"cmd": cmd, "timeout": timeout})
        return ExecResult(
            stdout=resp.get("stdout", ""),
            stderr=resp.get("stderr", ""),
            return_code=resp.get("rc", -1),
        )

    async def upload(
        self, remote_path: str, data: bytes | Path, *, mode: str | None = None, max_retries: int = 3
    ) -> None:
        if isinstance(data, Path):
            data = data.read_bytes()
        body: dict[str, Any] = {"path": remote_path, "content": base64.b64encode(data).decode()}
        if mode is not None:
            body["mode"] = mode
        payload_mb = len(body["content"]) / (1024 * 1024)
        upload_timeout = max(self._timeout, 60.0 + payload_mb * 2.0)
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = await self._post("/upload", body, timeout_override=upload_timeout)
                if not resp.get("ok"):
                    raise RuntimeError(f"upload to {remote_path} failed: {resp}")
                return
            except (TimeoutError, OSError, RuntimeError) as exc:
                last_err = exc
                if attempt < max_retries:
                    logger.warning("upload %s attempt %d/%d: %s", remote_path, attempt, max_retries, exc)
                    await asyncio.sleep(2.0 * attempt)
        raise RuntimeError(f"upload to {remote_path} failed after {max_retries} attempts: {last_err}")

    async def download(self, remote_path: str, *, max_retries: int = 3) -> bytes:
        import urllib.parse

        url = f"{self._base}/download?path={urllib.parse.quote(remote_path)}"
        return await self._request(
            label=f"download {remote_path}", url=url, method="GET", timeout=self._timeout, max_retries=max_retries
        )

    async def health(self) -> bool:
        try:
            await self._request(label="health", url=f"{self._base}/health", method="GET", timeout=5, max_retries=1)
            return True
        except (ConnectionError, OSError, TimeoutError, RuntimeError):
            return False

    async def _post(
        self, path: str, body: dict[str, Any], *, timeout_override: float | None = None, max_retries: int = 4
    ) -> dict[str, Any]:
        url = f"{self._base}{path}"
        payload = json.dumps(body).encode()
        if timeout_override is not None:
            http_timeout = timeout_override
        else:
            cmd_timeout = body.get("timeout")
            http_timeout = (
                max(self._timeout, cmd_timeout + 30) if isinstance(cmd_timeout, (int, float)) else self._timeout
            )
        raw = await self._request(
            label=f"POST {path}",
            url=url,
            method="POST",
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=http_timeout,
            max_retries=max_retries,
        )
        return json.loads(raw)

    async def _request(
        self,
        *,
        label: str,
        url: str,
        method: str,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: float,
        max_retries: int,
    ) -> bytes:
        session = await self._ensure_session()
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                async with session.request(method, url, data=data, headers=headers, timeout=client_timeout) as resp:
                    body = await resp.read()
                    if resp.status >= 400:
                        raise RuntimeError(f"{label} failed (HTTP {resp.status}): {body.decode(errors='replace')}")
                    return body
            except RuntimeError:
                raise
            except (aiohttp.ClientError, TimeoutError, OSError) as exc:
                last_err = exc
                if attempt < max_retries:
                    wait = min(15.0, 2.0 ** (attempt - 1))
                    logger.warning("%s attempt %d/%d: %s — retry in %.1fs", label, attempt, max_retries, exc, wait)
                    await asyncio.sleep(wait)
                    continue
                raise ConnectionError(f"{label} failed after {max_retries} attempts: {last_err}") from last_err
        raise ConnectionError(f"{label} unreachable")


# ── Image builder — AWS CodeBuild + ECR caching ─────────────────────


class ImageBuilder:
    """Build Docker images via CodeBuild → ECR with content-hash caching."""

    _lock = threading.Lock()
    _inflight_builds: dict[str, threading.Event] = {}
    _build_semaphore: threading.Semaphore | None = None
    _build_semaphore_size: int = 0

    @staticmethod
    def get_ecr_image_tag(environment_dir: str | Path, environment_name: str) -> str:
        h = hashlib.sha256()
        root = Path(environment_dir)
        for p in sorted(root.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(root)).encode())
                h.update(p.read_bytes())
        return f"{environment_name}__{h.hexdigest()[:8]}"

    @staticmethod
    def image_exists_in_ecr(ecr_repository: str, tag: str, region: str | None = None) -> bool:
        boto3, _, ClientError = _require_aws_sdks()
        ecr_region = ImageBuilder._ecr_region(ecr_repository, fallback=region)
        ecr = boto3.client("ecr", region_name=ecr_region)
        repo_name = ecr_repository.split("/", 1)[1] if "/" in ecr_repository else ecr_repository
        try:
            _retry_with_backoff(
                lambda: ecr.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": tag}]),
                operation_name="ecr.describe_images",
                max_retries=5,
            )
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code == "ImageNotFoundException":
                return False
            if code == "RepositoryNotFoundException":
                raise RuntimeError(
                    f"ECR repository {ecr_repository!r} does not exist; create it (e.g. via Terraform) "
                    f"before building or mirroring images"
                ) from exc
            raise

    @staticmethod
    def _ecr_region(ecr_repository: str, fallback: str | None = None) -> str | None:
        """Extract the region from an ECR repo URL like '123.dkr.ecr.us-west-2.amazonaws.com/repo'."""
        parts = ecr_repository.split(".")
        if len(parts) >= 4 and parts[1] == "dkr" and parts[2] == "ecr":
            return parts[3]
        return fallback

    @staticmethod
    def list_ecr_tags(ecr_repository: str, region: str | None = None) -> set[str]:
        """Return all image tags present in an ECR repository (paginated ``list_images``)."""
        boto3, _, ClientError = _require_aws_sdks()
        ecr_region = ImageBuilder._ecr_region(ecr_repository, fallback=region)
        ecr = boto3.client("ecr", region_name=ecr_region)
        repo_name = ecr_repository.split("/", 1)[1] if "/" in ecr_repository else ecr_repository

        def _fetch_all_tags() -> set[str]:
            tags: set[str] = set()
            paginator = ecr.get_paginator("list_images")
            for page in paginator.paginate(
                repositoryName=repo_name,
                filter={"tagStatus": "TAGGED"},
            ):
                for img_id in page.get("imageIds", []):
                    if tag := img_id.get("imageTag"):
                        tags.add(tag)
            return tags

        try:
            return _retry_with_backoff(_fetch_all_tags, operation_name="ecr.list_images", max_retries=5)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "RepositoryNotFoundException":
                return set()
            raise

    @staticmethod
    def ecr_docker_login(ecr_repository: str, region: str | None = None) -> None:
        """Authenticate the local Docker daemon against an ECR registry."""
        ecr_region = ImageBuilder._ecr_region(ecr_repository, fallback=region)
        registry = ecr_repository.split("/")[0]
        region_flag = f" --region {ecr_region}" if ecr_region else ""
        cmd = f"aws ecr get-login-password{region_flag} | docker login --username AWS --password-stdin {registry}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ECR docker login failed: {result.stderr.strip()}")
        logger.info("ECR docker login succeeded for %s", registry)

    @staticmethod
    def docker_push_to_ecr(local_image: str, ecr_repository: str, tag: str) -> str:
        """Tag a local Docker image and push it to ECR. Returns the ECR URL."""
        ecr_url = f"{ecr_repository}:{tag}"
        subprocess.run(["docker", "tag", local_image, ecr_url], check=True, capture_output=True)
        result = subprocess.run(["docker", "push", ecr_url], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"docker push {ecr_url} failed: {result.stderr.strip()}")
        logger.info("Pushed %s -> %s", local_image, ecr_url)
        return ecr_url

    @classmethod
    def _run_deduped_build(
        cls, *, cfg: EcsFargateConfig, tag: str, image_url: str, force: bool, build: Callable[[], None]
    ) -> str:
        """Run ``build`` for ``tag`` at most once across threads, returning the ECR ``image_url``.

        Concurrent callers wait on a shared in-flight event outside ``cls._lock`` (so one waiter
        can't stall other tags' builds), then confirm via ECR -- a failed build also sets the event.
        """
        ecr_repo = cfg.ecr_repository
        if not force and cls.image_exists_in_ecr(ecr_repo, tag, cfg.region):
            logger.info("ECR cache hit — skipping build: %s", image_url)
            return image_url

        with cls._lock:
            event = cls._inflight_builds.get(tag)
            is_builder = event is None
            if is_builder:
                event = threading.Event()
                cls._inflight_builds[tag] = event
            if cls._build_semaphore is None:
                cls._build_semaphore = threading.Semaphore(cfg.build_parallelism)
                cls._build_semaphore_size = cfg.build_parallelism
            elif cls._build_semaphore_size != cfg.build_parallelism:
                # Init once: never replace a live semaphore (would drift the permit cap).
                logger.warning(
                    "build_parallelism=%d ignored; build semaphore already initialized at %d",
                    cfg.build_parallelism,
                    cls._build_semaphore_size,
                )
            semaphore = cls._build_semaphore

        if not is_builder:
            event.wait()  # wait outside the lock so one waiter can't block other tags' builds
            if cls.image_exists_in_ecr(ecr_repo, tag, cfg.region):
                return image_url
            raise RuntimeError(f"Concurrent build for {image_url} failed; image is not in ECR")

        try:
            semaphore.acquire()
            try:
                if not force and cls.image_exists_in_ecr(ecr_repo, tag, cfg.region):
                    return image_url
                build()
                return image_url
            finally:
                semaphore.release()
        finally:
            with cls._lock:
                cls._inflight_builds.pop(tag, None)
            event.set()

    @classmethod
    def ensure_image_built(cls, *, cfg: EcsFargateConfig, environment_name: str, force_build: bool = False) -> str:
        ecr_repo = cfg.ecr_repository
        env_dir = cfg.environment_dir
        if not ecr_repo or not env_dir:
            raise ValueError("ecr_repository and environment_dir are required for image building")
        tag = cls.get_ecr_image_tag(env_dir, environment_name)
        image_url = f"{ecr_repo}:{tag}"
        return cls._run_deduped_build(
            cfg=cfg,
            tag=tag,
            image_url=image_url,
            force=force_build,
            build=lambda: cls._build_and_push(
                cfg=cfg, environment_name=environment_name, tag=tag, image_url=image_url
            ),
        )

    @classmethod
    def ensure_mirrored(cls, *, cfg: EcsFargateConfig, src_image: str, force: bool = False) -> str:
        """Ensure ``src_image`` is present in the ECR mirror tag, mirroring it on demand if not.

        Mirrors via a self-contained CodeBuild job (privileged DinD: login, pull, retag, push).
        Concurrent callers dedup on a shared in-flight event, like :meth:`ensure_image_built`.
        """
        ecr_repo = cfg.ecr_repository
        if not ecr_repo:
            raise ValueError("ecr_repository is required to mirror images")
        _validate_image_ref(src_image)
        tag = _sanitize_id(src_image)
        image_url = f"{ecr_repo}:{tag}"

        def _mirror() -> None:
            logger.info("Mirroring %s -> %s via CodeBuild ...", src_image, image_url)
            buildspec = cls._generate_mirror_buildspec(cfg, src_image, image_url)
            cls.run_buildspec_via_codebuild(
                cfg=cfg,
                buildspec=buildspec,
                job_label=f"mirror::{tag}",
                timeout_minutes=cfg.codebuild_build_timeout,
            )
            logger.info("Mirrored OK: %s -> %s", src_image, image_url)

        return cls._run_deduped_build(cfg=cfg, tag=tag, image_url=image_url, force=force, build=_mirror)

    @staticmethod
    def _generate_mirror_buildspec(cfg: EcsFargateConfig, src_image: str, ecr_url: str) -> str:
        ecr_registry = (cfg.ecr_repository or "").split("/")[0]
        ecr_region = ImageBuilder._ecr_region(cfg.ecr_repository or "", fallback="$AWS_DEFAULT_REGION")
        pre_build_cmds = [
            f"aws ecr get-login-password --region {ecr_region}"
            f" | docker login --username AWS --password-stdin {ecr_registry}",
        ]
        if cfg.dockerhub_secret_arn:
            pre_build_cmds.append(
                f"DOCKERHUB_CREDS=$(aws secretsmanager get-secret-value"
                f" --secret-id {cfg.dockerhub_secret_arn}"
                f" --query SecretString --output text --region $AWS_DEFAULT_REGION)"
                f' && DH_USER=$(echo "$DOCKERHUB_CREDS" | python3 -c'
                """ "import sys,json;print(json.load(sys.stdin)['username'])")"""
                f' && if [ -n "$DH_USER" ]; then echo "$DOCKERHUB_CREDS" | python3 -c'
                """ "import sys,json;print(json.load(sys.stdin)['password'])" """
                f'| docker login -u "$DH_USER" --password-stdin; fi'
                f' || echo "Docker Hub login failed — continuing without auth"'
            )
        pull_cmd = (
            f"for i in 1 2 3; do docker pull --platform linux/amd64 {src_image} && break; "
            f'echo "pull failed ($i/3), retry in 30s"; sleep 30; done'
        )
        pre_yaml = "\n".join(f"      - {c}" for c in pre_build_cmds)
        return (
            "version: 0.2\nphases:\n  pre_build:\n    commands:\n"
            f"{pre_yaml}\n  build:\n    commands:\n"
            f"      - {pull_cmd}\n"
            f"      - docker tag {src_image} {ecr_url}\n"
            f"  post_build:\n    commands:\n      - docker push {ecr_url}\n"
        )

    @staticmethod
    def _upload_build_context(cfg: EcsFargateConfig, environment_name: str, nonce: str) -> str:
        boto3, *_ = _require_aws_sdks()
        env_dir = Path(cfg.environment_dir or ".")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in env_dir.rglob("*"):
                if item.is_file():
                    zf.write(item, arcname=str(item.relative_to(env_dir)))
        buf.seek(0)
        s3 = boto3.client("s3", region_name=cfg.region)
        s3_prefix = cfg.s3_prefix or "ecs-sandbox"
        s3_key = f"{s3_prefix}/codebuild/{environment_name}-{nonce}.zip"
        body = buf.read()
        _retry_with_backoff(
            lambda: s3.put_object(Bucket=cfg.s3_bucket, Key=s3_key, Body=body),
            operation_name="s3.put_object(build_context)",
            max_retries=5,
        )
        return s3_key

    @staticmethod
    def _resolve_codebuild_project(cfg: EcsFargateConfig, cb: Any) -> str:
        _, _, ClientError = _require_aws_sdks()
        if cfg.codebuild_project:
            return cfg.codebuild_project
        if not cfg.codebuild_service_role:
            raise RuntimeError("codebuild_project or codebuild_service_role is required")
        # One shared project (created idempotently); source + buildspec are overridden per build,
        # so a generic NO_SOURCE project serves every build instead of accumulating per-build ones.
        project_name = "ecs-sandbox-build"
        try:
            _retry_with_backoff(
                lambda: cb.create_project(
                    name=project_name,
                    source={"type": "NO_SOURCE", "buildspec": "version: 0.2"},
                    artifacts={"type": "NO_ARTIFACTS"},
                    environment={
                        "type": "LINUX_CONTAINER",
                        "image": "aws/codebuild/amazonlinux-x86_64-standard:5.0",
                        "computeType": cfg.codebuild_compute_type,
                        "privilegedMode": True,
                    },
                    serviceRole=cfg.codebuild_service_role,
                    timeoutInMinutes=cfg.codebuild_build_timeout,
                ),
                operation_name="codebuild.create_project",
                max_retries=5,
            )
        except ClientError as e:
            if "already exists" not in str(e).lower():
                raise
        return project_name

    @staticmethod
    def _generate_buildspec(cfg: EcsFargateConfig, repo_name: str, tag: str, image_url: str) -> str:
        ecr_registry = (cfg.ecr_repository or "").split("/")[0]
        ecr_region = ImageBuilder._ecr_region(cfg.ecr_repository or "", fallback="$AWS_DEFAULT_REGION")
        pre_build_cmds = [
            f"aws ecr get-login-password --region {ecr_region}"
            f" | docker login --username AWS --password-stdin {ecr_registry}",
        ]
        if cfg.dockerhub_secret_arn:
            pre_build_cmds.append(
                f"DOCKERHUB_CREDS=$(aws secretsmanager get-secret-value"
                f" --secret-id {cfg.dockerhub_secret_arn}"
                f" --query SecretString --output text --region $AWS_DEFAULT_REGION)"
                f' && DH_USER=$(echo "$DOCKERHUB_CREDS" | python3 -c'
                """ "import sys,json;print(json.load(sys.stdin)['username'])")"""
                f' && if [ -n "$DH_USER" ]; then echo "$DOCKERHUB_CREDS" | python3 -c'
                """ "import sys,json;print(json.load(sys.stdin)['password'])" """
                f'| docker login -u "$DH_USER" --password-stdin; fi'
                f' || echo "Docker Hub login failed — continuing without auth"'
            )
        pre_yaml = "\n".join(f"      - {c}" for c in pre_build_cmds)
        build_cmd = (
            f"for i in 1 2 3; do docker build -t {repo_name}:{tag} . && break; "
            f'echo "build failed ($i/3), retry in 30s"; sleep 30; done'
        )
        return (
            "version: 0.2\nphases:\n  pre_build:\n    commands:\n"
            f"{pre_yaml}\n  build:\n    commands:\n"
            f"      - {build_cmd}\n      - docker tag {repo_name}:{tag} {image_url}\n"
            f"  post_build:\n    commands:\n      - docker push {image_url}\n"
        )

    @staticmethod
    def _poll_codebuild(cb: Any, build_id: str, image_url: str) -> None:
        while True:
            time.sleep(10 + random.uniform(0, 5))
            build = _retry_with_backoff(
                lambda: cb.batch_get_builds(ids=[build_id])["builds"][0],
                operation_name=f"BatchGetBuilds({build_id})",
                max_retries=8,
                base_delay=2.0,
                max_delay=120.0,
            )
            status = build["buildStatus"]
            if status == "SUCCEEDED":
                logger.info("CodeBuild succeeded: %s", build_id)
                return
            if status in ("FAILED", "FAULT", "STOPPED", "TIMED_OUT"):
                phases = build.get("phases", [])
                failed = [p for p in phases if p.get("phaseStatus") not in (None, "SUCCEEDED")]
                ctx = "; ".join(f"{p['phaseType']}: {p.get('phaseStatus')}" for p in failed) or status
                raise RuntimeError(f"CodeBuild failed for {image_url}: {ctx} (build: {build_id})")
            logger.debug("CodeBuild %s — phase=%s status=%s", build_id, build.get("currentPhase"), status)

    @classmethod
    def _build_and_push(cls, *, cfg: EcsFargateConfig, environment_name: str, tag: str, image_url: str) -> None:
        boto3, *_ = _require_aws_sdks()
        ecr_repo = cfg.ecr_repository or ""
        repo_name = ecr_repo.split("/", 1)[1] if "/" in ecr_repo else ecr_repo
        nonce = uuid.uuid4().hex[:8]
        logger.info("Building image via CodeBuild: %s", image_url)
        s3_key = cls._upload_build_context(cfg, environment_name, nonce)
        try:
            cb = boto3.client("codebuild", region_name=cfg.region)
            project_name = cls._resolve_codebuild_project(cfg, cb)
            buildspec = cls._generate_buildspec(cfg, repo_name, tag, image_url)
            resp = _retry_with_backoff(
                lambda: cb.start_build(
                    projectName=project_name,
                    sourceTypeOverride="S3",
                    sourceLocationOverride=f"{cfg.s3_bucket}/{s3_key}",
                    buildspecOverride=buildspec,
                    timeoutInMinutesOverride=cfg.codebuild_build_timeout,
                    privilegedModeOverride=True,
                    environmentTypeOverride="LINUX_CONTAINER",
                    imageOverride="aws/codebuild/amazonlinux-x86_64-standard:5.0",
                    computeTypeOverride=cfg.codebuild_compute_type,
                ),
                operation_name="codebuild.start_build",
                max_retries=5,
            )
            build_id = resp["build"]["id"]
            logger.info("CodeBuild started: %s", build_id)
            cls._poll_codebuild(cb, build_id, image_url)
        finally:
            # The build-context zip is only needed during the CodeBuild job.
            _delete_s3_object(cfg, s3_key)

    @classmethod
    def run_buildspec_via_codebuild(
        cls,
        *,
        cfg: EcsFargateConfig,
        buildspec: str,
        job_label: str = "harness-build",
        timeout_minutes: int | None = None,
    ) -> None:
        """Run a self-contained buildspec via CodeBuild (privileged DinD, ``NO_SOURCE``)."""
        boto3, *_ = _require_aws_sdks()
        cb = boto3.client("codebuild", region_name=cfg.region)
        project_name = cls._resolve_codebuild_project(cfg, cb)
        timeout = timeout_minutes or cfg.codebuild_build_timeout

        logger.info("Starting CodeBuild harness build: %s (timeout=%dm)", job_label, timeout)
        resp = _retry_with_backoff(
            lambda: cb.start_build(
                projectName=project_name,
                sourceTypeOverride="NO_SOURCE",
                buildspecOverride=buildspec,
                timeoutInMinutesOverride=timeout,
                privilegedModeOverride=True,
                environmentTypeOverride="LINUX_CONTAINER",
                imageOverride="aws/codebuild/amazonlinux-x86_64-standard:5.0",
                computeTypeOverride=cfg.codebuild_compute_type,
            ),
            operation_name="codebuild.start_build(harness)",
            max_retries=5,
        )
        build_id = resp["build"]["id"]
        logger.info("CodeBuild harness build started: %s (build=%s)", job_label, build_id)
        cls._poll_codebuild(cb, build_id, job_label)


# ── Core sandbox ─────────────────────────────────────────────────────

_active_sandboxes: dict[int, Any] = {}
_cleanup_lock = threading.RLock()
_atexit_registered = False
_PROCESS_NONCE = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
_exec_server_url_cache: dict[str, str] = {}

_task_def_cache: dict[str, str] = {}
_task_def_cache_lock = threading.Lock()
_task_def_inflight: dict[str, threading.Event] = {}

# Env keys whose values vary per invocation. Routed via RunTask containerOverrides (not baked into
# the task def) so the task-def hash cache stays stable across invocations. OutsideEndpoint routing
# keys are merged in dynamically at call time; this set holds the rest.
_PER_INVOCATION_ENV_KEYS: frozenset[str] = frozenset({"_NEL_EFS_SESSION"})


def _compute_task_def_hash(payload: dict[str, Any]) -> str:
    # Strip logConfiguration before hashing: it's a visibility annotation with no behavioral
    # effect, so task defs differing only in log group/prefix should share a cache entry.
    def _strip_log_cfg(containers: list) -> list:
        return [{k: v for k, v in c.items() if k != "logConfiguration"} for c in containers]

    canonical = {
        k: (_strip_log_cfg(v) if k == "containerDefinitions" else v) for k, v in payload.items() if k != "family"
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def _emergency_cleanup() -> None:
    with _cleanup_lock:
        for sb in list(_active_sandboxes.values()):
            try:
                sb._sync_stop()
            except Exception:
                logger.debug("Emergency cleanup failed for sandbox %s", id(sb), exc_info=True)


class EcsFargateSandbox:
    """ECS Fargate sandbox — async :class:`Sandbox` protocol."""

    def __init__(self, spec: SandboxSpec, *, ecs_config: EcsFargateConfig) -> None:
        self._spec = spec
        self._cfg = ecs_config
        self._task_arn: str | None = None
        self._task_def_arn: str | None = None
        self._task_ip: str | None = None
        self._ssh_key_file: str | None = None
        self._ssh_tunnel: SshTunnel | None = None
        self._exec_client: ExecClient | None = None
        self._started = False
        self._stopped = False
        self._ecs: Any = None
        self._ec2: Any = None
        self._ssm: Any = None
        self._runtime_container_env: dict[str, str] = {}
        self._s3_artifacts: list[str] = []  # transient S3 keys to delete on cleanup
        self._ssh_tunnel_port: int | None = None
        self._agent_forward_port: int | None = None
        self._outside_endpoints: list[OutsideEndpoint] = []
        self._outside_endpoint_routing = _OutsideEndpointRouting.empty()
        self._run_id = uuid.uuid4().hex[:12]

    # ── Protocol properties ──────────────────────────────────────────

    @property
    def spec(self) -> SandboxSpec:
        return self._spec

    def resolved_endpoint_url(self, env_var: str) -> str | None:
        return self._outside_endpoint_routing.resolved_endpoint_url(env_var)

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped

    @property
    def container_ip(self) -> str | None:
        return self._task_ip

    # ── Extra properties ─────────────────────────────────────────────

    @property
    def task_arn(self) -> str | None:
        return self._task_arn

    @property
    def local_port(self) -> int | None:
        if self._ssh_tunnel:
            try:
                return self._ssh_tunnel.local_port
            except RuntimeError:
                pass
        return None

    @property
    def ssh_tunnel(self) -> SshTunnel | None:
        return self._ssh_tunnel

    @property
    def exec_client(self) -> ExecClient | None:
        return self._exec_client

    @property
    def model_tunnel_port(self) -> int | None:
        return self._ssh_tunnel_port

    # ── Protocol async lifecycle ─────────────────────────────────────

    async def start(self, *, outside_endpoints: list[OutsideEndpoint] | None = None) -> None:
        if self._started:
            return
        self._outside_endpoints = outside_endpoints or []
        self._outside_endpoint_routing = _OutsideEndpointRouting.empty(self._outside_endpoints)
        sidecar = self._cfg.ssh_sidecar
        if sidecar and sidecar.exec_server_port is None:
            _OutsideEndpointRouting.for_agent_server(self._outside_endpoints)
        try:
            await asyncio.to_thread(self._do_start)
            self._started = True
        except Exception:
            if self._exec_client is not None:
                await self._exec_client.close()
            await asyncio.to_thread(self._cleanup)
            raise

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self._exec_client is not None:
            await self._exec_client.close()
        await asyncio.to_thread(self._cleanup)
        self._unregister_from_cleanup()

    async def exec(
        self,
        command: str,
        timeout_sec: float = 180,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        self._require_exec_client()
        shell_cmd = command
        if env:
            exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
            shell_cmd = f"export {exports} && {shell_cmd}"
        if cwd:
            shell_cmd = f"cd {shlex.quote(cwd)} && {shell_cmd}"
        if user is not None:
            if isinstance(user, int):
                shell_cmd = f'su -s /bin/bash "$(getent passwd {user} | cut -d: -f1)" -c {shlex.quote(shell_cmd)}'
            else:
                shell_cmd = f"su -s /bin/bash {shlex.quote(str(user))} -c {shlex.quote(shell_cmd)}"
        try:
            return await self._exec_client.exec(shell_cmd, timeout=int(timeout_sec))  # type: ignore[union-attr]
        except ConnectionError:
            if self._ssh_tunnel and not self._ssh_tunnel.is_open:
                logger.warning("SSH tunnel dead — attempting reconnect before re-raising")
                try:
                    await self.reconnect_tunnel()
                    sidecar = self._cfg.ssh_sidecar
                    if sidecar and sidecar.exec_server_port is not None:
                        health_url = f"http://127.0.0.1:{self._ssh_tunnel.local_port}/health"  # type: ignore[union-attr]
                        # wait_ready blocks (sleep + urlopen); offload so it can't stall the event loop.
                        await asyncio.to_thread(
                            self._ssh_tunnel.wait_ready,  # type: ignore[union-attr]
                            health_url=health_url,
                            timeout=60.0,
                        )
                        old_client = self._exec_client
                        self._exec_client = ExecClient(port=self._ssh_tunnel.local_port)  # type: ignore[union-attr]
                        if old_client is not None:
                            await old_client.close()
                        return await self._exec_client.exec(shell_cmd, timeout=int(timeout_sec))
                except Exception as reconnect_err:
                    logger.warning("Tunnel reconnect failed: %s", reconnect_err)
            raise

    async def upload(self, local_path: Path, remote_path: str) -> None:
        self._require_exec_client()
        local = Path(local_path)
        if local.is_dir():
            for child in local.rglob("*"):
                if child.is_file():
                    await self.upload(child, f"{remote_path}/{child.relative_to(local)}")
            return
        if local.stat().st_size > 512 * 1024 and self._cfg.s3_bucket:
            # Map the local temp name to the requested remote filename: the S3 tar is keyed by
            # basename and extracted into dest_dir, so otherwise it lands under the local basename.
            await self._upload_via_s3(
                [local],
                os.path.dirname(remote_path) or "/tmp",
                arcnames={local: os.path.basename(remote_path)},
            )
        else:
            await self._exec_client.upload(remote_path, local)  # type: ignore[union-attr]

    async def download(self, remote_path: str, local_path: Path) -> None:
        self._require_exec_client()
        data = await self._exec_client.download(remote_path)  # type: ignore[union-attr]
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def resolve_outside_endpoint(self, url: str) -> str:
        return self._outside_endpoint_routing.resolve_url(url)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ── Extra public methods ─────────────────────────────────────────

    async def reconnect_tunnel(self) -> None:
        if self._stopped or not self._started:
            raise RuntimeError("Cannot reconnect tunnel on a stopped/unstarted sandbox")
        sidecar = self._cfg.ssh_sidecar
        if sidecar is None:
            return
        if self._ssh_tunnel:
            self._ssh_tunnel.close()
            self._ssh_tunnel = None
        await asyncio.to_thread(self._open_tunnel, sidecar)

    # ── Sync start (runs via asyncio.to_thread) ──────────────────────

    def _do_start(self) -> None:
        cfg = self._cfg
        sidecar = cfg.ssh_sidecar
        if sidecar is None:
            raise ValueError("ssh_sidecar must be configured")
        self._init_aws_clients()

        built_image: str | None = None
        env_dir = cfg.environment_dir or self._spec.environment_dir
        if cfg.ecr_repository and env_dir:
            per_task_cfg = _dc_replace(cfg, environment_dir=env_dir)
            built_image = ImageBuilder.ensure_image_built(
                cfg=per_task_cfg, environment_name=_sanitize_id(self._spec.image or "sandbox")
            )
        elif (
            cfg.auto_mirror
            and cfg.ecr_repository
            and self._spec.image
            and not cfg.image_template
            and not _is_ecr_image_ref(self._spec.image)
        ):
            # Pull the bare/public image into the ECR mirror on demand so a missing entry self-heals.
            ImageBuilder.ensure_mirrored(cfg=cfg, src_image=self._spec.image)
        image = self._resolve_image(built_image)

        if not sidecar.private_key_secret_arn or not sidecar.public_key_secret_arn:
            raise ValueError("ssh_sidecar private_key_secret_arn and public_key_secret_arn are required")
        self._ssh_key_file = download_secret_to_file(sidecar.private_key_secret_arn, cfg.region)
        ssh_public_key_value = download_secret_to_string(sidecar.public_key_secret_arn, cfg.region)

        has_exec_server = sidecar.exec_server_port is not None
        if not has_exec_server:
            self._outside_endpoint_routing = _OutsideEndpointRouting.for_agent_server(self._outside_endpoints)
            self._ssh_tunnel_port = self._outside_endpoint_routing.agent_tunnel_port
        else:
            self._outside_endpoint_routing = _OutsideEndpointRouting.for_exec_server(self._outside_endpoints, sidecar)

        command = self._build_container_command(sidecar)
        env = self._build_env_vars()
        stable_env, self._runtime_container_env = self._split_env(env)
        log_region = cfg.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        sidecar_def = build_ssh_sidecar_container(
            sidecar,
            public_key_value=ssh_public_key_value,
            max_lifetime_sec=cfg.max_task_lifetime_sec,
            log_group=cfg.log_group,
            log_region=log_region,
            log_stream_prefix=cfg.log_stream_prefix or "ecs-sandbox",
        )
        self._task_def_arn = self._register_task_definition(
            image=image, command=command, env=stable_env, sidecar_def=sidecar_def
        )
        self._task_arn = self._run_task(self._task_def_arn)
        self._register_for_cleanup()
        self._wait_for_running()
        self._task_ip = self._get_task_public_ip()
        self._wait_for_ssh_ready(self._task_ip, sidecar.sshd_port, sidecar.ssh_ready_timeout_sec)
        self._open_tunnel(sidecar)

        if has_exec_server:
            health_url = f"http://127.0.0.1:{self._ssh_tunnel.local_port}/health"  # type: ignore[union-attr]
            self._ssh_tunnel.wait_ready(health_url=health_url, timeout=sidecar.ssh_ready_timeout_sec)  # type: ignore[union-attr]
            self._exec_client = ExecClient(port=self._ssh_tunnel.local_port)  # type: ignore[union-attr]

    # ── Internal helpers ─────────────────────────────────────────────

    def _init_aws_clients(self) -> None:
        boto3, Config, _ = _require_aws_sdks()
        boto_cfg = Config(connect_timeout=30, read_timeout=60, retries={"max_attempts": 8, "mode": "adaptive"})
        self._ecs = boto3.client("ecs", region_name=self._cfg.region, config=boto_cfg)
        self._ec2 = boto3.client("ec2", region_name=self._cfg.region, config=boto_cfg)
        self._ssm = boto3.client("ssm", region_name=self._cfg.region, config=boto_cfg)

    def _resolve_image(self, built_image: str | None = None) -> str:
        if built_image:
            return built_image
        cfg = self._cfg
        if cfg.image_template:
            sanitized = _sanitize_id(self._spec.image or "sandbox")
            fmt_keys = {
                "task_id": self._spec.image or "",
                "task_id_sanitized": sanitized,
                **(self._spec.env or {}),
            }
            try:
                return cfg.image_template.format_map(fmt_keys)
            except KeyError as exc:
                raise ValueError(
                    f"ecs.image_template placeholder {exc} not found in "
                    f"available keys: {sorted(fmt_keys)}. "
                    f"Hint: use sandbox.image_template (resolved via seed "
                    f"metadata) instead of ecs.image_template for task-specific "
                    f"placeholders like {{task_id}}."
                ) from exc
        if self._spec.image:
            # An existing ECR reference is used verbatim; re-prefixing it would corrupt the tag.
            if _is_ecr_image_ref(self._spec.image):
                return self._spec.image
            # Bare/public names route to the ECR mirror tag (avoids origin-registry rate limits).
            if cfg.ecr_repository:
                return f"{cfg.ecr_repository}:{_sanitize_id(self._spec.image)}"
            return self._spec.image
        if not cfg.task_definition:
            raise ValueError(
                "No image available: set image on SandboxSpec, image_template, "
                "ecr_repository + environment_dir, or task_definition"
            )
        return ""

    def _upload_exec_server(self) -> str:
        cfg = self._cfg
        if not cfg.s3_bucket:
            raise ValueError("s3_bucket is required for exec server upload")
        cache_key = f"{cfg.s3_bucket}/{self._run_id}"
        if cache_key in _exec_server_url_cache:
            return _exec_server_url_cache[cache_key]
        boto3, *_ = _require_aws_sdks()
        s3 = boto3.client("s3", region_name=cfg.region)
        prefix = cfg.s3_prefix or "ecs-sandbox"
        key = f"{prefix}/{self._run_id}-{_PROCESS_NONCE}/_exec_server/exec_server.py"
        _retry_with_backoff(
            lambda: s3.put_object(Bucket=cfg.s3_bucket, Key=key, Body=EXEC_SERVER_SCRIPT.encode()),
            operation_name="s3.put_object(exec_server)",
            max_retries=5,
        )
        url = s3.generate_presigned_url("get_object", Params={"Bucket": cfg.s3_bucket, "Key": key}, ExpiresIn=21600)
        _exec_server_url_cache[cache_key] = url
        logger.info("Uploaded exec server → s3://%s/%s", cfg.s3_bucket, key)
        return url

    def _build_container_command(self, sidecar: SshSidecarConfig) -> list[str] | None:
        if sidecar.exec_server_port is None:
            return None
        exec_port = sidecar.exec_server_port or 19542
        hostname = re.sub(r"[^A-Za-z0-9._-]", "-", self._spec.image or "sandbox")[:63]
        setup = (
            f"hostname {shlex.quote(hostname)} 2>/dev/null || true; "
            f"echo '{_EXEC_SERVER_B64}' | base64 -d > /tmp/_exec_server.py; "
            "if ! command -v python3 >/dev/null 2>&1; then "
            "  if command -v apt-get >/dev/null 2>&1; then "
            "    apt-get update -qq && apt-get install -y -qq --no-install-recommends python3 bash; "
            "  elif command -v apk >/dev/null 2>&1; then "
            "    apk add --no-cache python3 bash; "
            "  elif command -v yum >/dev/null 2>&1; then "
            "    yum install -y python3 bash; "
            "  elif command -v dnf >/dev/null 2>&1; then "
            "    dnf install -y python3 bash; "
            "  fi; "
            "fi; "
            "if ! command -v python3 >/dev/null 2>&1; then "
            "  echo 'FATAL: exec_server bootstrap failed — python3 not available' >&2; "
            "  exit 1; "
            "fi; "
            f"TB_EXEC_PORT={exec_port} TB_EXEC_BIND=127.0.0.1 "
            "exec python3 /tmp/_exec_server.py"
        )
        return ["sh", "-lc", setup]

    def _build_env_vars(self) -> dict[str, str]:
        env: dict[str, str] = dict(self._spec.env)
        cfg = self._cfg
        if cfg.extra_env:
            for k, v in cfg.extra_env.items():
                env[k] = self._render_env_value(v)
        env.update(self._outside_endpoint_routing.env_overrides())
        return env

    def _split_env(self, env: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
        runtime_keys = _PER_INVOCATION_ENV_KEYS | set(self._outside_endpoint_routing.env_overrides().keys())
        stable = {k: v for k, v in env.items() if k not in runtime_keys}
        runtime = {k: v for k, v in env.items() if k in runtime_keys}
        return stable, runtime

    def _render_env_value(self, value: str) -> str:
        if self._ssh_tunnel_port is not None:
            value = value.replace("{ssh_tunnel_port}", str(self._ssh_tunnel_port))
        if self._task_ip:
            value = value.replace("{task_ip}", self._task_ip)
        elif "{task_ip}" in value:
            # _task_ip is only known after the task is RUNNING, i.e. after env is baked into the
            # task-def; fail loudly rather than baking a literal "{task_ip}".
            raise ValueError(
                "{task_ip} is not available in container env: the task IP is only known after the "
                "sandbox starts. Resolve the container's own address at runtime instead."
            )
        value = value.replace("{image}", self._spec.image or "")
        return value

    # ── Task definition registration ─────────────────────────────────

    def _register_task_definition(
        self, *, image: str, command: list[str] | None, env: dict[str, str], sidecar_def: dict[str, Any]
    ) -> str:
        cfg = self._cfg
        log_region = cfg.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        log_cfg: dict[str, Any] | None = None
        if cfg.log_group:
            log_cfg = {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": cfg.log_group,
                    "awslogs-region": log_region,
                    "awslogs-stream-prefix": cfg.log_stream_prefix or "ecs-sandbox",
                    "awslogs-create-group": "true",
                },
            }

        _, _, ClientError = _require_aws_sdks()
        base: dict[str, Any] | None = None
        if cfg.task_definition:
            try:
                base = _retry_with_backoff(
                    lambda: self._ecs.describe_task_definition(taskDefinition=cfg.task_definition)["taskDefinition"],
                    operation_name="ecs.describe_task_definition",
                    max_retries=5,
                )
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") == "ClientException":
                    logger.warning("Base task definition %s not found, registering from scratch", cfg.task_definition)
                else:
                    raise

        if base is not None:
            return self._register_from_base(
                base=base, image=image, command=command, env=env, sidecar_def=sidecar_def, log_cfg=log_cfg
            )
        return self._register_from_scratch(
            image=image, command=command, env=env, sidecar_def=sidecar_def, log_cfg=log_cfg
        )

    def _register_from_base(
        self,
        *,
        base: dict,
        image: str,
        command: list[str] | None,
        env: dict[str, str],
        sidecar_def: dict,
        log_cfg: dict | None,
    ) -> str:
        cfg = self._cfg
        containers = list(base.get("containerDefinitions") or [])
        target = next((cd for cd in containers if cd.get("name") == cfg.container_name), None)
        if target is None:
            raise RuntimeError(
                f"Base task-def has no container '{cfg.container_name}'. "
                f"Available: {[c.get('name') for c in containers]}"
            )
        if image:
            target["image"] = image
        if command is not None:
            target["command"] = command
            target.pop("entryPoint", None)
        if env:
            existing = {e["name"]: e["value"] for e in target.get("environment", [])}
            existing.update(env)
            target["environment"] = [{"name": k, "value": v} for k, v in sorted(existing.items())]
        if log_cfg:
            target["logConfiguration"] = log_cfg
        target["dependsOn"] = [{"containerName": "ssh-tunnel", "condition": "HEALTHY"}]
        containers = [c for c in containers if c.get("name") != "ssh-tunnel"]
        containers.append(sidecar_def)

        task_volumes, mount_points = self._build_efs_volumes()
        if mount_points:
            existing_mounts = target.get("mountPoints") or []
            target["mountPoints"] = existing_mounts + mount_points

        family = self._make_family_name()
        cpu = max(int(base.get("cpu") or "256"), int(cfg.cpu))
        memory = max(int(base.get("memory") or "512"), int(cfg.memory))
        _validate_fargate_cpu_memory(cpu, memory)
        payload: dict[str, Any] = {
            "family": family,
            "networkMode": base.get("networkMode", "awsvpc"),
            "requiresCompatibilities": base.get("requiresCompatibilities", ["FARGATE"]),
            "cpu": str(cpu),
            "memory": str(memory),
            "containerDefinitions": containers,
        }
        ephemeral = _ephemeral_storage_block(
            (base.get("ephemeralStorage") or {}).get("sizeInGiB") or cfg.ephemeral_storage_gib
        )
        if ephemeral is not None:
            payload["ephemeralStorage"] = ephemeral
        for k in ("taskRoleArn", "executionRoleArn", "runtimePlatform", "volumes"):
            if base.get(k) is not None:
                payload[k] = base[k]
        if task_volumes:
            existing_vols = payload.get("volumes") or []
            payload["volumes"] = existing_vols + task_volumes
        if cfg.execution_role_arn:
            payload["executionRoleArn"] = cfg.execution_role_arn
        if cfg.task_role_arn:
            payload["taskRoleArn"] = cfg.task_role_arn
        return self._do_register(payload)

    def _register_from_scratch(
        self, *, image: str, command: list[str] | None, env: dict[str, str], sidecar_def: dict, log_cfg: dict | None
    ) -> str:
        cfg = self._cfg
        if not cfg.execution_role_arn:
            raise RuntimeError("execution_role_arn required when no base task definition provided")
        container_def: dict[str, Any] = {
            "name": cfg.container_name,
            "essential": True,
            "dependsOn": [{"containerName": "ssh-tunnel", "condition": "HEALTHY"}],
        }
        if image:
            container_def["image"] = image
        if command is not None:
            container_def["command"] = command
        if cfg.container_port:
            container_def["portMappings"] = [{"containerPort": cfg.container_port, "protocol": "tcp"}]
        if env:
            container_def["environment"] = [{"name": k, "value": v} for k, v in sorted(env.items())]
        if log_cfg:
            container_def["logConfiguration"] = log_cfg

        task_volumes, mount_points = self._build_efs_volumes()
        if mount_points:
            container_def["mountPoints"] = mount_points

        _validate_fargate_cpu_memory(int(cfg.cpu), int(cfg.memory))
        payload: dict[str, Any] = {
            "family": self._make_family_name(),
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": cfg.cpu,
            "memory": cfg.memory,
            "executionRoleArn": cfg.execution_role_arn,
            "containerDefinitions": [container_def, sidecar_def],
        }
        if task_volumes:
            payload["volumes"] = task_volumes
        if cfg.task_role_arn:
            payload["taskRoleArn"] = cfg.task_role_arn
        ephemeral = _ephemeral_storage_block(cfg.ephemeral_storage_gib)
        if ephemeral is not None:
            payload["ephemeralStorage"] = ephemeral
        return self._do_register(payload)

    def _build_efs_volumes(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build EFS volume defs + mount points from spec.volumes.

        A volume may name its own filesystem/access point or inherit the provider-level
        ``efs_filesystem_id`` / ``efs_access_point_id`` defaults.
        """
        cfg = self._cfg
        task_volumes: list[dict[str, Any]] = []
        mount_points: list[dict[str, Any]] = []
        for i, vol in enumerate(self._spec.volumes):
            if not vol.is_efs:
                continue
            filesystem_id = vol.efs_filesystem_id or cfg.efs_filesystem_id
            if not filesystem_id:
                raise ValueError(
                    f"EFS volume at {vol.container_path!r} has no filesystem id; set it on the volume "
                    f"or configure the provider's efs_filesystem_id"
                )
            access_point_id = vol.efs_access_point_id or cfg.efs_access_point_id
            vol_name = f"efs-{i}"
            efs_cfg: dict[str, Any] = {
                "fileSystemId": filesystem_id,
                "transitEncryption": "ENABLED",
            }
            if access_point_id:
                efs_cfg["authorizationConfig"] = {
                    "accessPointId": access_point_id,
                    "iam": "ENABLED",
                }
            elif vol.efs_root_directory:
                efs_cfg["rootDirectory"] = vol.efs_root_directory
            task_volumes.append({"name": vol_name, "efsVolumeConfiguration": efs_cfg})
            mount_points.append(
                {
                    "sourceVolume": vol_name,
                    "containerPath": vol.container_path,
                    "readOnly": vol.readonly,
                }
            )
        return task_volumes, mount_points

    def _ssm_lookup_task_def(self, h: str) -> str | None:
        _, _, ClientError = _require_aws_sdks()
        param_name = f"/{self._cfg.ssm_project}/task-defs/{h}"
        try:
            resp = self._ssm.get_parameter(Name=param_name)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code == "ParameterNotFound":
                return None
            logger.warning("SSM GetParameter %s failed (%s); falling through to register", param_name, code)
            return None

        arn = resp["Parameter"]["Value"]
        try:
            desc = self._ecs.describe_task_definition(taskDefinition=arn)
        except ClientError as exc:
            logger.warning(
                "Cached task def %s no longer describable (%s); re-registering",
                arn,
                exc.response["Error"]["Code"],
            )
            return None
        if desc["taskDefinition"]["status"] != "ACTIVE":
            logger.info("SSM cache entry %s is not ACTIVE; re-registering (hash %s)", arn, h)
            return None
        logger.info("Reusing task def from SSM cache: %s (hash %s)", arn, h)
        return arn

    def _ssm_write_task_def(self, h: str, arn: str) -> None:
        _, _, ClientError = _require_aws_sdks()
        param_name = f"/{self._cfg.ssm_project}/task-defs/{h}"
        try:
            self._ssm.put_parameter(Name=param_name, Value=arn, Type="String", Overwrite=True)
            logger.info("Wrote SSM task-def cache entry: %s -> %s", param_name, arn)
        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            logger.warning("SSM PutParameter %s failed (%s); cache entry not written", param_name, code)

    def _do_register(self, payload: dict[str, Any]) -> str:
        h = _compute_task_def_hash(payload)

        while True:
            with _task_def_cache_lock:
                if h in _task_def_cache:
                    arn = _task_def_cache[h]
                    logger.info("Reusing cached task def %s (hash %s)", arn, h)
                    return arn
                if h in _task_def_inflight:
                    event = _task_def_inflight[h]
                else:
                    event = threading.Event()
                    _task_def_inflight[h] = event
                    break
            event.wait()

        try:
            arn = self._ssm_lookup_task_def(h) or self._register_task_def_fresh(payload, h)
            with _task_def_cache_lock:
                _task_def_cache[h] = arn
            return arn
        finally:
            self._release_inflight(h, event)

    def _register_task_def_fresh(self, payload: dict[str, Any], h: str) -> str:
        resp = _retry_with_backoff(
            lambda: self._ecs.register_task_definition(**payload),
            operation_name="register_task_definition",
            max_retries=25,
        )
        arn = resp["taskDefinition"]["taskDefinitionArn"]
        logger.info("Registered task def: %s (hash %s)", arn, h)
        self._ssm_write_task_def(h, arn)
        return arn

    def _release_inflight(self, h: str, event: threading.Event) -> None:
        with _task_def_cache_lock:
            _task_def_inflight.pop(h, None)
        event.set()

    def _make_family_name(self) -> str:
        nonce = uuid.uuid4().hex[:12]
        raw = f"{self._cfg.task_definition_family_prefix}-{_sanitize_id(self._spec.image or 'sandbox')}-{nonce}"
        family = re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:255]
        if not family or not re.match(r"^[A-Za-z0-9]", family):
            family = f"ecs_{family}"
        return family

    # ── Run task + wait ──────────────────────────────────────────────

    def _run_task(self, task_def_arn: str) -> str:
        cfg = self._cfg
        run_kwargs: dict[str, Any] = {
            "cluster": cfg.cluster,
            "taskDefinition": task_def_arn,
            "launchType": "FARGATE",
            "networkConfiguration": {
                "awsvpcConfiguration": {
                    "subnets": cfg.subnets,
                    "securityGroups": cfg.security_groups,
                    "assignPublicIp": "ENABLED" if cfg.assign_public_ip else "DISABLED",
                }
            },
        }
        if self._runtime_container_env:
            run_kwargs["overrides"] = {
                "containerOverrides": [
                    {
                        "name": cfg.container_name,
                        "environment": [
                            {"name": k, "value": v} for k, v in sorted(self._runtime_container_env.items())
                        ],
                    }
                ]
            }
        has_efs = any(v.is_efs for v in self._spec.volumes)
        if cfg.platform_version:
            run_kwargs["platformVersion"] = cfg.platform_version
        elif has_efs:
            run_kwargs["platformVersion"] = "1.4.0"

        last_failures: Any = None
        for attempt in range(1, cfg.run_task_max_retries + 1):
            try:
                resp = _retry_with_backoff(
                    lambda: self._ecs.run_task(**run_kwargs), operation_name="run_task", max_retries=3
                )
            except Exception as exc:
                if not _is_retryable_error(exc) or attempt >= cfg.run_task_max_retries:
                    raise
                delay = min(60.0, 2.0 ** min(6, attempt - 1)) + random.random() * 2
                logger.warning(
                    "run_task failed (%d/%d): %s — retry in %.1fs", attempt, cfg.run_task_max_retries, exc, delay
                )
                time.sleep(delay)
                continue
            failures = resp.get("failures") or []
            if not failures:
                tasks = resp.get("tasks") or []
                if not tasks:
                    raise RuntimeError("run_task returned no tasks")
                task_arn = tasks[0]["taskArn"]
                logger.info("Started ECS task: %s", task_arn)
                return task_arn
            last_failures = failures
            reasons = " | ".join(str(f.get("reason", "")) for f in failures)
            if not any(m in reasons.lower() for m in _RETRYABLE_MESSAGES) or attempt >= cfg.run_task_max_retries:
                raise RuntimeError(f"run_task failures: {failures}")
            delay = min(60.0, 2.0 ** min(6, attempt - 1)) + random.random() * 2
            logger.warning(
                "run_task capacity issue (%d/%d): %s — retry in %.1fs",
                attempt,
                cfg.run_task_max_retries,
                reasons,
                delay,
            )
            time.sleep(delay)
        raise RuntimeError(f"run_task failed after {cfg.run_task_max_retries} retries: {last_failures}")

    def _wait_for_running(self) -> None:
        cfg = self._cfg
        start = time.monotonic()
        poll = 5.0
        last_status = ""
        while True:
            elapsed = time.monotonic() - start
            if elapsed > cfg.startup_timeout_sec:
                raise TimeoutError(f"ECS task not RUNNING after {elapsed:.0f}s (last: {last_status})")
            try:
                resp = self._ecs.describe_tasks(cluster=cfg.cluster, tasks=[self._task_arn])
            except Exception as exc:
                if _is_retryable_error(exc):
                    time.sleep(poll + random.random() * 3)
                    continue
                raise
            tasks = resp.get("tasks") or []
            if not tasks:
                raise RuntimeError("ECS task disappeared")
            status = tasks[0].get("lastStatus", "UNKNOWN")
            if status == "RUNNING":
                logger.info("ECS task RUNNING after %.0fs", elapsed)
                return
            if status == "STOPPED":
                raise RuntimeError(f"ECS task stopped: {tasks[0].get('stoppedReason')}")
            if status != last_status:
                logger.info("ECS task %s (%.0fs)", status, elapsed)
                last_status = status
            time.sleep(poll + random.random() * 3)
            poll = min(15.0, poll + 0.5)

    def _get_task_public_ip(self) -> str:
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._ecs.describe_tasks(cluster=self._cfg.cluster, tasks=[self._task_arn])
                tasks = resp.get("tasks") or []
                if not tasks:
                    raise RuntimeError("Task not found")
                eni_id = None
                for att in tasks[0].get("attachments") or []:
                    if att.get("type") == "ElasticNetworkInterface":
                        for d in att.get("details") or []:
                            if d.get("name") == "networkInterfaceId":
                                eni_id = d["value"]
                                break
                    if eni_id:
                        break
                if not eni_id:
                    for att in tasks[0].get("attachments") or []:
                        for d in att.get("details") or []:
                            if d.get("name") == "privateIPv4Address" and d.get("value"):
                                return d["value"]
                    raise RuntimeError("No ENI/IP yet")
                iface = self._ec2.describe_network_interfaces(NetworkInterfaceIds=[eni_id])["NetworkInterfaces"][0]
                pub = (iface.get("Association") or {}).get("PublicIp")
                if pub:
                    logger.info("Container public IP: %s", pub)
                    return pub
                priv = iface.get("PrivateIpAddress")
                # Keep retrying for the public IP to propagate before falling back to the private IP
                # on the last attempt (the orchestrator is outside the VPC and can't reach private).
                if self._cfg.assign_public_ip and attempt < max_retries:
                    raise RuntimeError(f"ENI {eni_id} has no public IP yet")
                if priv:
                    logger.info("Container private IP: %s", priv)
                    return priv
                raise RuntimeError(f"ENI {eni_id} has no IP")
            except Exception as exc:
                if attempt >= max_retries:
                    raise
                if _is_retryable_error(exc):
                    time.sleep(min(15.0, 2.0**attempt + random.random()))
                else:
                    logger.warning("get_task_ip attempt %d/%d: %s", attempt, max_retries, exc)
                    time.sleep(min(15.0, 3.0 + attempt * 2))
        raise RuntimeError("get_task_ip exhausted retries")

    @staticmethod
    def _wait_for_ssh_ready(host: str, port: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        logger.info("Waiting for SSH at %s:%d", host, port)
        while time.monotonic() < deadline:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5.0)
                    s.connect((host, port))
                    s.settimeout(5.0)
                    if b"SSH" in s.recv(256):
                        logger.info("SSH ready at %s:%d", host, port)
                        return
            except OSError:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"SSH not ready at {host}:{port} after {timeout:.0f}s")

    def _open_tunnel(self, sidecar: SshSidecarConfig) -> None:
        assert self._task_ip is not None and self._ssh_key_file is not None
        if sidecar.exec_server_port is not None:
            self._ssh_tunnel = SshTunnel(
                host=self._task_ip,
                port=sidecar.sshd_port,
                user="root",
                key_file=self._ssh_key_file,
                forward_port=sidecar.exec_server_port,
                reverses=self._outside_endpoint_routing.reverse_specs,
            )
            self._ssh_tunnel.open()
        else:
            remote_host, remote_port = self._outside_endpoint_routing.agent_tunnel_target()
            assert self._ssh_tunnel_port is not None
            self._agent_forward_port = _free_port()
            container_port = self._cfg.container_port
            if not container_port:
                raise ValueError("container_port is required in agent-server mode")
            self._ssh_tunnel = SshTunnel(
                host=self._task_ip,
                port=sidecar.sshd_port,
                user="root",
                key_file=self._ssh_key_file,
                forwards=[f"{self._agent_forward_port}:localhost:{container_port}"],
                reverses=[f"{self._ssh_tunnel_port}:{remote_host}:{remote_port}"],
                local_port_override=self._agent_forward_port,
            )
            self._ssh_tunnel.open()

    # ── Cleanup ──────────────────────────────────────────────────────

    def _cleanup(self) -> None:
        if self._ssh_tunnel:
            try:
                self._ssh_tunnel.close()
            except Exception:
                logger.debug("Failed to close SSH tunnel", exc_info=True)
            self._ssh_tunnel = None
        if self._task_arn and self._ecs:
            try:
                _retry_with_backoff(
                    lambda: self._ecs.stop_task(
                        cluster=self._cfg.cluster, task=self._task_arn, reason="sandbox cleanup"
                    ),
                    operation_name="stop_task",
                    max_retries=10,
                )
                logger.info("Stopped ECS task: %s", self._task_arn)
            except Exception as exc:
                logger.warning("Failed to stop task %s: %s", self._task_arn, exc)
        if self._ssh_key_file:
            try:
                os.remove(self._ssh_key_file)
            except Exception:
                logger.debug("Failed to remove SSH key file %s", self._ssh_key_file, exc_info=True)
            self._ssh_key_file = None
        for key in self._s3_artifacts:
            _delete_s3_object(self._cfg, key)
        self._s3_artifacts = []

    def _sync_stop(self) -> None:
        """Synchronous stop for emergency cleanup (atexit handler)."""
        if self._stopped:
            return
        self._stopped = True
        self._cleanup()

    def _require_exec_client(self) -> None:
        if self._exec_client is None:
            raise RuntimeError(
                "exec()/upload()/download() require exec-server mode "
                "(ssh_sidecar.exec_server_port). In agent-server mode use sandbox.ssh_tunnel."
            )

    async def _upload_via_s3(
        self, paths: list[Path], dest_dir: str, *, arcnames: dict[Path, str] | None = None
    ) -> None:
        cfg = self._cfg
        if not cfg.s3_bucket:
            raise ValueError("s3_bucket is required for S3 staging")

        names = arcnames or {}

        def _pack() -> bytes:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for p in paths:
                    if p.is_file():
                        tar.add(str(p), arcname=names.get(p, p.name))
                    elif p.is_dir():
                        for child in p.rglob("*"):
                            if child.is_file():
                                tar.add(str(child), arcname=str(child.relative_to(p)))
            buf.seek(0)
            return buf.read()

        body = await asyncio.to_thread(_pack)
        boto3, *_ = _require_aws_sdks()
        s3 = boto3.client("s3", region_name=cfg.region)
        prefix = cfg.s3_prefix or "ecs-sandbox"
        nonce = uuid.uuid4().hex[:12]
        key = f"{prefix}/{self._run_id}/upload-{nonce}.tar.gz"
        await asyncio.to_thread(
            _retry_with_backoff,
            lambda: s3.put_object(Bucket=cfg.s3_bucket, Key=key, Body=body),
            operation_name="s3.put_object(upload)",
            max_retries=5,
        )
        self._s3_artifacts.append(key)  # remove on cleanup (the container downloads it during the run)
        url = await asyncio.to_thread(
            s3.generate_presigned_url,
            "get_object",
            Params={"Bucket": cfg.s3_bucket, "Key": key},
            ExpiresIn=21600,
        )
        dl_cmd = (
            f"mkdir -p {shlex.quote(dest_dir)} && TGZ=/tmp/_upload_$$.tar.gz && "
            f"( curl -sf -L --max-time 300 -o $TGZ {shlex.quote(url)} 2>/dev/null || "
            f"python3 -c 'import urllib.request as u,sys;u.urlretrieve(sys.argv[1],sys.argv[2])' "
            f"{shlex.quote(url)} $TGZ ) && "
            f"tar xzf $TGZ -C {shlex.quote(dest_dir)} && rm -f $TGZ && echo ok"
        )
        result = await self._exec_client.exec(dl_cmd, timeout=360)  # type: ignore[union-attr]
        if "ok" not in result.stdout:
            raise RuntimeError(
                f"S3 upload extraction failed (rc={result.return_code}): {result.stderr or result.stdout}"
            )

    # ── Atexit cleanup ───────────────────────────────────────────────

    def _register_for_cleanup(self) -> None:
        global _atexit_registered
        with _cleanup_lock:
            _active_sandboxes[id(self)] = self
            if not _atexit_registered:
                atexit.register(_emergency_cleanup)
                _atexit_registered = True

    def _unregister_from_cleanup(self) -> None:
        with _cleanup_lock:
            _active_sandboxes.pop(id(self), None)
