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
"""Schema for the ``sidecar:`` block of a NeMo Gym config.

NVCF endpoints sit behind AWS Global Accelerator, which drops a connection that carries no
application data for 340s. Gym's aiohttp client cannot send HTTP/2 PINGs, so Gym points its model
base URLs at a local sidecar that forwards over HTTP/2 and sends the PINGs itself.

This module imports only pydantic, so it is safe to import from Gym's config machinery.
"""

import ipaddress
import re
from typing import List, Optional, Union
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, field_validator, model_validator


H2_PING_SIDECAR_KEY_NAME = "sidecar"

# AWS Global Accelerator's fixed idle timeout. A ping interval at or above it defeats the sidecar.
GLOBAL_ACCELERATOR_IDLE_TIMEOUT_SECONDS = 340.0

# AWS ALB's default client keep-alive: it closes a connection with a GOAWAY once it is this old.
LOAD_BALANCER_KEEP_ALIVE_SECONDS = 3600.0

_DEFAULT_LISTEN = "127.0.0.1:1250"
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|s|m|h)")
_DURATION_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_MEMORY_LIMIT_PATTERN = re.compile(r"^\d+(B|KiB|MiB|GiB|TiB)?$")


def parse_duration_seconds(value: str) -> float:
    """Seconds in a Go-style duration such as ``60s``, ``15m`` or ``1m30s``."""
    text = value.strip()
    parts = _DURATION_PART.findall(text)
    if not parts or "".join(number + unit for number, unit in parts) != text:
        raise ValueError(f"{value!r} is not a duration like '60s', '15m' or '1m30s'")
    return sum(float(number) * _DURATION_UNIT_SECONDS[unit] for number, unit in parts)


def _normalize_duration(value: Union[str, int, float]) -> str:
    if isinstance(value, bool):
        raise ValueError("a duration must be a string like '60s' or a number of seconds")
    if isinstance(value, (int, float)):
        value = f"{value}s"
    parse_duration_seconds(value)
    return value.strip()


def upstream_origin(url: str) -> str:
    """``scheme://host[:port]`` of an https URL; raises ``ValueError`` for anything else."""
    parts = urlsplit(url)
    if parts.scheme != "https" or not parts.netloc:
        raise ValueError(f"{url!r} must be an https:// URL (the sidecar speaks HTTP/2 over TLS)")
    return f"https://{parts.netloc.lower()}"


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` (a name or IP literal, optionally in brackets) only accepts local connections."""
    host = host.strip().strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class SidecarInstanceConfig(BaseModel, extra="forbid"):
    """One proxy: a loopback listener that forwards to a single HTTPS upstream."""

    name: str = "nvcf"
    """Used in log file names and error messages."""

    upstream: str
    """HTTPS URL to forward to. Only ``scheme://host[:port]`` is used; a path such as ``/v1`` is
    dropped, because clients keep sending it on every request."""

    listen: str = _DEFAULT_LISTEN
    """``host:port`` the sidecar binds. Keep it on loopback: the sidecar adds no authentication."""

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _NAME_PATTERN.match(value):
            raise ValueError(f"{value!r} must contain only letters, digits, '_' and '-'")
        return value

    @field_validator("upstream")
    @classmethod
    def _check_upstream(cls, value: str) -> str:
        return upstream_origin(value)

    @field_validator("listen")
    @classmethod
    def _check_listen(cls, value: str) -> str:
        host, _, port = value.rpartition(":")
        if not host or not port.isdigit() or not 0 < int(port) < 65536:
            raise ValueError(f"{value!r} must look like '127.0.0.1:1250'")
        return value


class H2PingSidecarConfig(BaseModel, extra="forbid"):
    """Configuration of the HTTP/2 PING sidecar that ``gym env start`` can run."""

    enabled: bool = False
    """Master switch. When false Gym starts nothing and leaves every URL untouched."""

    binary: Optional[str] = None
    """Path to the compiled ``h2-ping-sidecar``. Defaults to a build in Gym's cache directory (see
    ``build_if_missing``). It only needs to exist on the node that runs ``gym env start``."""

    build_if_missing: bool = True
    """Run ``go build`` on the launching node when the binary does not exist yet. Needs Go on PATH."""

    instances: List[SidecarInstanceConfig] = Field(default_factory=list)
    """One entry per upstream the sidecar forwards to; required when ``enabled``. A model URL is routed through
    an instance when its host matches the instance's ``upstream``. Give each instance its own ``listen`` port."""

    ping_interval: str = "60s"
    """Send an HTTP/2 PING after this much read-idle time. Must stay well below 340s."""

    ping_timeout: str = "15s"
    """Close the upstream connection if a PING is not acknowledged within this time."""

    shutdown_grace: str = "60s"
    """On shutdown, let in-flight requests finish for up to this long before the sidecar exits."""

    retry_body_limit: int = Field(default=16 * 1024 * 1024, ge=0)
    """Request bodies up to this many bytes are buffered in memory so a request refused with a graceful
    GOAWAY can be re-sent on a fresh connection. Larger bodies stream through and are not retried. Peak
    memory is roughly this limit times the number of concurrent requests. ``0`` disables buffering."""

    max_conn_age: str = "50m"
    """Retire the upstream connection after 90-100% of this age: in-flight requests finish on it and new
    requests use a fresh connection, at the cost of one TLS handshake. Keep it below the load balancer's
    client keep-alive limit (3600s by default) so large bodies never meet its GOAWAY. ``0s`` disables."""

    insecure_skip_verify: bool = False
    """Skip upstream TLS verification. For tests only."""

    gomemlimit: Optional[str] = None
    """Soft memory limit for the Go runtime, passed as ``GOMEMLIMIT`` (for example ``1GiB``)."""

    gomaxprocs: Optional[int] = Field(default=None, ge=1)
    """CPU threads the Go runtime may use, passed as ``GOMAXPROCS``. Only needed to cap the Go runtime below the
    container's CPU quota; setting it disables the runtime's automatic updates."""

    startup_timeout_seconds: float = Field(default=30.0, gt=0)
    """Wait this long for each sidecar to bind its port before failing the run."""

    log_dir: Optional[str] = None
    """Directory for ``h2ping-<instance>-<host>.log``. Defaults to ``nemo_gym_log_dir``, then a temp dir."""

    rewrite_base_urls: bool = True
    """Rewrite ``policy_base_url`` and every ``*base_url`` under ``responses_api_models`` that points at
    an instance's upstream so it targets that instance's local address instead."""

    @field_validator("ping_interval", "ping_timeout", "shutdown_grace", "max_conn_age", mode="before")
    @classmethod
    def _check_duration(cls, value: Union[str, int, float]) -> str:
        return _normalize_duration(value)

    @field_validator("gomemlimit")
    @classmethod
    def _check_gomemlimit(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not _MEMORY_LIMIT_PATTERN.match(value):
            raise ValueError(f"{value!r} must look like '1GiB', '512MiB' or a byte count")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "H2PingSidecarConfig":
        if self.enabled and not self.instances:
            raise ValueError(
                "instances is required when the sidecar is enabled: list each upstream to forward to, e.g. "
                "instances: [{name: judge, upstream: 'https://host.example.com', listen: '127.0.0.1:1250'}]"
            )
        interval = parse_duration_seconds(self.ping_interval)
        if not 0 < interval < GLOBAL_ACCELERATOR_IDLE_TIMEOUT_SECONDS:
            raise ValueError(
                f"ping_interval {self.ping_interval!r} must be above 0 and below the "
                f"{GLOBAL_ACCELERATOR_IDLE_TIMEOUT_SECONDS:g}s Global Accelerator idle limit it exists to defeat"
            )
        if parse_duration_seconds(self.ping_timeout) <= 0:
            raise ValueError("ping_timeout must be above 0")
        max_conn_age = parse_duration_seconds(self.max_conn_age)
        if max_conn_age >= LOAD_BALANCER_KEEP_ALIVE_SECONDS:
            raise ValueError(
                f"max_conn_age {self.max_conn_age!r} must be below the {LOAD_BALANCER_KEEP_ALIVE_SECONDS:g}s load "
                "balancer client keep-alive it exists to stay ahead of (use '0s' to disable rotation)"
            )

        names = [instance.name for instance in self.instances]
        if len(set(names)) != len(names):
            raise ValueError(f"instance names must be unique, got {names}")
        listens = [instance.listen for instance in self.instances]
        if len(set(listens)) != len(listens):
            raise ValueError(f"instances must listen on different addresses, got {listens}")
        return self
