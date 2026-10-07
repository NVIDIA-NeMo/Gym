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
"""Start, supervise and stop the h2-ping-sidecar for ``gym env start``, and point model URLs at it.

``H2PingSidecarManager`` is the entry point. It runs the proxy as an ordinary child process of
the process that starts the Gym servers (``RunHelper``), so the proxy is always on the same node
as the model servers that call it, including when ``RunHelper`` itself runs inside a Ray actor.
"""

import hashlib
import os
import re
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from omegaconf.errors import OmegaConfBaseException

from nemo_gym.global_config import (
    CACHE_DIR_KEY_NAME,
    NEMO_GYM_LOG_DIR_KEY_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
)
from nemo_gym.h2_ping_sidecar.config import (
    H2_PING_SIDECAR_KEY_NAME,
    H2PingSidecarConfig,
    SidecarInstanceConfig,
    is_loopback_host,
)


SIDECAR_SOURCE_DIR = Path(__file__).resolve().parent.parent / "tools" / "sidecar"
SIDECAR_BINARY_NAME = "h2-ping-sidecar"

_MODEL_SERVER_TYPE = "responses_api_models"
_BASE_URL_KEY_SUFFIX = "base_url"
_READY_POLL_INTERVAL_SEC = 0.1
_KILL_REAP_TIMEOUT_SEC = 5.0
# Extra time past `shutdown_grace` before the supervisor gives up waiting for the process to exit.
_STOP_SLACK_SEC = 5.0
_LOG_TAIL_LINES = 20
_BUILD_TIMEOUT_SEC = 300.0


class SidecarError(RuntimeError):
    """The sidecar could not be configured, started or kept running."""


# ---------------------------------------------------------------------------------------------
# Config, instances and binary
# ---------------------------------------------------------------------------------------------


def h2_ping_sidecar_config_from_global_config(global_config_dict: Any) -> H2PingSidecarConfig:
    """Build the config from the optional top-level ``sidecar:`` block; absent means disabled."""
    block: Any = None
    if global_config_dict is not None:
        try:
            block = global_config_dict.get(H2_PING_SIDECAR_KEY_NAME)
        except (AttributeError, TypeError):
            block = None
    if block is None:
        return H2PingSidecarConfig()
    # OmegaConf nodes are Mapping-like but not dicts; resolve them (the block may hold `${...}`) first.
    if isinstance(block, DictConfig):
        block = OmegaConf.to_container(block, resolve=True)
    return H2PingSidecarConfig.model_validate(block)


def _local_go_version() -> str:
    """``go env GOVERSION`` of the Go on PATH, or an empty string when there is none."""
    go = shutil.which("go")
    if go is None:
        return ""
    try:
        result = subprocess.run(
            [go, "env", "GOVERSION"],
            env={**os.environ, "GOTOOLCHAIN": "local"},
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _source_digest() -> str:
    """Short hash of the Go sources the binary is built from.

    A binary built from older sources must never be reused, so the hash is part of the cache path.
    """
    digest = hashlib.sha256()
    sources = sorted(path for path in SIDECAR_SOURCE_DIR.glob("*.go") if not path.name.endswith("_test.go"))
    for path in sources + [SIDECAR_SOURCE_DIR / "go.mod"]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:12]


def default_binary_path(global_config_dict: Any) -> Path:
    """Where the sidecar built from the current sources lives: ``<cache>/<sources hash>/<go version>/``.

    The proxy is mostly Go's own ``net/http`` and ``crypto/tls``, whose fixes ship as Go releases, so
    a binary built by a Go that has since been replaced is not reused: with Go on PATH the exact
    version is part of the path. Without Go nothing can be built, so any binary already built from
    these sources is used (the newest one), which lets a node without Go run a prebuilt cache.
    """
    cache_dir = global_config_dict.get(CACHE_DIR_KEY_NAME) or "cache"
    sources = Path(cache_dir).expanduser().resolve() / SIDECAR_BINARY_NAME / _source_digest()
    go_version = re.sub(r"[^A-Za-z0-9._-]", "_", _local_go_version())
    if go_version:
        return sources / go_version / SIDECAR_BINARY_NAME
    built = sorted(sources.glob(f"*/{SIDECAR_BINARY_NAME}"), key=lambda path: path.stat().st_mtime)
    return built[-1] if built else sources / "no-go" / SIDECAR_BINARY_NAME


def build_sidecar(output_path: Path) -> None:
    """Compile the bundled Go source into ``output_path``.

    The compiler writes to a private temporary file that is moved into place afterwards, so a
    concurrent run sharing the cache never executes a half-written binary.
    """
    go = shutil.which("go")
    if go is None:
        raise SidecarError(
            f"The h2-ping-sidecar binary {output_path} is missing and `go` is not on PATH, so it cannot be built. "
            f"Install Go (see go.mod in {SIDECAR_SOURCE_DIR} for the minimum version) or set "
            f"`{H2_PING_SIDECAR_KEY_NAME}.binary` to a prebuilt h2-ping-sidecar."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.{uuid4().hex}.partial")
    print(f"Building h2-ping-sidecar -> {output_path}")
    try:
        result = subprocess.run(
            [go, "build", "-buildvcs=false", "-trimpath", "-o", str(partial), "."],
            cwd=SIDECAR_SOURCE_DIR,
            env={**os.environ, "CGO_ENABLED": "0", "GOTOOLCHAIN": "local"},
            capture_output=True,
            text=True,
            errors="replace",
            timeout=_BUILD_TIMEOUT_SEC,
        )
        if result.returncode != 0:
            raise SidecarError(f"`go build` of the h2-ping-sidecar failed:\n{result.stdout}{result.stderr}")
        os.replace(partial, output_path)
    finally:
        partial.unlink(missing_ok=True)


def resolve_binary(config: H2PingSidecarConfig, global_config_dict: Any) -> Path:
    """The sidecar executable, building it first when allowed and missing."""
    path = Path(config.binary).expanduser() if config.binary else default_binary_path(global_config_dict)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists() and config.build_if_missing:
        build_sidecar(path)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SidecarError(
            f"h2-ping-sidecar binary {path} does not exist or is not executable. Set "
            f"`{H2_PING_SIDECAR_KEY_NAME}.binary` to a built binary, or set `build_if_missing: true` "
            f"to build it from {SIDECAR_SOURCE_DIR} (needs Go)."
        )
    return path


def resolve_log_dir(config: H2PingSidecarConfig, global_config_dict: Any) -> str:
    configured = config.log_dir or global_config_dict.get(NEMO_GYM_LOG_DIR_KEY_NAME)
    if configured:
        return str(Path(configured).expanduser().resolve())
    return os.path.join(tempfile.gettempdir(), f"nemo_gym_h2ping-{os.getuid()}")


# ---------------------------------------------------------------------------------------------
# Base URL rewriting
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RewrittenUrl:
    location: str
    old: str
    new: str


def _local_url(url: str, instances_by_origin: Dict[str, SidecarInstanceConfig], location: str) -> Optional[str]:
    try:
        parts = urlsplit(url)
    except ValueError as e:
        raise SidecarError(f"{location} is not a valid URL ({url!r}): {e}") from e
    if parts.scheme != "https" or not parts.netloc:
        return None
    instance = instances_by_origin.get(f"https://{parts.netloc.lower()}")
    if instance is None:
        return None
    return urlunsplit(("http", instance.listen, parts.path, parts.query, parts.fragment))


def rewrite_base_urls(global_config_dict: DictConfig, instances: List[SidecarInstanceConfig]) -> List[RewrittenUrl]:
    """Point every model URL aimed at an instance's upstream at that instance's local address, in place.

    Covers any ``*base_url`` key (a string or a list of strings) of a model server under
    ``responses_api_models``; top-level keys such as ``policy_base_url`` are left as they are. The
    *resolved* value is what gets matched, so a URL that comes from a resolver such as
    ``${oc.env:JUDGE_URL}`` or from an alias such as ``${policy_base_url}`` is routed through the
    sidecar too, and is materialized as a literal in memory for that field only. A field that aliases
    a whole list is rewritten as a resolved copy, so the list it points at is never changed.
    Raises ``SidecarError`` for a value that is not a parseable URL.
    """
    by_origin = {instance.upstream: instance for instance in instances}
    changes: List[RewrittenUrl] = []

    def rewrite_value(container: Any, key: Any, location: str) -> None:
        try:
            old = container[key]
        except OmegaConfBaseException:
            # A value that cannot be resolved cannot be loaded by the server either; leave it for it to report.
            return
        if not isinstance(old, str):
            return
        new = _local_url(old, by_origin, location)
        if new is not None:
            container[key] = new
            changes.append(RewrittenUrl(location, old, new))

    def rewrite_aliased_list(container: Any, key: Any, location: str, value: ListConfig) -> None:
        resolved = list(value)
        for index, item in enumerate(resolved):
            if not isinstance(item, str):
                continue
            new = _local_url(item, by_origin, f"{location}[{index}]")
            if new is not None:
                resolved[index] = new
                changes.append(RewrittenUrl(f"{location}[{index}]", item, new))
        if any(change.location.startswith(f"{location}[") for change in changes):
            container[key] = resolved

    with open_dict(global_config_dict):
        for top_level_path in list(global_config_dict.keys()):
            if top_level_path in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS:
                continue
            try:
                top_level_value = global_config_dict[top_level_path]
            except OmegaConfBaseException:
                continue
            if not isinstance(top_level_value, DictConfig):
                continue
            model_servers = top_level_value.get(_MODEL_SERVER_TYPE)
            if not isinstance(model_servers, DictConfig):
                continue
            for server_name, server_config in model_servers.items():
                if not isinstance(server_config, DictConfig):
                    continue
                for key in list(server_config.keys()):
                    if not str(key).endswith(_BASE_URL_KEY_SUFFIX):
                        continue
                    location = f"{top_level_path}.{_MODEL_SERVER_TYPE}.{server_name}.{key}"
                    try:
                        value = server_config[key]
                    except OmegaConfBaseException:
                        continue
                    if isinstance(value, ListConfig):
                        if OmegaConf.is_interpolation(server_config, key):
                            rewrite_aliased_list(server_config, key, location, value)
                        else:
                            for index in range(len(value)):
                                rewrite_value(value, index, f"{location}[{index}]")
                    else:
                        rewrite_value(server_config, key, location)
    return changes


# ---------------------------------------------------------------------------------------------
# Process supervision
# ---------------------------------------------------------------------------------------------


def sidecar_command(
    binary: str, config: H2PingSidecarConfig, instance: SidecarInstanceConfig, ready_path: str
) -> List[str]:
    command = [
        binary,
        "-listen",
        instance.listen,
        "-upstream",
        instance.upstream,
        "-ping-interval",
        config.ping_interval,
        "-ping-timeout",
        config.ping_timeout,
        "-shutdown-grace",
        config.shutdown_grace,
        "-retry-body-limit",
        str(config.retry_body_limit),
        "-max-conn-age",
        config.max_conn_age,
        "-ready-file",
        ready_path,
    ]
    if config.insecure_skip_verify:
        command.append("-insecure-skip-verify")
    return command


def sidecar_env(config: H2PingSidecarConfig) -> Dict[str, str]:
    """Environment for the child: ours, plus the Go runtime limits from the config."""
    env = dict(os.environ)
    if config.gomemlimit is not None:
        env["GOMEMLIMIT"] = config.gomemlimit
    if config.gomaxprocs is not None:
        env["GOMAXPROCS"] = str(config.gomaxprocs)
    return env


def _log_tail(log_path: str) -> str:
    try:
        lines = Path(log_path).read_text(errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(lines[-_LOG_TAIL_LINES:]) or "(empty log)"


class NodeSidecars:
    """The sidecar processes this run owns: one child process per instance."""

    def __init__(
        self,
        config: H2PingSidecarConfig,
        instances: List[SidecarInstanceConfig],
        binary: str,
        log_dir: str,
    ) -> None:
        self._config = config
        self._instances = instances
        self._binary = binary
        self._log_dir = log_dir
        self._host = socket.gethostname()
        # Distinct owners (concurrent runs on one host) must never share or delete one another's
        # readiness marker or log, even with the same instance names in the same log directory.
        self._ready_id = uuid4().hex
        self._processes: Dict[str, subprocess.Popen] = {}
        self._log_paths: Dict[str, str] = {}

    def start(self) -> List[str]:
        """Launch every instance and wait until each has bound its port; returns one summary line each."""
        if not os.access(self._binary, os.X_OK):
            raise SidecarError(f"{self._host}: h2-ping-sidecar binary {self._binary} is not executable.")
        os.makedirs(self._log_dir, exist_ok=True)
        try:
            for instance in self._instances:
                self._spawn(instance)
            for instance in self._instances:
                self._wait_ready(instance)
        except BaseException:
            self.stop()
            raise
        return [f"{self._host}: {i.name} listening on {i.listen} -> {i.upstream}" for i in self._instances]

    def _ready_path(self, instance: SidecarInstanceConfig) -> str:
        return os.path.join(self._log_dir, f"h2ping-{instance.name}-{self._host}-{self._ready_id}.ready")

    def _spawn(self, instance: SidecarInstanceConfig) -> None:
        ready_path = self._ready_path(instance)
        # A stale file from a crashed run would pass for readiness.
        Path(ready_path).unlink(missing_ok=True)
        log_path = os.path.join(self._log_dir, f"h2ping-{instance.name}-{self._host}-{self._ready_id}.log")
        self._log_paths[instance.name] = log_path
        log_file: IO[bytes] = open(log_path, "ab")
        try:
            # Own session: Ctrl-C reaches the servers but not the sidecar, so requests the servers
            # are still finishing can complete. `stop` ends it explicitly afterwards.
            self._processes[instance.name] = subprocess.Popen(
                sidecar_command(self._binary, self._config, instance, ready_path),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=sidecar_env(self._config),
                start_new_session=True,
            )
        finally:
            log_file.close()

    def _wait_ready(self, instance: SidecarInstanceConfig) -> None:
        process = self._processes[instance.name]
        ready_path = self._ready_path(instance)
        deadline = time.monotonic() + self._config.startup_timeout_seconds
        while True:
            # The file holds the PID and is written only after the bind succeeded, so a port that
            # something else already holds can't pass for ours.
            if os.path.exists(ready_path):
                return
            code = process.poll()
            if code is not None:
                tail = _log_tail(self._log_paths[instance.name])
                # The sidecar logs "cannot bind" when it loses the port; other exits (bad flag, bad
                # upstream) say something else, so only suggest a busy port when that is what happened.
                hint = f" (is {instance.listen} already in use?)" if "cannot bind" in tail else ""
                raise SidecarError(
                    f"{self._host}: h2-ping-sidecar `{instance.name}` exited with code {code} before it was "
                    f"ready{hint}. Log {self._log_paths[instance.name]}:\n{tail}"
                )
            if time.monotonic() >= deadline:
                raise SidecarError(
                    f"{self._host}: h2-ping-sidecar `{instance.name}` was not ready after "
                    f"{self._config.startup_timeout_seconds:g}s. Log {self._log_paths[instance.name]}:\n"
                    f"{_log_tail(self._log_paths[instance.name])}"
                )
            time.sleep(_READY_POLL_INTERVAL_SEC)

    def failures(self) -> List[str]:
        """One message per sidecar that is no longer running."""
        messages = []
        for name, process in self._processes.items():
            code = process.poll()
            if code is not None:
                messages.append(
                    f"{self._host}: h2-ping-sidecar `{name}` exited with code {code}. "
                    f"Log {self._log_paths[name]}:\n{_log_tail(self._log_paths[name])}"
                )
        return messages

    def stop(self) -> None:
        """SIGTERM every sidecar (it drains in-flight requests), then SIGKILL stragglers."""
        from nemo_gym.h2_ping_sidecar.config import parse_duration_seconds

        for process in self._processes.values():
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + parse_duration_seconds(self._config.shutdown_grace) + _STOP_SLACK_SEC
        for process in self._processes.values():
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=_KILL_REAP_TIMEOUT_SEC)
                except subprocess.TimeoutExpired:
                    pass
        for instance in self._instances:
            Path(self._ready_path(instance)).unlink(missing_ok=True)
        self._processes = {}


# ---------------------------------------------------------------------------------------------
# Orchestrator side
# ---------------------------------------------------------------------------------------------


class H2PingSidecarManager:
    """Owns the sidecar processes a run started."""

    def __init__(
        self,
        config: H2PingSidecarConfig,
        instances: List[SidecarInstanceConfig],
        binary: Path,
        log_dir: str,
    ) -> None:
        self.config = config
        self.instances = instances
        self._binary = str(binary)
        self._log_dir = log_dir
        self._sidecars: Optional[NodeSidecars] = None

    @property
    def log_dir(self) -> str:
        return self._log_dir

    def start(self) -> None:
        self._sidecars = NodeSidecars(self.config, self.instances, self._binary, self._log_dir)
        try:
            for line in self._sidecars.start():
                print(f"h2-ping-sidecar {line}")
        except BaseException:
            self.stop()
            raise

    def check(self) -> None:
        """Raise if any sidecar has stopped. Called from Gym's poll loop."""
        messages = self._sidecars.failures() if self._sidecars is not None else []
        if messages:
            raise SidecarError("h2-ping-sidecar stopped unexpectedly:\n" + "\n".join(messages))

    def stop(self) -> None:
        if self._sidecars is not None:
            self._sidecars.stop()
            self._sidecars = None


def start_h2_ping_sidecar(global_config_dict: DictConfig) -> Optional[H2PingSidecarManager]:
    """Start the sidecar when ``sidecar.enabled`` and point model URLs at it.

    Returns the manager to poll and stop later, or None when the feature is off. Raises
    ``SidecarError`` (after stopping anything it started) when it cannot be brought up. Call it after
    ``initialize_ray()`` and before the config is serialized for the servers.

    Everything that can be checked without side effects is checked first, so a bad config fails with
    no process started and the live config untouched. Once the proxy is running, any later failure or
    interrupt stops it before the error propagates, because nothing else owns it yet.
    """
    config = h2_ping_sidecar_config_from_global_config(global_config_dict)
    if not config.enabled:
        return None

    instances = list(config.instances)
    if config.rewrite_base_urls:
        # Dry run on a copy: surfaces malformed URLs before a proxy exists.
        rewrite_base_urls(deepcopy(global_config_dict), instances)
    for instance in instances:
        host = instance.listen.rpartition(":")[0]
        if not is_loopback_host(host):
            print(
                f"WARNING: h2-ping-sidecar instance `{instance.name}` listens on {instance.listen}, which is not a "
                "loopback address. The proxy has no authentication, so anything that can reach this address can "
                f"send requests, with its own credentials, to {instance.upstream}. Use 127.0.0.1 unless you "
                "have a reason not to."
            )

    manager = H2PingSidecarManager(
        config,
        instances,
        resolve_binary(config, global_config_dict),
        resolve_log_dir(config, global_config_dict),
    )
    manager.start()

    try:
        if config.rewrite_base_urls:
            changes = rewrite_base_urls(global_config_dict, instances)
            for change in changes:
                print(f"h2-ping-sidecar: {change.location}: {change.old} -> {change.new}")
            if not changes:
                print(
                    "WARNING: h2-ping-sidecar is running but no model base URL points at its upstream "
                    f"({', '.join(i.upstream for i in instances)}), so nothing is routed through it."
                )
    except BaseException:
        manager.stop()
        raise
    return manager
