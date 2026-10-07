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

``H2PingSidecarManager`` is the orchestrator-side entry point. On the launching node it runs the
proxy as an ordinary child process. For ``nodes: all`` (or a list of node IPs) it places one small
Ray actor per node, pinned to that node, and the actor runs the same ``NodeSidecars`` code there.
"""

import os
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
    POLICY_BASE_URL_KEY_NAME,
)
from nemo_gym.h2_ping_sidecar.config import (
    H2_PING_SIDECAR_KEY_NAME,
    H2PingSidecarConfig,
    SidecarInstanceConfig,
    is_loopback_host,
    upstream_origin,
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


def resolve_instances(config: H2PingSidecarConfig, global_config_dict: Any) -> List[SidecarInstanceConfig]:
    """The configured instances, or one derived from ``policy_base_url`` when none are configured."""
    if config.instances:
        return list(config.instances)

    policy_base_url = global_config_dict.get(POLICY_BASE_URL_KEY_NAME)
    if not isinstance(policy_base_url, str) or not policy_base_url:
        raise SidecarError(
            f"`{H2_PING_SIDECAR_KEY_NAME}` has no `instances` and `{POLICY_BASE_URL_KEY_NAME}` is not set, "
            "so there is nothing to forward to. Add `instances: [{name: ..., upstream: https://...}]`."
        )
    try:
        return [SidecarInstanceConfig(name="policy", upstream=upstream_origin(policy_base_url))]
    except ValueError as e:
        raise SidecarError(
            f"`{H2_PING_SIDECAR_KEY_NAME}` has no `instances`, so the upstream is taken from "
            f"`{POLICY_BASE_URL_KEY_NAME}`, but {e}"
        ) from e


def default_binary_path(global_config_dict: Any) -> Path:
    cache_dir = global_config_dict.get(CACHE_DIR_KEY_NAME) or "cache"
    return Path(cache_dir).expanduser().resolve() / SIDECAR_BINARY_NAME / SIDECAR_BINARY_NAME


def build_sidecar(output_path: Path) -> None:
    """Compile the bundled Go source into ``output_path``."""
    go = shutil.which("go")
    if go is None:
        raise SidecarError(
            "The h2-ping-sidecar binary is missing and `go` is not on PATH, so it cannot be built. Install Go "
            f"(see go.mod in {SIDECAR_SOURCE_DIR} for the minimum version) or set "
            f"`{H2_PING_SIDECAR_KEY_NAME}.binary` to a prebuilt h2-ping-sidecar."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Building h2-ping-sidecar -> {output_path}")
    result = subprocess.run(
        [go, "build", "-buildvcs=false", "-trimpath", "-o", str(output_path), "."],
        cwd=SIDECAR_SOURCE_DIR,
        env={**os.environ, "CGO_ENABLED": "0"},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=_BUILD_TIMEOUT_SEC,
    )
    if result.returncode != 0:
        raise SidecarError(f"`go build` of the h2-ping-sidecar failed:\n{result.stdout}{result.stderr}")


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

    Covers ``policy_base_url`` and any ``*base_url`` key (a string or a list of strings) of a model
    server under ``responses_api_models``. The *resolved* value is what gets matched, so a URL that
    comes from a resolver such as ``${oc.env:POLICY_URL}`` is routed through the sidecar too, and is
    materialized as a literal in memory. An alias such as ``${policy_base_url}`` stays an alias: once
    its root is rewritten it resolves to an ``http://`` URL, which no instance matches. A field that
    aliases a whole list is rewritten as a resolved copy, so the list it points at is never changed.
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
        # `in` would resolve the value; `keys()` does not, so an unresolvable policy URL is left for the servers.
        if POLICY_BASE_URL_KEY_NAME in global_config_dict.keys():
            rewrite_value(global_config_dict, POLICY_BASE_URL_KEY_NAME, POLICY_BASE_URL_KEY_NAME)

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
# Node-side process supervision (runs in this process, or inside a pinned Ray actor)
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
    """The sidecar processes of one node. Plain Python so it works locally and inside a Ray actor."""

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
        # Distinct owners (concurrent runs, or two actors) must never share or delete one another's
        # readiness marker, even with the same instance names in the same log directory.
        self._ready_id = uuid4().hex
        self._processes: Dict[str, subprocess.Popen] = {}
        self._log_paths: Dict[str, str] = {}

    def start(self) -> List[str]:
        """Launch every instance and wait until each has bound its port; returns one summary line each."""
        if not os.access(self._binary, os.X_OK):
            raise SidecarError(
                f"{self._host}: h2-ping-sidecar binary {self._binary} is not executable here. With "
                "`nodes: all` the binary must be at the same path on every node (a shared filesystem)."
            )
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
        log_path = os.path.join(self._log_dir, f"h2ping-{instance.name}-{self._host}.log")
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
                raise SidecarError(
                    f"{self._host}: h2-ping-sidecar `{instance.name}` exited with code {code} before it was ready "
                    f"(is {instance.listen} already in use?). Log {self._log_paths[instance.name]}:\n"
                    f"{_log_tail(self._log_paths[instance.name])}"
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


def _get_ray():
    import ray

    return ray


def select_ray_nodes(config: H2PingSidecarConfig, alive_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The alive Ray nodes the sidecar should run on, per ``nodes``."""
    if config.nodes == "all":
        selected = list(alive_nodes)
        if not selected:
            raise SidecarError("`nodes: all` found no alive Ray nodes.")
        return selected

    wanted = list(config.nodes)
    by_ip = {node["NodeManagerAddress"]: node for node in alive_nodes}
    missing = [ip for ip in wanted if ip not in by_ip]
    if missing:
        raise SidecarError(f"`nodes` names {missing}, which are not alive Ray nodes. Alive: {sorted(by_ip)}")
    return [by_ip[ip] for ip in wanted]


class H2PingSidecarManager:
    """Owns every sidecar a run started, wherever it runs."""

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
        self._local: Optional[NodeSidecars] = None
        self._actors: Dict[str, Any] = {}

    @property
    def log_dir(self) -> str:
        return self._log_dir

    def start(self) -> None:
        try:
            if self.config.nodes == "local":
                self._start_local()
            else:
                self._start_on_ray_nodes()
        except BaseException:
            self.stop()
            raise

    def _start_local(self) -> None:
        self._local = NodeSidecars(self.config, self.instances, self._binary, self._log_dir)
        for line in self._local.start():
            print(f"h2-ping-sidecar {line}")

    def _start_on_ray_nodes(self) -> None:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        ray = _get_ray()
        if not ray.is_initialized():
            raise SidecarError("`nodes` other than `local` needs a Ray cluster, but Ray is not initialized.")
        nodes = select_ray_nodes(self.config, [node for node in ray.nodes() if node.get("Alive")])

        actor_class = ray.remote(num_cpus=0)(NodeSidecars)
        for node in nodes:
            ip = node["NodeManagerAddress"]
            self._actors[ip] = actor_class.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False),
            ).remote(self.config, self.instances, self._binary, self._log_dir)

        timeout = self.config.startup_timeout_seconds * len(self.instances) + 60.0
        pending = {ip: actor.start.remote() for ip, actor in self._actors.items()}
        errors = []
        for ip, ref in pending.items():
            try:
                for line in ray.get(ref, timeout=timeout):
                    print(f"h2-ping-sidecar {line}")
            except Exception as e:
                errors.append(f"node {ip}: {e}")
        if errors:
            raise SidecarError("h2-ping-sidecar failed to start:\n" + "\n".join(errors))

    def check(self) -> None:
        """Raise if any sidecar has stopped. Called from Gym's poll loop."""
        messages: List[str] = []
        if self._local is not None:
            messages.extend(self._local.failures())
        if self._actors:
            ray = _get_ray()
            for ip, actor in self._actors.items():
                try:
                    messages.extend(ray.get(actor.failures.remote(), timeout=30))
                except Exception as e:
                    messages.append(f"node {ip}: sidecar supervisor is unreachable: {e}")
        if messages:
            raise SidecarError("h2-ping-sidecar stopped unexpectedly:\n" + "\n".join(messages))

    def stop(self) -> None:
        if self._local is not None:
            self._local.stop()
            self._local = None
        if self._actors:
            ray = _get_ray()
            from nemo_gym.h2_ping_sidecar.config import parse_duration_seconds

            timeout = parse_duration_seconds(self.config.shutdown_grace) + _STOP_SLACK_SEC + 30.0
            for ip, actor in self._actors.items():
                try:
                    ray.get(actor.stop.remote(), timeout=timeout)
                except Exception as e:
                    print(f"WARNING: could not stop the h2-ping-sidecar on node {ip}: {e}")
                try:
                    ray.kill(actor)
                except Exception:
                    pass
            self._actors = {}


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

    instances = resolve_instances(config, global_config_dict)
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
