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
"""Starting, watching, and stopping the servers of a Gym run: the supervisor behind `gym env start`."""

import asyncio
import shlex
from pathlib import Path
from signal import SIGINT
from subprocess import Popen, TimeoutExpired
from threading import Thread
from time import monotonic, sleep, time
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple

import requests
import uvicorn
from devtools import pprint
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo_gym import PARENT_DIR, _resolve_under_cwd_or_install
from nemo_gym.client.server_client import ServerClient, ServerStatus
from nemo_gym.config.config_types import ConfigError
from nemo_gym.config.global_config import (
    DRY_RUN_KEY_NAME,
    HEAD_SERVER_KEY_NAME,
    MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    SERVER_SPINUP_TIMEOUT_SECONDS_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)
from nemo_gym.orchestration.h2_ping_sidecar.launcher import H2PingSidecarManager, start_h2_ping_sidecar
from nemo_gym.orchestration.server_venv import get_venv_path, run_command, setup_env_command
from nemo_gym.server_runtime.ray_support import entrypoint_ray_enabled, initialize_ray, ray_is_installed
from nemo_gym.server_runtime.registry_server import HeadServer, ServerInstanceDisplayConfig
from nemo_gym.telemetry.config import MemoryProfilingConfig
from nemo_gym.telemetry.memory import MemoryProfiler, ServerMemoryTarget, process_tree_memory_supported
from nemo_gym.telemetry.metrics import record_active_servers
from nemo_gym.telemetry.setup import (
    configure_telemetry_env,
    get_telemetry,
    init_telemetry,
    is_telemetry_metrics_enabled,
    memory_profiling_config_from_env,
    shutdown_telemetry,
    telemetry_config_from_global_config,
)
from nemo_gym.telemetry.startup import STARTUP_SETUP_DONE_NS_ENV, SupervisorStartup


# Grace period after SIGINT before escalating to SIGKILL. Kept short so Ctrl-C is responsive.
_GRACEFUL_SHUTDOWN_TIMEOUT_SEC: int = 1


# Grace period after SIGKILL for the kernel to reap the child and avoid <defunct> entries.
_FORCE_KILL_REAP_TIMEOUT_SEC: int = 2


# Model servers name their upstream endpoint inconsistently: `openai_base_url` (openai_model,
# azure_openai_model), `base_url` (inference_provider, vllm_model, and the local vLLM servers),
# `anthropic_base_url`. Matching on the shape of the key rather than a fixed list also covers
# configs that carry a provider URL literally instead of interpolating `policy_base_url`, which
# most inference_provider variants do.
_MODEL_SERVER_TYPE = "responses_api_models"


_BASE_URL_KEY_SUFFIX = "base_url"


_ENDPOINT_PROBE_TIMEOUT_SEC: float = 5.0


_ENDPOINT_POLL_INTERVAL_SEC: float = 3.0


# What a probe found. The distinction that matters is whether waiting could change the answer: a
# name that does not resolve will not start resolving, while a refused connection may be a server
# that is still coming up, and so may a gateway status from a proxy or load balancer in front of it.
_ENDPOINT_ANSWERING = "answering"


_ENDPOINT_REFUSED = "refused"


_ENDPOINT_STARTING = "starting"


_ENDPOINT_UNRESOLVABLE = "unresolvable"


# A proxy or load balancer answers these while nothing healthy is behind it yet, for example in
# front of vLLM replicas that are still loading weights, so they mean "not yet" rather than "here".
_ENDPOINT_STARTING_STATUS_CODES = frozenset({502, 503, 504})


def _collect_model_endpoints(global_config_dict: DictConfig) -> List[Tuple[str, str]]:
    """The upstream model endpoints named in the resolved config, as (config key, url) pairs.

    `wait_for_spinup` only polls Gym's own servers, and the Gym-side model server answers as soon
    as it starts whether or not anything is behind it, so these are the URLs nothing checks.

    `base_url` is typed `Union[str, List[str]]` on vllm_model and the local vLLM servers, where the
    list form spreads load across replicas. The local ones default to an empty list and are filled
    in after Gym launches vLLM, so an empty list means "not yet" rather than "misconfigured" and is
    skipped, as is an unset value.

    The key travels with the url because the endpoint that fails may be a judge or user model, and
    a message naming `policy_base_url` would then be wrong.
    """
    endpoints: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for top_level_path, top_level_value in global_config_dict.items():
        if top_level_path in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS or not isinstance(top_level_value, DictConfig):
            continue

        model_servers = top_level_value.get(_MODEL_SERVER_TYPE)
        if not isinstance(model_servers, DictConfig):
            continue

        for server_config_dict in model_servers.values():
            if not isinstance(server_config_dict, DictConfig):
                continue

            for key, value in server_config_dict.items():
                if not str(key).endswith(_BASE_URL_KEY_SUFFIX):
                    continue

                urls = (
                    [value] if isinstance(value, str) else list(value) if isinstance(value, (list, ListConfig)) else []
                )
                for url in urls:
                    if not isinstance(url, str) or not url:
                        continue
                    # `http://x/v1` and `http://x/v1/` probe the same place.
                    trimmed = url.rstrip("/")
                    if trimmed in seen:
                        continue
                    seen.add(trimmed)
                    endpoints.append((str(key), url))

    return endpoints


def _endpoint_probe_url(base_url: str) -> str:
    """What to GET to find out whether `base_url` is being served.

    `GET /v1/models` is part of the OpenAI API, so most endpoints behind a `/v1` base URL answer it,
    but nothing here depends on that: any HTTP response other than a gateway status counts as
    answering, so an endpoint without it replies 404 and still passes. URLs that do not end in `/v1`
    are probed at their root.
    """
    trimmed = base_url.rstrip("/")
    return f"{trimmed}/models" if trimmed.endswith("/v1") else trimmed


def _is_dns_failure(error: BaseException) -> bool:
    """Whether a `requests` connection error was a name that does not resolve.

    `requests` reports this as a `ConnectionError` like any other, with the resolver failure nested
    inside, so the cause chain has to be walked rather than the class inspected.
    """
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if type(error).__name__ in ("NameResolutionError", "gaierror"):
            return True
        nested = next((arg for arg in getattr(error, "args", ()) if isinstance(arg, BaseException)), None)
        error = nested or error.__cause__ or error.__context__
    return False


def _probe_endpoint(base_url: str, timeout_seconds: float = _ENDPOINT_PROBE_TIMEOUT_SEC) -> str:
    """Whether anything answers at `base_url`, and if not, whether waiting could help.

    Answering is the bar, not healthy: a 401 or 404 means something is there, and requiring a 200
    would reject endpoints that need auth. The exception is 502, 503 and 504, which a proxy or load
    balancer returns while the server behind it is still starting; those are waited on like a
    refused connection. A completed TLS handshake counts too, even against a certificate this
    process does not trust, which is why `SSLError` is checked before `ConnectionError` it inherits
    from.

    The probe connects directly, ignoring `HTTP_PROXY` and the like, because model requests go
    through Gym's aiohttp client, which ignores them too. Through a proxy, an endpoint the proxy
    cannot reach would come back as 502 or 504 and be waited on until the timeout.
    """
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(_endpoint_probe_url(base_url), timeout=timeout_seconds)
        if response.status_code in _ENDPOINT_STARTING_STATUS_CODES:
            return _ENDPOINT_STARTING
        return _ENDPOINT_ANSWERING
    except requests.exceptions.SSLError:
        return _ENDPOINT_ANSWERING
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        return _ENDPOINT_UNRESOLVABLE if _is_dns_failure(e) else _ENDPOINT_REFUSED
    except requests.exceptions.RequestException:
        # Not probeable at all, such as a bad scheme or a redirect loop. Not a reason to hold up
        # startup; the request path reports it if it matters.
        return _ENDPOINT_ANSWERING


def _wait_for_model_endpoints(
    endpoints: List[Tuple[str, str]],
    timeout_seconds: float,
    poll_interval_seconds: float = _ENDPOINT_POLL_INTERVAL_SEC,
    monotonic=monotonic,
    sleep_fn=sleep,
) -> List[Tuple[str, str]]:
    """Wait for every endpoint to answer. Returns the ones that never did.

    Unresolvable names are reported once and not waited on. Refused connections and gateway
    statuses are waited on, because an inference server can take minutes to load weights and
    someone who starts thirty seconds early should not have to start over.
    """
    if timeout_seconds <= 0 or not endpoints:
        return []

    # The clock starts before the first round, not after it. That round is not free: an
    # unresolvable name costs a full resolver timeout, so several of them would otherwise push the
    # real wall time well past the bound the caller asked for.
    started_at = monotonic()

    waiting = []
    for key, url in endpoints:
        result = _probe_endpoint(url)
        if result == _ENDPOINT_UNRESOLVABLE:
            print(
                f"Model endpoint {url} ({key}) does not resolve. Waiting cannot fix a hostname, so it is not retried."
            )
        elif result in (_ENDPOINT_REFUSED, _ENDPOINT_STARTING):
            waiting.append((key, url))
    if not waiting:
        return []

    print(
        f"Waiting for {len(waiting)} model endpoint(s) to answer: "
        + ", ".join(f"{url} ({key})" for key, url in waiting)
        + f"\nIf an inference server is still loading weights this is expected. Waiting up to {timeout_seconds:.0f}s..."
    )

    poll_count = 0
    while True:
        if monotonic() - started_at >= timeout_seconds:
            return waiting

        sleep_fn(poll_interval_seconds)
        waiting = [(key, url) for key, url in waiting if _probe_endpoint(url) != _ENDPOINT_ANSWERING]
        if not waiting:
            print(f"All model endpoints are answering after {monotonic() - started_at:.0f}s.")
            return []

        poll_count += 1
        if poll_count % 10 == 0:
            print(f"  still waiting, {monotonic() - started_at:.0f}s elapsed")


# Matches the value `GlobalConfigDictParser` sets, so a caller that builds a config without going
# through the parser gets the same behaviour instead of silently skipping the check.
_DEFAULT_MODEL_ENDPOINT_READINESS_TIMEOUT_SEC: float = 600.0


def _model_endpoint_timeout_seconds(global_config_dict: DictConfig) -> float:
    """How long to wait for model endpoints, from config. 0 or a negative value skips the check.

    An unset key falls back to the same default the config parser applies. A value that is not a
    number is a configuration mistake, so it is reported as one rather than surfacing as a
    `ValueError` traceback from `float()`.
    """
    value = global_config_dict.get(MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME)
    if value is None:
        return _DEFAULT_MODEL_ENDPOINT_READINESS_TIMEOUT_SEC

    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(
            f"`{MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME}` must be a number of seconds, got {value!r}. "
            "Set it to 0 to skip waiting for model endpoints."
        ) from None


_DEFAULT_SERVER_SPINUP_TIMEOUT_SEC: float = 600.0


# Longest single readiness probe of a Gym server. A probe is capped further by the time left before the deadline.
_SERVER_PROBE_TIMEOUT_SEC: float = 5.0


def _server_spinup_timeout_seconds(global_config_dict: DictConfig) -> float:
    """How long to wait for Gym servers to become ready, from config. 0 or a negative value waits forever.

    An unset or null key falls back to the same default the config parser applies.
    Environment interpolation yields a string, so numeric strings are accepted.
    Anything else is reported as a `ConfigError` rather than a `TypeError` traceback mid-startup.
    """
    value = global_config_dict.get(SERVER_SPINUP_TIMEOUT_SECONDS_KEY_NAME)
    if value is None:
        return _DEFAULT_SERVER_SPINUP_TIMEOUT_SEC

    try:
        return float(value)
    except (TypeError, ValueError):
        raise ConfigError(
            f"`{SERVER_SPINUP_TIMEOUT_SECONDS_KEY_NAME}` must be a number of seconds, got {value!r}. "
            "Set it to 0 to wait for Gym servers without a limit."
        ) from None


def _resolve_server_dir(rel_path: Path) -> Path:
    """Resolve a relative server dir (e.g. ``resources_servers/<name>``) to an absolute path.

    Searches NEMO_GYM_EXTRA_ROOTS, the current working directory (a user's local server), then the Gym
    install root (``PARENT_DIR``) where built-in servers live in both editable and wheel installs. A
    directory counts as a server only if it ships an install marker for one of our two venv setups. This
    lets ``gym env test`` find and run built-in (and plugin) servers from any cwd, not just a repo checkout.
    """
    return _resolve_under_cwd_or_install(
        rel_path, validator=lambda d: (d / "requirements.txt").exists() or (d / "pyproject.toml").exists()
    )


class _ConfiguredServer(NamedTuple):
    top_level_path: str
    server_type: str
    name: str
    config: DictConfig
    entrypoint_fpath: Path
    dir_path: Path


def _configured_servers(global_config_dict: DictConfig) -> Iterator[_ConfiguredServer]:
    """Yield each server instance in the config that has an entrypoint to launch."""
    top_level_paths = [k for k in global_config_dict.keys() if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS]

    # TODO there is a better way to resolve this that uses nemo_gym/config/global_config.py::ServerInstanceConfig
    for top_level_path in top_level_paths:
        server_config_dict = global_config_dict[top_level_path]
        if not isinstance(server_config_dict, DictConfig):
            continue

        first_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[first_key]
        if not isinstance(server_config_dict, DictConfig):
            continue
        second_key = list(server_config_dict)[0]
        server_config_dict = server_config_dict[second_key]
        if not isinstance(server_config_dict, DictConfig):
            continue

        if "entrypoint" not in server_config_dict:
            continue

        # TODO: This currently only handles relative entrypoints. Later on we can resolve the absolute path.
        entrypoint_fpath = Path(server_config_dict.entrypoint)
        assert not entrypoint_fpath.is_absolute()

        # Resolve cwd-first (a local server), else the install location for built-ins.
        dir_path = _resolve_server_dir(Path(first_key, second_key))

        yield _ConfiguredServer(top_level_path, first_key, second_key, server_config_dict, entrypoint_fpath, dir_path)


# Ray reads this when it's imported. Set to "0", it makes Ray calls fail instead of starting a private cluster.
RAY_ENABLE_AUTO_CONNECT_ENV_VAR_NAME = "RAY_ENABLE_AUTO_CONNECT"


def _ray_server_names(configured_servers: List[_ConfiguredServer]) -> List[str]:
    """Names of the configured servers that declare `ray_enabled = True`, so this run must start or join Ray.

    A declaration that can't be read from the entrypoint's source, such as one inherited from another module,
    doesn't count. Such a server fails at startup with an error asking it to declare `ray_enabled = True` itself.
    """
    ray_server_names = [
        server.top_level_path
        for server in configured_servers
        if entrypoint_ray_enabled(server.dir_path / server.entrypoint_fpath) is True
    ]
    if ray_server_names and not ray_is_installed():
        raise ConfigError(
            f"These servers use Ray, but Ray isn't installed where `gym` runs: {', '.join(ray_server_names)}. "
            "Install with:\n  pip install nemo-gym[ray]"
        )
    return ray_server_names


def _server_launch_command(
    dir_path: Path,
    global_config_dict: DictConfig,
    server_name: str,
    entrypoint_fpath: Path,
) -> str:
    """Shell command that sets up a server's venv and starts its entrypoint with that venv's own interpreter.

    The interpreter is named explicitly instead of relying on the bare `python` that `source bin/activate`
    puts on PATH: a venv copied or moved after creation still names its original prefix in `bin/activate`,
    so activating it would silently run a different interpreter than `uv_venv_dir` selected.

    The middle line records when venv setup finished.
    The server uses it to separate venv setup from interpreter start and import time in its startup stages (see `nemo_gym.telemetry.startup`).
    It reads the clock with the venv's own interpreter because `date +%s%N` does not work on macOS.
    """
    venv_python_fpath = get_venv_path(dir_path, global_config_dict) / "bin" / "python"
    return f"""{setup_env_command(dir_path, global_config_dict, server_name)} \\
    && export {STARTUP_SETUP_DONE_NS_ENV}=$({shlex.quote(str(venv_python_fpath))} -c 'import time; print(time.time_ns())') \\
    && {shlex.quote(str(venv_python_fpath))} {shlex.quote(str(entrypoint_fpath))}"""


class ServerSpinupTimeoutError(RuntimeError):
    """Gym servers did not become ready within `server_spinup_timeout_seconds`."""


def _startup_outcome(error: BaseException) -> str:
    """How a startup that raised `error` ended, as the startup telemetry reports it."""
    if isinstance(error, ServerSpinupTimeoutError):
        return "timeout"
    if isinstance(error, KeyboardInterrupt):
        return "interrupted"
    return "failed"


class RunHelper:  # pragma: no cover
    _head_server: uvicorn.Server
    _head_server_thread: Thread
    _head_server_instance: HeadServer

    _processes: Dict[str, Popen]
    _server_instance_display_configs: List[ServerInstanceDisplayConfig]
    _server_client: ServerClient
    _memory_profiler: MemoryProfiler | None
    _memory_profiling_config: MemoryProfilingConfig
    _telemetry_metrics_enabled: bool
    _h2_ping_sidecar: H2PingSidecarManager | None
    # Created by `_start`, so its clock starts with the startup it measures.
    _startup: Optional[SupervisorStartup] = None

    def start(self, global_config_dict_parser_config: GlobalConfigDictParserConfig) -> None:
        """Start the head server and every configured server, and wait until all of them are ready.

        Any failure or interrupt before readiness shuts down everything started so far, then re-raises.
        Callers reach their own `shutdown()` only after this returns.
        The spawned servers have no process group or atexit handler, so nothing else would stop them.
        """
        self._processes = dict()
        self._head_server = None
        self._h2_ping_sidecar = None
        try:
            self._start(global_config_dict_parser_config)
        except BaseException as error:
            # Before shutdown, which flushes telemetry, so the failed startup's spans are exported.
            if self._startup is not None:
                self._startup.abort(_startup_outcome(error), error)
            self.shutdown()
            raise

    def _start(self, global_config_dict_parser_config: GlobalConfigDictParserConfig) -> None:
        self._startup = SupervisorStartup()
        global_config_dict = get_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)

        # Fail fast before starting Ray if nothing is configured to run (covers env run and the
        # e2e rollout-collection path, which both start servers via this method).
        GlobalConfigDictParser().raise_on_no_server_instances(global_config_dict)
        self._server_spinup_timeout_seconds = _server_spinup_timeout_seconds(global_config_dict)
        self._startup.mark("load_config")

        # Translate the `telemetry:` block into NEMO_GYM_OTEL_* env vars *before* anything is
        # spawned. run_command copies os.environ into every server process, and that copy is
        # the only channel these settings have — the servers share no memory with this one.
        # Also mints the run id they all report, so a backend can group one run's processes.
        telemetry_config = telemetry_config_from_global_config(global_config_dict)
        configure_telemetry_env(telemetry_config)
        init_telemetry(server_name="supervisor", server_type="supervisor")
        self._memory_profiler = None
        self._memory_profiling_config = memory_profiling_config_from_env(telemetry_config.memory_profiling)
        self._telemetry_metrics_enabled = is_telemetry_metrics_enabled()
        self._startup.begin_trace()
        self._startup.mark("init_telemetry")

        configured_servers = list(_configured_servers(global_config_dict))

        # Start or join Ray only when a configured server may use it. Servers inherit the cluster address through
        # the config dict below, so this has to happen before any of them are spawned.
        # Note: initialize_ray modifies the global config dict - updates `ray_head_node_address`
        ray_server_names = _ray_server_names(configured_servers)
        if ray_server_names:
            print(f"Initializing Ray for servers that use it: {', '.join(ray_server_names)}")
            initialize_ray()
            self._startup.mark("init_ray")

        # Start the HTTP/2 PING sidecar (if `sidecar.enabled`) and point model URLs at it. This
        # must come before the config is serialized below, which is how every server learns its URLs.
        # A dry run only builds venvs and talks to no model, so it starts nothing.
        if not global_config_dict[DRY_RUN_KEY_NAME]:
            self._h2_ping_sidecar = start_h2_ping_sidecar(global_config_dict)
            if self._h2_ping_sidecar is not None:
                self._startup.mark("start_h2_ping_sidecar")

        # Assume Nemo Gym Run is for a single agent.
        config_dict_yaml_str = OmegaConf.to_yaml(global_config_dict)

        # We always run the head server in this `run` command.
        self._head_server, self._head_server_thread, self._head_server_instance = HeadServer.run_webserver()
        self._startup.mark("start_head_server")

        self._server_instance_display_configs: List[ServerInstanceDisplayConfig] = []

        start_time = time()

        for (
            top_level_path,
            first_key,
            second_key,
            server_config_dict,
            entrypoint_fpath,
            dir_path,
        ) in configured_servers:
            # Name the venv in the logs: the server runs that venv's interpreter (see _server_launch_command).
            print(f"Starting `{top_level_path}` from venv {get_venv_path(dir_path, global_config_dict)}")
            command = _server_launch_command(dir_path, global_config_dict, top_level_path, entrypoint_fpath)

            extra_env = {
                NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME: config_dict_yaml_str,
                NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME: top_level_path,
            }
            if top_level_path not in ray_server_names:
                # A server that uses Ray without declaring it then fails on its first Ray call, rather than
                # starting a private cluster in each of its processes.
                extra_env[RAY_ENABLE_AUTO_CONNECT_ENV_VAR_NAME] = "0"
            extra_env |= self._startup.server_env(top_level_path, first_key)
            # Server entrypoints are executed as files (``python app.py``). In a source
            # checkout, put the component root on PYTHONPATH as well as the server directory
            # so package-qualified imports such as ``responses_api_agents.foo.helpers``
            # resolve from the same checkout instead of an older installed Gym distribution.
            # In a wheel install that root is the parent's site-packages; adding it would
            # shadow each server venv's own dependencies, so skip it there.
            process = run_command(
                command,
                dir_path,
                server_name=top_level_path,
                project_root=dir_path.parent.parent if (PARENT_DIR / "pyproject.toml").is_file() else None,
                extra_env=extra_env,
            )
            self._processes[top_level_path] = process
            # In dry run mode, wait for each setup command to finish before starting the next.
            # This installs uv virtual environments serially, which significantly reduces uv
            # cache size. For Nemotron's set of environments, parallel installation can produce
            # a cache 10-20GB larger than serial installation.
            if global_config_dict[DRY_RUN_KEY_NAME]:
                print("DRY_RUN enabled: setup commands are run serially")
                process.communicate()

            host = server_config_dict.get("host")
            port = server_config_dict.get("port")

            self._server_instance_display_configs.append(
                ServerInstanceDisplayConfig(
                    process_name=top_level_path,
                    server_type=first_key,
                    name=second_key,
                    dir_path=str(dir_path),
                    entrypoint=str(entrypoint_fpath),
                    host=host,
                    port=port,
                    url=f"http://{host}:{port}" if host and port else None,
                    pid=process.pid,
                    config_path=top_level_path,
                    start_time=start_time,
                )
            )

        self._head_server_instance.set_server_instances(
            [inst.model_dump(mode="json") for inst in self._server_instance_display_configs]
        )

        # `gym.servers.active` is a gauge, so exactly one process may write it or the
        # exported value is just whoever wrote last. The supervisor is the only process
        # that knows the fleet size, so it is the only writer (see telemetry/metrics.py).
        record_active_servers(len(self._server_instance_display_configs))
        self._startup.mark("spawn_servers")

        self._server_client = ServerClient(
            head_server_config=ServerClient.load_head_server_config(),
            global_config_dict=global_config_dict,
        )

        print("Waiting for head server to spin up")
        poll_count = 0
        while True:
            status = self._server_client.poll_for_status(HEAD_SERVER_KEY_NAME)
            if status == "success":
                break

            if poll_count % 10 == 0:  # Print every 30s
                print(f"Head server is not up yet (status `{status}`). Sleeping...")

            poll_count += 1
            sleep(3)
        self._startup.mark("wait_for_head_server")

        print("Waiting for servers to spin up")
        if global_config_dict[DRY_RUN_KEY_NAME]:
            self.wait_for_dry_run_spinup()
        else:
            self.wait_for_server_readiness(global_config_dict)

    def wait_for_server_readiness(self, global_config_dict: DictConfig) -> None:
        """Mark the head ready only after every managed server and model endpoint is reachable."""
        self.wait_for_spinup()
        if self._startup is not None:
            self._startup.mark("wait_for_servers")
        self._start_memory_profiler()
        self.wait_for_model_endpoints(global_config_dict)
        if self._startup is not None:
            self._startup.mark("wait_for_model_endpoints")
        self._head_server_instance.mark_ready()
        if self._startup is not None:
            self._startup.finish()

    def display_server_instance_info(self) -> None:
        if not self._server_instance_display_configs:
            print("No server instances to display.")
            return

        print(f"""
{"#" * 100}
#
# Server Instances
#
{"#" * 100}
""")

        for i, inst in enumerate(self._server_instance_display_configs, 1):
            print(f"[{i}] {inst.process_name} ({inst.server_type}/{inst.name})")
            pprint(inst.model_dump(mode="json", exclude={"start_time", "status", "uptime_seconds"}))
        print(f"{'#' * 100}\n")

    def poll(self) -> None:
        if not self._head_server_thread.is_alive():
            raise RuntimeError("Head server finished unexpectedly!")

        if self._h2_ping_sidecar is not None:
            self._h2_ping_sidecar.check()

        for process_name, process in self._processes.items():
            if process.poll() is not None:
                proc_out, proc_err = process.communicate()
                print_str = f"Process `{process_name}` finished unexpectedly!"

                if isinstance(proc_out, bytes):
                    proc_out = proc_out.decode("utf-8")
                    print_str = f"""{print_str}
Process `{process_name}` stdout:
{proc_out}
"""
                if isinstance(proc_err, bytes):
                    proc_err = proc_err.decode("utf-8")
                    print_str = f"""{print_str}
Process `{process_name}` stderr:
{proc_err}"""

                raise RuntimeError(print_str)

    def wait_for_dry_run_spinup(self) -> None:
        """Wait for every dry-run process to finish, and fail if any of them did.

        A dry run builds each server's venv and exits, so unlike poll() a finished process is the
        expected outcome here and only the exit code separates success from failure.
        The code has to be checked: uv creates the venv before installing into it, so a failed
        install still leaves an interpreter and an activate script behind.
        That venv then satisfies skip_venv_if_present on the next run, and the first symptom is an
        ImportError from a server long after the install that caused it.
        """
        sleep_interval = 3

        failures: List[Tuple[str, int]] = []
        remaining_processes = list(self._processes.items())
        while remaining_processes:
            for i in reversed(range(len(remaining_processes))):
                process_name, process = remaining_processes[i]
                returncode = process.poll()
                if returncode is not None:
                    if returncode != 0:
                        failures.append((process_name, returncode))
                    remaining_processes.pop(i)

            if remaining_processes:
                sleep(sleep_interval)

        if failures:
            raise RuntimeError(
                "Dry run failed to set up "
                + ("1 server" if len(failures) == 1 else f"{len(failures)} servers")
                + ". Each server's output is above, prefixed with its name:\n"
                + "\n".join(f"  `{name}` exited with {code}" for name, code in failures)
            )

    def wait_for_spinup(self) -> None:
        sleep_interval = 3
        poll_count = 0
        successful_servers = []
        total_servers = len(self._server_instance_display_configs)
        timeout_seconds = self._server_spinup_timeout_seconds
        deadline = monotonic() + timeout_seconds if timeout_seconds > 0 else None

        # Until we spin up or error out.
        while True:
            self.poll()
            statuses = self.check_http_server_statuses(successful_servers, deadline=deadline)
            if self._startup is not None:
                for name, status in statuses:
                    if status == "success":
                        self._startup.server_ready(name)
            successful_servers.extend(s for s, status in statuses if status == "success")

            waiting = []
            for name, status in statuses:
                if status != "success":
                    waiting.append(name)

            if len(successful_servers) != total_servers:
                if deadline is not None and monotonic() >= deadline:
                    raise ServerSpinupTimeoutError(
                        f"Timed out after {timeout_seconds:g}s waiting for Gym servers to become ready: "
                        f"{', '.join(waiting)}\n"
                        "This wait covers dependency installation and any local model download or loading "
                        "that a server does before it opens its port.\n"
                        f"Raise the limit with `++{SERVER_SPINUP_TIMEOUT_SECONDS_KEY_NAME}=<seconds>`, "
                        "or set it to 0 to wait without a limit."
                    )
                if poll_count % 10 == 0:  # Print every sleep_interval * poll_count = 3 * 10 = 30s
                    print(
                        f"""Checking for HTTP server statuses.
{len(successful_servers)} / {total_servers} servers ready. Waiting for servers to spin up: {waiting}"""
                    )
                poll_count += 1
            else:
                print(f"All {len(successful_servers)} / {total_servers} servers ready! Polling every 60s")
                self.display_server_instance_info()
                return

            if deadline is None:
                sleep(sleep_interval)
            else:
                sleep(max(0.0, min(sleep_interval, deadline - monotonic())))

    def shutdown(self) -> None:
        memory_profiler = getattr(self, "_memory_profiler", None)
        if memory_profiler is not None:
            memory_profiler.stop()
            self._memory_profiler = None

        # Before the servers go, the gauge should read zero once the fleet is torn down.
        # The metric provider also needs an explicit flush or the last interval is lost.
        record_active_servers(0)
        shutdown_telemetry()

        print("Sending interrupt signals to servers...")
        for process in self._processes.values():
            process.send_signal(SIGINT)

        print("Waiting for processes to finish...")
        killed_process_names: List[str] = []
        unreaped_process_names: List[str] = []
        for process_name, process in self._processes.items():
            try:
                process.wait(timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_SEC)
            except TimeoutExpired:
                process.kill()
                killed_process_names.append(process_name)
                # Reap the child after SIGKILL to avoid leaving a <defunct> entry.
                try:
                    process.wait(timeout=_FORCE_KILL_REAP_TIMEOUT_SEC)
                except TimeoutExpired:
                    unreaped_process_names.append(process_name)

        if killed_process_names:
            print(
                f"""Some processes ({", ".join(killed_process_names)}) didn't shutdown within the {_GRACEFUL_SHUTDOWN_TIMEOUT_SEC}s timeout, killing instead. You may see messages like:
```bash
rpc_client.h:203: Failed to connect to GCS within 60 seconds. GCS may have been killed. It's either GCS is terminated by `ray stop` or is killed unexpectedly. If it is killed unexpectedly, see the log file gcs_server.out. https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#logging-directory-structure. The program will terminate.
```
"""
            )
        if unreaped_process_names:
            print(
                f"WARNING: processes ({', '.join(unreaped_process_names)}) did not exit "
                f"within {_FORCE_KILL_REAP_TIMEOUT_SEC}s after SIGKILL; "
                "they may remain as zombies until this process exits."
            )
        self._processes = dict()

        # After the servers, so requests they were still finishing can drain through it. It gets its
        # own, longer grace period (`sidecar.shutdown_grace`).
        h2_ping_sidecar = getattr(self, "_h2_ping_sidecar", None)
        if h2_ping_sidecar is not None:
            print("Stopping h2-ping-sidecar...")
            h2_ping_sidecar.stop()
            self._h2_ping_sidecar = None

        # None before the head server starts and after an earlier shutdown.
        if self._head_server is not None:
            self._head_server.should_exit = True
            self._head_server_thread.join()

        self._head_server = None
        self._head_server_thread = None

        print("NeMo Gym finished!")

    def run_forever(self) -> None:
        if self._server_client.global_config_dict[DRY_RUN_KEY_NAME]:
            self.shutdown()
            return

        async def sleep():
            # Indefinitely
            while True:
                self.poll()
                await asyncio.sleep(60)

        try:
            asyncio.run(sleep())
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def wait_for_model_endpoints(self, global_config_dict: DictConfig) -> None:
        """Block until every model endpoint named in the config answers, then return.

        Reads the bound from `model_endpoint_readiness_timeout_seconds`, where 0 skips the check
        entirely. Raises `ConfigError` naming the endpoints that never answered and the config key
        each came from. It raises rather than exits because `RunHelper` is imported and driven as a
        library, so the caller decides what an unreachable endpoint means; the CLI entrypoints turn
        it into an exit.
        `start()` shuts down the servers it already spawned before the error reaches the caller.
        """
        timeout_seconds = _model_endpoint_timeout_seconds(global_config_dict)
        unreachable = _wait_for_model_endpoints(_collect_model_endpoints(global_config_dict), timeout_seconds)
        if not unreachable:
            return

        listed = "\n".join(f"  - {url} (from `{key}`)" for key, url in unreachable)
        raise ConfigError(
            f"""{len(unreachable)} model endpoint(s) never answered within {timeout_seconds:.0f}s:
{listed}

Nothing accepted a connection there, or only a proxy or load balancer answered with 502, 503 or 504.
The server was never started, is still starting up, or the URL in your config does not match where
it is listening.
  - Check the config key named beside each URL. Verify with `curl -i <base_url>/models` for a
    `/v1` endpoint, or `curl -i <base_url>` otherwise.
  - Any other response, including 401 or 404, means something is listening.
  - Raise `{MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME}` to wait longer, or set it to 0 to skip this check."""
        )

    def check_http_server_statuses(
        self, successful_servers: List[str], *, deadline: Optional[float] = None
    ) -> List[Tuple[str, ServerStatus]]:
        """Probe every server not yet ready.

        With a `deadline` from `monotonic()`, each probe is capped at the time left before it.
        Servers still unprobed once it passes are reported as `timeout` without a request.
        """
        statuses = []
        for server_instance_display_config in self._server_instance_display_configs:
            name = server_instance_display_config.config_path

            # No need to re-poll successfully spun up servers.
            if name in successful_servers:
                continue

            probe_timeout_seconds = _SERVER_PROBE_TIMEOUT_SEC
            if deadline is not None:
                remaining_seconds = deadline - monotonic()
                if remaining_seconds <= 0:
                    statuses.append((name, "timeout"))
                    continue
                probe_timeout_seconds = min(probe_timeout_seconds, remaining_seconds)

            status = self._server_client.poll_for_status(name, timeout_seconds=probe_timeout_seconds)
            statuses.append((name, status))

        return statuses

    def _start_memory_profiler(self) -> None:
        memory_config = getattr(self, "_memory_profiling_config", None)
        if memory_config is None or not memory_config.enabled:
            return
        if not process_tree_memory_supported():
            print("Memory profiling requires Linux procfs and is disabled on this host.")
            return
        if not getattr(self, "_telemetry_metrics_enabled", False):
            print("Memory profiling is enabled but telemetry metrics are disabled.")
            return
        telemetry = get_telemetry()
        if telemetry is None or not telemetry.is_exporting:
            print(
                "Memory profiling is enabled but telemetry is not exporting. "
                "Install nemo-gym[telemetry] and enable telemetry metrics."
            )
            return
        targets = [
            ServerMemoryTarget(
                name=instance.process_name,
                server_type=instance.server_type,
                pid=instance.pid,
            )
            for instance in self._server_instance_display_configs
        ]
        self._memory_profiler = MemoryProfiler(
            targets,
            interval_seconds=memory_config.interval_seconds,
        )
        self._memory_profiler.start()
