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
import json
import os
import shlex
import sys
from glob import glob
from pathlib import Path
from shutil import rmtree
from signal import SIGINT
from subprocess import Popen, TimeoutExpired
from tempfile import TemporaryDirectory
from threading import Thread
from time import monotonic, sleep, time
from typing import Dict, List, Optional, Tuple

import requests
import rich
import uvicorn
from devtools import pprint
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import Field
from rich.table import Table
from tqdm.auto import tqdm

from nemo_gym import PARENT_DIR, _resolve_under_cwd_or_install, component_search_roots
from nemo_gym.cli.setup_command import get_venv_path, run_command, setup_env_command
from nemo_gym.cli.utils import (
    exit_cleanly_on_config_error,
    exit_unknown_component,
    fuzzy_matches,
    print_no_matches,
    print_rich_table,
    render_component_inspection,
)
from nemo_gym.config_types import BaseNeMoGymCLIConfig, ConfigError
from nemo_gym.environment.manifest import EnvironmentKind, IntegrationProfile
from nemo_gym.environment.onboarding import (
    EnvironmentOnboardingError,
    VerifierReport,
    prepare_verifier_run,
)
from nemo_gym.environment.publication import finalize_publication
from nemo_gym.environment.scaffold import scaffold_environment, scaffold_resources_server
from nemo_gym.environment.validation import EnvironmentValidationReport, validate_environment
from nemo_gym.global_config import (
    COMPONENT_NAME_KEY_NAME,
    DRY_RUN_KEY_NAME,
    JSON_OUTPUT_KEY_NAME,
    MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME,
    NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
    NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME,
    NEMO_GYM_RESERVED_TOP_LEVEL_KEYS,
    QUERY_KEY_NAME,
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_global_config_dict,
)
from nemo_gym.registry import (
    EnvironmentCatalogEntry,
    discover_environment_catalog,
    read_environment_details,
    resolve_catalog_entry,
)
from nemo_gym.server_status import StatusCommand
from nemo_gym.server_utils import (
    HEAD_SERVER_KEY_NAME,
    HeadServer,
    ServerClient,
    ServerInstanceDisplayConfig,
    ServerStatus,
    initialize_ray,
)


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
# that is still coming up.
_ENDPOINT_ANSWERING = "answering"
_ENDPOINT_REFUSED = "refused"
_ENDPOINT_UNRESOLVABLE = "unresolvable"


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
    but nothing here depends on that: any HTTP response counts as answering, so an endpoint without
    it replies 404 and still passes. URLs that do not end in `/v1` are probed at their root.
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
    would reject endpoints that need auth. A completed TLS handshake counts too, even against a
    certificate this process does not trust, which is why `SSLError` is checked before
    `ConnectionError` it inherits from.
    """
    try:
        requests.get(_endpoint_probe_url(base_url), timeout=timeout_seconds)
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

    Unresolvable names are reported once and not waited on. Refused connections are waited on,
    because an inference server can take minutes to load weights and someone who starts thirty
    seconds early should not have to start over.
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
        elif result == _ENDPOINT_REFUSED:
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


class RunConfig(BaseNeMoGymCLIConfig):
    """
    Start NeMo Gym servers for agents, models, and resources.

    Examples:

    ```bash
    config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\\
    responses_api_models/openai_model/configs/openai_model.yaml"
    gym env start "+config_paths=[${config_paths}]"
    ```
    """

    entrypoint: str = Field(
        description="Entrypoint for this command. This must be a relative path with 2 parts. Should look something like `responses_api_agents/simple_agent`."
    )


class TestConfig(RunConfig):
    """
    Test a specific server module by running its pytest suite and optionally validating example data.

    Examples:

    ```bash
    gym env test +entrypoint=resources_servers/example_single_tool_call
    ```
    """

    should_validate_data: bool = Field(
        default=False,
        description="Whether or not to validate the example data (examples, metrics, rollouts, etc) for this server.",
    )

    _dir_path: Path  # initialized in model_post_init

    def model_post_init(self, context):  # pragma: no cover
        self._dir_path = Path(self.entrypoint)
        assert not self.dir_path.is_absolute()

        return super().model_post_init(context)

    @property
    def dir_path(self) -> Path:
        return self._dir_path

    @property
    def resolved_dir_path(self) -> Path:
        """Absolute server dir resolved against the cwd, then the Gym install root.

        Use this for filesystem access (reading data, running the suite); use ``dir_path`` (the
        relative entrypoint) for display and example commands shown to the user.
        """
        return _resolve_server_dir(self._dir_path)


class RunHelper:  # pragma: no cover
    _head_server: uvicorn.Server
    _head_server_thread: Thread
    _head_server_instance: HeadServer

    _processes: Dict[str, Popen]
    _server_instance_display_configs: List[ServerInstanceDisplayConfig]
    _server_client: ServerClient

    def start(self, global_config_dict_parser_config: GlobalConfigDictParserConfig) -> None:
        global_config_dict = get_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)

        # Fail fast before starting Ray if nothing is configured to run (covers env run and the
        # e2e rollout-collection path, which both start servers via this method).
        GlobalConfigDictParser().raise_on_no_server_instances(global_config_dict)

        # Initialize Ray cluster in the main process
        # Note: This function will modify the global config dict - update `ray_head_node_address`
        initialize_ray()

        # Assume Nemo Gym Run is for a single agent.
        escaped_config_dict_yaml_str = shlex.quote(OmegaConf.to_yaml(global_config_dict))

        # We always run the head server in this `run` command.
        self._head_server, self._head_server_thread, self._head_server_instance = HeadServer.run_webserver()

        top_level_paths = [k for k in global_config_dict.keys() if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS]

        self._processes: Dict[str, Popen] = dict()
        self._server_instance_display_configs: List[ServerInstanceDisplayConfig] = []

        start_time = time()

        # TODO there is a better way to resolve this that uses nemo_gym/global_config.py::ServerInstanceConfig
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

            command = f"""{setup_env_command(dir_path, global_config_dict, top_level_path)} \\
    && {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME}={escaped_config_dict_yaml_str} \\
    {NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME}={shlex.quote(top_level_path)} \\
    python {str(entrypoint_fpath)}"""

            process = run_command(command, dir_path, server_name=top_level_path)
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

        print("Waiting for servers to spin up")
        if global_config_dict[DRY_RUN_KEY_NAME]:
            self.wait_for_dry_run_spinup()
        else:
            self.wait_for_spinup()
            self.wait_for_model_endpoints(global_config_dict)

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

        # Until we spin up or error out.
        while True:
            self.poll()
            statuses = self.check_http_server_statuses(successful_servers)
            successful_servers.extend(s for s, status in statuses if status == "success")

            waiting = []
            for name, status in statuses:
                if status != "success":
                    waiting.append(name)

            if len(successful_servers) != total_servers:
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

            sleep(sleep_interval)

    def shutdown(self) -> None:
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

        The servers spawned above are shut down first. They hold ports and have neither a process
        group nor an atexit handler, and every caller reaches its own `shutdown()` only after
        `start()` returns.
        """
        timeout_seconds = _model_endpoint_timeout_seconds(global_config_dict)
        unreachable = _wait_for_model_endpoints(_collect_model_endpoints(global_config_dict), timeout_seconds)
        if not unreachable:
            return

        listed = "\n".join(f"  - {url} (from `{key}`)" for key, url in unreachable)
        self.shutdown()
        raise ConfigError(
            f"""{len(unreachable)} model endpoint(s) never answered within {timeout_seconds:.0f}s:
{listed}

Nothing accepted a connection there. The server was never started, is still starting up, or the URL
in your config does not match where it is listening.
  - Check the config key named beside each URL. Verify with `curl -i <base_url>/models` for a
    `/v1` endpoint, or `curl -i <base_url>` otherwise.
  - Any response, including 401 or 404, means something is listening.
  - Raise `{MODEL_ENDPOINT_READINESS_TIMEOUT_KEY_NAME}` to wait longer, or set it to 0 to skip this check."""
        )

    def check_http_server_statuses(self, successful_servers: List[str]) -> List[Tuple[str, ServerStatus]]:
        statuses = []
        for server_instance_display_config in self._server_instance_display_configs:
            name = server_instance_display_config.config_path

            # No need to re-poll successfully spun up servers.
            if name in successful_servers:
                continue

            status = self._server_client.poll_for_status(name)
            statuses.append((name, status))

        return statuses


@exit_cleanly_on_config_error
def prefetch(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
):  # pragma: no cover
    """
    Pre-warm per-server venvs without starting any servers.

    Accepts the same config format as 'gym env start'. For each server in the config,
    creates its isolated venv and installs dependencies serially — similar to how
    RL's prefetch_venvs.py installs actor venvs at container build time.

    No Ray is initialised and no server processes are started. Intended for use in
    Dockerfile builds so venvs are ready at runtime with no network access needed.

    Configuration Parameter:
        config_paths (List[str]): Paths to YAML configuration files. Specify via Hydra: `+config_paths="[file1.yaml,file2.yaml]"`

    Examples:

    ```bash
    # Pre-warm venvs for a specific config
    config_paths="responses_api_models/local_vllm_model/configs/local_vllm_model.yaml,\\
    resources_servers/math/configs/math.yaml"
    gym env prefetch "+config_paths=[${config_paths}]"
    ```
    """
    global_config_dict = get_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)
    GlobalConfigDictParser().raise_on_no_server_instances(global_config_dict)

    top_level_paths = [k for k in global_config_dict.keys() if k not in NEMO_GYM_RESERVED_TOP_LEVEL_KEYS]

    seen_dirs: set = set()
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

        dir_path = _resolve_server_dir(Path(first_key, second_key))
        if dir_path in seen_dirs:
            print(f"Skipping duplicate: {dir_path}")
            continue
        seen_dirs.add(dir_path)
        print(f"Pre-warming venv for: {dir_path}")

        # Run only the setup (venv creation + dep install) — no server start.
        # Serial execution reduces uv cache size (same reasoning as dry_run in RunHelper.start).
        setup_cmd = setup_env_command(dir_path, global_config_dict, top_level_path)
        process = run_command(setup_cmd, dir_path, server_name=top_level_path)
        returncode = process.wait()
        if returncode != 0:
            print(f"ERROR: prefetch failed for {dir_path} (exit {returncode})")
            raise SystemExit(returncode)

    print("Prefetch complete.")


@exit_cleanly_on_config_error
def run(
    global_config_dict_parser_config: Optional[GlobalConfigDictParserConfig] = None,
):  # pragma: no cover
    """
    Start NeMo Gym servers for agents, models, and resources.

    This command reads configuration from YAML files specified via `+config_paths` and starts all configured servers.
    The configuration files should define server instances with their entrypoints and settings.

    Configuration Parameter:
        config_paths (List[str]): Paths to YAML configuration files. Specify via Hydra: `+config_paths="[file1.yaml,file2.yaml]"`

    Examples:

    ```bash
    # Start servers with specific configs
    config_paths="resources_servers/example_single_tool_call/configs/example_single_tool_call.yaml,\\
    responses_api_models/openai_model/configs/openai_model.yaml"
    gym env start "+config_paths=[${config_paths}]"
    ```
    """
    global_config_dict = get_global_config_dict(global_config_dict_parser_config=global_config_dict_parser_config)
    # Just here for help
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    rh = RunHelper()
    rh.start(global_config_dict_parser_config)
    rh.run_forever()


def _validate_data_single(test_config: TestConfig) -> None:  # pragma: no cover
    if not test_config.should_validate_data:
        return

    # We have special data checks for resources servers
    if test_config.dir_path.parts[0] != "resources_servers":
        return

    # Check that the required examples and example metrics are present. Read from the resolved dir
    # (built-ins live under the install root) while messages reference the relative entrypoint.
    example_fpath = test_config.resolved_dir_path / "data/example.jsonl"
    assert example_fpath.exists(), (
        f"A jsonl file containing 5 examples is required for the {test_config.dir_path} resources server. The file must be found at {example_fpath}. Usually this example data is just the first 5 examples of your train dataset."
    )
    with open(example_fpath) as f:
        count = sum(1 for _ in f)
    assert count == 5, f"Expected 5 examples at {example_fpath} but got {count}."

    server_type_name = test_config.dir_path.parts[-1]
    example_metrics_fpath = test_config.resolved_dir_path / "data/example_metrics.json"
    assert (
        example_metrics_fpath.exists()
    ), f"""You must run the example data validation for the example data found at {example_fpath}.
Your command should look something like the following (you should update this command with your actual server config path):
```bash
gym dataset collate "+config_paths=[{test_config._dir_path}/configs/{server_type_name}.yaml]" \\
    +output_dirpath=data/{server_type_name} \\
    +mode=example_validation
```
and your config must include an agent server config with an example dataset like:
```yaml
example_multi_step_simple_agent:
  responses_api_agents:
    simple_agent:
      ...
      datasets:
      - name: example
        type: example
        jsonl_fpath: resources_servers/example_multi_step/data/example.jsonl
```

See `resources_servers/example_multi_step/configs/example_multi_step.yaml` for a full config example.
"""
    with open(example_metrics_fpath) as f:
        example_metrics = json.load(f)
    assert example_metrics["Number of examples"] == 5, (
        f"Expected 5 examples in the metrics at {example_metrics_fpath}, but got {example_metrics['Number of examples']}"
    )

    conflict_paths = glob(str(test_config.resolved_dir_path / "data/*conflict*"))
    conflict_paths_str = "\n- ".join([""] + conflict_paths)
    assert not conflict_paths, f"Found {len(conflict_paths)} conflicting paths: {conflict_paths_str}"

    example_rollouts_fpath = test_config.resolved_dir_path / "data/example_rollouts.jsonl"
    assert example_rollouts_fpath.exists(), f"""You must run the example data through your agent and provide the example rollouts at `{example_rollouts_fpath}`.

Your commands should look something like:
```bash
# Server spinup
example_multi_step_config_paths="responses_api_models/openai_model/configs/openai_model.yaml,\
resources_servers/example_multi_step/configs/example_multi_step.yaml"
gym env start "+config_paths=[${{example_multi_step_config_paths}}]"

# Collect the rollouts
gym eval run --no-serve +agent_name=example_multi_step_simple_agent \
    +input_jsonl_fpath=resources_servers/example_multi_step/data/example.jsonl \
    +output_jsonl_fpath=resources_servers/example_multi_step/data/example_rollouts.jsonl \
    +limit=null

# View your rollouts
head -1 resources_servers/example_multi_step/data/example_rollouts.jsonl
```
"""
    with open(example_rollouts_fpath) as f:
        count = sum(1 for _ in f)
    assert count == 5, f"Expected 5 example rollouts in {example_rollouts_fpath}, but got {count}"

    print(f"The data for {test_config.dir_path} has been successfully validated!")


def _test_single(test_config: TestConfig, global_config_dict: DictConfig) -> Popen:  # pragma: no cover
    # Eventually we may want more sophisticated testing here, but this is sufficient for now.
    prefix = test_config.entrypoint.replace("/", "\\/")
    resolved_dir = test_config.resolved_dir_path
    pytest_args = ""
    junit_dir_value = os.environ.get("GYM_CI_JUNIT_DIR", "").strip()
    if junit_dir_value:
        junit_dir = Path(junit_dir_value).expanduser().resolve()
        junit_dir.mkdir(parents=True, exist_ok=True)
        entrypoint = test_config.entrypoint.replace("\\", "/").strip("/")
        report_name = f"{entrypoint.replace('/', '__')}.xml"
        junit_path = junit_dir / report_name
        junit_prefix = entrypoint.replace("/", ".")
        pytest_args = f" --junitxml={shlex.quote(str(junit_path))} --junit-prefix={shlex.quote(junit_prefix)}"
    command = f"""{setup_env_command(resolved_dir, global_config_dict, prefix)} && pytest{pytest_args}"""
    # Generated server tests import `resources_servers.<name>...`, so the project root (the dir
    # holding the server-type dirs) must be on PYTHONPATH when running from outside a repo checkout.
    return run_command(command, resolved_dir, project_root=resolved_dir.parent.parent)


def test():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    test_config = TestConfig.model_validate(global_config_dict)

    proc = _test_single(test_config, global_config_dict)
    return_code = proc.wait()
    if return_code != 0:
        print(f"You can run detailed tests via `cd {test_config.entrypoint} && source .venv/bin/activate && pytest`.")
        exit(return_code)

    try:
        _validate_data_single(test_config)
    except AssertionError as e:
        print(f"""{e}
Data validation failed for {test_config.entrypoint}. You can rerun just the data validation like:
```bash
gym env test +entrypoint={test_config.entrypoint} +should_validate_data=true
```
""")
        exit(1)


def _display_list_of_paths(paths: List[Path]) -> str:  # pragma: no cover
    paths = list(map(str, paths))
    return "".join(f"\n- {p}" for p in paths)


def _format_pct(count: int, total: int) -> str:  # pragma: no cover
    return f"{count} / {total} ({100 * count / total:.2f}%)"


class TestAllConfig(BaseNeMoGymCLIConfig):
    """
    Run tests for all server modules in the project.

    Examples:

    ```bash
    gym env test
    ```
    """

    fail_on_total_and_test_mismatch: bool = Field(
        default=False,
        description="Fail if the number of server modules doesn't match the number with tests (default: False).",
    )
    delete_venvs_after_each_test: bool = Field(
        default=False,
        description="Delete each server venv after its tests have been run (default: False).",
    )
    num_shards: int = Field(
        default=1,
        ge=1,
        description="Total number of shards to split the server suite across (default: 1 = no sharding). "
        "Used to parallelize the suite across CI runners.",
    )
    shard_index: int = Field(
        default=0,
        ge=0,
        description="Which shard (0-based) this invocation runs; must be < num_shards (default: 0).",
    )


def _select_shard(dir_paths: List[Path], shard_index: int, num_shards: int) -> List[Path]:
    """Deterministically select this shard's subset of modules.

    Round-robin (stride) over a sorted list spreads heavy modules across shards more evenly than
    contiguous chunks, which balances wall-time when the suite is parallelized across CI runners.
    """
    if num_shards <= 1:
        return dir_paths
    assert 0 <= shard_index < num_shards, (
        f"shard_index ({shard_index}) must be in [0, num_shards) for num_shards={num_shards}"
    )
    return sorted(dir_paths, key=str)[shard_index::num_shards]


def _delete_server_venv(dir_path: Path, global_config_dict: DictConfig) -> None:
    venv_path = get_venv_path(dir_path, global_config_dict)
    print(f"Deleting {venv_path} since `delete_venvs_after_each_test=true`")
    rmtree(venv_path, ignore_errors=True)


def test_all():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    test_all_config = TestAllConfig.model_validate(global_config_dict)

    # Discover server modules across every component-search root: NEMO_GYM_EXTRA_ROOTS (plugins), the cwd
    # (a user's project), and the Gym install root (built-ins, under PARENT_DIR in editable and wheel
    # installs). Entrypoints are kept relative; earlier roots shadow later ones for same-named modules. This
    # lets `gym env test` discover and run built-in and plugin servers from any cwd, not only a repo checkout.
    server_type_dirs = ("resources_servers", "responses_api_agents", "responses_api_models")
    seen_rel_paths: set[str] = set()
    candidate_dir_paths: List[str] = []
    for root in component_search_roots():
        for server_type_dir in server_type_dirs:
            for module_path in sorted((root / server_type_dir).glob("*")):
                if "pycache" in module_path.name or not module_path.is_dir():
                    continue
                rel_path = f"{server_type_dir}/{module_path.name}"
                if rel_path in seen_rel_paths:
                    continue
                seen_rel_paths.add(rel_path)
                candidate_dir_paths.append(rel_path)
    print(f"Found {len(candidate_dir_paths)} total modules:{_display_list_of_paths(candidate_dir_paths)}\n")
    dir_paths: List[Path] = list(map(Path, candidate_dir_paths))
    dir_paths = [p for p in dir_paths if (_resolve_server_dir(p) / "README.md").exists()]
    print(f"Found {len(dir_paths)} modules to test:{_display_list_of_paths(dir_paths)}\n")

    # Keep the full list for the total-vs-tested mismatch check below, then narrow to this shard.
    full_dir_paths = dir_paths
    dir_paths = _select_shard(dir_paths, test_all_config.shard_index, test_all_config.num_shards)
    if test_all_config.num_shards > 1:
        print(
            f"Shard {test_all_config.shard_index + 1}/{test_all_config.num_shards}: "
            f"testing {len(dir_paths)} of {len(full_dir_paths)} modules:{_display_list_of_paths(dir_paths)}\n"
        )

    tests_passed: List[Path] = []
    tests_failed: List[Path] = []
    tests_missing: List[Path] = []
    tests_unrecognized: List[Path] = []
    data_validation_failed: List[Path] = []
    times_taken: List[Tuple[float, Path]] = []
    for dir_path in tqdm(dir_paths, desc="Running tests"):
        start_time = time()

        test_config = TestConfig(
            entrypoint=str(dir_path),
            should_validate_data=True,  # Test all always validates data.
        )
        proc = _test_single(test_config, global_config_dict)
        return_code = proc.wait()

        match return_code:
            case 0:
                tests_passed.append(dir_path)
            case 1 | 2:
                tests_failed.append(dir_path)
            case 5:
                tests_missing.append(dir_path)
            case _:
                tests_unrecognized.append(dir_path)

        try:
            _validate_data_single(test_config)
        except AssertionError:
            data_validation_failed.append(dir_path)

        if test_all_config.delete_venvs_after_each_test:
            _delete_server_venv(_resolve_server_dir(dir_path), global_config_dict)

        times_taken.append((time() - start_time, dir_path))

    times_taken.sort(reverse=True)
    table = Table(title="Times taken per test (sorted from highest to lowest)")
    table.add_column("Server path")
    table.add_column("Time taken (s)")
    for time_taken, dir_path in times_taken:
        table.add_row(str(dir_path), f"{time_taken:.2f}")
    print_rich_table(table)

    print(f"""Found {len(candidate_dir_paths)} total modules:{_display_list_of_paths(candidate_dir_paths)}

Found {len(dir_paths)} modules to test:{_display_list_of_paths(dir_paths)}

Tests passed {_format_pct(len(tests_passed), len(dir_paths))}:{_display_list_of_paths(tests_passed)}

Tests failed {_format_pct(len(tests_failed), len(dir_paths))}:{_display_list_of_paths(tests_failed)}

Tests missing {_format_pct(len(tests_missing), len(dir_paths))}:{_display_list_of_paths(tests_missing)}

Tests that returned unrecognized exit codes {_format_pct(len(tests_unrecognized), len(dir_paths))}:{_display_list_of_paths(tests_unrecognized)}

Data validation failed {_format_pct(len(data_validation_failed), len(dir_paths))}:{_display_list_of_paths(data_validation_failed)}
""")

    if tests_failed or tests_missing or tests_unrecognized:
        print(f"""You can rerun just the server with failed, missing or unrecognized tests results like:
```bash
gym env test +entrypoint={(tests_failed + tests_missing + tests_unrecognized)[0]}
```
""")
    if data_validation_failed:
        print(f"""You can rerun just the server with failed data validation like:
```bash
gym env test +entrypoint={data_validation_failed[0]} +should_validate_data=true
```
""")

    if test_all_config.fail_on_total_and_test_mismatch:
        # Compare against the full (unsharded) module list — every module must be testable
        # regardless of how many shards we split the run into.
        extra_candidates = [p for p in candidate_dir_paths if Path(p) not in full_dir_paths]
        assert (
            len(candidate_dir_paths) == len(full_dir_paths)
        ), f"""Mismatch on the number of total modules found ({len(candidate_dir_paths)}) and the number of actual modules tested ({len(full_dir_paths)})!

Extra candidate paths:{_display_list_of_paths(extra_candidates)}"""

    if tests_missing or tests_failed or data_validation_failed:
        exit(1)


class InitEnvironmentConfig(BaseNeMoGymCLIConfig):
    scaffold_kind: EnvironmentKind
    scaffold_name: str
    profile: IntegrationProfile = IntegrationProfile.CUSTOM_GYM_VERIFIER
    reuse_verifier: Optional[str] = None
    reward_range: Optional[Tuple[float, float]] = None
    higher_is_better: Optional[bool] = None


def _command_overrides() -> DictConfig:
    """Parse the already-translated local command flags without resolving Gym runtime config."""
    return OmegaConf.from_dotlist([token.lstrip("+") for token in sys.argv[1:] if "=" in token])


@exit_cleanly_on_config_error
def init_environment() -> None:
    """Create a manifest-backed environment or benchmark skeleton."""
    config = InitEnvironmentConfig.model_validate(_command_overrides())
    result = scaffold_environment(
        kind=config.scaffold_kind,
        name=config.scaffold_name,
        profile=config.profile,
        reuse_verifier=config.reuse_verifier,
        reward_range=config.reward_range,
        higher_is_better=config.higher_is_better,
    )

    if result.created:
        rich.print(f"[green]✓[/green] Created {result.asset_dir}")
        for path in result.created:
            print(f"  {path}")
    else:
        rich.print(f"[green]✓[/green] {result.asset_dir} is already up to date.")


def init_resources_server():  # pragma: no cover
    """
    Initialize a new resources server with template files and directory structure.

    Examples:

    ```bash
    gym env init --resources-server my_server
    ```
    """
    config_dict = get_global_config_dict()
    run_config = RunConfig.model_validate(config_dict)

    dirpath = Path(run_config.entrypoint).resolve()

    if dirpath.exists():
        print(f"Folder already exists: {dirpath}. Exiting init.")
        exit()

    checkout_root = PARENT_DIR if (PARENT_DIR / "pyproject.toml").is_file() else None
    scaffold_resources_server(directory=dirpath, checkout_root=checkout_root)


@exit_cleanly_on_config_error
def dump_config():  # pragma: no cover
    """
    Display the resolved Hydra configuration for debugging purposes.

    Examples:

    ```bash
    gym env resolve "+config_paths=[<config1>,<config2>]"
    ```
    """
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            hide_secrets=True,
        ),
    )

    # Just here for help
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    print(OmegaConf.to_yaml(global_config_dict, resolve=True))


class ManifestCommandConfig(BaseNeMoGymCLIConfig):
    onboarding_name: Optional[str] = None
    catalog_kind: Optional[EnvironmentKind] = None
    manifest_path: Optional[Path] = None
    sync: bool = False
    update_expected: bool = False


_MANIFEST_VALIDATE_KEYS = frozenset({"onboarding_name", "catalog_kind", "manifest_path", "sync", "json", "verbose"})
_MANIFEST_TEST_KEYS = frozenset({"onboarding_name", "catalog_kind", "update_expected", "json", "verbose"})
_MANIFEST_PUBLISH_KEYS = frozenset({"onboarding_name", "catalog_kind", "json", "verbose"})


def _reject_manifest_command_extras(command_dict: DictConfig, allowed: frozenset[str]) -> None:
    extras = sorted(set(command_dict) - allowed)
    if extras:
        raise ConfigError("Manifest-backed commands do not accept runtime config overrides: " + ", ".join(extras))


def _manifest_entry(config: ManifestCommandConfig) -> Optional[EnvironmentCatalogEntry]:
    if config.onboarding_name is not None and config.manifest_path is not None:
        raise ConfigError("select a catalog name or --manifest, not both")
    if config.manifest_path is not None and config.catalog_kind is not None:
        raise ConfigError("--kind is only valid when selecting a catalog name")
    if config.onboarding_name is None:
        return None
    return resolve_catalog_entry(config.onboarding_name, config.catalog_kind)


def _print_validation_report(report: EnvironmentValidationReport, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(report.to_dict()))
        return
    rich.print(
        f"[green]✓[/green] Manifest: {report.name} {report.version} "
        f"kind={report.kind} profile={report.declared_profile} inferred={report.inferred_profile}"
    )
    print(f"    Profile evidence: {report.profile_evidence}")
    rich.print("[green]✓[/green] Components:")
    for component in report.components:
        print(f"    {component.role}: {component.name} -> {component.implementation} [{component.boundary}]")
    rich.print("[green]✓[/green] Datasets:")
    for dataset in report.datasets:
        rich.print(f"    {dataset.name}: {dataset.rows} rows ({dataset.type})")
    if report.synchronized_fields:
        rich.print(f"[green]✓[/green] Synchronized: {', '.join(report.synchronized_fields)}")
    for warning in report.warnings:
        rich.print(f"[yellow]Warning:[/yellow] {warning}", file=sys.stderr)


@exit_cleanly_on_config_error
def validate() -> None:
    """Validate a manifest-backed workload or a legacy Gym config without starting services."""
    command_dict = _command_overrides()
    command_config = ManifestCommandConfig.model_validate(command_dict)
    entry = _manifest_entry(command_config)
    if command_config.manifest_path is not None or entry is not None:
        _reject_manifest_command_extras(command_dict, _MANIFEST_VALIDATE_KEYS)
        manifest_path = command_config.manifest_path if entry is None else entry.manifest_path
        if manifest_path is None:
            raise ConfigError(
                f"'{entry.name}' has no manifest; validate its config with --environment or --benchmark."
            )
        report = validate_environment(
            manifest_path,
            entry.config_path if entry is not None else None,
            sync=command_config.sync,
        )
        _print_validation_report(report, json_output=command_dict.get(JSON_OUTPUT_KEY_NAME, False))
        return
    unsupported = [
        key for key in ("catalog_kind", "sync", "update_expected", "json") if command_dict.get(key) is not None
    ]
    if unsupported:
        raise ConfigError("The following options require a workload name or --manifest: " + ", ".join(unsupported))

    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        ),
    )
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)
    rich.print("[green]✓[/green] Config is valid.")


@exit_cleanly_on_config_error
def test_environment_manifest() -> None:
    """Exercise a manifest-backed workload's verifier fixture without starting services."""
    command_dict = _command_overrides()
    command_config = ManifestCommandConfig.model_validate(command_dict)
    entry = _manifest_entry(command_config)
    if entry is None:
        raise ConfigError("gym env test requires a catalog name or --resources-server")
    _reject_manifest_command_extras(command_dict, _MANIFEST_TEST_KEYS)
    report = _run_manifest_verifier(entry, update_expected=command_config.update_expected)
    if command_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps(report.to_dict()))
        return
    rich.print(
        f"[green]✓[/green] Verifier: {report.name} ({report.resources_server}) {len(report.cases)} cases passed."
    )


@exit_cleanly_on_config_error
def publish_environment_manifest() -> None:
    """Run local publication checks and confirm a workload is cataloged."""
    command_dict = _command_overrides()
    command_config = ManifestCommandConfig.model_validate(command_dict)
    entry = _manifest_entry(command_config)
    if entry is None:
        raise ConfigError("gym env publish requires a catalog name")
    _reject_manifest_command_extras(command_dict, _MANIFEST_PUBLISH_KEYS)
    if entry.manifest_path is None:
        raise ConfigError(f"'{entry.name}' has no manifest and cannot be published.")

    validation = validate_environment(entry.manifest_path, entry.config_path)
    verifier = _run_manifest_verifier(entry, update_expected=False, validation=validation)
    report = finalize_publication(entry, validation, verifier)
    if command_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps(report.to_dict()))
        return
    rich.print(
        f"[green]✓[/green] Publication checks passed for {report.kind} {report.name} {report.version}; "
        f"catalog status={report.status} ({report.verifier_cases} verifier cases)."
    )


def _run_manifest_verifier(
    entry: EnvironmentCatalogEntry,
    *,
    update_expected: bool,
    validation: EnvironmentValidationReport | None = None,
) -> VerifierReport:
    spec = prepare_verifier_run(entry, validation)
    setup_config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    server_dir = Path(spec.server_dir)
    prefix = f"resources_servers/{spec.resources_server}".replace("/", "\\/")

    with TemporaryDirectory(prefix="nemo-gym-verifier-") as temporary_dir:
        request_path = Path(temporary_dir) / "request.json"
        result_path = Path(temporary_dir) / "result.json"
        request_path.write_text(
            json.dumps({"spec": spec.to_dict(), "update_expected": update_expected}),
            encoding="utf-8",
        )
        python = get_venv_path(server_dir, setup_config) / "bin" / "python"
        runner = (
            f"{shlex.quote(str(python))} -m nemo_gym.environment._verifier_runner "
            f"--request {shlex.quote(str(request_path))} --result {shlex.quote(str(result_path))}"
        )
        command = f"{setup_env_command(server_dir, setup_config, prefix)} && {runner}"
        process = run_command(
            command,
            server_dir,
            server_name=spec.resources_server,
            project_root=PARENT_DIR if (PARENT_DIR / "pyproject.toml").is_file() else None,
            global_config_dict=setup_config,
            stdout_target=sys.stderr,
            stderr_target=sys.stderr,
        )
        return_code = process.wait()
        if not result_path.is_file():
            raise EnvironmentOnboardingError(
                f"Verifier runner exited with code {return_code} without producing a result."
            )
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise EnvironmentOnboardingError(f"Could not read verifier result: {error}") from error
        if not isinstance(result, dict) or not result.get("ok"):
            message = result.get("error") if isinstance(result, dict) else None
            raise EnvironmentOnboardingError(message or f"Verifier runner failed with code {return_code}.")
        if return_code != 0:
            raise EnvironmentOnboardingError(f"Verifier runner exited with code {return_code}.")
        try:
            return VerifierReport.from_dict(result["report"])
        except (KeyError, TypeError, ValueError) as error:
            raise EnvironmentOnboardingError(f"Verifier runner returned an invalid report: {error}") from error


def _inspect_environment(
    name: str,
    entries: Tuple[EnvironmentCatalogEntry, ...],
    global_config_dict: DictConfig,
) -> None:
    """Render one entry from the unified environment catalog."""
    kind = global_config_dict.get("catalog_kind")
    matching_names = {entry.name: entry for entry in entries}
    if name not in matching_names:
        exit_unknown_component(name, matching_names, "environment")
        return
    entry = resolve_catalog_entry(name, kind, entries=entries)
    parsed = read_environment_details(entry.config_path)
    details = {"config": str(entry.config_path.resolve()), "status": entry.status}
    if entry.manifest_path is not None:
        details["manifest"] = str(entry.manifest_path.resolve())
    for label, value in (
        ("version", entry.version),
        ("profile", entry.integration_profile),
        ("modality", entry.modality),
        ("licensing", entry.licensing),
        ("lifecycle", entry.lifecycle),
    ):
        if value:
            details[label] = value
    if parsed["resources_servers"]:
        details["resources servers"] = ", ".join(parsed["resources_servers"])
    if parsed["agent"]:
        details["agent"] = parsed["agent"]
    if parsed["datasets"]:
        details["datasets"] = ", ".join(parsed["datasets"])

    description = entry.description or parsed["description"]
    if parsed["value"]:  # surface `value` as a trailing line of the description
        description = f"{description}\nValue: {parsed['value']}" if description else f"Value: {parsed['value']}"

    render_component_inspection(
        json_output=global_config_dict.get(JSON_OUTPUT_KEY_NAME, False),
        name=name,
        type_noun=entry.kind,
        domain=entry.domain or parsed["domain"],
        description=description,
        details=details,
        usage=(
            f"gym env start --resources-server {entry.resources_server_selector} --model-type vllm_model"
            if entry.resources_server_selector
            else f"gym env start --{entry.kind} {name} --model-type vllm_model"
        ),
    )


def _catalog_payload(entry: EnvironmentCatalogEntry) -> Dict[str, object]:
    return {
        "name": entry.name,
        "kind": entry.kind,
        "status": entry.status,
        "domain": entry.domain,
        "description": entry.description,
        "version": entry.version,
        "integration_profile": entry.integration_profile,
        "modality": entry.modality,
        "licensing": entry.licensing,
        "lifecycle": entry.lifecycle,
    }


@exit_cleanly_on_config_error
def list_environments() -> None:
    """List or inspect the manifest and legacy environment/benchmark catalog."""
    global_config_dict = _command_overrides()
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    discovered = discover_environment_catalog()
    entries = list(discovered)

    name = global_config_dict.get(COMPONENT_NAME_KEY_NAME)
    if name:
        _inspect_environment(name, discovered, global_config_dict)
        return

    query = global_config_dict.get(QUERY_KEY_NAME)
    if query:
        entries = [
            entry for entry in entries if fuzzy_matches(query, entry.name, entry.domain or "", entry.description or "")
        ]
    for field in ("domain", "catalog_kind", "modality", "licensing", "status", "lifecycle"):
        expected = global_config_dict.get(field)
        if expected is None:
            continue
        attribute = "kind" if field == "catalog_kind" else field
        missing = sum(getattr(entry, attribute) is None for entry in entries)
        if missing:
            noun = "entry" if missing == 1 else "entries"
            print(
                f"Warning: {missing} catalog {noun} {'has' if missing == 1 else 'have'} no "
                f"{attribute} metadata and could not be matched.",
                file=sys.stderr,
            )
        entries = [entry for entry in entries if getattr(entry, attribute) == expected]

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps([_catalog_payload(entry) for entry in entries]))
        return

    if not entries:
        print_no_matches("environments", query)
        return

    coverage = sum(entry.manifest_path is not None for entry in discovered)
    title = (
        f"Catalog entries matching '{query}' ({len(entries)}; manifests {coverage}/{len(discovered)})"
        if query
        else f"Available environments and benchmarks ({len(entries)}; manifests {coverage}/{len(discovered)})"
    )
    table = Table(title=title)
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Lifecycle")
    table.add_column("Domain")
    table.add_column("Description")
    for entry in entries:
        table.add_row(
            entry.name,
            entry.kind,
            entry.status,
            entry.lifecycle or "",
            entry.domain or "",
            entry.description or "",
        )

    print_rich_table(table)


@exit_cleanly_on_config_error
def status():  # pragma: no cover
    global_config_dict = get_global_config_dict()
    BaseNeMoGymCLIConfig.model_validate(global_config_dict)

    status_cmd = StatusCommand()
    servers = status_cmd.discover_servers()

    if global_config_dict.get(JSON_OUTPUT_KEY_NAME, False):
        print(json.dumps([server.model_dump(mode="json") for server in servers]))
        return

    status_cmd.display_status(servers)


class PipListConfig(RunConfig):
    format: Optional[str] = Field(
        default=None,
        description="Output format for pip list. Options: 'columns' (default), 'freeze', 'json'",
    )
    outdated: bool = Field(
        default=False,
        description="List outdated packages",
    )


@exit_cleanly_on_config_error
def pip_list():  # pragma: no cover
    """List packages installed in a server's virtual environment."""
    global_config_dict = get_global_config_dict()
    config = PipListConfig.model_validate(global_config_dict)

    dir_path = _resolve_server_dir(Path(config.entrypoint))
    venv_path = dir_path / ".venv"

    if not venv_path.exists():
        print(f"  Virtual environment not found at: {venv_path}")
        print("  Run tests or setup the server first using:")
        print(f"  gym env test +entrypoint={config.entrypoint}")
        exit(1)

    pip_list_cmd = "uv pip list"
    if config.format:
        pip_list_cmd += f" --format={config.format}"
    if config.outdated:
        pip_list_cmd += " --outdated"

    command = f"""cd {dir_path} \\
    && source .venv/bin/activate \\
    && {pip_list_cmd}"""

    # Only in the default human view: an explicit --format (json, freeze) is meant to be piped, so stdout
    # must carry nothing but `uv pip list` output.
    if not config.format:
        print(f"  Package list for: {config.entrypoint}")
        print(f"Virtual environment: {venv_path.absolute()}")
        print("-" * 72)

    proc = run_command(command, dir_path)
    return_code = proc.wait()
    exit(return_code)
