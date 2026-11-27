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
"""Start, watch and stop the OpenTelemetry Collector that `telemetry.collector` asks for.

`ServerSupervisor` starts it before its own telemetry and before any server, so nothing they export is dropped,
and stops it after the last server has exited, so the final values each server flushes on its way out are
collected. A collector that exits during the run is started again, up to `max_restarts` times with a growing
delay. That never fails the run: what is exported while no collector runs is lost, and the log says so.

Each collector process writes its JSONL into a scratch directory of its own, and once it has exited those files
are appended to the run's, as the Slurm step does. The file exporter truncates its file when it starts, so a
collector started on another's files would erase them.
"""

import getpass
import os
import secrets
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import IO, Any, Optional
from urllib.error import URLError
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, build_opener

import yaml

from nemo_gym.config.global_config import NEMO_GYM_LOG_DIR_KEY_NAME
from nemo_gym.telemetry.collector import COLLECTOR_FILE_SIGNALS, collector_document, node_metrics_filter
from nemo_gym.telemetry.config import CollectorConfig, TelemetryConfig


COLLECTOR_DIR = "otel"
COLLECTOR_CONFIG_NAME = "collector.yaml"
COLLECTOR_LOG_NAME = "collector.log"
# Where the unfinished last record of a killed collector goes, beside `<signal>.jsonl`.
TRUNCATED_SUFFIX = ".jsonl.truncated"
LISTEN_HOST = "127.0.0.1"
_LOG_TAIL_LINES = 20
_HEALTH_POLL_SEC = 0.2
_HEALTH_REQUEST_TIMEOUT_SEC = 1.0
_KILL_REAP_TIMEOUT_SEC = 5.0
_WATCHER_JOIN_TIMEOUT_SEC = 5.0
_COPY_CHUNK_BYTES = 1 << 20
_FIRST_RESTART_DELAY_SEC = 1.0
_MAX_RESTART_DELAY_SEC = 30.0
# A collector that ran this long before it exited is started again after the shortest delay.
_STABLE_RUN_SEC = 60.0
_OTLP_ENV = "OTEL_EXPORTER_OTLP"
# The OTel SDK's protocol when none is set; nemo-lens uses it too.
_SDK_DEFAULT_PROTOCOL = "grpc"
_HTTP_PROTOCOLS = ("http/protobuf", "http/json")
_EXPORTER_ENVS = ("NEMO_GYM_OTEL_EXPORTER", "NEMO_LENS_EXPORTER")
_PROXY_ENVS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "GRPC_PROXY")
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost")
# The health probe goes to loopback directly, whatever HTTP_PROXY says.
_DIRECT = build_opener(ProxyHandler({}))


class CollectorError(RuntimeError):
    """The collector could not be configured or started."""


def _literal(value: str) -> str:
    """`value` as written, not expanded: the collector expands `${...}` anywhere in its configuration."""
    return value.replace("$", "$$")


def render_local_collector_config(config: CollectorConfig, *, run_id: str, file_dir: str, remote_export: bool) -> str:
    """The collector's YAML for one run of `gym env start`, writing its JSONL into `file_dir`."""
    interval = f"{config.scrape_interval_seconds}s"
    # The job name is what scraped data carries as `service.name`, and so becomes its display identity.
    scrape_configs = [
        {
            "job_name": _literal(f"{config.component}/{name}"),
            "scrape_interval": interval,
            "static_configs": [{"targets": [target], "labels": {"gym_service": _literal(name)}}],
        }
        for name, target in config.scrape_targets.items()
    ]
    # The conventional job names, so the standard DCGM and node_exporter dashboards' queries apply unchanged.
    for job_name, port in (("dcgm", config.gpu_metrics_port), ("node", config.node_metrics_port)):
        if port is not None:
            scrape_configs.append(
                {
                    "job_name": job_name,
                    "scrape_interval": interval,
                    "static_configs": [{"targets": [f"localhost:{port}"]}],
                }
            )
    attributes = {
        **({"service.name": config.service_name} if config.service_name is not None else {}),
        "user": getpass.getuser(),
        "run_id": run_id,
        "host": socket.gethostname(),
        **({"slurm_job_id": os.environ["SLURM_JOB_ID"]} if os.environ.get("SLURM_JOB_ID") else {}),
    }
    resource_actions = [{"key": k, "value": _literal(v), "action": "upsert"} for k, v in attributes.items()]
    # `insert` only adds: a producer's own identity (`service.instance.id`, `service.name.override`, ...) stays.
    resource_actions += [
        {"key": _literal(k), "value": _literal(v), "action": "insert"} for k, v in config.resource_attributes.items()
    ]
    doc = collector_document(
        scrape_configs=scrape_configs,
        resource_actions=resource_actions,
        file_dir=file_dir,
        remote_export=remote_export,
        endpoint=config.endpoint,
        token_env=config.token_env,
        scrape_interval_seconds=config.scrape_interval_seconds,
        engine_rollout_ids=True,
        node_filter=node_metrics_filter(config.local_node_metrics, config.local_node_metrics_extra)
        if config.node_metrics_port is not None
        else None,
        otlp_grpc_port=config.otlp_grpc_port,
        otlp_http_port=config.otlp_http_port,
        health_port=config.health_port,
        listen_host=LISTEN_HOST,
    )
    return yaml.safe_dump(doc, sort_keys=False)


def _log_tail(log_path: Path) -> str:
    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(lines[-_LOG_TAIL_LINES:]) or "(empty log)"


def _ports_in_use(config: CollectorConfig) -> list[int]:
    """The collector's ports that something on this host already holds."""
    taken = []
    for port in (config.otlp_grpc_port, config.otlp_http_port, config.health_port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            # As the collector binds: connections a previous run left in TIME_WAIT do not hold the port.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((LISTEN_HOST, port))
            except OSError:
                taken.append(port)
    return taken


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _protocol(value: str) -> str:
    """`value` as nemo-lens reads it."""
    return value.strip().lower()


def _transport(protocol: str) -> str:
    """What nemo-lens builds for `protocol`: an HTTP exporter for the HTTP protocols, else a gRPC one."""
    return "http" if protocol in _HTTP_PROTOCOLS else "grpc"


def _collector_port(config: CollectorConfig, protocol: str) -> int:
    return config.otlp_http_port if _transport(protocol) == "http" else config.otlp_grpc_port


def _collector_endpoint(config: CollectorConfig, protocol: str, signal: Optional[str] = None) -> str:
    endpoint = f"http://{LISTEN_HOST}:{_collector_port(config, protocol)}"
    # A signal's own HTTP endpoint is used as given; the general one gets `/v1/<signal>` appended.
    return f"{endpoint}/v1/{signal}" if signal is not None and _transport(protocol) == "http" else endpoint


def _is_collector(endpoint: str, config: CollectorConfig, protocol: str) -> bool:
    try:
        parts = urlsplit(endpoint)
        return parts.hostname in _LOOPBACK_HOSTS and parts.port == _collector_port(config, protocol)
    except ValueError:
        return False


def _bypass_proxies_for_loopback() -> None:
    """Keep exporters to the collector off any proxy the environment names; both spellings of NO_PROXY get it."""
    if not any(_env(name) or _env(name.lower()) for name in _PROXY_ENVS):
        return
    entries: list[str] = []
    for name in ("no_proxy", "NO_PROXY"):
        entries += [entry.strip() for entry in os.environ.get(name, "").split(",") if entry.strip()]
    merged = list(dict.fromkeys(entries + list(_LOOPBACK_HOSTS)))
    if merged != entries:
        os.environ["no_proxy"] = os.environ["NO_PROXY"] = ",".join(merged)


def point_exporters_at_collector(config: CollectorConfig) -> None:
    """Make the collector the OTLP destination of this process and the servers it starts, unless one is set.

    Reads the environment `configure_telemetry_env` has written, so `telemetry.otlp_endpoint` and
    `telemetry.otlp_protocol` count as set, and resolves each signal as nemo-lens does: its own protocol, else the
    general one, else gRPC. Without an endpoint, the collector's port for the protocol is used, HTTP when there is
    none, and a signal whose own protocol needs the other port gets its own endpoint. An endpoint that is already set
    is kept, with a warning for each signal it sends somewhere other than the collector.
    """
    exporter = next((_env(name) for name in _EXPORTER_ENVS if _env(name)), "otlp")
    if exporter != "otlp":
        print(
            f"WARNING: The telemetry exporter is {exporter!r}, not 'otlp', so Gym's own telemetry does not reach the "
            "collector. The collector still scrapes its targets."
        )
        return
    _bypass_proxies_for_loopback()
    general_endpoint = _env(f"{_OTLP_ENV}_ENDPOINT")
    general_protocol = _protocol(_env(f"{_OTLP_ENV}_PROTOCOL"))
    supplied = not general_endpoint
    if supplied:
        general_protocol = general_protocol or "http/protobuf"
        general_endpoint = _collector_endpoint(config, general_protocol)
        os.environ[f"{_OTLP_ENV}_PROTOCOL"] = general_protocol
        os.environ[f"{_OTLP_ENV}_ENDPOINT"] = general_endpoint
    elsewhere: dict[tuple[str, str], list[str]] = {}
    for signal in ("traces", "metrics", "logs"):
        endpoint_env = f"{_OTLP_ENV}_{signal.upper()}_ENDPOINT"
        protocol = (
            _protocol(_env(f"{_OTLP_ENV}_{signal.upper()}_PROTOCOL")) or general_protocol or _SDK_DEFAULT_PROTOCOL
        )
        endpoint = _env(endpoint_env)
        if not endpoint and supplied and _transport(protocol) != _transport(general_protocol):
            os.environ[endpoint_env] = _collector_endpoint(config, protocol, signal)
            continue
        endpoint = endpoint or general_endpoint
        if not _is_collector(endpoint, config, protocol):
            elsewhere.setdefault((endpoint, protocol), []).append(signal)
    for (endpoint, protocol), signals in elsewhere.items():
        print(
            f"WARNING: Gym exports its {', '.join(signals)} to {endpoint} over {protocol}, which is not the "
            f"collector's {_transport(protocol)} port ({_collector_port(config, protocol)}), so they do not reach it. "
            "The collector still scrapes its targets."
        )


def _end_of_last_line(source: IO[bytes], size: int) -> int:
    """The offset just past the last newline in `source`, or 0 when it has none."""
    end = size
    while end > 0:
        start = max(0, end - _COPY_CHUNK_BYTES)
        source.seek(start)
        index = source.read(end - start).rfind(b"\n")
        if index >= 0:
            return start + index + 1
        end = start
    return 0


def _write_all(destination: IO[bytes], data: bytes) -> None:
    """Write `data` to an unbuffered file, which can write less than it is given."""
    view = memoryview(data)
    while view:
        view = view[destination.write(view) :]


def _copy_prefix(source: IO[bytes], destination: IO[bytes], length: int) -> None:
    source.seek(0)
    remaining = length
    while remaining > 0:
        chunk = source.read(min(_COPY_CHUNK_BYTES, remaining))
        if not chunk:
            break
        _write_all(destination, chunk)
        remaining -= len(chunk)


def _roll_back(sizes: dict[Path, Optional[int]]) -> None:
    """Return each file to its size in `sizes`; one that did not exist is removed."""
    for path, size in sizes.items():
        try:
            if size is None:
                path.unlink(missing_ok=True)
            else:
                os.truncate(path, size)
        except OSError as e:
            print(
                f"WARNING: Could not undo a failed append to {path}: {e}. It may end in a partial or repeated record."
            )


class LocalCollector:
    """The collector processes of one run, one at a time, and the thread that starts the next when one exits."""

    def __init__(self, config: CollectorConfig, *, binary: str, run_dir: Path, run_id: str, remote_export: bool):
        self._config = config
        self._binary = binary
        self._run_id = run_id
        self._remote_export = remote_export
        self.run_dir = run_dir
        self.config_path = run_dir / COLLECTOR_CONFIG_NAME
        self.log_path = run_dir / COLLECTOR_LOG_NAME
        self._scratch = Path(tempfile.mkdtemp(prefix="gym-otel-"))
        self._spawned = 0
        self._process: Optional[subprocess.Popen] = None
        self._started_at = 0.0
        self._stopping = threading.Event()
        # Held to start a process, so none starts once `stop` has begun.
        self._process_lock = threading.Lock()
        # Held to append exited collectors' files to the run directory.
        self._files_lock = threading.Lock()
        self._watcher: Optional[threading.Thread] = None
        self.restarts = 0

    @property
    def pid(self) -> Optional[int]:
        return self._process.pid if self._process is not None else None

    def start(self) -> None:
        """Start the collector and wait until it answers its health check; raise `CollectorError` if it does not."""
        try:
            with self._process_lock:
                self._spawn()
            error = self._wait_healthy()
            if error is not None:
                raise CollectorError(
                    f"The OpenTelemetry Collector {error}. Config {self.config_path}, log {self.log_path}:\n"
                    f"{_log_tail(self.log_path)}"
                )
        except OSError as e:
            self.stop()
            raise CollectorError(f"The OpenTelemetry Collector {self._binary} could not be started: {e}") from e
        except BaseException:
            self.stop()
            raise
        self._watcher = threading.Thread(target=self._watch, name="otel-collector-watcher", daemon=True)
        self._watcher.start()

    def _spawn(self) -> None:
        """Start a collector writing into a scratch directory no earlier collector used."""
        self._spawned += 1
        file_dir = self._scratch / f"{self._spawned:04d}"
        file_dir.mkdir()
        self.config_path.write_text(
            render_local_collector_config(
                self._config, run_id=self._run_id, file_dir=str(file_dir), remote_export=self._remote_export
            )
        )
        with open(self.log_path, "ab") as log:
            # Own session: Ctrl-C reaches the servers but not the collector, which has to outlive them to receive
            # what they flush on exit. `stop` ends it afterwards.
            self._process = subprocess.Popen(
                [self._binary, "--config", str(self.config_path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        self._started_at = time.monotonic()

    def _wait_healthy(self) -> Optional[str]:
        """None once the running collector answers its health check, else what went wrong."""
        process = self._process
        url = f"http://{LISTEN_HOST}:{self._config.health_port}/"
        timeout = self._config.health_check_timeout_seconds
        deadline = time.monotonic() + timeout
        while True:
            code = process.poll()
            if code is not None:
                return f"exited with code {code} before it was ready"
            try:
                with _DIRECT.open(url, timeout=_HEALTH_REQUEST_TIMEOUT_SEC) as response:
                    if response.status == 200:
                        return None
            except (URLError, OSError):
                pass
            if time.monotonic() >= deadline:
                return f"did not answer its health check at {url} within {timeout:g}s"
            if self._stopping.wait(_HEALTH_POLL_SEC):
                return "was stopped before it was ready"

    def _terminate(self, process: subprocess.Popen) -> None:
        """SIGTERM, so it flushes; SIGKILL after `shutdown_timeout_seconds`."""
        if process.poll() is not None:
            return
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=self._config.shutdown_timeout_seconds)
        except subprocess.TimeoutExpired:
            print(
                f"WARNING: The OpenTelemetry Collector did not exit within "
                f"{self._config.shutdown_timeout_seconds:g}s of SIGTERM; killing it. What it had not written is lost."
            )
            process.kill()
            try:
                process.wait(timeout=_KILL_REAP_TIMEOUT_SEC)
            except subprocess.TimeoutExpired:
                pass

    def _watch(self) -> None:
        delay = _FIRST_RESTART_DELAY_SEC
        while True:
            process = self._process
            if process is not None:
                code = process.wait()
                what = f"exited with code {code}"
                if time.monotonic() - self._started_at >= _STABLE_RUN_SEC:
                    delay = _FIRST_RESTART_DELAY_SEC
            else:
                what = "could not be started"
            if self._stopping.is_set():
                return
            self._collect()
            if self.restarts >= self._config.max_restarts:
                print(
                    f"WARNING: The OpenTelemetry Collector {what} after {self.restarts} restart(s), the most "
                    "`telemetry.collector.max_restarts` allows. The run goes on without it; nothing exported from "
                    f"now on is collected. Log {self.log_path}:\n{_log_tail(self.log_path)}"
                )
                return
            print(
                f"WARNING: The OpenTelemetry Collector {what}; starting it again in {delay:g}s. What is exported "
                f"until then is lost. Log {self.log_path}:\n{_log_tail(self.log_path)}"
            )
            if self._stopping.wait(delay):
                return
            delay = min(delay * 2, _MAX_RESTART_DELAY_SEC)
            with self._process_lock:
                if self._stopping.is_set():
                    return
                self.restarts += 1
                try:
                    self._spawn()
                except OSError as e:
                    self._process = None
                    print(f"WARNING: The OpenTelemetry Collector {self._binary} could not be started again: {e}")
                    continue
            error = self._wait_healthy()
            if error is None:
                print(f"The OpenTelemetry Collector is running again (pid {self._process.pid}).")
            elif not self._stopping.is_set():
                print(f"WARNING: The restarted OpenTelemetry Collector {error}; stopping it. Log {self.log_path}")
                self._terminate(self._process)

    def _collect(self) -> bool:
        """Append the files of every exited collector to the run directory, oldest first.

        Returns False when that failed: the files from there on stay in the scratch directory, in order.
        """
        with self._files_lock:
            running = self._process is not None and self._process.poll() is None
            for file_dir in sorted(path for path in self._scratch.iterdir() if path.is_dir()):
                if running and file_dir.name == f"{self._spawned:04d}":
                    break
                try:
                    for name in COLLECTOR_FILE_SIGNALS:
                        self._append(file_dir / f"{name}.jsonl", name)
                    file_dir.rmdir()
                except OSError as e:
                    print(
                        f"WARNING: Could not append the collector's files from {file_dir} to {self.run_dir}: {e}. "
                        "They, and any later ones, are left where they are."
                    )
                    return False
            return True

    def _append(self, source_path: Path, name: str) -> None:
        """Append one exited collector's file to the run's, then remove it; on failure, change neither."""
        if not source_path.exists():
            return
        destination_path = self.run_dir / f"{name}.jsonl"
        truncated_path = self.run_dir / f"{name}{TRUNCATED_SUFFIX}"
        # Undone on failure, so appending the same file again later neither repeats nor splits a record.
        sizes = {path: path.stat().st_size if path.exists() else None for path in (destination_path, truncated_path)}
        try:
            with open(source_path, "rb") as source:
                size = os.fstat(source.fileno()).st_size
                # A killed collector can leave its last record unfinished. It is kept apart, so every line of the
                # run's file is a whole record.
                complete = _end_of_last_line(source, size)
                # Unbuffered, so a failed write is not written again by a flush on close.
                with open(destination_path, "ab", buffering=0) as destination:
                    _copy_prefix(source, destination, complete)
                if complete < size:
                    source.seek(complete)
                    with open(truncated_path, "ab", buffering=0) as destination:
                        _write_all(destination, source.read() + b"\n")
            source_path.unlink()
        except OSError:
            _roll_back(sizes)
            raise
        if complete < size:
            print(
                f"WARNING: A collector that exited left an unfinished {name} record ({size - complete} bytes); "
                f"it is in {truncated_path}, not in {name}.jsonl."
            )

    def stop(self) -> None:
        """Ask the collector to flush and exit, and keep its files. Starts no other collector afterwards."""
        self._stopping.set()
        with self._process_lock:
            process = self._process
        if process is not None:
            self._terminate(process)
        watcher = self._watcher
        if watcher is not None and watcher is not threading.current_thread():
            watcher.join(timeout=_WATCHER_JOIN_TIMEOUT_SEC)
        if self._collect():
            shutil.rmtree(self._scratch, ignore_errors=True)


def _path_component(value: str) -> str:
    """`value` as one directory name: a run id is any string, `/` and `..` included."""
    component = quote(value, safe="-_.~@+=,")
    return f"_{component}" if component in ("", ".", "..") else component


def collector_instance() -> str:
    """`<host>-<pid>-<random>`: one supervisor's name, even beside another with the same host name and PID in
    another container."""
    return f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(4)}"


def collector_run_dir(config: CollectorConfig, run_id: str, global_config_dict: Any, instance: str) -> Path:
    """`<output>/otel/<run id>/<instance>/`: a run id can be shared, for one Slurm job's processes on several nodes."""
    base = config.output_dir or global_config_dict.get(NEMO_GYM_LOG_DIR_KEY_NAME) or os.getcwd()
    return Path(base).expanduser().resolve() / COLLECTOR_DIR / _path_component(run_id) / _path_component(instance)


def start_local_collector(
    telemetry_config: TelemetryConfig, run_id: Optional[str], global_config_dict: Any
) -> Optional[LocalCollector]:
    """Start the collector when `telemetry.collector` is set and telemetry is on; returns it to stop later.

    Call after `configure_telemetry_env`, which returns `run_id`, and before anything exports: this also points
    the exporters at the collector. Raises `CollectorError` when the collector cannot start.
    """
    config = telemetry_config.collector
    if config is None:
        return None
    if run_id is None:
        print(
            "WARNING: telemetry.collector is set but telemetry is off (`telemetry.enabled`, NEMO_GYM_OTEL_ENABLED), "
            "so no collector is started."
        )
        return None
    binary = shutil.which(os.path.expanduser(config.binary))
    if binary is None:
        raise CollectorError(
            f"telemetry.collector.binary {config.binary!r} is not an executable file or a command on PATH. "
            "Install the OpenTelemetry Collector (`otelcol-contrib`), set the binary's path, or remove "
            "telemetry.collector to run without one."
        )
    taken = _ports_in_use(config)
    if taken:
        raise CollectorError(
            f"Port(s) {', '.join(map(str, taken))} on {LISTEN_HOST} are in use, so the OpenTelemetry Collector "
            "cannot listen there. Another run's collector may hold them: give this run its own "
            "telemetry.collector.otlp_grpc_port, otlp_http_port and health_port."
        )
    remote_export = config.remote_export
    if remote_export and not os.environ.get(config.token_env):
        remote_export = False
        print(
            f"WARNING: {config.token_env} is missing or empty, so the collector exports nothing remotely. "
            "Telemetry is still written locally."
        )
    run_dir = collector_run_dir(config, run_id, global_config_dict, collector_instance())
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Claimed, so no other supervisor writes here; kept across this one's restarts.
        run_dir.mkdir()
    except FileExistsError:
        raise CollectorError(f"{run_dir} already exists; another collector claimed it.") from None
    collector = LocalCollector(config, binary=binary, run_dir=run_dir, run_id=run_id, remote_export=remote_export)
    collector.start()
    point_exporters_at_collector(config)
    print(f"OpenTelemetry Collector running (pid {collector.pid}), writing to {run_dir}")
    return collector
