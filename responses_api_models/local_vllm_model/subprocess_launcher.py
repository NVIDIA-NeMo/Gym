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
"""Opt-in, single-node vLLM launcher using only its public CLI/HTTP API."""

import asyncio
import json
import logging
import os
import re
import shutil
import socket
import sys
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Literal

from aiohttp import ClientError, ClientTimeout
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from nemo_gym.server_utils import request


LOG = logging.getLogger(__name__)


class VLLMSubprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    executable: str = "vllm"
    port: int | None = Field(default=None, ge=1, le=65535)
    startup_timeout_seconds: float = Field(default=600, gt=0, allow_inf_nan=False)
    probe_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    shutdown_timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    request_timeout_seconds: float = Field(default=2, gt=0, allow_inf_nan=False)
    poll_interval_seconds: float = Field(default=0.25, gt=0, allow_inf_nan=False)
    log_dir: Path = Path("results/local_vllm")


class PDWorkerPlacement(BaseModel):
    """Internal, validated placement supplied by the P/D lifecycle owner."""

    model_config = ConfigDict(extra="forbid")
    role: Literal["prefill", "decode"]
    gpu_indices: list[StrictInt] = Field(min_length=1)
    rpc_port: int = Field(ge=1, le=65535)
    side_channel_port: int = Field(ge=1, le=65535)


def normalize_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in kwargs.items():
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", key):
            raise ValueError(f"Invalid vLLM option name: {key!r}")
        key = key.replace("-", "_")
        if key in normalized:
            raise ValueError(f"Duplicate vLLM option: {key}")
        normalized[key] = value
    return normalized


def validate_managed_kwargs(kwargs: dict[str, Any], env: dict[str, str]) -> None:
    """Fail closed on topology and options that bypass launcher ownership/readiness."""
    for key in ("data_parallel_size", "tensor_parallel_size"):
        value = kwargs.get(key, 1)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer; got {value!r}")
    if type(kwargs.get("pipeline_parallel_size", 1)) is not int or kwargs.get("pipeline_parallel_size", 1) != 1:
        raise ValueError("The single-node subprocess launcher requires pipeline_parallel_size=1")
    for key in ("distributed_executor_backend", "data_parallel_backend"):
        if kwargs.get(key, "mp") != "mp":
            raise ValueError(f"Subprocess launcher requires {key}=mp")
    reserved = {"host", "port", "uds", "root_path", "ssl_keyfile", "ssl_certfile", "api_key", "config"}
    unsupported = {
        "kv_transfer_config",
        "ec_transfer_config",
        "headless",
        "nnodes",
        "node_rank",
        "master_addr",
        "master_port",
        "api_server_count",
    }
    for key, value in kwargs.items():
        if key in reserved:
            raise ValueError(f"{key} is managed by the subprocess launcher and cannot be overridden")
        if key in unsupported or (
            key.startswith("data_parallel_") and key not in {"data_parallel_size", "data_parallel_backend"}
        ):
            if value is not None and value is not False:
                raise ValueError(f"{key} is not supported by the single-node, non-PD launcher")
        if "ray" in key:
            raise ValueError(f"Legacy Ray option {key} cannot be used with launcher=subprocess")
    if any(key.startswith("VLLM_RAY_") for key in env):
        raise ValueError("Remove VLLM_RAY_* environment overrides when selecting launcher=subprocess")


def topology_manifest(kwargs: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    """Validate declared visibility without importing CUDA or claiming GPU allocation."""
    tp, dp = kwargs.get("tensor_parallel_size", 1), kwargs.get("data_parallel_size", 1)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    devices = [device.strip() for device in visible.split(",")] if visible is not None else None
    if devices is not None and (
        not all(re.fullmatch(r"(?:[0-9]+|GPU-[\w-]+|MIG-[\w/-]+)", device) for device in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain unique GPU ordinals/UUIDs from the test allocation")
    required = tp * dp
    if required > 1 and devices is None:
        raise ValueError("Multi-GPU serving requires explicit scheduler-provided CUDA_VISIBLE_DEVICES")
    if devices is not None and len(devices) < required:
        raise ValueError(f"TP={tp} × DP={dp} requires {required} visible GPUs, got {len(devices)}")
    return {
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": 1,
        "data_parallel_size": dp,
        "enable_expert_parallel": kwargs.get("enable_expert_parallel", False),
        "required_gpus": required,
        "visibility_validated": devices is not None,
        "allocation_validated": False,
        "ranks": [
            {
                "dp_rank": rank,
                "local_gpu_indices": list(range(rank * tp, (rank + 1) * tp)),
                "visible_devices": devices[rank * tp : (rank + 1) * tp] if devices is not None else None,
            }
            for rank in range(dp)
        ],
    }


def kwargs_to_argv(kwargs: dict[str, Any], supported_flags: set[str]) -> list[str]:
    argv = []
    for key, value in kwargs.items():
        if value is None:
            continue  # null explicitly means use the executable's default.
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            flag = flag if value else "--no-" + key.replace("_", "-")
            if flag not in supported_flags:
                raise ValueError(
                    f"Selected vLLM executable does not advertise {flag}; omit the option to use its default"
                )
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            if not value:
                raise ValueError(f"Empty list for {flag}: use null to omit it")
            items = [json.dumps(item) if isinstance(item, (dict, list, bool)) else str(item) for item in value]
            if any(item.startswith("-") for item in items):
                raise ValueError(f"List items for {flag} must not start with '-' (ambiguous CLI options)")
            argv.extend([flag, *items])
        else:
            encoded = json.dumps(value) if isinstance(value, dict) else str(value)
            argv.append(f"{flag}={encoded}")
    return argv


def redacted_argv(argv: list[str]) -> list[str]:
    result, secret = [], False
    for arg in argv:
        if arg.startswith("--"):
            name = arg.split("=", 1)[0].lower()
            secret = any(part in name for part in ("key", "token", "header", "password", "secret"))
            result.append(name + "=****" if secret and "=" in arg else arg)
        else:
            result.append("****" if secret else arg)
    return result


class OwnedProcess:
    """A supervisor remains alive to clean up even if the Gym owner is killed."""

    def __init__(self, shutdown_timeout: float):
        self.shutdown_timeout = shutdown_timeout
        self.process: asyncio.subprocess.Process | None = None

    async def start(self, argv: list[str], env: dict[str, str], log_path: Path) -> None:
        if self.process is not None:
            raise RuntimeError("OwnedProcess is single-use")
        supervisor = Path(__file__).with_name("process_supervisor.py")
        with log_path.open("xb") as log:
            # Shield the spawn so cancellation cannot lose a newly created process handle.
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    sys.executable,
                    "-u",
                    str(supervisor),
                    "--shutdown-timeout",
                    str(self.shutdown_timeout),
                    "--",
                    *argv,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            )
            try:
                self.process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                self.process = await spawn
                await self.stop()
                raise

    async def stop(self) -> None:
        if self.process is None:
            return
        # Closing the lifetime pipe asks the supervisor to clean up its group.
        self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=self.shutdown_timeout + 5)
        except TimeoutError as exc:
            # Never kill the supervisor here: it is still the group's cleanup owner.
            raise RuntimeError(f"Supervisor {self.process.pid} did not finish cleanup; inspect its log") from exc


class VLLMSubprocessLauncher:
    def __init__(
        self,
        *,
        config: VLLMSubprocessConfig,
        model: str,
        kwargs: dict[str, Any],
        env: dict[str, str],
        api_key: str,
        cache_dir: str,
        show_stats: bool = False,
        pd_placement: PDWorkerPlacement | None = None,
    ):
        if os.name != "posix":
            raise ValueError("Managed subprocess serving currently requires POSIX process groups")
        self.config = config
        self.model = model
        self.kwargs = normalize_kwargs(kwargs)
        validate_managed_kwargs(self.kwargs, env)
        self.env = os.environ.copy() | env
        # Retire router connections (50s inference / 90s health pools) before
        # Uvicorn closes them. Its 5s default races with sparse agent turns.
        # This is an idle HTTP timeout, not the generation deadline.
        self.env.setdefault("VLLM_HTTP_TIMEOUT_KEEP_ALIVE", "120")
        validate_managed_kwargs(self.kwargs, self.env)
        inherited_visibility = os.environ.get("CUDA_VISIBLE_DEVICES")
        if inherited_visibility is not None and self.env.get("CUDA_VISIBLE_DEVICES") != inherited_visibility:
            raise ValueError("Do not override inherited CUDA_VISIBLE_DEVICES; allocate GPUs before starting Gym")
        self.pd_placement = pd_placement
        if pd_placement is not None:
            if inherited_visibility is None:
                raise ValueError("PD requires scheduler-provided CUDA_VISIBLE_DEVICES")
            topology_manifest(self.kwargs, self.env)
            devices = inherited_visibility.split(",")
            indices = pd_placement.gpu_indices
            required = self.kwargs.get("tensor_parallel_size", 1) * self.kwargs.get("data_parallel_size", 1)
            if (
                len(indices) != required
                or len(set(indices)) != len(indices)
                or any(index < 0 or index >= len(devices) for index in indices)
            ):
                raise ValueError("PD GPU indices must select exactly TP × DP distinct inherited GPUs")
            if pd_placement.side_channel_port + self.kwargs.get("data_parallel_size", 1) - 1 > 65535:
                raise ValueError("PD side-channel port range exceeds 65535")
            if any(key.startswith("VLLM_NIXL_SIDE_CHANNEL_") for key in env):
                raise ValueError("PD launcher owns NIXL side-channel addresses; remove environment overrides")
            self.env["CUDA_VISIBLE_DEVICES"] = ",".join(devices[index].strip() for index in indices)
            self.env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = "127.0.0.1"
            self.env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(pd_placement.side_channel_port)
        self.topology = topology_manifest(self.kwargs, self.env)
        executable = shutil.which(config.executable, path=self.env.get("PATH"))
        if not executable:
            raise ValueError(
                f"vLLM executable not found: {config.executable!r}; install it separately or set subprocess.executable"
            )
        self.executable = os.path.abspath(executable)
        # Keep console scripts (e.g. ninja) beside the selected executable visible.
        self.env["PATH"] = os.pathsep.join([str(Path(self.executable).parent), self.env.get("PATH", "")])
        self.api_key = api_key
        self.env["VLLM_API_KEY"] = api_key
        self.cache_dir = cache_dir
        self.show_stats = show_stats
        self.owner = OwnedProcess(config.shutdown_timeout_seconds)
        self.run_dir: Path | None = None
        self.base_url: str | None = None

    async def _probe(self, args: list[str], label: str, *, executable: str | None = None) -> str:
        owner = OwnedProcess(self.config.shutdown_timeout_seconds)
        path = self.run_dir / f"{label}.log"
        try:
            await owner.start([executable or self.executable, *args], self.env, path)
            try:
                code = await asyncio.wait_for(owner.process.wait(), timeout=self.config.probe_timeout_seconds)
            except TimeoutError as exc:
                raise RuntimeError(f"vLLM {label} probe timed out; see {path}") from exc
            if code != 0:
                raise RuntimeError(f"vLLM {label} probe exited with {code}; see {path}")
            return path.read_text(errors="replace")
        finally:
            await owner.stop()

    def _prepare_run(self, port: int, *, host: str, prefix: str, dry_run: bool) -> None:
        """Reserve the serving endpoint before creating this run's output directory."""
        # SO_REUSEADDR tolerates TIME_WAIT from previous runs, but not a live listener.
        with socket.socket() as reservation:
            reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if not dry_run:
                reservation.bind((host, port))
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = Path(mkdtemp(prefix=prefix, dir=self.config.log_dir)).resolve()
        self.base_url = f"http://{host}:{port}/v1"

    async def start(self, port: int, *, dry_run: bool = False) -> str:
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("Serving port must be between 1 and 65535")
        if self.run_dir is not None:
            raise RuntimeError("VLLMSubprocessLauncher is single-use")
        self._prepare_run(port, host="127.0.0.1", prefix="launch-", dry_run=dry_run)
        try:
            version = (await self._probe(["--version"], "version")).strip()
            help_text = await self._probe(["serve", "--help=all"], "help")
            flags = set(re.findall(r"--[a-z][a-z0-9-]*", help_text))
            kwargs = dict(self.kwargs)
            checkpoint = kwargs.pop("model", self.model)
            if not isinstance(checkpoint, str) or not checkpoint or checkpoint.startswith("-"):
                raise ValueError("vllm_serve_kwargs.model must be a nonempty checkpoint name/path, not an option")
            names = kwargs.setdefault("served_model_name", [self.model])
            if self.model not in ([names] if isinstance(names, str) else names):
                raise ValueError("served_model_name must include config.model (the name Gym sends to vLLM)")
            kwargs.update(host="127.0.0.1", port=port, distributed_executor_backend="mp", data_parallel_backend="mp")
            for dimension in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
                kwargs.setdefault(dimension, 1)
            if self.topology["data_parallel_size"] > 1:
                # One native DP group, not independently launched dense replicas.
                kwargs["data_parallel_size_local"] = self.topology["data_parallel_size"]
                kwargs["api_server_count"] = 1
                for flag in ("--data-parallel-size-local", "--api-server-count"):
                    if flag not in flags:
                        raise ValueError(f"Selected vLLM executable does not advertise {flag}")
            if self.pd_placement:
                for flag in ("--kv-transfer-config", "--data-parallel-rpc-port"):
                    if flag not in flags:
                        raise ValueError(f"Selected vLLM executable does not advertise {flag}")
                kwargs["data_parallel_rpc_port"] = self.pd_placement.rpc_port
                kwargs["kv_transfer_config"] = {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_both",
                    "kv_load_failure_policy": "fail",
                }
            kwargs.setdefault("download_dir", self.cache_dir)
            if not self.show_stats:
                kwargs.setdefault("disable_log_stats", True)
            argv = [self.executable, "serve", checkpoint, *kwargs_to_argv(kwargs, flags)]
            manifest = {
                "launcher": "subprocess",
                "scope": "single-node NIXL PD MP, PP=1" if self.pd_placement else "single-node non-PD MP, PP=1",
                "topology": self.topology,
                "dry_run": dry_run,
                "vllm_version": version,
                "argv_redacted": redacted_argv(argv),
                "base_url": self.base_url,
                "cuda_visible_devices": self.env.get("CUDA_VISIBLE_DEVICES"),
                "http_timeout_keep_alive": self.env["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"],
                "startup_timeout_seconds": self.config.startup_timeout_seconds,
                "shutdown_timeout_seconds": self.config.shutdown_timeout_seconds,
                "pd_placement": self.pd_placement.model_dump() if self.pd_placement else None,
            }
            (self.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if dry_run:
                return self.base_url
            LOG.info("Starting managed vLLM; logs and runtime manifest: %s", self.run_dir)
            await self.owner.start(argv, self.env, self.run_dir / "server.log")
            await self._await_ready()
            return self.base_url
        except BaseException:
            await self.stop()
            raise

    async def _await_ready(self) -> None:
        last_error = "no response"
        try:
            async with asyncio.timeout(self.config.startup_timeout_seconds):
                while True:
                    if self.owner.process.returncode is not None:
                        raise RuntimeError(
                            f"vLLM exited with {self.owner.process.returncode}; see {self.run_dir / 'server.log'}"
                        )
                    try:
                        response = await request(
                            "GET",
                            f"{self.base_url}/models",
                            _max_connection_retries=0,
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            timeout=ClientTimeout(total=self.config.request_timeout_seconds),
                        )
                        async with response:
                            last_error = f"HTTP {response.status}"
                            if response.status == 200:
                                body = await response.json()
                                models = body.get("data") if isinstance(body, dict) else None
                                if isinstance(models, list) and any(
                                    isinstance(item, dict) and item.get("id") == self.model for item in models
                                ):
                                    if self.owner.process.returncode is None:
                                        return
                                last_error = "expected model absent from /v1/models"
                    except (ClientError, TimeoutError, ValueError) as exc:
                        last_error = type(exc).__name__
                    await asyncio.sleep(self.config.poll_interval_seconds)
        except TimeoutError as exc:
            raise RuntimeError(f"vLLM readiness timed out ({last_error}); see {self.run_dir / 'server.log'}") from exc

    async def stop(self) -> None:
        await self.owner.stop()
