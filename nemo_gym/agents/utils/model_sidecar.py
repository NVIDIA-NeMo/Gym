# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Start a Gym model server inside a declared task sandbox sidecar."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from http.client import HTTPConnection
from math import isfinite
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, model_validator

from nemo_gym.client.server_client import apply_rollout_prefix
from nemo_gym.sandbox.api import AsyncSandbox, AsyncSandboxSidecar
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxResources, SandboxSidecarSpec


class ModelSidecarConfig(BaseModel):
    """A preinstalled Gym model server in a sandbox-owned container."""

    image: str = Field(min_length=1)
    # A task-sandbox route provisioned independently from the framework's model endpoint.
    upstream_base_url: str | None = None
    name: str = "gym-model"
    python: str = "/opt/nemo_gym_venv/bin/python"
    remote_dir: str = "/tmp/nemo-gym-model"
    port: int = Field(default=19001, ge=1024, le=65535)
    registry_port: int = Field(default=19000, ge=1024, le=65535)
    start_timeout_s: float = Field(default=45, gt=0, allow_inf_nan=False)
    shutdown_timeout_s: float = Field(default=15, gt=0, allow_inf_nan=False)
    cpu: float = Field(default=1, gt=0)
    memory_mib: int = Field(default=2048, gt=0)
    provider_options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_paths(self) -> ModelSidecarConfig:
        if not PurePosixPath(self.remote_dir).is_absolute() or ".." in PurePosixPath(self.remote_dir).parts:
            raise ValueError("remote_dir must be an absolute normalized sandbox path")
        if self.port == self.registry_port:
            raise ValueError("model and registry must use different ports")
        if self.upstream_base_url is not None:
            url = urlsplit(self.upstream_base_url)
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username is not None
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "upstream_base_url must be an HTTP(S) endpoint without credentials, query or fragment"
                )
        return self

    def sidecars(self) -> list[SandboxSidecarSpec]:
        """Declare the model port that the task sandbox may reach."""
        return [
            SandboxSidecarSpec(
                name=self.name,
                image=self.image,
                resources=SandboxResources(cpu=self.cpu, memory_mib=self.memory_mib),
                provider_options=self.provider_options,
                ports=(self.port,),
            )
        ]


def model_only_config(resolved_config: Mapping[str, Any], config: ModelSidecarConfig) -> dict[str, Any]:
    """Copy the resolved VLLM model, preserving sampling, conversion and failure policy.

    The input is Gym's merged config with any profile overrides applied.
    Agent/resource nodes and host registry addresses never enter the sidecar config.
    """
    try:
        node = resolved_config["policy_model"]["responses_api_models"]["vllm_model"]
    except KeyError:
        raise ValueError("model_sidecar requires policy_model.responses_api_models.vllm_model") from None
    model = (
        OmegaConf.to_container(node, resolve=True, throw_on_missing=True)
        if isinstance(node, DictConfig)
        else deepcopy(node)
    )
    if not isinstance(model, dict):
        raise ValueError("resolved policy_model.responses_api_models.vllm_model must be a mapping")
    if "${" in json.dumps(model):
        raise ValueError("model config must be resolved before staging")
    if config.upstream_base_url is not None:
        if model.get("endpoint_file") or model.get("generation_cut_control_url"):
            raise ValueError(
                "model_sidecar requires a static upstream without an endpoint file or generation-cut control URL"
            )
        # The task route uses network authorization. Do not stage the framework endpoint's key or headers.
        model.update(base_url=config.upstream_base_url, api_key="gym", default_headers={})
    model.update(
        host="0.0.0.0",
        port=config.port,
        num_workers=1,
        entrypoint="app.py",
    )
    return {
        "head_server": {"host": "127.0.0.1", "port": config.registry_port},
        "policy_model": {"responses_api_models": {"vllm_model": model}},
    }


_ID = re.compile(r"[A-Za-z0-9._-]+\Z")
_LOG = logging.getLogger(__name__)


_PREPARE = "from pathlib import Path; import sys; Path(sys.argv[1]).mkdir(mode=0o700, parents=True)"
_STOP = """from pathlib import Path
import sys
root = Path(sys.argv[1])
if root.is_dir():
    (root / "stop").touch()
    (root / "config.json").unlink(missing_ok=True)
"""
_READY = "NEMO_GYM_MODEL_SIDECAR_READY"


class ModelSidecarError(RuntimeError):
    """The sandbox model server could not start or quiesce."""


def _check_exec(result: SandboxExecResult, operation: str) -> None:
    if result.error_type is not None or result.return_code != 0:
        raise ModelSidecarError(f"Model sidecar {operation} failed ({result.error_type or result.return_code})")


@dataclass
class ModelSidecar:
    """A ready model listener owned by an agent session."""

    _origin: str
    registry_port: int
    rollout_id: str
    _stop: Callable[[], Awaitable[None]] = field(repr=False)
    _shutdown_task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)

    @property
    def base_url(self) -> str:
        """The task-local API URL with the active rollout context."""
        return f"{apply_rollout_prefix(self._origin, self.rollout_id)}/v1"

    async def shutdown(self) -> None:
        """Confirm sidecar shutdown once while the sandbox is still reachable."""
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._stop())
        await asyncio.shield(self._shutdown_task)


async def start_model_sidecar(
    sandbox: AsyncSandbox,
    resolved_config: Mapping[str, Any],
    config: ModelSidecarConfig,
    *,
    rollout_id: str,
    run_timeout_s: float,
) -> ModelSidecar:
    """Stage a model-only config and wait for its model server and registry.

    An unsuccessful start requests shutdown before returning control to the caller.
    """
    if not _ID.fullmatch(rollout_id):
        raise ValueError("rollout_id must be a valid Gym correlation key")
    if not isfinite(run_timeout_s) or run_timeout_s <= 0:
        raise ValueError("run_timeout_s must be finite and positive")
    token = uuid4().hex
    config = config.model_copy(update={"remote_dir": f"{config.remote_dir}/{token}"})
    staged = model_only_config(resolved_config, config)
    sidecar: AsyncSandboxSidecar = sandbox.sidecar(config.name)
    origin = sidecar.url(config.port)
    run: asyncio.Task[SandboxExecResult] | None = None

    async def stop() -> None:
        try:
            async with asyncio.timeout(config.shutdown_timeout_s):
                result = await sidecar.exec(
                    config.python, "-c", _STOP, config.remote_dir, timeout_s=config.shutdown_timeout_s
                )
                _check_exec(result, "stop request")
                if run is not None:
                    _check_exec(await asyncio.shield(run), "shutdown")
        finally:
            if run is not None:
                if run.done() and not run.cancelled():
                    run.exception()
                else:
                    # This only releases the exec waiter. A lost stop request leaves a quiescence gap.
                    run.cancel()

    ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
    marker = f"{_READY} {token}"
    pending = ""

    def on_stdout(text: str) -> None:
        nonlocal pending
        *lines, pending = (pending + text).split("\n")
        for line in lines:
            if line == marker and not ready.done():
                ready.set_result(None)

    try:
        async with asyncio.timeout(config.start_timeout_s):
            result = await sidecar.exec(
                config.python, "-c", _PREPARE, config.remote_dir, timeout_s=config.start_timeout_s
            )
            _check_exec(result, "prepare")
            # The stage contains credentials. Use the container's file transfer, not the exec argv.
            with TemporaryDirectory(prefix="nemo-gym-model-sidecar-") as local:
                staged_path = Path(local) / "config.json"
                staged_path.write_text(json.dumps(staged))
                await sidecar.upload(staged_path, f"{config.remote_dir}/config.json")
            run = asyncio.create_task(
                sidecar.exec(
                    config.python,
                    "-u",
                    "-m",
                    "nemo_gym.agents.utils.model_sidecar",
                    config.remote_dir,
                    token,
                    str(config.start_timeout_s),
                    str(config.shutdown_timeout_s / 3),
                    str(run_timeout_s),
                    timeout_s=config.start_timeout_s + run_timeout_s + config.shutdown_timeout_s,
                    on_stdout=on_stdout,
                )
            )
            await asyncio.wait({run, ready}, return_when=asyncio.FIRST_COMPLETED)
            if run.done():
                _check_exec(run.result(), "startup")
                raise ModelSidecarError("Model sidecar exited before readiness")
            return ModelSidecar(origin, config.registry_port, rollout_id, stop)
    except BaseException:
        cleanup = asyncio.create_task(stop())
        # The caller may release the sandbox as soon as this method returns or raises.
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        try:
            cleanup.result()
        except Exception:
            _LOG.exception("Could not confirm shutdown after a failed model-sidecar start")
        raise
    finally:
        ready.cancel()


def _serve(root: Path, token: str, start_timeout_s: float, grace_s: float, run_timeout_s: float) -> int:
    from nemo_gym.config.global_config import (
        NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME,
        NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME,
        GlobalConfigDictParserConfig,
        get_global_config_dict,
        reset_global_config_dict,
    )
    from nemo_gym.server_runtime.supervisor.server_supervisor import ServerStartupStopped, ServerSupervisor

    # A sidecar may inherit a worker's Gym config. Load only the explicitly staged config.
    os.environ.pop(NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME, None)
    os.environ.pop(NEMO_GYM_CONFIG_DICT_FILE_ENV_VAR_NAME, None)
    reset_global_config_dict()
    staged_path = root / "config.json"
    stop_path = root / "stop"
    try:
        if stop_path.exists():
            return 0
        config = json.loads(staged_path.read_text())
    finally:
        staged_path.unlink(missing_ok=True)
    config.update(
        head_server_spinup_timeout_seconds=start_timeout_s,
        server_spinup_timeout_seconds=start_timeout_s,
        server_shutdown_timeout_seconds=grace_s,
        # Listener readiness only. No upstream probe or model invocation during bootstrap.
        model_endpoint_readiness_timeout_seconds=0,
    )

    supervisor = ServerSupervisor(use_current_environment=True, stop_requested=stop_path.exists)
    try:
        supervisor.start(
            GlobalConfigDictParserConfig(
                initial_global_config_dict=OmegaConf.create(config),
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                skip_exporters=True,
            )
        )
    except ServerStartupStopped as error:
        return 0 if error.clean else 1

    try:
        resolved = get_global_config_dict()
        registry_port = resolved.head_server.port
        model_port = resolved.policy_model.responses_api_models.vllm_model.port
        for port in (registry_port, model_port):
            connection = HTTPConnection("127.0.0.1", port, timeout=min(1, start_timeout_s))
            try:
                connection.request("GET", "/readyz")
                if connection.getresponse().status != 200:
                    raise ModelSidecarError(f"Model sidecar listener on port {port} is not ready")
            finally:
                connection.close()
        if not stop_path.exists():
            supervisor.poll()
            print(f"{_READY} {token}", flush=True)
        deadline = monotonic() + run_timeout_s
        while not stop_path.exists():
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            supervisor.poll()
            sleep(min(0.1, remaining))
    finally:
        clean = supervisor.shutdown()
    return 0 if clean else 1


if __name__ == "__main__":
    _root, _token, _start, _grace, _run = sys.argv[1:]
    try:
        _exit = _serve(Path(_root), _token, float(_start), float(_grace), float(_run))
    except Exception as error:
        # Startup errors may contain values from the resolved model config.
        print(f"Model sidecar process failed ({type(error).__name__})", file=sys.stderr)
        _exit = 1
    raise SystemExit(_exit)
