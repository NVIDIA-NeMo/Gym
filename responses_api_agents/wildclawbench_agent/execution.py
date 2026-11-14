# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host execution leases shared by the native agent and its resources owner."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import signal
import socket
import subprocess
from pathlib import Path
from typing import Literal

import anyio
from pydantic import BaseModel, Field, model_validator

from nemo_gym.config import ModelServerRef
from responses_api_agents.wildclawbench_agent.data import digest, task_spec, verify_source, workspace_digest


class ExecutionConfig(BaseModel):
    source_root: str
    worker_python: str
    output_root: str
    model_server: ModelServerRef
    model_name: str
    context_window: int = Field(default=131072, gt=0)
    max_tokens: int = Field(default=8192, gt=0)
    thinking: str = "high"
    sandbox_runtime: str = "docker"
    judge_model: str = "openai/gpt-5.4"
    network_access: Literal["inherit"] = "inherit"

    @model_validator(mode="after")
    def validate_execution(self) -> ExecutionConfig:
        if self.context_window <= 4 * self.max_tokens:
            raise ValueError("Context must exceed four completion budgets for native compaction")
        if self.sandbox_runtime not in {"docker", "enroot"}:
            raise ValueError("sandbox_runtime must be docker or enroot")
        return self


def image_identity(config: ExecutionConfig) -> str:
    if config.sandbox_runtime == "enroot":
        return "sha256:" + digest(Path(config.source_root) / ".cache/wildclawbench-ubuntu_v1.3.sqsh")
    value = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", "wildclawbench-ubuntu:v1.3"],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    ).stdout.strip()
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", value):
        raise ValueError("Docker did not return an immutable sandbox image ID")
    return value


def episode_output(config: ExecutionConfig, key: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{64}", key):
        raise ValueError("Invalid native execution key")
    root = Path(config.output_root).absolute()
    output = root / key
    if output.resolve().parent != root.resolve():
        raise ValueError("Native execution path escapes output root")
    return output


def read_lease(config: ExecutionConfig, key: str) -> dict:
    lease = json.loads((episode_output(config, key) / "lease.json").read_text())
    if lease["owner_host"] != socket.gethostname() or lease["execution"] != config.model_dump(mode="json"):
        raise ValueError("Native resources and agent must share one host, execution config and workspace")
    return lease


def create_lease(config: ExecutionConfig, key: str, metadata: dict, prompt: str, episode: dict, task: dict) -> Path:
    root = Path(config.source_root)
    verify_source(root)
    expected = task_spec(root, metadata["task_id"])
    expected_prompt = expected.pop("prompt")
    expected["workspace_sha256"] = workspace_digest(root, expected["workspace"])
    if expected != metadata or expected_prompt != prompt:
        raise ValueError("Prepared task, assets or actor prompt changed; prepare again")
    output = episode_output(config, key)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.chmod(0o700)
    lease = dict(
        execution=config.model_dump(mode="json"),
        owner_host=socket.gethostname(),
        metadata=metadata,
        prompt=prompt,
        episode_id=episode,
        task_id=task,
    )
    with (output / "lease.json").open("x") as stream:
        json.dump(lease, stream, indent=2)
    return output


class NativeProcess:
    """Run a bounded native stage; the resources owner, not this process, owns its container."""

    def __init__(self, config: ExecutionConfig, output: Path, image: str):
        self.config, self.output, self.image = config, output, image
        self.process: asyncio.subprocess.Process | None = None

    async def stop(self) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(self.process.wait(), timeout=45)
        except TimeoutError:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.wait_for(self.process.wait(), timeout=10)

    async def invoke(self, stage: str, timeout: float) -> dict:
        if stage not in {"agent", "verify", "cleanup"}:
            raise ValueError("Unknown native stage")
        config_file = self.output / "worker.json"
        if not config_file.exists() and stage == "cleanup":
            return {}
        env = worker_environment(self.config, self.output)
        if self.config.sandbox_runtime == "docker":
            env["DOCKER_IMAGE"] = self.image
        command = [
            self.config.worker_python,
            "-m",
            "responses_api_agents.wildclawbench_agent.worker",
            str(config_file),
            "--" + stage,
        ]
        with (self.output / ("worker-" + stage + ".log")).open("a") as log:
            if self.process is not None and self.process.returncode is None:
                raise RuntimeError("Another native stage is still running")
            self.process = None
            launch = asyncio.create_task(
                asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=log, start_new_session=True)
            )
            try:
                self.process = await asyncio.shield(launch)
                await asyncio.wait_for(self.process.wait(), timeout=timeout)
            finally:
                with anyio.CancelScope(shield=True):
                    # Cancellation can arrive after the OS child starts but before asyncio returns its handle.
                    # Finish acquiring that handle before stop; otherwise close could leave an unowned worker.
                    if self.process is None:
                        self.process = await asyncio.shield(launch)
                    await self.stop()
        if self.process.returncode:
            raise RuntimeError(f"Native {stage} stage failed; inspect {self.output}")
        if stage == "cleanup":
            (self.output / "resources_closed.json").write_text('{"cleanup_confirmed":true}\n')
            return {}
        return json.loads((self.output / ("agent_result.json" if stage == "agent" else "result.json")).read_text())


def worker_environment(config: ExecutionConfig, output: Path) -> dict[str, str]:
    """Keep cache/home mounts away from the host's home directories."""
    env = dict(os.environ)
    storage = output / "storage"
    for name, folder in {
        "XDG_CACHE_HOME": "cache",
        "XDG_CONFIG_HOME": "config",
        "XDG_DATA_HOME": "data",
        "XDG_STATE_HOME": "state",
        "HF_HOME": "huggingface",
        "UV_CACHE_DIR": "uv",
        "PIP_CACHE_DIR": "pip",
        "TMPDIR": "tmp",
        "FLASHINFER_WORKSPACE_BASE": "flashinfer",
    }.items():
        directory = storage / folder
        directory.mkdir(parents=True, exist_ok=True)
        env[name] = str(directory)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PATH"] = f"{Path(config.worker_python).parent}:{env['PATH']}"
    env["DOCKER_IMAGE"] = "wildclawbench-ubuntu:v1.3"
    env["DOCKER_NETWORK_MODE"] = ""
    # Empty values also block native python-dotenv from restoring hidden overrides.
    for name in ("WILDCLAW_OPENCLAW_MODEL_PARAMS", "WILDCLAW_OPENCLAW_THINKING", "OPENCLAW_IMAGE_MODEL"):
        env[name] = ""
    env["JUDGE_MODEL"] = config.judge_model
    if config.sandbox_runtime == "enroot":
        commands = storage / "bin"
        commands.mkdir(parents=True, exist_ok=True)
        wrapper = commands / "docker"
        wrapper.write_text(
            "#!/bin/sh\nexec "
            + shlex.quote(config.worker_python)
            + ' -m responses_api_agents.wildclawbench_agent.enroot_docker "$@"\n'
        )
        wrapper.chmod(0o700)
        env["PATH"] = f"{commands}:{env['PATH']}"
        env["WILDCLAW_ROOT"] = config.source_root
        env["WILDCLAW_ENROOT_IMAGE"] = str(Path(config.source_root) / ".cache/wildclawbench-ubuntu_v1.3.sqsh")
        env["ENROOT_MOUNT_HOME"] = "n"
        for name, folder in {
            "WILDCLAW_ENROOT_STATE_DIR": "state",
            "ENROOT_CACHE_PATH": "cache",
            "ENROOT_DATA_PATH": "data",
            "ENROOT_RUNTIME_PATH": "runtime",
            "ENROOT_TEMP_PATH": "tmp",
            "ENROOT_CONFIG_PATH": "config",
        }.items():
            directory = storage / "enroot" / folder
            directory.mkdir(parents=True, exist_ok=True)
            env[name] = str(directory)
    return env
