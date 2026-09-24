# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the actual sandbox program in a separate process with filesystem transport."""

import asyncio
import os
import shlex
import shutil
import sys
from pathlib import Path

import pytest

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.miniswe_sandboxed_agent import harness as module


class ProcessSandbox:
    def __init__(self, directory):
        self.directory = directory
        self.commands = []
        self.runners = []

    async def upload(self, local, remote):
        shutil.copyfile(local, remote)

    async def download(self, remote, local):
        shutil.copyfile(remote, local)

    async def exec(self, command, *, timeout_s=30, **kwargs):
        self.commands.append(command)
        if command == "command -v setsid":
            return SandboxExecResult("setsid", "", 0)
        if command.startswith("setsid --fork bash -c "):
            script = shlex.split(command)[4]
            with (self.directory / "runner.log").open("wb") as stream:
                process = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    script,
                    stdout=stream,
                    stderr=stream,
                    cwd=kwargs.get("cwd"),
                    start_new_session=True,
                )
            self.runners.append(process)
            return SandboxExecResult("", "", 0)
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=kwargs.get("cwd"),
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)


@pytest.fixture
async def runner_factory(tmp_path, monkeypatch):
    sandboxes = []

    async def install(harness):
        remote = Path(harness.remote_directory)
        (remote / "venv/bin").mkdir(parents=True)
        python = remote / "venv/bin/python"
        python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        python.chmod(0o755)
        await harness.sandbox.upload(Path(module.__file__).with_name("sandbox_runner.py"), str(remote / "runner.py"))

    monkeypatch.setattr(module.MiniSWEHarness, "_install_runner", install)

    async def make(**kwargs):
        root = tmp_path / f"sandbox-{len(sandboxes)}"
        root.mkdir()
        sandbox = ProcessSandbox(root)
        sandboxes.append(sandbox)
        context = kwargs["context"]
        context.workdir = str(root)
        harness = module.MiniSWEHarness(sandbox=sandbox, **kwargs)
        harness.remote_directory = str(root / "runtime")
        await harness.setup()
        return harness

    yield make
    for sandbox in sandboxes:
        for process in sandbox.runners:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:
                    pass
            await process.wait()
