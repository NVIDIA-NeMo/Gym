# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the sandbox program as one process against a local Gym model endpoint."""

import asyncio
import json
import os
import shlex
import shutil
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from aiohttp import ClientResponseError

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
        is_runner = command.startswith("setsid --fork --wait bash -c ")
        if is_runner:
            command = shlex.split(command)[5]
        process = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=kwargs.get("cwd"),
            start_new_session=True,
        )
        if is_runner:
            self.runners.append(process)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()
            raise
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)


@pytest.fixture
async def runner_factory(tmp_path, monkeypatch):
    sandboxes = []
    servers = []
    pending_queries = []
    loop = asyncio.get_running_loop()

    async def install(harness):
        remote = Path(harness.remote_directory)
        (remote / "venv/bin").mkdir(parents=True)
        python = remote / "venv/bin/python"
        python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        python.chmod(0o755)
        await harness.sandbox.upload(Path(module.__file__).with_name("sandbox_runner.py"), str(remote / "runner.py"))

    monkeypatch.setattr(module.MiniSWEHarness, "_install_runner", install)

    async def make(**kwargs):
        query = kwargs.pop("query")
        session_id = kwargs["context"].session_id

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                assert self.path.endswith("/v1/responses"), self.path
                assert self.headers["x-session-id"] == session_id
                params = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                try:
                    future = asyncio.run_coroutine_threadsafe(query(params), loop)
                    pending_queries.append(future)
                    response = future.result(timeout=60)
                    payload = response.model_dump(mode="json")
                    status = 200
                except ClientResponseError as error:
                    payload = json.loads(error.response_content)
                    status = error.status
                except Exception as error:
                    payload = {"error": {"message": str(error)}}
                    status = 503
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        root = tmp_path / f"sandbox-{len(sandboxes)}"
        root.mkdir()
        sandbox = ProcessSandbox(root)
        sandboxes.append(sandbox)
        context = kwargs["context"]
        context.workdir = str(root)
        harness = module.MiniSWEHarness(
            sandbox=sandbox,
            model_base_url=f"http://127.0.0.1:{server.server_port}/ng-rollout/test/v1",
            **kwargs,
        )
        harness.remote_directory = str(root / "runtime")
        await harness.setup()
        return harness

    yield make
    for future in pending_queries:
        if not future.done():
            future.cancel()
    for server in servers:
        server.shutdown()
        server.server_close()
    for sandbox in sandboxes:
        for process in sandbox.runners:
            if process.returncode is None:
                try:
                    os.killpg(process.pid, 9)
                except ProcessLookupError:
                    pass
            await process.wait()
