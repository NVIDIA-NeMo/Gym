# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the pinned non-TTY CLI's process identity and forced termination."""

import asyncio
import json
import os
import shutil
import signal
import sys
from pathlib import Path

import pytest
from aiohttp import web
from fastapi import Request

from nemo_gym.agent_utils.sandbox_session import HARNESS_RUNNER_SCRIPT, shell_assignments
from nemo_gym.agent_utils.supervisor_client import SUPERVISOR_SCRIPT
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.openclaw_agent.tests.test_sandbox import seed, setup, staged  # noqa: F401


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process supervision required")
async def test_pinned_non_tty_cli_stays_in_supervised_group_and_stops(setup, tmp_path: Path) -> None:
    entrypoint = os.environ.get("GYM_TEST_OPENCLAW_ENTRYPOINT")
    node = os.environ.get("GYM_TEST_OPENCLAW_NODE") or shutil.which("node")
    if entrypoint is None or node is None:
        pytest.skip("Set GYM_TEST_OPENCLAW_ENTRYPOINT and GYM_TEST_OPENCLAW_NODE to the pinned runtime")
    assert json.loads(Path(entrypoint).with_name("package.json").read_text())["version"] == "2026.6.11"
    requested = asyncio.Event()
    release = asyncio.Event()

    async def hold_model_call(request: web.Request) -> web.Response:
        await request.read()
        requested.set()
        await release.wait()
        return web.json_response({"error": "test finished"}, status=503)

    app = web.Application()
    app.router.add_post("/{path:.*}", hold_model_call)
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        agent, sandbox = setup
        agent.server_client._build_server_base_url.return_value = f"http://127.0.0.1:{runner.addresses[0][1]}"
        directory = tmp_path / "session"
        (directory / "home/.openclaw").mkdir(parents=True)
        workdir = tmp_path / "task"
        workdir.mkdir()
        request = Request({"type": "http", "session": {}, "path_params": {"rollout_id": "openclaw-smoke-a2"}})
        body = seed()
        body.sandbox_access.workdir = str(workdir)
        sandbox.expected_workdir = str(workdir)
        await agent.seed_agent_session(request, body)
        state = agent._session_records[body.agent_session_id].state
        state.session.session_dir = str(directory)
        await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="Wait for the model response."))
        payload = staged(sandbox)
        payload["command"][:2] = [node, entrypoint]
        identities = directory / "node-identities.jsonl"
        preload = directory / "identity.cjs"
        preload.write_text(
            "const fs=require('node:fs');"
            "const stat=fs.readFileSync('/proc/self/stat','utf8').split(') ').pop().split(' ');"
            f"fs.appendFileSync({json.dumps(str(identities))},JSON.stringify({{pid:process.pid,pgid:Number(stat[2])}})+'\\n');"
        )
        payload["env"]["NODE_OPTIONS"] = f"--require={preload}"
        payload["env"]["NODE_COMPILE_CACHE"] = str(directory / "compile-cache")
        payload["env"]["HOME"] = str(tmp_path / "task-home")
        (directory / "home/.openclaw/openclaw.json").write_text(
            sandbox.files[f"{directory}/home/.openclaw/openclaw.json"]
        )
        (directory / "prompt.txt").write_text(payload.pop("prompt"))
        (directory / "params.sh").write_text(shell_assignments(payload))
        process = await asyncio.create_subprocess_exec(
            "bash",
            str(SUPERVISOR_SCRIPT),
            "--timeout",
            "90",
            "--cleanup-timeout",
            "1",
            "--receipt",
            str(directory / "cleanup.json"),
            "--stop-file",
            str(directory / "stop.request"),
            "--",
            "bash",
            str(HARNESS_RUNNER_SCRIPT),
            str(directory / "params.sh"),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            # A real model HTTP request proves the CLI completed startup. Holding its
            # response makes the forced-stop check independent of model throughput.
            await asyncio.wait_for(requested.wait(), 60)
            worker = json.loads((directory / "runtime.json").read_text())["pid"]
            observed = [json.loads(line) for line in identities.read_text().splitlines()]
            assert observed
            assert all(item["pgid"] == worker for item in observed), observed
            (directory / "stop.request").touch()
            process.send_signal(signal.SIGTERM)
            stdout, stderr = await asyncio.wait_for(process.communicate(), 15)
            assert process.returncode == 0, (stdout, stderr)
            receipt = json.loads((directory / "cleanup.json").read_text())
            assert receipt["cleanup_confirmed"] is True
            assert receipt["timed_out"] is False
            for item in observed:
                stat = Path(f"/proc/{item['pid']}/stat")
                assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        finally:
            (directory / "stop.request").touch()
            if process.returncode is None:
                process.send_signal(signal.SIGTERM)
            try:
                if process.returncode is None:
                    await asyncio.wait_for(process.communicate(), 15)
            finally:
                # Stop this test's worker group before awaiting pipe EOF on failure.
                # Killing only the supervisor could leave inherited output pipes open.
                runtime = directory / "runtime.json"
                if runtime.exists():
                    try:
                        os.killpg(json.loads(runtime.read_text())["pid"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                # Failure-path cleanup is limited to node PIDs recorded by this test.
                if identities.exists():
                    for line in identities.read_text().splitlines():
                        try:
                            os.kill(json.loads(line)["pid"], signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                if process.returncode is None:
                    process.kill()
                await asyncio.wait_for(process.communicate(), 5)
    finally:
        release.set()
        await runner.cleanup()
