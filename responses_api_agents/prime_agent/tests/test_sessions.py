# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prime Agent sessions: seed refusals, request mapping, and real supervised runs of a fake Prime Agent CLI."""

import asyncio
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from nemo_gym.agents import AgentSeedSessionRequest, SandboxSession, check_agent_seed_rejected
from nemo_gym.agents.agent_outcome import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.prime_agent.app import PrimeAgent, PrimeAgentConfig
from responses_api_agents.prime_agent.sandbox import CLI, PRIME_AGENT_VERSION, PrimeSession


FAKE_PRIME = r"""
import json, os, subprocess, sys, time
from pathlib import Path
args = sys.argv[2:]
if args == ["--version"]:
    print("__VERSION__")
    sys.exit(0)
agent_dir = Path(os.environ["PRIME_AGENT_CODING_AGENT_DIR"])
mode = os.environ.get("TEST_MODE", "normal")
(agent_dir / "seen.json").write_text(json.dumps({
    "cli": sys.argv[1], "args": args, "cwd": os.getcwd(),
    "models": json.loads((agent_dir / "models.json").read_text()),
    "env": {k: v for k, v in os.environ.items() if k.startswith(("PRIME_AGENT_", "PI_"))},
}))
subprocess.run(["bash", "-c", 'printf "%s\n%s\n" "$HOME" "$TASK_TEST" > proof.txt'], check=True)
def end(message):
    print(json.dumps({"type": "message_end", "message": message}), flush=True)
usage = {"input": 10, "output": 2, "cacheRead": 1}
end({"role": "assistant", "content": [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "ls"}}],
     "usage": usage, "stopReason": "toolUse"})
end({"role": "toolResult", "toolCallId": "c1", "content": [{"type": "text", "text": "a.txt"}]})
if mode == "sleep":
    time.sleep(60)
if mode == "error":
    end({"role": "assistant", "content": [], "usage": usage, "stopReason": "error",
         "errorMessage": "503 Service Unavailable"})
else:
    end({"role": "assistant", "content": [{"type": "text", "text": "Done."}], "usage": usage, "stopReason": "stop"})
""".replace("__VERSION__", PRIME_AGENT_VERSION)


class LocalSandbox:
    """Run every sandbox command on this host, except the installer, which writes the fake runtime."""

    def __init__(self):
        self.disconnected = False

    async def upload(self, source, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    async def download(self, source, destination):
        shutil.copyfile(source, destination)

    async def exec(self, command, *, cwd=None, timeout_s=30, env=None):
        argv = shlex.split(command)
        if len(argv) > 1 and argv[1].endswith("/install_prime_runtime.sh"):
            assert argv[3] == PRIME_AGENT_VERSION
            node = Path(argv[2]) / "bin" / "node"
            node.parent.mkdir(parents=True, exist_ok=True)
            node.write_text(f"#!{sys.executable}\n" + FAKE_PRIME)
            node.chmod(0o755)
            return SandboxExecResult("", "", 0)
        process = await asyncio.create_subprocess_exec(
            "sh", "-c", command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        return SandboxExecResult(stdout.decode(errors="replace"), stderr.decode(errors="replace"), process.returncode)

    async def disconnect(self):
        self.disconnected = True

    async def stop(self):
        pytest.fail("A borrowed sandbox must not be stopped")


def agent(**kwargs) -> PrimeAgent:
    kwargs.setdefault("model_server", {"type": "responses_api_models", "name": "policy_model"})
    return PrimeAgent(
        config=PrimeAgentConfig(
            host="localhost", port=8080, name="prime_agent", entrypoint="app.py", model="policy", **kwargs
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def seed(*, workdir="/repo", tools=None, user=None, borrowed=True) -> AgentSeedSessionRequest:
    access = {
        "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"id": "task"}},
        "workdir": workdir,
    }
    if user is not None:
        access["user"] = user
    return AgentSeedSessionRequest(
        agent_session_id="session-1",
        episode_id={"rollout_id": "rollout-1"},
        task_id={"taskset": "test", "task_id": "task"},
        tool_accesses=tools or [],
        sandbox_access=access if borrowed else None,
    )


def body(**fields) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "Use bash."}, {"role": "user", "content": "Fix the bug."}], **fields
    )


@pytest.mark.parametrize(
    ("server", "request_seed", "reason"),
    [
        (
            {},
            seed(
                tools=[
                    {
                        "kind": "mcp",
                        "name": "task_tools",
                        "required": True,
                        "connection": {"transport": "streamable_http", "url": "http://r/mcp"},
                    }
                ]
            ),
            "Prime Agent sessions use Prime Agent's own tools; required tool grants are unsupported",
        ),
        ({"model_server": None}, seed(), "Prime Agent sessions require a sandbox-reachable Gym model_server"),
        ({"extra_args": ["--x"]}, seed(), "Prime Agent sessions do not support command, extra_args, or env overrides"),
        ({}, seed(user="nobody"), "Prime Agent does not support a SandboxAccess user override"),
        ({}, seed(workdir="/tmp"), "Prime Agent task workdir must be absolute and separate from its session storage"),
        ({}, seed(borrowed=False), "Prime Agent requires sandbox_access or a configured sandbox_provider"),
    ],
)
def test_unsupported_seeds_are_final_refusals(server, request_seed, reason):
    prime = agent(**server)
    assert check_agent_seed_rejected(prime.setup_webserver(), request_seed) == reason
    assert not prime._session_records


def test_session_request_mapping_and_refusals():
    prime = agent(system_prompt="Be brief.")
    assert prime._session_conversation(body(instructions="Answer in English.")) == (
        "Fix the bug.",
        "Be brief.\n\nAnswer in English.\n\nUse bash.",
    )
    for fields in ({"temperature": 0.5}, {"max_output_tokens": 10}, {"model": "other"}):
        with pytest.raises(HTTPException) as error:
            prime._session_conversation(body(**fields))
        assert error.value.status_code == 422


def test_run_without_resources_server_is_refused():
    with pytest.raises(HTTPException) as error:
        asyncio.run(agent().run(MagicMock(cookies={}), MagicMock()))
    assert error.value.status_code == 422


async def activate(tmp_path: Path, mode: str, *, timeout: float = 20):
    workdir = tmp_path / "task repo"
    workdir.mkdir()
    request = seed(workdir=str(workdir))
    request.sandbox_access.env = {
        "TEST_MODE": mode,
        "TASK_TEST": "task-value",
        "HOME": str(tmp_path / "task-home"),
        "PRIME_AGENT_CODING_AGENT_DIR": "must-not-inherit",
    }
    sandbox = LocalSandbox()
    state = PrimeSession(
        request=request,
        provider_name="local",
        sandbox_id="task",
        runtime=str(tmp_path / "runtime"),
        session=SandboxSession(
            sandbox=sandbox, session_dir=str(tmp_path / "prime session"), workdir=str(workdir), harness="Prime Agent"
        ),
    )
    prime = agent(timeout=timeout, thinking="high")
    asset = tmp_path / "asset.tgz"
    asset.write_bytes(b"asset")
    # The agent server downloads Node.js and the release package; the fake installer stands in for both.
    with patch(
        "responses_api_agents.miniswe_sandboxed_agent.bootstrap.cached_download", AsyncMock(return_value=asset)
    ) as download:
        await state.install(timeout=10)
    assert [call.args[0].rsplit("/", 1)[1] for call in download.await_args_list] == [
        f"node-v22.23.2-linux-{'arm64' if os.uname().machine == 'aarch64' else 'x64'}.tar.gz",
        "prime-agent-0.7.0.tgz",
    ]
    state.activation_request = body()
    with patch.object(
        PrimeAgent, "resolve_model_base_url", return_value="http://model.invalid/ng-rollout/rollout-1/v1"
    ):
        response = await asyncio.wait_for(prime._activate_session(state, state.activation_request), timeout=60)
    seen = json.loads((Path(state.agent_dir) / "seen.json").read_text())
    close = await asyncio.wait_for(prime._close_agent_session_state(state), timeout=60)
    return response, close, seen, state, sandbox


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_real_session_run_maps_output_and_keeps_the_task_environment(tmp_path):
    response, close, seen, state, sandbox = await activate(tmp_path, "normal")
    assert response.status == "completed"
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
    assert (response.usage.input_tokens, response.usage.output_tokens) == (22, 4)
    assert seen["cli"] == f"{state.runtime}/{CLI}"
    args = seen["args"]
    assert args[:5] == ["--print", "--mode", "json", "--no-session", "--daemon-socket"]
    assert args[args.index("--provider") + 1] == "nemo" and args[args.index("--model") + 1] == "policy"
    assert args[args.index("--append-system-prompt") + 1] == "Use bash." and args[-1] == "Fix the bug."
    assert seen["models"]["providers"]["nemo"]["baseUrl"] == "http://model.invalid/ng-rollout/rollout-1/v1"
    assert seen["env"]["PRIME_AGENT_CODING_AGENT_DIR"] == state.agent_dir
    assert (Path(state.session.workdir) / "proof.txt").read_text().splitlines() == [
        str(tmp_path / "task-home"),
        "task-value",
    ]
    assert close.cleanup_confirmed and sandbox.disconnected and not Path(state.session.session_dir).exists()


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_model_error_reports_the_upstream_status(tmp_path):
    response, close, _, _, _ = await activate(tmp_path, "error")
    assert response.status == "failed"
    outcome = json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])
    assert outcome == {"failure_kind": "model_request_error", "status_code": 503}
    assert close.cleanup_confirmed


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform != "linux", reason="Shared process supervisor requires Linux")
async def test_timeout_keeps_partial_work_as_incomplete(tmp_path):
    response, close, _, state, _ = await activate(tmp_path, "sleep", timeout=3)
    assert state.session.cleanup["timed_out"] and response.status == "incomplete"
    assert close.agent_observations.records[1].outcome == "timeout"
