# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Agent's sandbox mode: the same turn loop, run in the sandbox its environment lends."""

import asyncio
import importlib.metadata
import json
import os
import subprocess
import sys
import tarfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web
from omegaconf import OmegaConf
from packaging.requirements import Requirement

from nemo_gym.agents import AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from responses_api_agents.simple_agent import sandbox as sandbox_module
from responses_api_agents.simple_agent import sandbox_runner
from responses_api_agents.simple_agent.app import (
    ModelServerRef,
    SimpleAgent,
    SimpleAgentConfig,
    SimpleAgentSessionState,
)


ROLLOUT = "rollout-a1"
# A sandbox has no Gym settings in its environment; neither do the runner processes these tests start.
SANDBOX_ENV = {name: value for name, value in os.environ.items() if not name.startswith(("NEMO_GYM", "PYTHON"))}
MODEL_PATH = f"/ng-rollout/{ROLLOUT}/v1/responses"


def _reply(text: str) -> dict[str, Any]:
    return NeMoGymResponse(
        id="resp-1",
        created_at=0,
        model="policy",
        object="response",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "msg-1",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
        usage={
            "input_tokens": 7,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 3,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 10,
        },
    ).model_dump(mode="json")


@asynccontextmanager
async def _model_server(replies: list[tuple[int, dict[str, Any]]]) -> AsyncIterator[tuple[int, list[Any]]]:
    """A model server on loopback that answers each request with the next reply and records what it was sent."""
    seen: list[tuple[str, dict[str, Any], str | None]] = []

    async def respond(request: web.Request) -> web.Response:
        seen.append((request.path, await request.json(), request.headers.get("x-session-id")))
        status, body = replies[min(len(seen), len(replies)) - 1]
        return web.json_response(body, status=status)

    app = web.Application()
    app.router.add_post("/{tail:.*}", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield site._server.sockets[0].getsockname()[1], seen
    finally:
        await runner.cleanup()


def _config(**overrides: Any) -> SimpleAgentConfig:
    return SimpleAgentConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="app.py",
        name="simple",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_model_retries=0,
        **overrides,
    )


def _client(port: int) -> ServerClient:
    return ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=port),
        global_config_dict=OmegaConf.create(
            {"policy_model": {"responses_api_models": {"model": {"host": "127.0.0.1", "port": port}}}}
        ),
    )


def _seed(**fields: Any) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session",
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task_id=TaskId(taskset="example", task_id="0"),
        **fields,
    )


ACCESS = SandboxAccess(
    connection=DirectSandboxConnection(provider_config_ref="attempt_sandboxes", descriptor={"id": "sb-1"}),
    workdir="/workspace",
)


class _LoopInProcess:
    """Stands in for the sandbox: hands the staged payload, as JSON, to the runner the sandbox would run."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def run(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> dict[str, Any]:
        staged = json.loads(json.dumps(payload))
        self.payloads.append(staged)
        return json.loads(json.dumps(await sandbox_runner.run_loop(staged)))


def _loop_arguments() -> dict[str, Any]:
    return {
        "model_url_path": MODEL_PATH,
        "resources_server_cookies": None,
        "tool_access": None,
        "resources_prefix": "",
        "in_session": True,
        "execute_tools": True,
        "invocation_id": "root",
        "rollout_id": ROLLOUT,
        "collect_trajectory": True,
        "activation": None,
    }


@pytest.mark.parametrize(
    "replies",
    [
        pytest.param([(200, _reply("The answer is B."))], id="answer"),
        # A rejected model call ends the session's turn with a failed response, in the sandbox as in process.
        pytest.param([(400, {"error": "context too long"})], id="model_rejects"),
    ],
)
async def test_the_sandboxed_loop_returns_what_the_in_process_loop_returns(replies: list) -> None:
    body = NeMoGymResponseCreateParamsNonStreaming(input="Which option?", temperature=0.2)
    async with _model_server(replies) as (port, seen):
        in_process = SimpleAgent(config=_config(), server_client=_client(port))
        expected = await in_process._create_session_episode(
            SimpleAgentSessionState(request=_seed(), tool_access=None, resources_cookies=None),
            body,
            **_loop_arguments(),
        )
        in_process_calls = list(seen)
        seen.clear()

        sandboxed = SimpleAgent(config=_config(sandboxed=True), server_client=_client(port))
        loop = _LoopInProcess()
        state = SimpleAgentSessionState(
            request=_seed(sandbox_access=ACCESS),
            tool_access=None,
            resources_cookies=None,
            sandbox_loop=loop,
            model_server_endpoint=("127.0.0.1", port),
        )
        actual = await sandboxed._create_session_episode(state, body, **_loop_arguments())

    # The same model requests, on the same rollout path, with the same session header.
    assert seen == in_process_calls
    assert [path for path, _, _ in seen] == [MODEL_PATH]
    response, trajectory, _, _ = actual
    expected_response, expected_trajectory, _, _ = expected
    # The same output items, status, usage and error.
    for field in ("output", "status", "usage", "error", "incomplete_details"):
        assert getattr(response, field) == getattr(expected_response, field), field
    assert [turn.model_dump(mode="json")["answer"] for turn in trajectory.turns] == [
        turn.model_dump(mode="json")["answer"] for turn in expected_trajectory.turns
    ]
    assert [call.response_id for call in trajectory.model_calls] == [
        call.response_id for call in expected_trajectory.model_calls
    ]
    # The sandbox runs the loop in process there, with the agent's own settings otherwise.
    (payload,) = loop.payloads
    assert payload["config"]["sandboxed"] is False
    assert payload["config"]["model_server"] == {"type": "responses_api_models", "name": "policy_model"}
    assert payload["model_server"] == {"host": "127.0.0.1", "port": port}
    assert payload["body"] == {"input": "Which option?", "temperature": 0.2}
    assert "activation" not in payload["loop"]


async def test_the_runner_unpacked_from_the_harness_archive_runs_the_loop(tmp_path: Path) -> None:
    """The archive holds everything the loop imports: the runner runs from it alone, as in the sandbox."""
    sandbox_module.harness_archive.cache_clear()
    archive = sandbox_module.harness_archive()
    with tarfile.open(archive) as members:
        names = set(members.getnames())
        members.extractall(tmp_path, filter="data")
    assert {
        "nemo_gym/__init__.py",
        "nemo_gym/agents/utils/sandbox_borrow.py",
        "responses_api_agents/simple_agent/app.py",
        "responses_api_agents/simple_agent/sandbox_runner.py",
    } <= names
    # The same sources give the same archive.
    sandbox_module.harness_archive.cache_clear()
    assert sandbox_module.harness_archive() == archive

    runner = tmp_path / "responses_api_agents/simple_agent/sandbox_runner.py"
    async with _model_server([(200, _reply("42"))]) as (port, seen):
        payload = {
            "config": _config().model_dump(mode="json", exclude_unset=True),
            "model_server": {"host": "127.0.0.1", "port": port},
            "body": {"input": "What is 6 * 7?"},
            "loop": {key: value for key, value in _loop_arguments().items() if key != "activation"},
        }
        (tmp_path / "input.json").write_text(json.dumps(payload))
        # -E -s as in the sandbox; from / so nothing but the archive provides nemo_gym and the agent. The model
        # server runs on this event loop, so the runner is awaited, not waited for.
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-E",
            "-s",
            str(runner),
            str(tmp_path / "input.json"),
            str(tmp_path / "output.json"),
            cwd="/",
            env=SANDBOX_ENV,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
        assert process.returncode == 0, stderr.decode(errors="replace")

    output = json.loads((tmp_path / "output.json").read_text())
    assert output["response"]["output"][0]["content"][0]["text"] == "42"
    assert [path for path, _, _ in seen] == [MODEL_PATH]
    check = subprocess.run(
        [
            sys.executable,
            "-E",
            "-s",
            "-c",
            f"import sys; sys.argv = ['x']; sys.path.insert(0, {str(tmp_path)!r}); import nemo_gym; print(nemo_gym.__file__)",
        ],
        cwd="/",
        env=SANDBOX_ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert check.stdout.strip().startswith(str(tmp_path)), check.stdout


def test_the_runner_reports_what_the_loop_raised(tmp_path: Path) -> None:
    payload = {
        "config": _config().model_dump(mode="json", exclude_unset=True),
        "model_server": {"host": "127.0.0.1", "port": 9},
        "body": {"input": "hi"},
        # An argument the loop does not take fails it before any model call.
        "loop": {"model_url_path": MODEL_PATH, "in_session": True, "max_steps": 1},
    }
    (tmp_path / "input.json").write_text(json.dumps(payload))

    status = sandbox_runner.main(["runner", str(tmp_path / "input.json"), str(tmp_path / "output.json")])

    output = json.loads((tmp_path / "output.json").read_text())
    assert status == 1
    assert output["error_type"] == "TypeError"
    assert "max_steps" in output["error"]


def test_the_runtime_holds_every_distribution_the_loop_imports() -> None:
    """Importing the runner loads only the runtime's requirements and their dependencies, never numpy or pandas."""
    probe = (
        "import json, sys\n"
        "import responses_api_agents.simple_agent.sandbox_runner\n"
        "import responses_api_agents.simple_agent.app\n"
        "print(json.dumps(sorted({m.__file__ for m in list(sys.modules.values()) if getattr(m, '__file__', None)})))\n"
    )
    loaded = json.loads(
        subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True, timeout=120).stdout
    )
    owners: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        for file in distribution.files or []:
            owners[str(Path(file.locate()).resolve())] = distribution.metadata["Name"].lower().replace("_", "-")
    imported = {owners[path] for path in (str(Path(f).resolve()) for f in loaded) if path in owners} - {"nemo-gym"}

    assert not imported & {"numpy", "pandas", "scipy", "pyarrow"}
    covered: set[str] = set()
    pending = [name.lower() for name in sandbox_module.RUNTIME_REQUIREMENTS]
    while pending:
        name = pending.pop()
        if name in covered:
            continue
        covered.add(name)
        try:
            requirements = importlib.metadata.requires(name) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        for line in requirements:
            requirement = Requirement(line)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name.lower().replace("_", "-"))
    assert imported <= covered, sorted(imported - covered)


def test_a_sandboxed_agent_refuses_a_seed_that_lends_no_sandbox() -> None:
    agent = SimpleAgent(config=_config(sandboxed=True), server_client=_client(1))

    detail = check_agent_seed_rejected(agent.setup_webserver(), _seed())

    assert "lends none" in detail


def test_an_in_process_agent_still_refuses_a_lent_sandbox() -> None:
    agent = SimpleAgent(config=_config(), server_client=_client(1))

    detail = check_agent_seed_rejected(agent.setup_webserver(), _seed(sandbox_access=ACCESS))

    assert "sandboxed: true" in detail


async def test_a_sandboxed_agent_whose_model_server_the_sandbox_cannot_reach_is_refused(monkeypatch) -> None:
    agent = SimpleAgent(config=_config(sandboxed=True), server_client=_client(1))
    monkeypatch.setattr(
        "responses_api_agents.simple_agent.app.sandbox_server_url",
        lambda name, require_reachable: (_ for _ in ()).throw(ValueError("loopback address 127.0.0.1")),
    )

    with pytest.raises(Exception, match="cannot reach its model server"):
        await agent._seed_agent_session_state(_seed(sandbox_access=ACCESS))


async def test_a_sandboxed_agent_never_runs_its_loop_in_process() -> None:
    agent = SimpleAgent(config=_config(sandboxed=True), server_client=_client(1))

    with pytest.raises(RuntimeError, match="only in the sandbox"):
        await agent._create_episode(
            NeMoGymResponseCreateParamsNonStreaming(input="hi"), model_url_path=MODEL_PATH, in_session=True
        )
    assert agent._seed_restarts(_seed(sandbox_access=ACCESS)) is True
    assert SimpleAgent(config=_config(), server_client=_client(1))._seed_restarts(_seed()) is False
