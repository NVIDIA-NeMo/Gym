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
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess, SandboxExecResult
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

        sandboxed = SimpleAgent(config=_config(), server_client=_client(port))
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
    # The sandbox runs the loop in process there, with the agent's own settings.
    (payload,) = loop.payloads
    assert payload["config"]["model_server"] == {"type": "responses_api_models", "name": "policy_model"}
    # Only the settings the loop reads are staged in the sandbox.
    assert set(payload["config"]) == {
        "name",
        "host",
        "port",
        "entrypoint",
        "model_server",
        "max_model_retries",
    }
    assert payload["model_server"] == {"host": "127.0.0.1", "port": port}
    assert payload["body"] == {"input": "Which option?", "temperature": 0.2}
    assert "activation" not in payload["loop"]


async def test_the_runner_unpacked_from_the_harness_archive_runs_the_loop(tmp_path: Path) -> None:
    """The archive holds everything the loop imports: the runner runs from it alone, as in the sandbox."""
    sandbox_module._harness_bytes.cache_clear()
    archive = sandbox_module.harness_archive()
    with tarfile.open(archive) as members:
        names = set(members.getnames())
        members.extractall(tmp_path, filter="data")
    assert {
        "nemo_gym/__init__.py",
        "nemo_gym/server_runtime/sandbox_borrow.py",
        "responses_api_agents/simple_agent/app.py",
        "responses_api_agents/simple_agent/sandbox_runner.py",
    } <= names
    # The same sources give the same archive, and a removed or changed file is written again.
    content = archive.read_bytes()
    sandbox_module._harness_bytes.cache_clear()
    assert sandbox_module._harness_bytes() == content
    archive.write_bytes(b"tampered")
    assert sandbox_module.harness_archive().read_bytes() == content
    assert archive.parent.stat().st_mode & 0o077 == 0

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
    pending = [Requirement(line).name.lower() for line in sandbox_module.RUNTIME_REQUIREMENTS]
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


async def test_the_seed_decides_where_the_loop_runs(installed_runtime, monkeypatch) -> None:
    """A seed that lends a sandbox runs the loop in it; one that lends none runs it in process. No agent setting
    chooses: the environment that lends the sandbox does."""
    agent = SimpleAgent(config=_config(), server_client=_client(1))
    sandbox = _FakeSandbox()
    borrowed: list[SandboxAccess] = []

    async def borrow(access: SandboxAccess) -> _FakeSandbox:
        borrowed.append(access)
        return sandbox

    monkeypatch.setattr("responses_api_agents.simple_agent.app.borrow", borrow)
    monkeypatch.setattr(
        "responses_api_agents.simple_agent.app.sandbox_server_url",
        lambda name, require_reachable: "http://10.0.0.5:8000",
    )

    in_process = await agent._seed_agent_session_state(_seed())
    assert in_process.sandbox_loop is None
    assert borrowed == []

    lent = await agent._seed_agent_session_state(_seed(sandbox_access=ACCESS))
    assert borrowed == [ACCESS]
    assert lent.sandbox_loop is not None
    assert lent.model_server_endpoint == ("10.0.0.5", 8000)
    # The loop's runtime was installed in the borrowed sandbox and its runner checked there.
    assert sandbox.commands[-1].endswith(f"{sandbox_module.runner_path()} --check")


async def test_an_agent_whose_model_server_the_sandbox_cannot_reach_is_refused(monkeypatch) -> None:
    agent = SimpleAgent(config=_config(), server_client=_client(1))
    monkeypatch.setattr(
        "responses_api_agents.simple_agent.app.sandbox_server_url",
        lambda name, require_reachable: (_ for _ in ()).throw(ValueError("loopback address 127.0.0.1")),
    )

    with pytest.raises(Exception, match="cannot reach its model server"):
        await agent._seed_agent_session_state(_seed(sandbox_access=ACCESS))


async def test_a_session_lent_a_sandbox_never_runs_its_loop_in_process() -> None:
    agent = SimpleAgent(config=_config(), server_client=_client(1))
    # As a restore would rebuild it: lent a sandbox, but with no loop in it.
    state = SimpleAgentSessionState(request=_seed(sandbox_access=ACCESS), tool_access=None, resources_cookies=None)

    with pytest.raises(RuntimeError, match="only in that sandbox"):
        await agent._create_session_episode(
            state, NeMoGymResponseCreateParamsNonStreaming(input="hi"), **_loop_arguments()
        )
    # A loop in a borrowed sandbox is never checkpointed: its episode starts over after a crash.
    assert agent._seed_must_restart(_seed(sandbox_access=ACCESS)) is True


def test_a_model_servers_http_error_comes_back_from_the_sandbox_as_it_was_raised() -> None:
    from aiohttp import ClientResponseError, RequestInfo
    from multidict import CIMultiDict, CIMultiDictProxy
    from yarl import URL

    url = URL(f"http://10.0.0.5:8000{MODEL_PATH}")
    raised = ClientResponseError(
        RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url),
        (),
        status=502,
        message="Bad Gateway",
        headers=CIMultiDictProxy(CIMultiDict([("x-ng-failure", "report"), ("x-other", "1")])),
    )
    raised.response_content = b'{"detail": "upstream"}'
    try:
        raise raised
    except ClientResponseError as error:
        output = json.loads(json.dumps(sandbox_runner.error_output(error)))

    again = sandbox_module.loop_error(output)

    assert isinstance(again, ClientResponseError)
    assert (again.status, again.message, again.request_info.method) == (502, "Bad Gateway", "POST")
    assert again.headers["x-ng-failure"] == "report"
    assert again.response_content == b'{"detail": "upstream"}'
    assert isinstance(
        sandbox_module.loop_error({"error": "ValueError: bad", "error_type": "ValueError"}), RuntimeError
    )


class _ClosingLoop:
    def __init__(self) -> None:
        self.closed = 0

    async def close(self, *, timeout: float) -> None:
        self.closed += 1


async def test_retiring_a_session_lent_a_sandbox_stops_its_loop_and_releases_the_sandbox() -> None:
    agent = SimpleAgent(config=_config(), server_client=_client(1))
    loop = _ClosingLoop()
    state = SimpleAgentSessionState(
        request=_seed(sandbox_access=ACCESS),
        tool_access=None,
        resources_cookies=None,
        sandbox_loop=loop,
        model_server_endpoint=("10.0.0.5", 8000),
    )
    agent._install_restored_session("agent-session", state)

    await agent.retire_agent_session("agent-session")

    assert loop.closed == 1
    assert "agent-session" not in agent._session_records


async def test_an_agent_refuses_a_tool_grant_the_sandbox_cannot_reach(monkeypatch) -> None:
    agent = SimpleAgent(config=_config(), server_client=_client(1))
    monkeypatch.setattr(
        "responses_api_agents.simple_agent.app.sandbox_server_url", lambda name, require_reachable: "http://10.0.0.5:1"
    )
    grant = {"name": "mcqa.direct_http", "kind": "direct_http", "required": True, "base_url": "http://127.0.0.1:9"}

    with pytest.raises(Exception, match="cannot reach its tool grant"):
        await agent._seed_agent_session_state(_seed(sandbox_access=ACCESS, tool_accesses=[grant]))


def test_a_subclass_that_changes_the_agent_loop_refuses_a_lent_sandbox() -> None:
    class _OwnLoop(SimpleAgent):
        async def _create_episode(self, body, **kwargs):  # noqa: ANN001, ANN003, ANN202
            raise AssertionError("not reached")

    agent = _OwnLoop(config=_config(), server_client=_client(1))

    detail = check_agent_seed_rejected(agent.setup_webserver(), _seed(sandbox_access=ACCESS))

    assert "changes the agent loop" in detail


class _FakeSandbox:
    """A borrowed sandbox that records commands and keeps uploaded files in memory; a command naming ``fail`` fails."""

    def __init__(self, *, fail: str | None = None) -> None:
        self.commands: list[str] = []
        self.files: dict[str, bytes] = {}
        self.fail = fail

    async def exec(self, command: str, **kwargs: Any) -> SandboxExecResult:
        self.commands.append(command)
        failed = self.fail is not None and self.fail in command
        return SandboxExecResult(stdout="", stderr="no such module" if failed else "", return_code=1 if failed else 0)

    async def upload(self, source: Path, target: str) -> None:
        self.files[target] = Path(source).read_bytes()

    async def download(self, source: str, destination: Path) -> None:
        Path(destination).write_bytes(self.files[source])


@pytest.fixture
def installed_runtime(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The runtime counts as installed unless a test says otherwise; installs are recorded by their timeout."""
    installs: list[float] = []

    async def install(runtime: Any, sandbox: Any, *, timeout_s: float) -> None:
        installs.append(timeout_s)

    # The runtime is a frozen dataclass, so its class's methods are replaced.
    runtime_class = type(sandbox_module.RUNTIME)
    monkeypatch.setattr(runtime_class, "is_installed", lambda runtime, sandbox: _resolved(True))
    monkeypatch.setattr(runtime_class, "install", install)
    return installs


async def _resolved(value: Any) -> Any:
    return value


async def test_install_unpacks_this_servers_sources_beside_the_runtime_and_checks_the_runner(
    installed_runtime,
) -> None:
    sandbox = _FakeSandbox()
    loop = sandbox_module.SandboxLoop.borrowed(sandbox, workdir="/workspace")

    await loop.install(timeout=900)

    assert installed_runtime == []
    archive = f"{loop.session.session_dir}/harness.tar.gz"
    assert sandbox.files[archive] == sandbox_module.harness_archive().read_bytes()
    harness = sandbox_module.harness_dir()
    assert harness.startswith(sandbox_module.RUNTIME.directory)
    assert any(f"tar -xzf {archive} -C {harness}" in command for command in sandbox.commands)
    assert sandbox.commands[-1].endswith(f"{sandbox_module.runner_path()} --check")


async def test_install_builds_a_missing_runtime_and_fails_the_seed_when_the_runner_cannot_import(
    installed_runtime, monkeypatch
) -> None:
    monkeypatch.setattr(type(sandbox_module.RUNTIME), "is_installed", lambda runtime, sandbox: _resolved(False))
    sandbox = _FakeSandbox(fail="--check")
    loop = sandbox_module.SandboxLoop.borrowed(sandbox, workdir="/workspace")

    with pytest.raises(RuntimeError, match="could not import the agent loop in the sandbox: no such module"):
        await loop.install(timeout=123)

    assert installed_runtime == [123]


async def test_each_activation_stages_its_own_input_and_returns_what_the_runner_wrote(monkeypatch) -> None:
    sandbox = _FakeSandbox()
    loop = sandbox_module.SandboxLoop.borrowed(sandbox, workdir="/workspace")
    rearmed: list[str] = []
    commands: list[list[str]] = []

    async def begin_activation(*, session_dir: str, timeout: float) -> None:
        rearmed.append(session_dir)
        loop.session.session_dir = session_dir

    async def execute(*, stage_activation, collect, timeout: float, close_timeout: float) -> dict[str, Any]:
        command = await stage_activation()
        commands.append(command.argv)
        # The runner writes its output where the command says.
        sandbox.files[command.argv[-1]] = json.dumps({"response": {"answer": len(commands)}}).encode()
        return await collect()

    monkeypatch.setattr(loop.session, "begin_activation", begin_activation)
    monkeypatch.setattr(loop.session, "execute", execute)
    root = loop.session.session_dir

    first = await loop.run({"body": 1}, timeout=10, close_timeout=5)
    second = await loop.run({"body": 2}, timeout=10, close_timeout=5)

    assert (first, second) == ({"response": {"answer": 1}}, {"response": {"answer": 2}})
    assert rearmed == [f"{root}/activation-2"]
    assert commands[1] == [
        sandbox_module.RUNTIME.python,
        "-E",
        "-s",
        sandbox_module.runner_path(),
        f"{root}/activation-2/input.json",
        f"{root}/activation-2/output.json",
    ]
    assert json.loads(sandbox.files[f"{root}/activation-2/input.json"]) == {"body": 2}
    # Each activation's directory is made before its input is uploaded there.
    assert f"mkdir -p {root}/activation-2" in sandbox.commands


async def test_an_activation_raises_what_the_loop_raised_or_its_missing_output(monkeypatch) -> None:
    from aiohttp import ClientResponseError

    sandbox = _FakeSandbox()
    loop = sandbox_module.SandboxLoop.borrowed(sandbox, workdir="/workspace")
    written: dict[str, Any] | None = None

    async def execute(*, stage_activation, collect, timeout: float, close_timeout: float) -> dict[str, Any]:
        command = await stage_activation()
        if written is not None:
            sandbox.files[command.argv[-1]] = json.dumps(written).encode()
        return await collect()

    monkeypatch.setattr(loop.session, "execute", execute)
    monkeypatch.setattr(loop.session, "read_output_log", lambda: _resolved("Traceback: killed"))
    monkeypatch.setattr(loop.session, "begin_activation", lambda **kwargs: _resolved(None))

    with pytest.raises(RuntimeError, match="exited without output: Traceback: killed"):
        await loop.run({}, timeout=10, close_timeout=5)

    written = {
        "error": "ClientResponseError: 503",
        "error_type": "ClientResponseError",
        "http_error": {
            "url": "http://10.0.0.5:8000/v1/responses",
            "method": "POST",
            "status": 503,
            "message": "Service Unavailable",
            "headers": [],
            "content": None,
        },
    }
    with pytest.raises(ClientResponseError) as raised:
        await loop.run({}, timeout=10, close_timeout=5)
    assert raised.value.status == 503


async def test_close_releases_the_borrowed_sandbox_through_its_session(monkeypatch) -> None:
    loop = sandbox_module.SandboxLoop.borrowed(_FakeSandbox(), workdir="/workspace")
    closed: list[float] = []

    async def close(*, timeout: float) -> None:
        closed.append(timeout)

    monkeypatch.setattr(loop.session, "close", close)

    await loop.close(timeout=7)

    assert closed == [7]
    assert loop.session.owns_sandbox is False


def test_the_model_servers_address_must_name_a_host_and_a_port() -> None:
    assert sandbox_module.model_server_endpoint("http://10.0.0.5:8000") == ("10.0.0.5", 8000)
    with pytest.raises(ValueError, match="names no host and port"):
        sandbox_module.model_server_endpoint("http://10.0.0.5")
