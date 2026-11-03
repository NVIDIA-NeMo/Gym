# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import io
import json
import subprocess
import tarfile
import tomllib
from pathlib import Path
from shlex import quote
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from packaging.utils import canonicalize_name
from pydantic import ValidationError

import nemo_gym
from nemo_gym.agents import AgentCloseSessionRequest, SandboxSession
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.langgraph_agent import sandbox as module
from responses_api_agents.langgraph_agent.sandbox import LangGraphSandboxSession
from responses_api_agents.langgraph_agent.tests.test_app import MemorySandbox, _agent, _seed
from responses_api_agents.langgraph_agent.tests.test_app import sandbox_boundary as sandbox_boundary


def _state(sandbox: MemorySandbox) -> LangGraphSandboxSession:
    return LangGraphSandboxSession(
        request=_seed(),
        session=SandboxSession(
            sandbox=sandbox,
            session_dir="/tmp/graph session's runtime",
            workdir="/workspace/task",
            harness="LangGraph",
        ),
    )


@pytest.mark.parametrize(
    "field", ["sandbox_install_timeout_seconds", "sandbox_runner_timeout_seconds", "session_close_timeout_seconds"]
)
@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_timeouts_require_finite_positive_values(field, value):
    with pytest.raises(ValidationError):
        _agent(**{field: value})


def test_runtime_bundle_contains_current_sources_and_only_runtime_dependencies():
    bundle = module.build_runtime_bundle()
    with tarfile.open(fileobj=io.BytesIO(bundle.archive), mode="r:gz") as archive:
        names = archive.getnames()
        source = Path(nemo_gym.__file__).parent / "agents" / "utils" / "sandbox_session.py"
        assert archive.extractfile("runtime/nemo_gym/agents/utils/sandbox_session.py").read() == source.read_bytes()
        agent_dir = Path(module.__file__).parent
        for filename in ("workflows.py", "sandbox_runner.py"):
            assert (
                archive.extractfile(f"runtime/responses_api_agents/langgraph_agent/{filename}").read()
                == (agent_dir / filename).read_bytes()
            )
        project = tomllib.loads(archive.extractfile("runtime/pyproject.toml").read().decode())
    assert project["project"]["name"] == "nemo-gym"
    dependencies = project["project"]["dependencies"]
    for requirement in (agent_dir / "runtime-requirements.txt").read_text().splitlines():
        if requirement and not requirement.startswith("#"):
            assert requirement in dependencies
    assert all(not requirement.startswith(("pytest", "ruff", "mypy")) for requirement in dependencies)
    assert all(not name.startswith("runtime/responses_api_agents/langgraph_agent/app.py") for name in names)
    assert not any(part in name.split("/") for name in names for part in (".git", ".venv", ".env", ".aws"))
    assert len(bundle.digest) == 64


def test_runtime_bundle_constrains_dependencies_to_the_host_versions():
    constraints = dict(line.split("==", 1) for line in module.build_runtime_bundle().constraints.splitlines())
    names = {canonicalize_name(name) for name in constraints}
    # Base dependencies resolve to what this server runs, so episodes cannot drift between index releases.
    for dependency in ("pydantic", "fastapi", "aiohttp"):
        name = next(name for name in constraints if canonicalize_name(name) == dependency)
        assert constraints[name] == importlib.metadata.version(dependency)
    # The bundled source and the exact graph pins are never constrained to a conflicting host version.
    assert "nemo-gym" not in names
    assert not names & {"langgraph", "langchain-core"}
    assert all("+" not in version for version in constraints.values())


def test_agent_test_environment_installs_the_sandbox_graph_pins():
    # CI builds the agent's test environment from requirements.txt, so the graph tests run only with these pins.
    agent_dir = Path(module.__file__).parent
    runtime = {line for line in (agent_dir / "runtime-requirements.txt").read_text().splitlines() if line.strip()}
    host = {line.strip() for line in (agent_dir / "requirements.txt").read_text().splitlines()}
    assert runtime <= host


@pytest.fixture
def runtime_uv(monkeypatch, tmp_path):
    uv = tmp_path / "uv"
    uv.write_text("test uv executable", encoding="utf-8")
    monkeypatch.setattr(module, "uv_for_sandbox", AsyncMock(return_value=uv))
    return uv


async def test_runtime_install_uploads_exact_archive_and_validates_import(runtime_uv):
    sandbox = MemorySandbox()
    state = _state(sandbox)
    await state.install_runtime(install_timeout=37)
    uploaded_archive = sandbox.uploads[f"{state.session.session_dir}/runtime.tar.gz"]
    assert uploaded_archive == module.build_runtime_bundle().archive
    assert sandbox.uploads[f"{state.session.session_dir}/uv"] == runtime_uv.read_bytes()
    assert state.runtime_digest == module.build_runtime_bundle().digest
    constraints_path = f"{state.session.session_dir}/constraints.txt"
    assert sandbox.uploads[constraints_path].decode() == module.build_runtime_bundle().constraints
    install = [entry for entry in sandbox.commands if "pip install" in entry[0]]
    assert len(install) == 1
    command, directory, timeout = install[0]
    assert timeout == 37
    assert directory == state.session.session_dir
    assert f"--constraint {quote(constraints_path)}" in command
    assert "--python 3.13.14" in command
    assert " -I -c " in command
    assert "from responses_api_agents.langgraph_agent.workflows import run_workflow" in command
    assert "runtime.tar.gz" in command
    # Paths with whitespace and quotes must remain valid shell syntax.
    for command, _, _ in sandbox.commands:
        subprocess.run(["bash", "-n"], input=command, text=True, check=True, capture_output=True)
    assert sandbox.stops == sandbox.disconnects == 0


@pytest.mark.parametrize("phase", ["prepare", "chmod", "install"])
@pytest.mark.parametrize("result_kind", ["exit", "timeout", "no_exit_code"])
async def test_runtime_install_checks_every_exec_result(runtime_uv, monkeypatch, phase, result_kind):
    sandbox = MemorySandbox()
    state = _state(sandbox)
    original_exec = sandbox.exec
    failure = SandboxExecResult(
        return_code=1 if result_kind == "exit" else (0 if result_kind == "timeout" else None),
        error_type="timeout" if result_kind == "timeout" else None,
        stdout=None,
        stderr="installation did not finish",
    )

    async def execute(command, **kwargs):
        await original_exec(command, **kwargs)
        current_phase = (
            "prepare" if command.startswith("mkdir") else "chmod" if command.startswith("chmod") else "install"
        )
        if current_phase == phase:
            return failure
        return SandboxExecResult(return_code=0, stdout=None, stderr=None)

    monkeypatch.setattr(sandbox, "exec", execute)
    with pytest.raises(RuntimeError):
        await state.install_runtime(install_timeout=37)
    assert state.runtime_digest is None
    assert len(sandbox.commands) == {"prepare": 1, "chmod": 2, "install": 3}[phase]


@pytest.mark.parametrize("owned", [False, True])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failed_setup_preserves_original_error_and_retryable_cleanup(
    sandbox_boundary, monkeypatch, owned, cleanup_fails
):
    sandbox, _ = sandbox_boundary
    agent = _agent(sandbox_provider="runtime")
    seed = _seed(borrowed=not owned)
    request = SimpleNamespace(session={})
    monkeypatch.setattr(
        LangGraphSandboxSession, "install_runtime", AsyncMock(side_effect=RuntimeError("install failed"))
    )
    if cleanup_fails:
        sandbox.release_error = RuntimeError("cleanup unavailable")
    with pytest.raises(RuntimeError, match="install failed"):
        await agent.seed_agent_session(request, seed)
    if cleanup_fails:
        record = agent._session_records[seed.agent_session_id]
        assert record.closing and record.state is not None
        with pytest.raises(HTTPException) as error:
            await agent.seed_agent_session(request, seed)
        assert error.value.status_code == 409
    else:
        assert all(record.state is None for record in agent._session_records.values())
    sandbox.release_error = None
    await agent.close_agent_session(
        request, AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    )
    assert all(record.state is None for record in agent._session_records.values())
    expected_releases = 2 if cleanup_fails else 1
    assert (sandbox.stops, sandbox.disconnects) == ((expected_releases, 0) if owned else (0, expected_releases))


@pytest.mark.parametrize("output", [[], "bad", 1, None])
async def test_runner_output_requires_json_object(output):
    sandbox = MemorySandbox()
    sandbox.output = output
    with pytest.raises(TypeError, match="JSON object"):
        await _state(sandbox).collect_artifacts()


@pytest.mark.parametrize("missing", ["response", "runtime"])
async def test_runner_result_cannot_succeed_without_response_and_runtime(missing):
    sandbox = MemorySandbox()
    del sandbox.output[missing]
    state = _state(sandbox)
    with pytest.raises(TypeError, match="response and runtime"):
        await state.execute({}, timeout=10, close_timeout=5)
    await state.close(5)
    assert sandbox.disconnects == 1


async def test_borrowed_execution_honors_environment_and_user(sandbox_boundary, monkeypatch):
    sandbox, _ = sandbox_boundary
    agent = _agent()
    seed = _seed()
    seed.sandbox_access.env = {"TASK_DATA": "/workspace/task/data"}
    seed.sandbox_access.user = "1000"
    request = SimpleNamespace(session={})
    await agent.seed_agent_session(request, seed)
    state = agent._session_records[seed.agent_session_id].state
    observed = []
    original_exec = sandbox.exec

    async def execute(command, **kwargs):
        if "exec bash" in command:
            observed.append(kwargs)
        return await original_exec(command, **kwargs)

    monkeypatch.setattr(sandbox, "exec", execute)
    await state.execute({}, timeout=10, close_timeout=5)
    assert len(observed) == 1
    assert observed[0]["cwd"] == "/workspace/task"
    assert observed[0]["env"] == {"TASK_DATA": "/workspace/task/data"}
    assert observed[0]["user"] == "1000"
    await agent.close_agent_session(
        request, AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    )
    assert sandbox.stops == 0


async def test_connect_failure_closes_provider_without_releasing_an_unconnected_sandbox(sandbox_boundary, monkeypatch):
    from responses_api_agents.langgraph_agent import app

    sandbox, factory = sandbox_boundary
    provider = AsyncMock()
    monkeypatch.setattr(app, "create_provider", lambda _config: provider)
    factory.connect.side_effect = RuntimeError("connection unavailable")
    agent = _agent()
    with pytest.raises(RuntimeError, match="connection unavailable"):
        await agent.seed_agent_session(SimpleNamespace(session={}), _seed())
    provider.aclose.assert_awaited_once()
    assert sandbox.disconnects == sandbox.stops == 0
    assert not agent._session_records


@pytest.mark.parametrize("ttl,expected", [({}, 630), ({"ttl_s": 42}, 42), ({"ttl_s": None}, None)])
async def test_owned_sandbox_merges_metadata_and_keeps_config_unchanged(sandbox_boundary, monkeypatch, ttl, expected):
    from responses_api_agents.langgraph_agent import app

    sandbox, _ = sandbox_boundary
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_args: {"provider": "default", "environment": "eos"})
    config = {"image": "ubuntu:24.04", "workdir": "/workspace", "metadata": {"provider": "override"}, **ttl}
    agent = _agent(
        sandbox_provider="runtime",
        sandbox_config=config,
        sandbox_install_timeout_seconds=10,
        sandbox_runner_timeout_seconds=20,
    )
    seed = _seed(borrowed=False)
    request = SimpleNamespace(session={})
    await agent.seed_agent_session(request, seed)
    assert len(sandbox.started) == 1
    spec = sandbox.started[0]
    assert spec.ttl_s == expected
    assert spec.metadata == {
        "provider": "override",
        "environment": "eos",
        "harness": "langgraph",
        "agent-session": seed.agent_session_id,
    }
    assert agent.config.sandbox_config == config
    await agent.close_agent_session(
        request, AgentCloseSessionRequest(agent_session_id=seed.agent_session_id, episode_id=seed.episode_id)
    )
    assert sandbox.stops == 1


async def test_stage_uses_isolated_venv_python_and_transfers_payload_without_shell_interpolation():
    sandbox = MemorySandbox()
    state = _state(sandbox)
    state.session.env = {"PYTHONPATH": "/workspace/untrusted-package"}
    payload = {
        "workflow": "reflection",
        "config": {"max_reflections": 2},
        "request": {"input": "Question with shell text: $(touch sentinel); `echo sentinel`"},
        "model_url": "http://model-server:8080/ng-rollout/graph-a1/training-token-capture/v1/responses",
    }
    command = await state.stage_activation(payload)

    assert command.argv == [
        state.python,
        "-I",
        "-m",
        "responses_api_agents.langgraph_agent.sandbox_runner",
        f"{state.session.session_dir}/input.json",
        f"{state.session.session_dir}/output.json",
    ]
    # The supervisor's stdlib-runner path resolves the base interpreter and would lose this venv's dependencies.
    assert command.stdlib_runner is False
    assert json.loads(sandbox.uploads[f"{state.session.session_dir}/input.json"]) == payload
    assert not any("sentinel" in argument for argument in command.argv)
    assert not sandbox.commands
