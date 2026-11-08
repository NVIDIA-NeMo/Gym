# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise NOOA's typed launch and borrowed-sandbox conformance fixtures."""

import ast
import asyncio
import io
import json
import sys
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from scripts.harness_conformance.episode import _config, _install_sampling, check_runtime
from scripts.harness_conformance.nooa import BorrowedSandboxProbe, observer_module_name
from scripts.harness_conformance.provider import ANSWER, Probe
from scripts.harness_conformance.scenarios import SCENARIOS

from nemo_gym.core import ModelCallSampling, NeMoGymResponse, encode_model_sampling
from nemo_gym.rollouts.harness_capabilities import digest_file
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.server_runtime import ResourcesSeedSessionResponse


SCENARIO = {scenario.name: scenario for scenario in SCENARIOS}
PYTHON_CELL = {
    "type": "function",
    "name": "python_cell",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}
ATOM_ADAPTER = "responses_api_agents.nooa_agent.atom_adapter:invoke_atom"


def test_prepared_atom_keeps_settings_and_wraps_its_selected_adapter(tmp_path, probe_ports):
    prepared = {
        "nooa": {
            "agent_class": "nooa_atom.gym:AtomHarness",
            "invocation_adapter": ATOM_ADAPTER,
            "execution_mode": "sandboxed",
            "init_kwargs": {"turn_method": "handle_orchestrate"},
        },
        "settings_file": str(tmp_path / "settings.yaml"),
        "context_window": 32768,
        "max_policy_calls": None,
        "sandbox_runner_timeout_seconds": 417,
    }
    original = deepcopy(prepared)
    config = _config("nooa", tmp_path, probe_ports, 23, use_agent_sessions=True, prepared_agent=prepared)
    effective = config["probe_agent"]["responses_api_agents"]["nooa_agent"]
    assert effective["nooa"]["invocation_adapter"] != ATOM_ADAPTER
    assert effective["nooa"]["invocation_adapter"] == f"{observer_module_name(ATOM_ADAPTER)}:invoke_with_witness"
    assert effective["nooa"] | {"invocation_adapter": ATOM_ADAPTER} == original["nooa"]
    assert all(effective[key] == value for key, value in original.items() if key != "nooa")
    assert prepared == original


@pytest.mark.parametrize("scenario_name", ["tool_success", "tool_failure"])
async def test_prepared_atom_executes_scripted_probe_with_independent_witness(
    tmp_path, monkeypatch, probe_ports, scenario_name, caplog
):
    atom_gym = pytest.importorskip("nooa_atom.gym")
    from nooa.runtime.hooks import hooks_scope
    from scripts.harness_conformance import nooa
    from scripts.harness_conformance.nooa_observer import ExecutionWitness

    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from responses_api_agents.nooa_agent import atom_adapter
    from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
    from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    settings = tmp_path / "settings.yaml"
    settings.write_text("atom:\n  summarization:\n    policy: none\n  context_error_policy: raise\n")
    monkeypatch.setenv("NEMO_OO_SETTINGS", str(settings))
    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)
    original_factory = atom_gym.create_session_agent
    agents = []

    def factory(*args, **kwargs):
        agent = original_factory(*args, **kwargs)
        assert agent.context_error_policy == "raise"
        assert agent.get_summarization_status()["policy"] == "none"
        agents.append(agent)
        return agent

    monkeypatch.setattr(atom_gym, "create_session_agent", factory)
    production = AsyncMock(wraps=atom_adapter.invoke_atom)
    monkeypatch.setattr(atom_adapter, "invoke_atom", production)
    prepared = {
        "nooa": {
            "agent_class": "nooa_atom.gym:AtomHarness",
            "invocation_adapter": ATOM_ADAPTER,
            "execution_mode": "sandboxed",
            "init_kwargs": {"turn_method": "handle_batch", "sessions_dir": str(tmp_path / "sessions")},
        }
    }
    config = _config("nooa", tmp_path, probe_ports, 23, use_agent_sessions=True, prepared_agent=prepared)
    invocation = config["probe_agent"]["responses_api_agents"]["nooa_agent"]["nooa"]
    # Import the same immutable fixture bytes installed in the disposable worker.
    module = ModuleType(invocation["invocation_adapter"].split(":")[0])
    exec(nooa.observer_source(ATOM_ADAPTER), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    probe = BorrowedSandboxProbe(SCENARIO[scenario_name], tmp_path, invocation_adapter=ATOM_ADAPTER, atom=True)
    app = probe.model_app()
    _install_sampling(app, tmp_path, name="policy_model", required=True)
    sampling = ModelCallSampling(max_output_tokens=1024, temperature=1.0, top_p=1.0)
    model_path = f"/ng-rollout/atom-probe/ng-sampling/{encode_model_sampling(sampling)}/v1/responses"
    with TestClient(app) as model:

        async def post(**kwargs):
            assert kwargs["url_path"] == model_path
            response = model.post(model_path, json=kwargs["json"].model_dump(mode="json", exclude_none=True))
            assert response.status_code == 200, response.text
            return FakeHTTPResponse(response.json())

        runner = InProcessNOOARunner(
            invocation=NOOAInvocationConfig.model_validate(invocation),
            server_client=SimpleNamespace(post=post),
            model_server_name="policy_model",
            max_policy_calls=4,
            context_window=32768,
        )
        other = ExecutionWitness(tmp_path / "other.jsonl")
        with hooks_scope(other):
            result = await runner.run(
                NOOARunRequest(
                    responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                        input="Run the shell checks then finish.", max_output_tokens=1024
                    ),
                    model_url_path=model_path,
                )
            )
    assert result.termination_reason is None, result.termination_error
    assert result.return_value.solution_description == ANSWER
    production.assert_awaited_once()
    assert len(agents) == 1
    assert len(probe.attempts) == 3 and probe.finished and not probe.violations
    assert all(attempt["request"]["max_output_tokens"] == 1024 for attempt in probe.attempts)
    assert all(attempt["request"]["temperature"] == attempt["request"]["top_p"] == 1.0 for attempt in probe.attempts)
    assert all(call["executed"] and call["result_seen"] for call in probe.tool_calls)
    executions = [json.loads(line) for line in (tmp_path / "native_executions.jsonl").read_text().splitlines()]
    assert executions == [json.loads(line) for line in (tmp_path / "other.jsonl").read_text().splitlines()]
    by_id = {item["id"]: item for item in executions}
    assert [by_id[call["id"]]["exit_code"] for call in probe.tool_calls] == [int(scenario_name == "tool_failure"), 0]
    assert by_id[probe.finish_call["id"]]["exit_code"] == 0
    # Gym's ordinary projection remains present alongside the independent observer.
    assert {call.tool_name for call in result.trajectory.tool_calls} >= {"python_cell"}
    probe.save()
    assert json.loads((tmp_path / "witness.json").read_text())["invocation_adapter"] == ATOM_ADAPTER
    assert not any(record.levelname == "ERROR" for record in caplog.records)


async def test_observer_install_binds_source_to_configured_adapter_and_offline_runtime(tmp_path, monkeypatch):
    pytest.importorskip("nooa")
    from scripts.harness_conformance.nooa import BENCH_ADAPTER, install_observer, observer_source

    runtime = AsyncMock(return_value="/private-runtime/python")
    monkeypatch.setattr("responses_api_agents.nooa_agent.sandbox_runtime.prepare_nooa_runtime", runtime)
    staged = {}

    async def upload(source, destination):
        staged[destination] = source.read_text()

    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None, stdout="/site\n", stderr="")),
        upload=upload,
    )
    await install_observer(sandbox, invocation_adapter=ATOM_ADAPTER, wheelhouse_path=tmp_path)
    runtime.assert_awaited_once_with(sandbox, wheelhouse_path=tmp_path)
    module = observer_module_name(ATOM_ADAPTER)
    assert staged == {f"/site/{module}.py": observer_source(ATOM_ADAPTER)}
    assert module != observer_module_name(BENCH_ADAPTER)
    assert f"from {module} import invoke_with_witness" in sandbox.exec.call_args.args[0]


@pytest.mark.parametrize("error", [ValueError("production failure"), asyncio.CancelledError()])
async def test_observer_preserves_configured_adapter_failure_and_outer_hooks(tmp_path, monkeypatch, error):
    hooks = pytest.importorskip("nooa.runtime.hooks")
    from scripts.harness_conformance import nooa_observer

    production = AsyncMock(side_effect=error)
    selected = ModuleType("selected_test_adapter")
    selected.run = production
    monkeypatch.setitem(sys.modules, selected.__name__, selected)
    monkeypatch.setattr(nooa_observer, "INVOCATION_ADAPTER", "selected_test_adapter:run")
    observer = nooa_observer.ExecutionWitness(tmp_path / "other.jsonl")
    agent, params = object(), object()
    with hooks.hooks_scope(observer):
        with pytest.raises(type(error)) as caught:
            await nooa_observer.invoke_with_witness(agent, params)
        assert caught.value is error
        assert hooks.get_hooks() is observer
    production.assert_awaited_once_with(agent, params)


def test_nooa_runtime_records_locked_dependencies():
    adapter = Path(__file__).resolve().parents[3] / "responses_api_agents/nooa_agent"
    assert not (adapter / "requirements.txt").exists()

    runtime = check_runtime("nooa", use_agent_sessions=True)

    assert runtime["pyproject_sha256"] == digest_file(adapter / "pyproject.toml")
    assert runtime["lock_sha256"] == digest_file(adapter / "uv.lock")
    assert runtime["bootstrap_sha256"] == digest_file(adapter / "sandbox_runtime.py")
    assert runtime["installation"] == "agent_session"


@pytest.fixture
def probe_ports() -> dict[int | str, int]:
    # Support this stack's positional ports and main's named-port API from #4631.
    named = {"policy_model": 10001, "probe_resources": 10002, "probe_agent": 10003, "probe_environment": 10004}
    return {**dict(enumerate(named.values())), **named}


def test_nooa_config_selects_production_borrower_and_bench_agent(tmp_path, probe_ports):
    pytest.importorskip("nooa")
    from responses_api_agents.nooa_agent.config import NOOAAgentConfig

    config = _config("nooa", tmp_path, probe_ports, 23, use_agent_sessions=True)
    agent = NOOAAgentConfig.model_validate(config["probe_agent"]["responses_api_agents"]["nooa_agent"])

    assert agent.nooa.execution_mode == "sandboxed"
    assert agent.nooa.agent_class == "nooa_bench.bench_agent:BenchAgent"
    assert agent.nooa.invocation_adapter == f"{observer_module_name()}:invoke_with_witness"
    assert agent.sandbox_runner_timeout_seconds == 23
    assert agent.resources_server is None
    assert agent.model_server.name == "policy_model"
    assert config["probe_sandbox"] == {"conformance_local": {"concurrency": 1}}


async def test_resources_seed_supports_real_borrowed_execution_and_owner_cleanup(tmp_path, monkeypatch, probe_ports):
    monkeypatch.setattr("scripts.harness_conformance.nooa.install_observer", AsyncMock())
    probe = BorrowedSandboxProbe(SCENARIO["tool_success"], tmp_path)
    config = _config("nooa", tmp_path, probe_ports, 23, use_agent_sessions=True)
    session = {
        "resources_session_id": "resources-session",
        "episode_id": {"rollout_id": "episode", "attempt": 0},
        "task_id": {"taskset": "conformance", "task_id": "tool_success"},
        "task_data": {},
    }
    with TestClient(probe.resources_app()) as resources:
        seeded = resources.post("/seed_session", json=session)
        assert seeded.status_code == 200
        seed = ResourcesSeedSessionResponse.model_validate(seeded.json())
        assert seed.resources_session_id == session["resources_session_id"]
        assert seed.sandbox_access is not None
        assert seed.sandbox_access.workdir == str(tmp_path / "workspace")
        assert resources.cookies.get("probe_resources_session")
        access = seed.sandbox_access
        borrower = await AsyncSandbox.connect(
            access.connection.descriptor,
            provider=create_provider(config[access.connection.provider_config_ref]),
        )
        try:
            execution = await borrower.exec("printf borrowed > witness.txt; cat witness.txt", timeout_s=5)
            assert (execution.return_code, execution.stdout, execution.error_type) == (0, "borrowed", None)
            assert (tmp_path / "workspace/witness.txt").read_text() == "borrowed"
        finally:
            await borrower.disconnect()
        # Releasing the agent's client must leave the Resources-owned sandbox available for grading.
        owner_execution = await probe.sandbox.exec("cat witness.txt", timeout_s=5)
        assert owner_execution.stdout == "borrowed"
        close_body = {key: session[key] for key in ("resources_session_id", "episode_id")}
        closed = resources.post("/close_session", json=close_body)
        assert closed.status_code == 200 and closed.json() == {"resources_session_id": "resources-session"}
        with pytest.raises(RuntimeError, match="Sandbox has not been started"):
            await probe.sandbox.exec("true", timeout_s=5)
    assert probe.seeded == 1 and probe.closed == 1


@pytest.mark.parametrize("scenario_name", ["tool_success", "tool_failure"])
async def test_python_cell_executes_shell_recovery_and_submits_real_task_result(tmp_path, scenario_name):
    bench_agent = pytest.importorskip("nooa_bench.bench_agent")
    from nooa.tools.shell_tools import ShellTools

    shell = ShellTools(cwd=str(tmp_path), init_command="")
    probe = Probe(SCENARIO[scenario_name], tmp_path)
    history = [{"role": "user", "content": "Run the checks"}]
    body = {"model": "conformance-model", "input": history, "tools": [PYTHON_CELL]}
    with TestClient(probe.model_app()) as model:
        for index in range(2):
            response = model.post("/v1/responses", json=body)
            assert response.status_code == 200
            NeMoGymResponse.model_validate(response.json())
            (call,) = response.json()["output"]
            assert call["name"] == "python_cell"
            code = json.loads(call["arguments"])["code"]
            output = io.StringIO()
            expected_exit = probe.scenario.tool_exit_code if index == 0 else 0
            compiled = compile(code, "<conformance-cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            with redirect_stdout(output):
                if expected_exit:
                    with pytest.raises(RuntimeError, match="shell exited with code 7"):
                        await eval(compiled, {"self": SimpleNamespace(shell=shell)})
                else:
                    await eval(compiled, {"self": SimpleNamespace(shell=shell)})
            history.extend(
                [call, {"type": "function_call_output", "call_id": call["call_id"], "output": output.getvalue()}]
            )
        response = model.post("/v1/responses", json=body)
        assert response.status_code == 200
        (finish,) = response.json()["output"]
        submitted = []
        exec(json.loads(finish["arguments"])["code"], {"return_result": submitted.append})

    await shell.close()
    (result,) = submitted
    assert isinstance(result, bench_agent.TaskResult)
    assert result.solution_description == result.evidence == result.how_to_verify == ANSWER
    assert probe.finish_call["id"] == finish["call_id"]
    assert probe.finished and not probe.violations
    assert len(probe.tool_calls) == 2
    assert all(call["executed"] and call["result_seen"] for call in probe.tool_calls)
    assert all(call["outputs"] == [call["token"]] for call in probe.tool_calls)


async def test_observer_wraps_production_adapter_once_and_keeps_other_hooks(tmp_path, monkeypatch):
    hooks = pytest.importorskip("nooa.runtime.hooks")
    from scripts.harness_conformance import nooa_observer

    production = AsyncMock(return_value="production-result")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    agent = SimpleNamespace(shell=SimpleNamespace(cwd="/app"))
    params = SimpleNamespace(input="task")
    observer = nooa_observer.ExecutionWitness(tmp_path / "other.jsonl")

    async def execute(*args):
        context = hooks.call_before_hook(
            "before_code_execution", agent=agent, code="inspect(self)", execution_id="exec-1", tool_call_id="prefill-1"
        )
        hooks.call_after_hook(
            "after_code_execution",
            context=context,
            agent=agent,
            code="inspect(self)",
            result=None,
            execution_id="exec-1",
            exception=None,
        )
        context = hooks.call_before_hook(
            "before_code_execution",
            agent=agent,
            code="raise ValueError()",
            execution_id="exec-2",
            tool_call_id="failed-1",
        )
        hooks.call_after_hook(
            "after_code_execution",
            context=context,
            agent=agent,
            code="raise ValueError()",
            result=None,
            execution_id="exec-2",
            exception=ValueError("failed"),
        )
        return "production-result"

    production.side_effect = execute
    monkeypatch.setattr("responses_api_agents.nooa_agent.bench_agent_adapter.invoke_bench_agent", production)
    with hooks.hooks_scope(observer):
        assert await nooa_observer.invoke_with_witness(agent, params) == "production-result"
        assert hooks.get_hooks() is observer
    production.assert_awaited_once_with(agent, params)
    records = [json.loads(line) for line in (tmp_path / "native_executions.jsonl").read_text().splitlines()]
    assert [(record["id"], record["exit_code"]) for record in records] == [("prefill-1", 0), ("failed-1", 1)]
    assert records == [json.loads(line) for line in (tmp_path / "other.jsonl").read_text().splitlines()]


def test_native_witness_requires_execution_observation_separate_from_model_history(tmp_path):
    probe = BorrowedSandboxProbe(SCENARIO["tool_success"], tmp_path)
    content = "<sys>PythonOutput(tool_call_id='prefill-1', stdout='actual feedback')</sys>"
    body = {
        "input": [
            {
                "type": "function_call",
                "call_id": "prefill-1",
                "name": "python_cell",
                "arguments": '{"code":"inspect(self)"}',
            },
            {"role": "user", "content": content},
        ]
    }
    probe._observe_tool_results(body)
    assert probe.native_tool_calls == []
    execution = {"id": "prefill-1", "arguments": {"code": "inspect(self)"}, "exit_code": 1, "executed": True}
    (tmp_path / "native_executions.jsonl").write_text(json.dumps(execution) + "\n")
    probe._observe_tool_results(body)
    probe.save()
    (observed,) = json.loads((tmp_path / "witness.json").read_text())["native_tool_calls"]
    assert observed == {**execution, "name": "python_cell", "outputs": [content], "result_seen": True}
