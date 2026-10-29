# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise NOOA's typed launch and borrowed-sandbox conformance fixtures."""

import ast
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from scripts.harness_conformance.episode import _config
from scripts.harness_conformance.nooa import BorrowedSandboxProbe, observer_module_name
from scripts.harness_conformance.provider import ANSWER, Probe
from scripts.harness_conformance.scenarios import SCENARIOS

from nemo_gym.base_resources_server import ResourcesSeedSessionResponse
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import create_provider


SCENARIO = {scenario.name: scenario for scenario in SCENARIOS}
PYTHON_CELL = {
    "type": "function",
    "name": "python_cell",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}


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
    monkeypatch.setattr(nooa_observer, "invoke_bench_agent", production)
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
