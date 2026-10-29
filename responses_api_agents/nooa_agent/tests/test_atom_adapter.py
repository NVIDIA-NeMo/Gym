# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom through the NOOA adapter: the config resolves, the adapter hands over the task, Atom runs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from responses_api_agents.nooa_agent.atom_adapter import invoke_atom
from responses_api_agents.nooa_agent.config import NOOAAgentConfig, NOOAInvocationConfig
from responses_api_agents.nooa_agent.invocation import validate_invocation
from responses_api_agents.nooa_agent.result import finalize_run_result
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


ROOT = Path(__file__).resolve().parents[3]
atom_gym = pytest.importorskip("nooa_atom.gym", reason="nooa-atom not installed in this venv")


def atom_config() -> NOOAAgentConfig:
    configs_dir = ROOT / "responses_api_agents/nooa_agent/configs"
    base = yaml.safe_load((configs_dir / "nooa_agent.yaml").read_text())["nooa_agent"]["responses_api_agents"][
        "nooa_agent"
    ]
    atom = yaml.safe_load((configs_dir / "nooa_atom_agent.yaml").read_text())["nooa_atom_agent"][
        "responses_api_agents"
    ]["nooa_agent"]
    merged = {**base, **atom, "nooa": {**base["nooa"], **atom["nooa"]}}
    return NOOAAgentConfig(name="nooa", host="127.0.0.1", port=9000, **merged)


def test_atom_config_resolves_the_harness_and_adapter():
    config = atom_config()
    agent_class, adapter = validate_invocation(config.nooa)
    assert agent_class is atom_gym.AtomHarness and adapter is invoke_atom
    assert config.nooa.init_kwargs == {"turn_method": "handle_batch"}


@pytest.mark.asyncio
async def test_adapter_hands_the_latest_user_message_to_atom():
    agent = SimpleNamespace(solve=AsyncMock(return_value="result"))
    request = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "preamble"}, {"role": "user", "content": "fix the bug"}]
    )
    assert await invoke_atom(agent, request) == "result"
    agent.solve.assert_awaited_once_with("fix the bug")


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_method", ["handle_batch", "handle_orchestrate"])
async def test_atom_runs_through_the_runner_with_a_scripted_model(tmp_path: Path, monkeypatch, turn_method: str):
    """The real Atom session, Gym's model client as its only model, one scripted python_cell that finishes."""
    workdir = tmp_path / "task"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    config = atom_config()
    invocation = NOOAInvocationConfig.model_validate(
        config.nooa.model_dump()
        | {"init_kwargs": {"turn_method": turn_method, "sessions_dir": str(tmp_path / "sessions")}}
    )
    code = (
        "open('hello.txt', 'w').write('hello gym')\n"
        "return_result(Done(explanation='wrote the file', result=TaskResult("
        "solution_description='created hello.txt', evidence='file exists', how_to_verify='cat hello.txt', "
        "report='done')))"
    )
    calls: list[dict] = []

    async def post(**kwargs):
        calls.append(kwargs)
        index = len(calls) - 1
        output = NeMoGymResponseFunctionToolCall(
            id=f"function-{index}", call_id=f"call-{index}", name="python_cell", arguments=json.dumps({"code": code})
        )
        return FakeHTTPResponse(model_response(output, response_id=f"response-{index}"))

    runner = InProcessNOOARunner(
        invocation=invocation,
        server_client=SimpleNamespace(post=post),
        model_server_name=config.model_server.name,
        max_policy_calls=3,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input="Create hello.txt containing 'hello gym', then finish."
            ),
            model_url_path="/v1/responses",
            task_id="atom-task",
            rollout_id="atom-rollout",
        )
    )
    assert result.termination_reason is None, result.termination_error
    assert result.return_value.solution_description == "created hello.txt"
    assert (workdir / "hello.txt").read_text() == "hello gym"  # Atom's cell ran in the task directory
    assert len(calls) == 1  # one model call finished the task
    # The harness carried Gym's hooks into Atom's session loop: the executed cell is observed.
    tools = {call.tool_name for call in result.trajectory.tool_calls}
    assert "python_cell" in tools, tools
    response, observations = finalize_run_result(result)
    assert response.status == "completed"
    # Two gaps the current projection records for Atom: the scripted model carries no training
    # metadata, and Atom always ends with return_result() rather than a model-authored message, so
    # the projection appends a synthetic one and flags the terminal output as non-trainable. The
    # second gap is the RL blocker: token-id capture cannot attribute the scored response to a
    # captured call through that synthetic message (declared terminal_response_id is the fix).
    assert sorted(gap.code for gap in observations.gaps) == [
        "non_trainable_terminal_output",
        "public_assistant_metadata_unavailable",
    ]


@pytest.mark.asyncio
async def test_adapter_unwraps_atoms_turn_failure_to_the_strategy_error():
    """The runner classifies CodeAct stops and model rejections by the error the strategy raised;
    Atom's session wraps it, so the adapter hands the original back."""
    from nooa.errors import GenerationError
    from nooa_atom.session.session import TurnFailedError

    original = GenerationError("LLM API error after 3 retries")

    async def failing(task: str):
        try:
            raise original
        except GenerationError as error:
            raise TurnFailedError(str(error)) from error

    agent = SimpleNamespace(solve=failing)
    with pytest.raises(GenerationError) as info:
        await invoke_atom(agent, NeMoGymResponseCreateParamsNonStreaming(input="task"))
    assert info.value is original

    async def other(task: str):
        raise RuntimeError("not a turn failure")

    with pytest.raises(RuntimeError):
        await invoke_atom(SimpleNamespace(solve=other), NeMoGymResponseCreateParamsNonStreaming(input="task"))


@pytest.mark.asyncio
async def test_atom_root_and_child_sessions_route_through_gym(tmp_path: Path, monkeypatch):
    """A root that delegates: the child is its own model-calling session, both share Gym's client and
    cookies, and the child's result flows back into the root's. Ported from Elisa Martinez Abad's #5178."""
    monkeypatch.chdir(tmp_path)
    config = atom_config()
    code = [
        "child = await self.delegate('offline child', 'return child evidence')\n"
        "return_result(Done(explanation='root', result=TaskResult(solution_description='root', "
        "evidence=child.result.evidence, how_to_verify='offline', report='native root and child')))",
        "return_result(Done(explanation='child', result=TaskResult(solution_description='child', "
        "evidence='native child evidence', how_to_verify='offline', report='child')))",
    ]
    calls: list[dict] = []

    async def post(**kwargs):
        calls.append(kwargs)
        assert len(calls) <= 2
        out = NeMoGymResponseFunctionToolCall(
            id=f"f{len(calls)}",
            call_id=f"c{len(calls)}",
            name="python_cell",
            arguments=json.dumps({"code": code[len(calls) - 1]}),
        )
        return FakeHTTPResponse(model_response(out, response_id=f"r{len(calls)}"))

    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig.model_validate(
            config.nooa.model_dump() | {"init_kwargs": {"sessions_dir": str(tmp_path / "sessions")}}
        ),
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=2,
        context_window=1048576,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input="offline native fixture", max_output_tokens=131072
            ),
            model_url_path="/fixture/v1/responses",
            model_cookies={"session": "fixture"},
        )
    )
    assert result.return_value.evidence == "native child evidence"
    assert result.termination_reason is None
    assert len(calls) == len(result.trajectory.turns) == 2
    assert len(result.trajectory.invocations) >= 2  # the child is observed as its own invocation
    assert all(call["server_name"] == "policy" and call["url_path"] == "/fixture/v1/responses" for call in calls)
    assert all(call["json"].max_output_tokens == 131072 for call in calls)
    assert all(call["cookies"]["session"] == "fixture" for call in calls)


@pytest.mark.parametrize(
    "config_path", ["benchmarks/terminal_bench_2_1/atom.yaml", "benchmarks/swebench/pro/atom.yaml"]
)
def test_atom_benchmark_configs_route_every_benchmark_dataset_to_atom(
    config_path: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolve each benchmark dataset's agent the way Gym does (``resolve_dataset_agent``: taskset route or the
    declaring resources server) and require the Atom harness with the Atom adapter.

    Guards against overriding an agent block that nothing references, which silently runs the default agent."""
    from omegaconf import DictConfig, OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser, resolve_dataset_agent

    # The SWE config requires a prepared offline runtime; the path only has to be absolute here.
    monkeypatch.setenv("NOOA_SWE_RUNTIME_WHEELHOUSE", str(tmp_path / "wheelhouse"))
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(ROOT / config_path)])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    resolved = []
    for instance_name, block in config.items():
        servers = block.get("resources_servers") if isinstance(block, DictConfig) else None
        if not isinstance(servers, DictConfig) or not servers:
            continue
        for dataset in next(iter(servers.values())).get("datasets") or []:
            if dataset.get("type") != "benchmark":
                continue
            agent_name = resolve_dataset_agent(
                config, str(instance_name), pin=dataset.get("agent"), taskset=dataset.get("taskset")
            )
            agent = NOOAAgentConfig(
                name=agent_name,
                host="localhost",
                port=8001,
                **OmegaConf.to_container(config[agent_name].responses_api_agents.nooa_agent, resolve=True),
            )
            agent_class, adapter = validate_invocation(agent.nooa)
            assert agent_class is atom_gym.AtomHarness, (
                f"{dataset.get('name')} -> {agent_name} runs {agent.nooa.agent_class}"
            )
            assert adapter is invoke_atom
            assert agent.nooa.execution_mode == "sandboxed"
            assert agent.nooa.init_kwargs == {"turn_method": "handle_batch"}
            resolved.append((str(dataset.get("name")), agent_name))
    assert resolved, f"{config_path} declares no benchmark dataset"
