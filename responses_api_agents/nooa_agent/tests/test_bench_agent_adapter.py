# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from responses_api_agents.nooa_agent.bench_agent_adapter import invoke_bench_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [None, RuntimeError("model failed"), asyncio.CancelledError()])
async def test_adapter_restores_task_cwd_and_always_drains_cleanup(tmp_path, monkeypatch, error):
    monkeypatch.chdir(tmp_path)
    value = object()
    agent = SimpleNamespace(
        shell=SimpleNamespace(cwd="/wrong", close=AsyncMock()),
        _install_python_tools=MagicMock(),
        _solve_task=AsyncMock(return_value=value, side_effect=error),
        aclose=AsyncMock(),
    )
    request = NeMoGymResponseCreateParamsNonStreaming(input="unchanged canonical prompt")
    if error is None:
        assert await invoke_bench_agent(agent, request) is value
    else:
        with pytest.raises(type(error)):
            await invoke_bench_agent(agent, request)
    agent.shell.close.assert_awaited_once()
    agent._install_python_tools.assert_called_once_with(str(tmp_path))
    agent._solve_task.assert_awaited_once_with("unchanged canonical prompt")
    agent.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 2])
async def test_real_benchagent_delegation_uses_gym_trace_and_shared_call_budget(
    tmp_path, monkeypatch, limit: int | None
):
    pytest.importorskip("nooa_bench.bench_agent")
    from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
    from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response

    monkeypatch.chdir(tmp_path)
    code = [
        "child = await self.delegate('bounded fixture')\nreturn_result(TaskResult(solution_description='parent', evidence=child.evidence, how_to_verify='fixture'))",
        "return_result(TaskResult(solution_description='child', evidence='delegated evidence', how_to_verify='fixture'))",
    ]
    calls = []

    async def post(**kwargs):
        calls.append(kwargs)
        output = NeMoGymResponseFunctionToolCall(
            id=f"f{len(calls)}",
            call_id=f"c{len(calls)}",
            name="python_cell",
            arguments=json.dumps({"code": code[len(calls) - 1]}),
        )
        return FakeHTTPResponse(model_response(output, response_id=f"r{len(calls)}"))

    client = SimpleNamespace(post=post)
    invocation = NOOAInvocationConfig(
        agent_class="nooa_bench.bench_agent:BenchAgent",
        invocation_adapter="responses_api_agents.nooa_agent.bench_agent_adapter:invoke_bench_agent",
    )
    runner = InProcessNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="policy",
        max_policy_calls=limit,
        context_window=262144,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="fixture", max_output_tokens=32768),
            model_url_path="/v1/responses",
        )
    )
    assert result.return_value.evidence == "delegated evidence"
    assert len(calls) == len(result.trajectory.turns) == 2
    assert len(result.trajectory.invocations) >= 2
    assert all(call["json"].max_output_tokens == 32768 for call in calls)
    assert result.termination_reason is None


@pytest.mark.asyncio
async def test_default_summarizer_calls_share_gym_budget_and_capture(tmp_path):
    module = pytest.importorskip("nooa_bench.bench_agent")
    from responses_api_agents.nooa_agent.gym_llm import GymResponsesLLM, PolicyCallBudgetExceeded, RolloutLLMState
    from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response

    state = RolloutLLMState(max_policy_calls=1)
    captured = []

    async def post(**kwargs):
        output = {
            "type": "message",
            "id": "summary",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": '{"value":"summary"}', "annotations": []}],
        }
        return FakeHTTPResponse(model_response(output, response_id="summary"))

    llm = GymResponsesLLM(
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        model_url_path="/v1/responses",
        state=state,
        cookies={},
        on_call=captured.append,
        context_window=262144,
    )
    agent = module.BenchAgent(llm=llm, working_dir=str(tmp_path))
    try:
        assert agent._max_delegation_depth == 4 and len(agent._summarizers) == 1
        summarizer = agent._summarizers[0]
        assert summarizer._summary_llm() is llm
        assert await summarizer.summarize("synthetic history", 100) == "summary"
        assert len(captured) == len(state.calls) == 1
        with pytest.raises(PolicyCallBudgetExceeded):
            await llm.acall([{"role": "user", "content": "second"}])
        assert len(captured) == 1
    finally:
        await agent.aclose()


def test_benchagent_runtime_profile_changes_fingerprint_without_replacing_default():
    import io
    import tarfile
    from pathlib import Path

    from responses_api_agents.nooa_agent.sandbox_runtime import _runtime_archive

    root = Path(__file__).resolve().parents[3]
    default, first = _runtime_archive(root)
    selected, second = _runtime_archive(
        root, requirements_path=Path("responses_api_agents/nooa_agent/runtime/benchagent-requirements.txt")
    )
    assert first != second

    def requirements(blob):
        with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
            return archive.extractfile("runtime-requirements.txt").read().decode()

    # Public source revisions, not credentials.
    assert "051472343211914222e24ce36d8752f4e86bbe43" in requirements(default)  # pragma: allowlist secret
    actual = requirements(selected)
    assert actual.count("564a34014a354f11009cf7dda44a81b039a04273") == 3  # pragma: allowlist secret
    assert "subdirectory=packages/nooa-cli" in actual and "subdirectory=packages/nooa-bench" in actual
    assert "051472343211914222e24ce36d8752f4e86bbe43" not in actual  # pragma: allowlist secret
