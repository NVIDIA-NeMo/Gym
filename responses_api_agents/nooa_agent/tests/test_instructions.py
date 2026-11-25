# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark instructions reach Atom and wrappers on every model call."""

import json
from types import SimpleNamespace

import pytest

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from responses_api_agents.nooa_agent.atom_adapter import invoke_atom
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
from responses_api_agents.nooa_agent.task_agent import _task_instructions
from responses_api_agents.nooa_agent.tests.test_atom_adapter import atom_config
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


async def invoke_wrapper(agent, request):
    return await invoke_atom(agent, request)


@pytest.mark.parametrize("wrapped", [False, True])
async def test_first_and_subsequent_atom_requests_keep_instructions(tmp_path, monkeypatch, wrapped):
    monkeypatch.chdir(tmp_path)
    bodies = []
    code = [
        "print('first turn')",
        "return_result(Done(explanation='done', result=TaskResult(solution_description='done', evidence='done', "
        "how_to_verify='done', report='done')))",
    ]

    async def post(**kwargs):
        index = len(bodies)
        bodies.append(kwargs["json"])
        assert index < 2
        return FakeHTTPResponse(
            model_response(
                NeMoGymResponseFunctionToolCall(
                    id=f"item-{index}",
                    call_id=f"call-{index}",
                    name="python_cell",
                    arguments=json.dumps({"code": code[index]}),
                ),
                response_id=f"response-{index}",
            )
        )

    config = atom_config()
    invocation = config.nooa.model_dump()
    invocation["init_kwargs"] = {"turn_method": "handle_batch", "sessions_dir": str(tmp_path / "sessions")}
    if wrapped:
        invocation["invocation_adapter"] = f"{__name__}:invoke_wrapper"
    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig.model_validate(invocation),
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=2,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                instructions="benchmark policy",
                input=[
                    {"role": "system", "content": "apps are in /opt/apps"},
                    {"role": "developer", "content": [{"type": "input_text", "text": "save the document"}]},
                    {"role": "user", "content": "complete the task"},
                ],
            ),
            model_url_path="/v1/responses",
        )
    )
    assert result.termination_reason is None, result.termination_error
    assert len(bodies) == 2
    for body in bodies:
        assert body.instructions.startswith("benchmark policy\n\napps are in /opt/apps\n\nsave the document\n\n")
        assert "complete the task" in str(body.input)
        assert "apps are in /opt/apps" not in str(body.input)
    assert bodies[0].instructions == bodies[1].instructions
    # Real Atom's cached renderer drops the old live suffix; the bridge restores
    # it before the unedited first reply, then keeps the fresh context at the end.
    first = bodies[0].model_dump(mode="json")["input"]
    second = bodies[1].model_dump(mode="json")["input"]
    assert second[: len(first)] == first


def test_non_text_system_instructions_are_explicitly_rejected():
    request = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "system",
                "content": [{"type": "input_image", "image_url": "https://example.test/a.png", "detail": "auto"}],
            },
            {"role": "user", "content": "task"},
        ]
    )
    with pytest.raises(ValueError, match="text only"):
        _task_instructions(request)
