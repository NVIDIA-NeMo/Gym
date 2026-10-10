# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.runner import NOOARunFailure, NOOARunRequest
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response
from responses_api_agents.nooa_agent.tests.test_runner import make_runner


@pytest.mark.parametrize("recover", [False, True])
async def test_model_transport_failure_requires_a_successful_retry(recover: bool) -> None:
    runner, _ = make_runner()
    transport_error = ConnectionError("model unavailable")
    runner._server_client.post = AsyncMock(side_effect=[transport_error, FakeHTTPResponse(model_response())])

    async def invoke(agent, request):
        try:
            await agent.llm.acall([{"role": "user", "content": "first"}])
        except ConnectionError:
            pass
        if recover:
            await agent.llm.acall([{"role": "user", "content": "retry"}])
        return "fallback answer"

    runner._invocation_adapter = invoke
    request = NOOARunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
        model_url_path="/v1/responses",
    )
    if recover:
        result = await runner.run(request)
        assert result.return_value == "fallback answer"
        assert result.termination_reason is None
        assert runner._server_client.post.await_count == 2
    else:
        with pytest.raises(NOOARunFailure) as caught:
            await runner.run(request)
        assert caught.value.__cause__ is transport_error
        assert caught.value.result.termination_reason is None
        assert runner._server_client.post.await_count == 1
