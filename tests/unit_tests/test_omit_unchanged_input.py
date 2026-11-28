# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from nemo_gym.agents.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.responses_api_types import NeMoGymResponse, NeMoGymResponseFunctionToolCall
from nemo_gym.server_runtime.base_resources_server import BaseMultiRewardVerifyResponse, BaseRunRequest


class _Reply(BaseMultiRewardVerifyResponse):
    verifier_detail: dict


def _agent(*, change_input: str | None = None, error_response: bool = False):
    forwarded = []
    replies = []

    class Agent(SimpleResponsesAPIAgent):
        async def responses(self, body=...):
            raise NotImplementedError

        async def run(self, body: BaseRunRequest) -> _Reply:
            forwarded.append(body.model_dump())
            if error_response:
                return JSONResponse({"reward": 0, "mask_sample": True, "failure_reason": "judge failed"})
            if change_input == "replace":
                body.responses_create_params.input = "agent-rewritten prompt"
            elif change_input == "nested":
                body.responses_create_params.input[0].content[0]["text"] = "agent-rewritten prompt"
            result = _Reply(
                responses_create_params=body.responses_create_params,
                response=NeMoGymResponse.model_construct(
                    id="response-1",
                    output=[
                        NeMoGymResponseFunctionToolCall(
                            id="call-1",
                            type="function_call",
                            call_id="tool-1",
                            name="lookup",
                            arguments='{"key":"value"}',
                        )
                    ],
                ),
                reward=0.5,
                reward_components={"answer": 0.5},
                mask_sample=True,
                failure_reason="kept diagnostic",
                verifier_detail={"tool_result": "retained"},
            )
            replies.append(result)
            return result

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    agent = Agent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="test"),
        server_client=client,
    )
    return agent.setup_webserver(), forwarded, replies


_INPUT = [
    {
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Describe the image"},
            {"type": "input_image", "image_url": "data:image/png;base64,retained", "detail": "auto"},
        ],
    }
]


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("request_input", [_INPUT, "string prompt", [], ""])
@pytest.mark.parametrize("metadata", [None, {}, {"caller_input_annotation": "user-owned"}])
def test_run_omits_only_opted_in_unchanged_input(enabled, request_input, metadata):
    app, forwarded, replies = _agent()
    payload = {"responses_create_params": {"input": deepcopy(request_input), "metadata": metadata}}
    if enabled is not None:
        payload["omit_unchanged_input"] = enabled
    before = deepcopy(payload)
    with TestClient(app) as client:
        response = client.post("/run", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    expected = replies[0].model_dump(mode="json")
    assert "omit_unchanged_input" not in forwarded[0]
    assert "omit_unchanged_input" not in result
    assert payload == before
    if enabled and request_input:
        assert result.pop("input_omitted") is True
        assert result["responses_create_params"]["input"] == []
        result["responses_create_params"]["input"] = expected["responses_create_params"]["input"]
    assert result == expected
    # Neither the agent-owned response nor the caller's request was emptied.
    assert (
        replies[0].responses_create_params.model_dump(mode="json")["input"]
        == expected["responses_create_params"]["input"]
    )


@pytest.mark.parametrize("change_input", ["replace", "nested"])
def test_run_keeps_agent_changed_input(change_input):
    app, _, replies = _agent(change_input=change_input)
    with TestClient(app) as client:
        response = client.post(
            "/run", json={"omit_unchanged_input": True, "responses_create_params": {"input": _INPUT}}
        )
    assert response.status_code == 200, response.text
    assert response.json() == replies[0].model_dump(mode="json")
    assert "input_omitted" not in response.json()


def test_run_keeps_error_response():
    app, _, _ = _agent(error_response=True)
    with TestClient(app) as client:
        response = client.post(
            "/run", json={"omit_unchanged_input": True, "responses_create_params": {"input": "prompt"}}
        )
    assert response.json() == {"reward": 0, "mask_sample": True, "failure_reason": "judge failed"}


def test_run_rejects_non_boolean_opt_in():
    app, _, _ = _agent()
    with TestClient(app) as client:
        response = client.post(
            "/run", json={"omit_unchanged_input": "true", "responses_create_params": {"input": "prompt"}}
        )
    assert response.status_code == 422
