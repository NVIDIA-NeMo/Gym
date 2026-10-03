# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.simple_agent.tests.test_app import _make_agent, _mock_response


def _model_reply(output, **kwargs):
    return _mock_response(
        {
            "id": "response",
            "created_at": 1.0,
            "model": "model",
            "object": "response",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "output": output,
            **kwargs,
        }
    )


@pytest.fixture
def answer():
    return {
        "id": "answer",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "done", "annotations": []}],
    }


async def test_string_input_is_normalized_without_mutating_request(answer):
    agent, client = _make_agent(True)
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    original = body.model_dump(exclude_unset=True)
    client.post = AsyncMock(return_value=_model_reply([answer]))

    response, trajectory, _, _ = await agent._create_episode(
        body, model_url_path="/v1/responses", collect_trajectory=True
    )

    sent = client.post.await_args.kwargs["json"]
    assert body.model_dump(exclude_unset=True) == original
    assert sent is not body
    assert sent.input == [NeMoGymEasyInputMessage(role="user", content="hello")]
    assert trajectory.invocations[0].conversation == [*sent.input, *response.output]


@pytest.mark.parametrize("outcome", ["completed", "incomplete", "bad_arguments", "invalid_response"])
async def test_tool_episode_preserves_nested_request_and_prior_steps(answer, outcome):
    agent, client = _make_agent(True)
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look up x"},
                    {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"},
                ],
            }
        ],
        tools=[
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up a value",
                "strict": False,
                "parameters": {"type": "object", "properties": {"q": {"type": "string", "enum": ["x", "y"]}}},
            }
        ],
    )
    original = deepcopy(body)
    original_wire = body.model_dump(exclude_unset=True)
    tool_call = {
        "type": "function_call",
        "id": "fc-1",
        "call_id": "call-1",
        "name": "lookup",
        "arguments": "{broken" if outcome == "bad_arguments" else '{"q":"x"}',
        "status": "completed",
    }
    final_reply = (
        _mock_response({"output": "invalid"})
        if outcome == "invalid_response"
        else _model_reply(
            [answer], incomplete_details={"reason": "max_output_tokens"} if outcome == "incomplete" else None
        )
    )
    replies = [_model_reply([tool_call])]
    if outcome != "bad_arguments":
        replies.append(_mock_response(content="tool result"))
    client.post = AsyncMock(side_effect=[*replies, final_reply])

    episode = agent._create_episode(body, model_url_path="/v1/responses", collect_trajectory=True)
    if outcome == "invalid_response":
        with pytest.raises(RuntimeError, match="Received an invalid response from model server"):
            await episode
    else:
        response, trajectory, _, _ = await episode
        assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
        assert trajectory.invocations[0].status == ("incomplete" if outcome == "incomplete" else "completed")
        assert trajectory.invocations[0].conversation == [*body.input, *response.output]
        assert trajectory.turns[0].question == body.input
        assert trajectory.tool_calls[0].status == ("failed" if outcome == "bad_arguments" else "completed")

    assert body == original
    assert body.model_dump(exclude_unset=True) == original_wire
    calls = client.post.await_args_list
    if outcome == "bad_arguments":
        assert [call.kwargs["server_name"] for call in calls] == ["model", "model"]
    else:
        assert [call.kwargs["server_name"] for call in calls] == ["model", "resources", "model"]
        assert calls[1].kwargs["json"] == {"q": "x"}
    first, second = (calls[index].kwargs["json"] for index in (0, -1))
    for sent in (first, second):
        assert sent is not body and sent.input is not body.input
        # Reuse read-only nested data, but never the mutable per-step container.
        assert sent.tools is body.tools and sent.input[0] is body.input[0]
        assert sent.model_dump(exclude_unset=True) | {"input": original_wire["input"]} == original_wire
    assert first.input == body.input
    assert second.input is not first.input
    assert second.input[0] == body.input[0]
    assert second.input[1].model_dump(exclude_unset=True) == tool_call
    assert second.input[2].call_id == "call-1"
    if outcome == "bad_arguments":
        assert "Invalid tool call arguments" in second.input[2].output
    else:
        assert second.input[2].output == "tool result"
    assert len(second.input) == 3
