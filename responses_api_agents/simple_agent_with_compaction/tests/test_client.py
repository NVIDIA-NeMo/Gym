# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef
from nemo_gym.context_management import ContextHistoryConfig, ContextManagedResponsesClient
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.sink import CAPTURE_PARENT_HEADER


def answer(index, *, output=None, incomplete=None):
    return NeMoGymResponse(
        id=f"response-{index}",
        created_at=0,
        error=None,
        incomplete_details=incomplete,
        instructions=None,
        metadata=None,
        model="test",
        object="response",
        parallel_tool_calls=True,
        temperature=None,
        tool_choice="auto",
        tools=[],
        top_p=None,
        output=output
        if output is not None
        else [
            {
                "id": f"message-{index}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": f"answer {index}", "annotations": []}],
            }
        ],
    )


def http_response(payload, *, content=None, cookies=None, status=200):
    response = MagicMock(ok=status < 400, status=status, cookies=cookies or {})
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    response.content.read = AsyncMock(return_value=(content if content is not None else json.dumps(payload)).encode())
    return response


def observation(image, text="observation"):
    return {
        "role": "user",
        "type": "message",
        "content": [
            {"type": "input_image", "image_url": f"https://example.invalid/{image}.png", "detail": "auto"},
            {"type": "input_text", "text": text},
        ],
    }


def reasoning(index):
    return {
        "id": f"reason-{index}",
        "type": "reasoning",
        "summary": [],
        "content": [{"type": "reasoning_text", "text": f"private {index}"}],
    }


def make_client(*, config=None, request=None, responses=None, seed=()):
    transport = MagicMock(spec=ServerClient)
    queue = list(responses or [answer(index) for index in range(1, 101)])
    calls = []

    async def post(**kwargs):
        calls.append(deepcopy(kwargs))
        if kwargs["url_path"].endswith("/measure"):
            return http_response({"prompt_token_count": 10})
        return http_response(queue.pop(0).model_dump(mode="json"), cookies={"model_session": "kept"})

    transport.post = AsyncMock(side_effect=post)
    initial = request or NeMoGymResponseCreateParamsNonStreaming(input="task", max_output_tokens=8)
    client = ContextManagedResponsesClient(
        server_client=transport,
        model_server=ModelServerRef(name="policy", type="responses_api_models"),
        logical_rollout_id="group_g0",
        config=config or ContextHistoryConfig(),
        initial_request=initial,
        seed_observations=seed,
    )
    return client, transport, calls, initial


async def test_identity_and_full_history_adapter_share_selected_parent_state():
    explicit, _, explicit_calls, initial = make_client()
    full, _, full_calls, _ = make_client(request=initial)
    first = await explicit.create()
    assert (await full.create(initial)).id == first.id
    observation_item = {"type": "function_call_output", "call_id": "call-1", "output": "observed"}
    explicit.append_observation([observation_item])
    second = await explicit.create()
    full_body = NeMoGymResponseCreateParamsNonStreaming(
        **(
            initial.model_dump()
            | {
                "input": [
                    {"role": "user", "content": "task", "type": "message"},
                    *first.output,
                    observation_item,
                ]
            }
        )
    )
    second_full = await full.create(full_body)
    assert explicit.finish(second) == full.finish(second_full)
    assert [call["json"].model_dump() for call in explicit_calls] == [call["json"].model_dump() for call in full_calls]
    assert [json.loads(call["headers"][CAPTURE_PARENT_HEADER]) for call in full_calls] == [None, first.id]
    assert all(call["_retry"] is False for call in full_calls)
    assert all("/group_g0/training-token-capture/v1/responses" in call["url_path"] for call in full_calls)


async def test_full_history_adapter_accepts_original_source_after_policy_compaction():
    config = ContextHistoryConfig.model_validate(
        {
            "enabled": True,
            "policy": {"type": "recency", "config": {"images": {"enabled": True, "keep_last_groups": 1}}},
        }
    )
    client, _, calls, initial = make_client(config=config, seed=[observation("seed")])
    first = await client.create()
    full_history = [{"role": "user", "type": "message", "content": "task"}, observation("seed"), *first.output]
    full_history.append(observation("next"))
    second = await client.create(
        NeMoGymResponseCreateParamsNonStreaming.model_validate(initial.model_dump() | {"input": full_history})
    )
    full_history.extend([*second.output, observation("newest")])
    third = await client.create(
        NeMoGymResponseCreateParamsNonStreaming.model_validate(initial.model_dump() | {"input": full_history})
    )
    assert len(client.finish(third).selected_actions) == 3
    assert "seed.png" not in calls[-1]["json"].model_dump_json()
    assert "newest.png" in calls[-1]["json"].model_dump_json()


async def test_empty_initial_message_does_not_drop_final_output():
    request = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "type": "message", "content": []}])
    client, _, _, _ = make_client(request=request)
    final = await client.create()
    client.finish(final)
    assert len(client.output_items) == 1
    assert client.output_items[0]["id"] == "message-1"


async def test_retained_image_uses_original_source_and_result_has_no_segments():
    client, _, calls, _ = make_client(seed=[observation("A")])
    first = await client.create()
    client.append_observation([observation("B")])
    second = await client.create()
    assert "A.png" in calls[1]["json"].model_dump_json()
    assert "B.png" in calls[1]["json"].model_dump_json()
    result = client.finish(second).model_dump()
    assert set(result) == {"logical_rollout_id", "selected_actions", "outcome"}
    assert [item["response_id"] for item in result["selected_actions"]] == [first.id, second.id]


async def test_only_definite_responses_are_resampled_from_the_accepted_parent():
    client, _, calls, _ = make_client(config=ContextHistoryConfig(max_response_retries=1))
    first = await client.create(select_response=lambda response: response.id != "response-1")
    client.append_observation([{"role": "user", "content": "next"}])
    final = await client.create(select_response=lambda response: response.id != "response-3")
    result = client.finish(final)
    assert first.id == "response-2"
    assert [json.loads(call["headers"][CAPTURE_PARENT_HEADER]) for call in calls] == [None, None, first.id, first.id]
    assert [item.response_id for item in result.selected_actions] == ["response-2", "response-4"]
    assert "answer 1" not in json.dumps(client.output_items)
    assert "answer 3" not in json.dumps(client.output_items)
    assert calls[0]["json"] == calls[1]["json"]
    assert calls[2]["json"] == calls[3]["json"]
    assert all(call["_retry"] is False for call in calls)


@pytest.mark.parametrize("failure", ["transport", "read"])
async def test_ambiguous_failure_is_not_retried_and_cannot_finish(failure):
    client, transport, _, _ = make_client(config=ContextHistoryConfig(max_response_retries=10))
    if failure == "transport":
        transport.post.side_effect = ConnectionError("ack lost")
    else:
        response = http_response(answer(1).model_dump(mode="json"))
        response.read.side_effect = ConnectionError("ack lost")
        transport.post.side_effect = None
        transport.post.return_value = response
    with pytest.raises(ConnectionError):
        await client.create()
    assert transport.post.await_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        client.finish(answer(1))


@pytest.mark.parametrize("limit", ["retries", "calls"])
async def test_rejected_responses_obey_both_finite_budgets(limit):
    config = ContextHistoryConfig(max_response_retries=1 if limit == "retries" else 10, max_model_calls=2)
    client, transport, _, _ = make_client(config=config)

    async def reject(response):
        return False

    with pytest.raises(RuntimeError, match="resampling budget" if limit == "retries" else "model-call limit"):
        await client.create(select_response=reject)
    assert transport.post.await_count == 2
    assert client.output_items == []
    with pytest.raises(RuntimeError, match="closed"):
        await client.create()
