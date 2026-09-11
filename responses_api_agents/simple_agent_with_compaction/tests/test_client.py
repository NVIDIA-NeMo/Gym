# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.config_types import ModelServerRef
from nemo_gym.context_management import ContextGuardRejected, ContextHistoryConfig, ContextManagedResponsesClient
from nemo_gym.context_management.client import PARENT_HEADER
from nemo_gym.context_management.result import LogicalCCResult, capture_rollout_id
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.staging.rebuild import verify_and_linearize
from nemo_gym.token_id_capture.staging.records import RolloutManifest, RolloutReceipt


@pytest.mark.parametrize("suffix", ["", "_a" + "1" * 32])
def test_capture_scope_preserves_framework_attempt(suffix):
    owner = "group_g0" + suffix
    assert capture_rollout_id(owner, 2) == owner + "_s2"


@pytest.mark.parametrize("suffix", ["_a", "_a123", "_a" + "z" * 32, "_s0", "/other"])
def test_capture_scope_rejects_malformed_attempt(suffix):
    with pytest.raises(ValueError, match="framework logical owner"):
        capture_rollout_id("group_g0" + suffix, 0)


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
    assert [json.loads(call["headers"][PARENT_HEADER]) for call in full_calls] == [None, first.id]
    assert all(call["_retry"] is False for call in full_calls)
    assert all("/group_g0_s0/training-token-capture/v1/responses" in call["url_path"] for call in full_calls)


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
    assert len(client.finish(third).segments) == 3
    assert "seed.png" not in calls[-1]["json"].model_dump_json()
    assert "newest.png" in calls[-1]["json"].model_dump_json()


async def test_empty_initial_message_does_not_drop_final_output():
    request = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "type": "message", "content": []}])
    client, _, _, _ = make_client(request=request)
    final = await client.create()
    client.finish(final)
    assert len(client.output_items) == 1
    assert client.output_items[0]["id"] == "message-1"


async def test_full_history_empty_initial_event_is_not_duplicated_or_rejected_on_turn_two():
    initial = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "type": "message", "content": []}])
    client, _, calls, _ = make_client(request=initial)
    first = await client.create(initial)
    assert len(client.history.events) == 2  # The original empty item, then the selected response.
    second_body = NeMoGymResponseCreateParamsNonStreaming.model_validate(
        initial.model_dump() | {"input": [*initial.input, *first.output]}
    )
    second = await client.create(second_body)
    assert len(client.history.events) == 3
    assert len(client.finish(second).segments) == 1
    assert [item["id"] for item in client.output_items] == ["message-1", "message-2"]
    assert json.loads(calls[1]["headers"][PARENT_HEADER]) == first.id


@pytest.mark.parametrize("has_seed", [False, True])
async def test_only_current_opening_observation_is_pending_until_first_action(has_seed):
    config = ContextHistoryConfig.model_validate(
        {
            "enabled": True,
            "policy": {
                "type": "recency",
                "config": {"images": {"enabled": True, "protect_initial_context": False, "keep_last_groups": 0}},
            },
            "guards": {"max_active_images": 1},
        }
    )
    initial = NeMoGymResponseCreateParamsNonStreaming(input=[observation("initial")])
    client, _, calls, _ = make_client(config=config, request=initial, seed=[observation("seed")] if has_seed else [])
    await client.create()
    first_prompt = calls[0]["json"].model_dump_json()
    assert ("initial.png" in first_prompt) is not has_seed
    assert ("seed.png" in first_prompt) is has_seed
    final = await client.create()
    assert "initial.png" not in calls[1]["json"].model_dump_json()
    assert "seed.png" not in calls[1]["json"].model_dump_json()
    result = client.finish(final)
    assert [len(segment.media_occurrence_refs) for segment in result.segments] == [1, 0]


async def test_caller_config_mutation_cannot_change_an_active_rollout_policy():
    config = ContextHistoryConfig.model_validate(
        {
            "enabled": True,
            "policy": {"type": "recency", "config": {"images": {"enabled": True, "keep_last_groups": 1}}},
        }
    )
    client, _, calls, _ = make_client(config=config, seed=[observation("seed")])
    await client.create()
    config.policy.config.images.keep_last_groups = 0
    final = await client.create()
    assert client.config.policy.config.images.keep_last_groups == 1
    assert "seed.png" in calls[1]["json"].model_dump_json()
    assert len(client.finish(final).segments) == 1


@pytest.mark.parametrize("change", ["source", "temperature", "instructions", "metadata", "tools"])
async def test_full_history_adapter_rejects_unexplained_mutation_before_generation(change):
    client, _, calls, initial = make_client()
    first = await client.create()
    payload = initial.model_dump() | {
        "input": [
            {"role": "user", "content": "task", "type": "message"},
            *first.output,
        ]
    }
    if change == "source":
        payload["input"][0]["content"] = "silently rewritten"
    else:
        payload[change] = {
            "temperature": 0.7,
            "instructions": "changed",
            "metadata": {"extra_body": '{"add_generation_prompt":false}'},
            "tools": [{"type": "function", "name": "new_tool", "parameters": {"type": "object"}, "strict": False}],
        }[change]
    with pytest.raises(ValueError, match="append-only|unchanged"):
        await client.create(NeMoGymResponseCreateParamsNonStreaming.model_validate(payload))
    assert len(calls) == 1


@pytest.mark.parametrize(
    "options",
    [
        {"metadata": {"extra_body": '{"truncate_prompt_tokens":128}'}},
        {"previous_response_id": "hidden-history"},
        {"conversation": "hidden-history"},
        {"truncation": "auto"},
    ],
)
def test_client_rejects_engine_owned_history(options):
    with pytest.raises(ValueError, match="compaction|semantic history"):
        make_client(request=NeMoGymResponseCreateParamsNonStreaming(input="task", **options))


@pytest.mark.parametrize("chunk_size", [1, 2, 5])
async def test_immediately_preceding_reasoning_rewrite_starts_a_root_at_chunk_boundary(chunk_size):
    config = ContextHistoryConfig.model_validate(
        {
            "enabled": True,
            "policy": {"type": "recency", "config": {"reasoning": {"enabled": True, "keep_last_blocks": 0}}},
            "schedule": {"type": "turn_chunked_recency", "actions_per_chunk": chunk_size},
        }
    )
    responses = [answer(i, output=[reasoning(i), *answer(i).output]) for i in range(1, chunk_size + 2)]
    client, _, calls, _ = make_client(config=config, responses=responses)
    for _ in range(chunk_size + 1):
        last = await client.create()
    result = client.finish(last)
    assert [len(segment.selected_actions) for segment in result.segments] == [chunk_size, 1]
    assert calls[-1]["url_path"].endswith("/group_g0_s1/training-token-capture/v1/responses")
    assert json.loads(calls[-1]["headers"][PARENT_HEADER]) is None
    assert all(item["type"] != "reasoning" for item in calls[-1]["json"].model_dump()["input"])
    if chunk_size > 1:
        assert json.loads(calls[1]["headers"][PARENT_HEADER]) == "response-1"


async def test_pending_images_early_chunk_close_and_image_only_guard_do_not_probe():
    config = ContextHistoryConfig.model_validate(
        {
            "enabled": True,
            "policy": {
                "type": "recency",
                "config": {"images": {"enabled": True, "keep_last_groups": 1, "protect_initial_context": False}},
            },
            "schedule": {"type": "turn_chunked_recency", "actions_per_chunk": 5},
            "guards": {"max_active_images": 1},
        }
    )
    client, _, calls, _ = make_client(config=config, seed=[observation("seed")])
    await client.create()
    client.append_observation([observation("pending")])
    last = await client.create()
    result = client.finish(last)
    assert len(calls) == 2  # no token measurement for image-only guards
    assert len(result.segments) == 2
    assert "pending.png" in calls[-1]["json"].model_dump_json()
    assert "seed.png" not in calls[-1]["json"].model_dump_json()
    assert client.controller.chunk_records[0].early_close_reason == "guard:active_images"
    assert len(result.media_assets) == 2


async def test_reserved_budget_probe_remeasures_after_early_close_and_preserves_pending():
    config = ContextHistoryConfig.model_validate(
        {
            "enabled": True,
            "policy": {
                "type": "recency",
                "config": {"images": {"enabled": True, "keep_last_groups": 1, "protect_initial_context": False}},
            },
            "schedule": {"type": "turn_chunked_recency", "actions_per_chunk": 5},
            "guards": {"max_total_tokens": 20, "reserved_generation_tokens": 8},
        }
    )
    client, transport, calls, _ = make_client(config=config, seed=[observation("seed")])
    original = transport.post.side_effect

    async def measured(**kwargs):
        response = await original(**kwargs)
        if kwargs["url_path"].endswith("/measure"):
            image_count = sum(str(item).count("input_image") for item in kwargs["json"].input)
            return http_response({"prompt_token_count": 10 * image_count})
        return response

    transport.post.side_effect = measured
    await client.create()
    client.append_observation([observation("pending")])
    final = await client.create()
    assert [call["url_path"].split("/")[-1] for call in calls] == [
        "measure",
        "responses",
        "measure",
        "measure",
        "responses",
    ]
    assert [json.loads(call["headers"][PARENT_HEADER]) for call in calls] == [None, None, "response-1", None, None]
    assert len(client.finish(final).segments) == 2


async def test_pending_images_that_cannot_fit_fail_before_generation():
    config = ContextHistoryConfig(guards={"max_active_images": 0})
    client, _, calls, _ = make_client(config=config, seed=[observation("pending")])
    with pytest.raises(ContextGuardRejected):
        await client.create()
    assert calls == []
    with pytest.raises(RuntimeError, match="closed"):
        await client.create()


async def test_only_definite_responses_can_be_resampled_and_rejected_sibling_never_becomes_parent():
    client, _, calls, _ = make_client(config=ContextHistoryConfig(max_response_retries=1))
    accepted = await client.create(select_response=lambda response: response.id != "response-1")
    final = await client.create()
    result = client.finish(final)
    assert accepted.id == "response-2"
    assert [json.loads(call["headers"][PARENT_HEADER]) for call in calls] == [None, None, "response-2"]
    assert [item.response_id for item in result.segments[0].selected_actions] == ["response-2", "response-3"]
    assert "answer 1" not in json.dumps(client.output_items)


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


async def test_concurrent_create_and_finite_call_limits():
    import asyncio

    client, transport, _, _ = make_client(config=ContextHistoryConfig(max_model_calls=1))
    entered, release = asyncio.Event(), asyncio.Event()
    original = transport.post.side_effect

    async def blocked(**kwargs):
        entered.set()
        await release.wait()
        return await original(**kwargs)

    transport.post.side_effect = blocked
    pending = asyncio.create_task(client.create())
    await entered.wait()
    with pytest.raises(RuntimeError, match="pending"):
        await client.create()
    release.set()
    await pending
    with pytest.raises(RuntimeError, match="model-call limit"):
        await client.create()
    assert transport.post.await_count == 1


async def test_result_is_compact_token_free_and_checks_terminal_identity_and_shape():
    client, _, _, _ = make_client(seed=[observation("same"), observation("same")])
    first = await client.create()
    for _ in range(39):
        last = await client.create()
    with pytest.raises(ValueError, match="last selected"):
        client.finish(first)
    result = client.finish(last, outcome="max_steps")
    assert len(result.media_assets) == 1
    assert len(result.segments[0].media_occurrence_refs) == 2
    assert [len(action.new_media_occurrence_refs) for action in result.segments[0].selected_actions] == [2] + [0] * 39
    assert len(client.history.events) == 43
    assert len(client.controller.boundary_events) <= 1
    encoded = result.model_dump_json()
    assert len(encoded) < 20000
    for forbidden in ("prompt_token_ids", "generation_token_ids", "prepared", "lineage", "agent_input"):
        assert forbidden not in encoded
    payload = result.model_dump(mode="json")
    payload["segments"][0]["selected_actions"].append(payload["segments"][0]["selected_actions"][0])
    with pytest.raises(ValidationError, match="unique"):
        LogicalCCResult.model_validate(payload)


async def test_media_deltas_keep_repeated_occurrences_and_reject_reordering():
    client, _, _, _ = make_client(seed=[observation("first")])
    await client.create()
    client.append_observation([observation("second"), observation("first")])
    final = await client.create()
    result = client.finish(final)
    segment = result.segments[0]
    initial, added = [action.new_media_occurrence_refs for action in segment.selected_actions]
    assert len(initial) == 1 and len(added) == 2
    assert initial[0] == added[1] != added[0]
    assert segment.media_occurrence_refs == initial + added
    payload = result.model_dump(mode="json")
    payload["segments"][0]["selected_actions"][1]["new_media_occurrence_refs"].reverse()
    with pytest.raises(ValidationError, match="media deltas"):
        LogicalCCResult.model_validate(payload)


async def test_shared_client_uses_real_capture_ledger_and_worker_delta_chain(monkeypatch, tmp_path):
    from responses_api_models.vllm_model.tests.test_segment_capture import make_capture_harness

    harness = make_capture_harness(monkeypatch, tmp_path)
    client, transport, _, _ = make_client(config=ContextHistoryConfig(max_response_retries=1))

    async def bridge(**kwargs):
        assert kwargs["_retry"] is False
        result = harness.client.post(
            kwargs["url_path"], json=kwargs["json"].model_dump(mode="json"), headers=kwargs["headers"]
        )
        assert result.status_code == 200, result.text
        return http_response(result.json())

    transport.post.side_effect = bridge
    rejected = []

    def select(response):
        if not rejected:
            rejected.append(response.id)
            return False
        return True

    await client.create(select_response=select)
    client.append_observation([{"role": "user", "type": "message", "content": "continue"}])
    final = await client.create()
    result = client.finish(final)
    manifest = RolloutManifest.model_validate(await harness.ledger.manifest("group_g0_s0"))
    segment = result.segments[0]
    records = [harness.sink.records[item.staging_key] for item in manifest.records]
    terminal = next(record for record in manifest.records if record.response_id == final.id)
    receipt = RolloutReceipt(
        rollout_id=segment.capture_rollout_id,
        manifest=manifest.records,
        terminal_model_call_id=terminal.model_call_id,
        terminal_selection="declared",
    )
    rebuilt = verify_and_linearize(receipt, records)
    by_call = {record.model_call_id: record for record in manifest.records}
    assert [by_call[call_id].response_id for call_id in rebuilt.model_call_ids] == [
        action.response_id for action in segment.selected_actions
    ]
    assert rejected[0] not in [by_call[call_id].response_id for call_id in rebuilt.model_call_ids]
    assert len(manifest.records) == 3
    assert [record.parent_call_id for record in manifest.records[:2]] == [None, None]
    assert [call[0].mode for call in harness.worker_calls] == ["text", "text", "token_in"]
    harness.client.close()
    await harness.ledger.close()


async def test_image_policy_rewrite_creates_a_real_parentless_capture_segment(monkeypatch, tmp_path):
    from responses_api_models.vllm_model.tests.test_segment_capture import make_capture_harness

    harness = make_capture_harness(monkeypatch, tmp_path)
    config = ContextHistoryConfig.model_validate(
        {
            "enabled": True,
            "policy": {"type": "recency", "config": {"images": {"enabled": True, "keep_last_groups": 1}}},
        }
    )
    client, transport, _, _ = make_client(config=config, seed=[observation("old")])

    async def bridge(**kwargs):
        result = harness.client.post(
            kwargs["url_path"], json=kwargs["json"].model_dump(mode="json"), headers=kwargs["headers"]
        )
        assert result.status_code == 200, result.text
        return http_response(result.json())

    transport.post.side_effect = bridge
    first = await client.create()
    client.append_observation([observation("pending")])
    final = await client.create()
    result = client.finish(final)
    assert [segment.capture_rollout_id for segment in result.segments] == ["group_g0_s0", "group_g0_s1"]
    for segment, selected in zip(result.segments, [first, final], strict=True):
        manifest = RolloutManifest.model_validate(await harness.ledger.manifest(segment.capture_rollout_id))
        assert manifest.failures == []
        assert len(manifest.records) == 1 and manifest.records[0].parent_call_id is None
        record = manifest.records[0]
        row = verify_and_linearize(
            RolloutReceipt(
                rollout_id=segment.capture_rollout_id,
                manifest=manifest.records,
                terminal_model_call_id=record.model_call_id,
                terminal_selection="declared",
            ),
            [harness.sink.records[record.staging_key]],
        )
        assert record.response_id == selected.id
        assert row.token_mask == [0.0, 0.0, 1.0]
    assert [call[0].mode for call in harness.worker_calls] == ["text", "text"]
    harness.client.close()
    await harness.ledger.close()
