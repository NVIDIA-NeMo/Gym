# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest

from nemo_gym.core.responses_api_types import TOKEN_METADATA_FIELDS, NeMoGymResponseOutputMessageForTraining
from nemo_gym.observability.token_metadata import strip_duplicate_token_metadata
from nemo_gym.rollout_observability import AgentInvocation, TrajectoryRecord


IDS = {"prompt_token_ids": [1, 2, 3], "generation_token_ids": [4, 5], "generation_log_probs": [-0.1, -0.2]}


def _item(item_id="msg-1", **extra):
    return {"id": item_id, "type": "message", "role": "assistant", "content": [{"text": "ok"}], **extra}


def _result():
    """A rollout whose single model output is repeated in the trajectory and the agent observations."""
    return {
        "response": {"output": [_item(**IDS)]},
        "ng_trajectory": {
            "invocations": [{"invocation_id": "root", "conversation": [{"role": "user"}, _item(**IDS)]}],
            "turns": [
                {"question": [{"role": "user"}, _item("msg-0", **IDS)], "answer": [_item(**IDS)]},
            ],
            "model_calls": [
                {
                    "model_call_id": "call-1",
                    "request": {"input": [{"role": "user"}]},
                    "response": {"id": "resp-1", "output": [_item(**IDS)]},
                }
            ],
        },
        "ng_agent_observations": {"records": [{"conversation": [_item(**IDS)]}]},
    }


def test_token_metadata_stays_on_the_response_output_and_leaves_the_copies():
    result = _result()

    removed = strip_duplicate_token_metadata(result)

    assert result["response"]["output"][0] == _item(**IDS)
    copies = [
        result["ng_trajectory"]["invocations"][0]["conversation"][1],
        result["ng_trajectory"]["turns"][0]["answer"][0],
        result["ng_trajectory"]["model_calls"][0]["response"]["output"][0],
        result["ng_agent_observations"]["records"][0]["conversation"][0],
    ]
    assert copies == [_item()] * 4
    assert removed == 4 * len(IDS)


def test_a_copy_still_joins_to_its_canonical_item_by_id_and_keeps_its_other_fields():
    result = _result()

    strip_duplicate_token_metadata(result)

    answer = result["ng_trajectory"]["turns"][0]["answer"][0]
    assert answer["id"] == result["response"]["output"][0]["id"]
    assert answer["content"] == [{"text": "ok"}]


def test_a_repeated_item_with_another_id_is_left_alone():
    result = _result()

    strip_duplicate_token_metadata(result)

    # The question holds an earlier output (msg-0) that response.output does not carry.
    assert result["ng_trajectory"]["turns"][0]["question"][1] == _item("msg-0", **IDS)


def test_a_copy_keeps_its_complete_metadata_when_the_canonical_twin_holds_only_part_of_it():
    result = _result()
    del result["response"]["output"][0]["generation_log_probs"]

    removed = strip_duplicate_token_metadata(result)

    # Removing only the arrays the twin has would leave an item that no longer validates, so none is removed.
    assert result["ng_trajectory"]["turns"][0]["answer"][0] == _item(**IDS)
    assert removed == 0


def test_routed_experts_is_kept_once_like_the_token_ids():
    result = _result()
    experts = [[[1, 2]], [[3, 4]]]
    result["response"]["output"][0]["routed_experts"] = experts
    result["ng_trajectory"]["turns"][0]["answer"][0]["routed_experts"] = experts

    strip_duplicate_token_metadata(result)

    assert result["response"]["output"][0]["routed_experts"] == experts
    assert "routed_experts" not in result["ng_trajectory"]["turns"][0]["answer"][0]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda r: r["response"].update(output=[_item()]), id="output carries no token metadata"),
        pytest.param(lambda r: r["response"].update(output=[]), id="empty output"),
        pytest.param(lambda r: r["response"]["output"][0].pop("id"), id="canonical item has no id"),
        pytest.param(lambda r: r.pop("response"), id="no response"),
        pytest.param(lambda r: r.update(response=None), id="response is null"),
    ],
)
def test_a_record_without_canonical_token_metadata_is_left_untouched(mutate):
    result = _result()
    mutate(result)
    before = copy.deepcopy(result)

    assert strip_duplicate_token_metadata(result) == 0
    assert result == before


def test_stripping_twice_changes_nothing_more():
    result = _result()
    strip_duplicate_token_metadata(result)
    once = copy.deepcopy(result)

    assert strip_duplicate_token_metadata(result) == 0
    assert result == once


def test_a_record_without_a_trajectory_or_observations_is_fine():
    result = {"response": {"output": [_item(**IDS)]}}

    assert strip_duplicate_token_metadata(result) == 0


class _NotIterated(list):
    def __iter__(self):
        raise AssertionError("visited every element of a list of numbers")


def test_long_lists_of_numbers_on_a_replayed_copy_are_not_walked():
    result = _result()
    replayed = result["ng_trajectory"]["turns"][0]["question"][1]
    replayed["generation_token_ids"] = _NotIterated(range(100))
    replayed["generation_log_probs"] = _NotIterated([-0.1] * 100)

    removed = strip_duplicate_token_metadata(result)

    # The replayed item has no canonical twin, so it keeps its metadata and costs nothing to pass over.
    assert len(replayed["generation_token_ids"]) == 100
    assert removed == 4 * len(IDS)


def _training_item(**extra):
    """An output item as a full dump of the validated model writes it, with the null routed_experts placeholder."""
    return NeMoGymResponseOutputMessageForTraining(
        id="msg_1",
        role="assistant",
        status="completed",
        type="message",
        content=[{"type": "output_text", "text": "ok", "annotations": []}],
        **IDS,
        **extra,
    ).model_dump(mode="json")


def _record_with_a_validated_conversation(response_item, conversation_item):
    invocation = AgentInvocation(invocation_id="root", conversation=[conversation_item]).model_dump(mode="json")
    trajectory = TrajectoryRecord(task_id="t", rollout_id="r", invocations=[invocation]).model_dump(mode="json")
    return {"response": {"output": [response_item]}, "ng_trajectory": trajectory}


def _stripped_trajectory_still_validates(result):
    TrajectoryRecord.model_validate(json.loads(json.dumps(result["ng_trajectory"])))


def test_a_stripped_copy_that_carries_a_null_placeholder_still_validates_as_a_trajectory():
    result = _record_with_a_validated_conversation(_training_item(), _training_item())
    assert result["ng_trajectory"]["invocations"][0]["conversation"][0]["routed_experts"] is None

    removed = strip_duplicate_token_metadata(result)

    copy_ = result["ng_trajectory"]["invocations"][0]["conversation"][0]
    assert not TOKEN_METADATA_FIELDS.intersection(copy_)
    assert removed == len(TOKEN_METADATA_FIELDS)
    assert {field: result["response"]["output"][0][field] for field in IDS} == IDS
    _stripped_trajectory_still_validates(result)


def test_real_routed_experts_on_both_copies_are_kept_once_and_the_copy_validates():
    experts = [[[1, 2]], [[3, 4]]]
    result = _record_with_a_validated_conversation(
        _training_item(routed_experts=experts), _training_item(routed_experts=experts)
    )

    strip_duplicate_token_metadata(result)

    assert result["response"]["output"][0]["routed_experts"] == experts
    assert not TOKEN_METADATA_FIELDS.intersection(result["ng_trajectory"]["invocations"][0]["conversation"][0])
    _stripped_trajectory_still_validates(result)


def test_routed_experts_only_on_the_copy_keep_its_whole_group_so_it_still_validates():
    experts = [[[1, 2]], [[3, 4]]]
    result = _record_with_a_validated_conversation(_training_item(), _training_item(routed_experts=experts))

    removed = strip_duplicate_token_metadata(result)

    copy_ = result["ng_trajectory"]["invocations"][0]["conversation"][0]
    assert removed == 0
    assert copy_["routed_experts"] == experts
    assert {field: copy_[field] for field in IDS} == IDS
    _stripped_trajectory_still_validates(result)
