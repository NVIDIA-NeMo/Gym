# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from nemo_gym.observability.token_metadata import strip_duplicate_token_metadata


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


def test_metadata_is_never_removed_without_a_canonical_twin_that_holds_the_same_field():
    result = _result()
    del result["response"]["output"][0]["generation_log_probs"]

    strip_duplicate_token_metadata(result)

    answer = result["ng_trajectory"]["turns"][0]["answer"][0]
    assert answer == _item(generation_log_probs=IDS["generation_log_probs"])


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
