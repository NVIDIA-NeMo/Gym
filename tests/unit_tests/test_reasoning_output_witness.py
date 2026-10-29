# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

import pytest

from nemo_gym.token_id_capture.lineage import RolloutLineage
from nemo_gym.token_id_capture.reasoning_witness import make_witness, select_witness


def scenario():
    request = [{"role": "user", "content": "Solve this synthetic task."}]
    output = [{"type": "message", "role": "assistant", "content": "Working."}]
    second = request + output + [{"role": "user", "content": "Continue."}]
    reasoning = [{"type": "reasoning", "id": "rs_unique", "summary": [{"type": "summary_text", "text": "Plan."}]}]
    third = second + reasoning + [{"role": "user", "content": "Continue again."}]
    lineage = RolloutLineage()
    lineage.record("first", request + output, [1, 2], "digest-one", context_len=len(request))
    lineage.record("second", second + reasoning, [1, 2, 3], "digest-two", context_len=len(second))
    return lineage, make_witness("second", second, reasoning), third


def test_reproduces_real_ambiguity_and_exact_witness_resolves():
    lineage, witness, request = scenario()
    saved = lineage.by_call_id["second"].reasoning_witness
    lineage.by_call_id["second"].reasoning_witness = None
    status, _, reason = lineage.resolve_node(request)
    assert status.value == "unresolved" and reason == "ambiguous"
    lineage.by_call_id["second"].reasoning_witness = saved
    status, node, reason = lineage.resolve_node(request)
    assert status.value == "resolved" and reason == "reasoning_output_witness"
    assert node.call_id == "second"
    assert select_witness(request, [witness]) == "second"


@pytest.mark.parametrize("mutation", ["context", "id", "reasoning", "extra_assistant", "omit_reasoning"])
def test_changed_history_or_unattested_output_abstains(mutation):
    import copy

    _, witness, original = scenario()
    request = copy.deepcopy(original)
    if mutation == "context":
        request[0]["content"] = "Other task"
    elif mutation == "id":
        request[3]["id"] = "rs_other_branch"
    elif mutation == "reasoning":
        request[3]["summary"][0]["text"] = "Altered"
    elif mutation == "extra_assistant":
        request.append({"role": "assistant", "content": "Uncaptured"})
    else:
        del request[3]
    assert select_witness(request, [witness]) is None


def test_duplicate_witnesses_abstain_independent_of_order():
    _, witness, request = scenario()
    sibling = replace(witness, call_id="sibling")
    assert select_witness(request, [witness, sibling]) is None
    assert select_witness(request, [sibling, witness]) is None


def test_mixed_output_and_missing_ids_do_not_create_witness():
    assert make_witness("call", [], [{"type": "reasoning"}]) is None
    assert make_witness("call", [], [{"type": "reasoning", "id": "a"}, {"role": "assistant", "content": "x"}]) is None


@pytest.mark.parametrize(
    "inserted",
    [{"role": "user", "content": "Inserted"}, {"type": "function_call_output", "call_id": "x", "output": "Inserted"}],
)
def test_new_items_cannot_precede_or_interrupt_exact_output(inserted):
    from copy import deepcopy

    _, witness, request = scenario()
    before = deepcopy(request)
    before.insert(witness.context_len, inserted)
    assert select_witness(before, [witness]) is None
    output = [request[witness.context_len], {"type": "reasoning", "id": "rs_second", "summary": []}]
    multiple = make_witness("multi", request[: witness.context_len], output)
    interleaved = request[: witness.context_len] + [output[0], inserted, output[1]]
    assert select_witness(interleaved, [multiple]) is None
    reordered = request[: witness.context_len] + list(reversed(output))
    assert select_witness(reordered, [multiple]) is None


def test_consecutive_reasoning_turns_preserve_nullable_prior_context():
    from copy import deepcopy

    _, first, request = scenario()
    # The next recorded request has SDK-serialized optional nulls in a prior
    # reasoning item; an equivalent echo may omit only those nullable fields.
    second_request = deepcopy(request)
    second_request[3].update(content=None, encrypted_content=None)
    second_output = [
        {"type": "reasoning", "id": "rs_next", "summary": [{"type": "summary_text", "text": "Next plan"}]}
    ]
    second = make_witness("third", second_request, second_output)
    continuation = request + second_output + [{"role": "user", "content": "Proceed"}]
    assert select_witness(continuation, [first, second]) == "third"
    changed = deepcopy(continuation)
    changed[3]["summary"][0]["text"] = "Altered earlier reasoning"
    assert select_witness(changed, [first, second]) is None
