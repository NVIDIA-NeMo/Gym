# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import pytest

from nemo_gym.core import responses_api_types as sdk
from nemo_gym.metrics import looping


UNIT = "".join(random.Random(19).choices("abcdefghijklmnopqrstuvwxyz", k=4096))
FILLER = "".join(random.Random(23).choices("abcdefghijklmnopqrstuvwxyz", k=60_000))


def message(text):
    return {"type": "message", "id": "m", "content": [{"type": "output_text", "text": text}]}


def tools(pattern, changing=False):
    requests = [
        sdk.NeMoGymResponseFunctionToolCall(call_id=str(index), name=name, arguments="{}")
        for index, name in enumerate(pattern)
    ]
    results = [
        sdk.NeMoGymFunctionCallOutput(call_id=str(index), output=str(index) if changing else "result")
        for index in reversed(range(len(pattern)))
    ]
    return [*requests, *results]


@pytest.mark.parametrize(
    "text,expected",
    [
        (UNIT[:129] * 15, False),
        (UNIT[:129] * 16, True),
        ((UNIT[:64] + "x" * 2048) * 8, True),
        ((UNIT[:1100] + "\n\n") * 5 + (UNIT[1100:2200] + "\n\n") * 5, True),
        ((UNIT[:1100] + "\n\n") * 5 + (UNIT[1100:2200] + "\n\n") * 5 + FILLER, False),
    ],
)
def test_text_thresholds_inner_witness_and_final_paragraph_fraction(text, expected):
    assert looping.assess_looping([message(text)]).is_looping is expected


@pytest.mark.parametrize(
    "pattern,changing,expected",
    [("A" * 7, False, False), ("AB" * 4, False, True), ("ABABABABCDEFCDEFCDEF", False, True), ("AB" * 4, True, False)],
)
def test_tool_witness_request_order_and_changing_results(pattern, changing, expected):
    assert looping.assess_looping(tools(pattern, changing)).is_looping is expected


def test_gaps_break_cycles_but_do_not_erase_a_confirmed_positive():
    output = tools("A" * 9)
    output[4] = message("x" * 1024)
    output[4]["content"].extend([{"type": "unknown"}, *output[4]["content"]])
    assert looping.assess_looping(output).is_looping is None
    output[4]["content"][0]["text"] = UNIT[:129] * 16
    assessment = looping.assess_looping(output)
    assert assessment.reasons == (looping.LoopingReason.REPEATED_TEXT,)
    assert looping.assess_looping([*tools("A" * 8), *tools("B")]).is_looping is None  # Duplicate IDs are ambiguous.


def test_reasoning_prefers_content_and_ignores_prompt_text():
    reasoning = sdk.NeMoGymResponseReasoningItem(
        id="r",
        summary=[{"type": "summary_text", "text": "short"}],
        content=[{"type": "reasoning_text", "text": UNIT[:129] * 16}],
    )
    assert looping.assess_looping([reasoning, message("recovered")]).is_looping is True
    prompt = sdk.NeMoGymEasyInputMessage(role="user", content=UNIT * 1000)
    assert looping.assess_looping([prompt]).is_looping is False


def test_missing_caps_search_exhaustion_and_seven_copy_stress():
    assert looping.assess_looping([]).is_looping is False
    assert looping.assess_looping([message("x" * 2_000_001)]).is_looping is None
    assert looping.assess_looping([message((UNIT * 7 + "!") * 60)]).is_looping is None


def test_pooled_metrics_keep_unknowns_and_missing_snapshots_in_coverage():
    counts = looping.LoopingCounts()
    for value in (True, False, None):
        counts.observe(looping.LoopingAssessment(value))
    assert counts.metrics(expected=4) == {"looping/rate": 0.5, "looping/coverage": 0.5}
