# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from unittest.mock import AsyncMock, patch

import pytest

from resources_servers.partner_if.app import VerifyRequest
from resources_servers.partner_if.schema import validate_input
from resources_servers.partner_if.tests.test_app import contract, response, server, verdict


def frozen_row():
    data = {"verifier_metadata": {"partner_if": contract("frozen-synthetic")}}
    data["verifier_metadata"]["partner_if"]["evidence_mode"] = "frozen_tool_history"
    data["responses_create_params"] = {
        "tools": [
            {
                "type": "function",
                "name": "search",
                "description": "Search.",
                "parameters": {"type": "object"},
                "strict": False,
            }
        ],
        "input": [
            {"role": "user", "content": "Say HELLO using the supplied source."},
            {"type": "function_call", "call_id": "c1", "name": "search", "arguments": '{"query":"hello"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "Source a: HELLO"},
            {"role": "user", "content": "Now answer without calling more tools."},
        ],
    }
    return data


def test_explicit_frozen_mode_preserves_history_and_tools():
    params = frozen_row()["responses_create_params"]
    original = copy.deepcopy(params)
    validate_input(params, evidence_mode="frozen_tool_history")
    assert params == original
    with pytest.raises(ValueError):
        validate_input(params, evidence_mode="text_response")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["input"][2].update(call_id="orphan"),
        lambda p: p["input"].pop(2),
        lambda p: p["input"].insert(3, copy.deepcopy(p["input"][2])),
        lambda p: p["input"].insert(3, copy.deepcopy(p["input"][1])),
        lambda p: p["input"][1].update(name="undeclared"),
        lambda p: p["input"][1].update(arguments="not JSON"),
        lambda p: p["input"][1].update(arguments="[]"),
        lambda p: p["input"][2].update(output=[{"type": "input_image", "image_url": "x"}]),
        lambda p: p["tools"].append(copy.deepcopy(p["tools"][0])),
        lambda p: p.update(tools=[{"type": "web_search"}]),
        lambda p: p.update(previous_response_id="hidden"),
        lambda p: p["input"].insert(2, {"role": "assistant", "content": "Interrupted tool batch"}),
        lambda p: p["input"].append({"type": "item_reference", "id": "hidden"}),
    ],
)
def test_reject_malformed_or_hidden_frozen_history(mutate):
    params = frozen_row()["responses_create_params"]
    mutate(params)
    with pytest.raises(ValueError):
        validate_input(params, evidence_mode="frozen_tool_history")


async def test_judge_sees_recorded_tools_but_only_grades_new_answer():
    data = frozen_row()
    body = VerifyRequest.model_validate({**data, "response": response().model_dump()})
    before = body.model_dump()
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(verdict())),
    ) as call:
        output = await server().verify(body)
    assert output.reward == 1
    payload = json.loads(call.call_args.kwargs["json"]["input"][1]["content"])
    assert payload["evidence_sources"]["tool:0/name"]["lines"][0]["text"] == "search"
    assert payload["evidence_sources"]["context:2/output"]["lines"][0]["text"] == "Source a: HELLO"
    assert payload["evidence_sources"]["context:2/output"]["origin"] == "context"
    assert payload["evidence_sources"]["answer"]["lines"][0]["text"] == "HELLO"
    assert "recorded" in call.call_args.kwargs["json"]["input"][0]["content"].lower()
    assert body.model_dump() == before


async def test_a_new_tool_call_instead_of_the_answer_fails_every_item_without_the_judge():
    call = {"type": "function_call", "id": "f2", "call_id": "c2", "name": "search", "arguments": "{}"}
    body = VerifyRequest.model_validate({**frozen_row(), "response": response(output=[call]).model_dump()})
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock) as judge:
        output = await server().verify(body)
    judge.assert_not_called()
    assert output.reward == 0 and not output.mask_sample
    assert output.failure_kind == "partner_if:no_answer"
    assert output.if_result.status == "scored" and output.if_result.passed == 0
