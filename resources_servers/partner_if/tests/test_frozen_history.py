# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
import json
from unittest.mock import AsyncMock, patch

import pytest

from resources_servers.partner_if.app import VerifyRequest
from resources_servers.partner_if.package import validate_rows
from resources_servers.partner_if.tests.test_app import contract, response, server, verdict


def frozen_row():
    data = {"verifier_metadata": {"partner_if": contract("frozen-synthetic", data_kind="reported")}}
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
    data = frozen_row()
    original = copy.deepcopy(data)
    assert validate_rows([data]) == [original]
    data["verifier_metadata"]["partner_if"]["evidence_mode"] = "text_response"
    with pytest.raises(ValueError):
        validate_rows([data])


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
    data = frozen_row()
    mutate(data["responses_create_params"])
    with pytest.raises(ValueError):
        validate_rows([data])


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


@pytest.mark.parametrize("unexercised", [True, False])
async def test_new_text_judges_cannot_emit_the_retired_scope_field(unexercised):
    parsed = verdict()
    parsed["items"][1].update(unexercised=unexercised)
    body = VerifyRequest.model_validate({**frozen_row(), "response": response().model_dump()})
    body.verifier_metadata.partner_if.rubrics[1].constraint = "Every emitted citation is outside inline code."
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(parsed)),
    ):
        output = await server().verify(body)
    assert output.mask_sample and output.if_result.status == "unscorable"
    assert "unexercised" in output.grading_result.items[0].reason


async def test_new_text_judge_scores_boolean_items_without_scope_labels():
    parsed = verdict()
    body = VerifyRequest.model_validate({**frozen_row(), "response": response().model_dump()})
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=response(json.dumps(parsed)),
    ):
        output = await server().verify(body)
    assert output.if_result.passed == 2 and output.if_result.total == 2
    assert output.reward == 1 and output.judge_protocol == "partner-if-evidence-v5.2"
