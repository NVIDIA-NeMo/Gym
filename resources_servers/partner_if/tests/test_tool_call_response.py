# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Next-call grading must preserve native calls without running task tools."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from resources_servers.partner_if.app import VerifyRequest
from resources_servers.partner_if.tests.test_app import response, server


def tool_row():
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Call search with query=hello."}],
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search.",
                    "parameters": {"type": "object"},
                    "strict": False,
                }
            ],
            "tool_choice": "required",
        },
        "verifier_metadata": {
            "partner_if": {
                "schema_version": "partner-if-v4",
                "task_id": "next-call",
                "origin": "synthetic",
                "source": "authored",
                "rubric_version": "test-v1",
                "audit_signoff": True,
                "data_kind": "synthetic_smoke",
                "evidence_mode": "tool_call_response",
                "rubrics": [
                    {
                        "id": "plan",
                        "constraint": "The response calls search with query=hello.",
                        "verifier": {"id": "llm_judge_v1"},
                    }
                ],
            }
        },
    }


def function_call(arguments='{"query":"hello"}'):
    return {
        "type": "function_call",
        "id": "fc1",
        "call_id": "c1",
        "name": "search",
        "arguments": arguments,
        "status": "completed",
    }


def judgment(passed, source=None):
    return response(
        json.dumps(
            {
                "items": [
                    {
                        "id": "plan",
                        "passed": passed,
                        "reason": "Check the emitted call.",
                        "evidence": "Requested query is present." if passed else "Required valid call is absent.",
                        "references": [{"source": source, "start_line": 1, "end_line": 1}] if source else [],
                    }
                ]
            }
        )
    )


async def test_tool_only_response_reaches_shared_judge_with_typed_arguments():
    body = VerifyRequest.model_validate({**tool_row(), "response": response(output=[function_call()]).model_dump()})
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=judgment(True, "event:0/arguments/query"),
    ):
        result = await server().verify(body)
    assert result.reward == 1 and not result.mask_sample
    assert result.grading_result.dimensions["if"].partial_score == 1
    assert result.grading_result.dimensions["if"].binary_score == 1
    assert result.judge_protocol == "partner-if-evidence-v5.2"
    sources = json.loads(result.judge_request["input"][1]["content"])["evidence_sources"]
    assert sources["event:0/tool_name"]["lines"][0]["text"] == "search"
    assert sources["event:0/arguments/query"]["lines"][0]["text"] == "hello"
    assert sources["tool:0/name"]["origin"] == "context"
    assert result.grading_result.items[0].quotes[0].quote == "hello"


@pytest.mark.parametrize("output", [response("I cannot do that.").model_dump()["output"], [function_call("not JSON")]])
async def test_a_refusal_or_malformed_call_is_judged_not_masked(output):
    body = VerifyRequest.model_validate({**tool_row(), "response": response(output=output).model_dump()})
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=judgment(False),
    ):
        result = await server().verify(body)
    assert result.reward == 0 and not result.mask_sample
    assert result.if_result.status == "scored"
    if output and output[0]["type"] == "function_call":
        sources = json.loads(result.judge_request["input"][1]["content"])["evidence_sources"]
        assert sources["event:0/arguments"]["lines"][0]["text"] == "not JSON"


@pytest.mark.parametrize(
    "override,kind",
    [
        ({"output": []}, "partner_if:no_answer"),
        ({"output": response(" ").model_dump()["output"]}, "partner_if:no_answer"),
        # Simple Agent marks a completed reply without a message or call as incomplete; the reply is still complete.
        (
            {"output": [], "status": "incomplete", "metadata": {"ng_termination_reason": "empty_output"}},
            "partner_if:no_answer",
        ),
        (
            {
                "output": [{"type": "reasoning", "id": "r1", "summary": []}],
                "status": "incomplete",
                "metadata": {"ng_termination_reason": "incomplete_reasoning"},
            },
            "partner_if:no_answer",
        ),
        ({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}, "partner_if:truncated"),
        (
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "metadata": {"ng_termination_reason": "empty_output"},
            },
            "partner_if:truncated",
        ),
    ],
)
async def test_a_reply_without_an_answer_or_call_fails_every_item_without_the_judge(override, kind):
    body = VerifyRequest.model_validate({**tool_row(), "response": response(**override).model_dump()})
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock) as call:
        result = await server().verify(body)
    call.assert_not_called()
    assert result.reward == 0 and not result.mask_sample
    assert result.failure_kind == kind and result.if_result.status == "scored"


@pytest.mark.parametrize(
    "override",
    [
        {"status": "incomplete", "metadata": {"ng_termination_reason": "max_steps"}},
        {"output": [function_call() | {"status": "in_progress"}]},
        {"output": [{"type": "function_call_output", "call_id": "c1", "output": "not actually executed"}]},
    ],
)
async def test_incomplete_or_non_single_response_capture_stays_unscorable(override):
    body = VerifyRequest.model_validate({**tool_row(), "response": response(**override).model_dump()})
    result = await server().verify(body)
    assert result.mask_sample and result.if_result.status == "unscorable"


async def test_recorded_calls_remain_context_and_new_calls_keep_their_order():
    row = tool_row()
    row["responses_create_params"]["input"] += [
        function_call(),
        {"type": "function_call_output", "call_id": "c1", "output": "old result"},
        {"role": "user", "content": "Call search with query=hello again."},
    ]
    calls = [function_call('{"query":"wrong"}'), function_call() | {"id": "fc2", "call_id": "c2"}]
    body = VerifyRequest.model_validate({**row, "response": response(output=calls).model_dump()})
    with patch(
        "resources_servers.partner_if.grading_service.call_judge",
        new_callable=AsyncMock,
        return_value=judgment(False),
    ):
        result = await server().verify(body)
    sources = json.loads(result.judge_request["input"][1]["content"])["evidence_sources"]
    assert sources["context:1/arguments"]["origin"] == "context"
    assert sources["event:0/arguments/query"]["lines"][0]["text"] == "wrong"
    assert sources["event:1/arguments/query"]["lines"][0]["text"] == "hello"
