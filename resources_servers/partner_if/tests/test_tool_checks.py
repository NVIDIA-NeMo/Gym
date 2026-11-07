# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool selection, argument validation and infrastructure masking through partner_if."""

from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest

from resources_servers.partner_if.app import VerifyRequest
from resources_servers.partner_if.checks import CheckContext, resolve
from resources_servers.partner_if.engine import evaluate
from resources_servers.partner_if.evidence import EvidenceBundle, EvidenceUnavailable
from resources_servers.partner_if.grading_contract import GradingContract, VerifierSpec
from resources_servers.partner_if.tests.test_app import response, server


def tool(name="ipython"):
    return {
        "type": "function",
        "name": name,
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
            "additionalProperties": False,
        },
        "strict": False,
    }


def contract():
    return GradingContract(
        task_id="synthetic-tool-selection",
        origin="synthetic",
        source="authored test",
        rubric_version="test-v1",
        audit_signoff=True,
        data_kind="synthetic_smoke",
        evidence_mode="tool_call_response",
        rubrics=[
            {
                "id": "advertised_tool_call",
                "constraint": "Emit exactly one schema-valid call to an advertised function.",
                "verifier": {"id": "single_advertised_tool_call_v1"},
            }
        ],
    )


def function_call(name="ipython", arguments='{"code":"print(1)"}'):
    return {
        "type": "function_call",
        "id": "fc1",
        "call_id": "c1",
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


async def verify(output, **overrides):
    body = VerifyRequest.model_validate(
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "Only call advertised tools. Execute print(1) with ipython."}],
                "tools": [tool()],
                "tool_choice": "auto",
            },
            "verifier_metadata": {"partner_if": contract().model_dump()},
            "response": response(output=output, **overrides).model_dump(),
        }
    )
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock) as judge:
        result = await server().verify(body)
    judge.assert_not_called()
    return result


@pytest.mark.parametrize(
    "name,arguments,expected",
    [
        ("ipython", '{"code":"await send()"}', 1),
        ("ipython", '{"code":"send()"}', 1),
        ("ipython", '{"code":"not valid Python!"}', 1),
        ("ipython", '{"code":""}', 1),
        ("unlisted.send", '{"code":"print(1)"}', 0),
        ("ipython", "not JSON", 0),
        ("ipython", '"JSON string"', 0),
        ("ipython", "[]", 0),
        ("ipython", "null", 0),
        ("ipython", "{}", 0),
        ("ipython", '{"code":123}', 0),
        ("ipython", '{"code":"print(1)","extra":true}', 0),
    ],
)
async def test_new_call_is_scored_without_a_judge_or_code_execution(name, arguments, expected):
    result = await verify([function_call(name, arguments)])
    assert result.reward == expected and not result.mask_sample
    assert result.if_result.status == "scored"
    assert result.judge_request is None and result.judge_response is None


@pytest.mark.parametrize(
    "output",
    [[], response("I cannot do that.").model_dump()["output"], [function_call(), function_call() | {"call_id": "c2"}]],
)
async def test_no_call_and_multiple_calls_fail(output):
    result = await verify(output)
    assert result.reward == 0 and not result.mask_sample


async def test_token_cutoff_fails_but_provider_and_capture_errors_are_masked():
    cutoff = await verify([], status="incomplete", incomplete_details={"reason": "max_output_tokens"})
    assert cutoff.reward == 0 and not cutoff.mask_sample and cutoff.failure_kind == "partner_if:truncated"
    for overrides in [
        {"status": "failed", "error": {"code": "server_error", "message": "provider failed"}},
        {"status": "incomplete", "metadata": {"ng_termination_reason": "max_steps"}},
    ]:
        result = await verify([function_call()], **overrides)
        assert result.mask_sample and result.failure_kind == "partner_if:unscorable"
    result = await verify([function_call() | {"status": "in_progress"}])
    assert result.mask_sample and result.failure_kind == "partner_if:unscorable"


def run_check(evidence):
    checker, params = resolve(VerifierSpec(id="single_advertised_tool_call_v1"))
    return checker.run(params, CheckContext(evidence))[0]


def test_registry_and_new_calls_are_used_instead_of_historical_or_textual_mentions():
    evidence = EvidenceBundle(
        tools=[tool("kernel"), tool("another_kernel")],
        context=[function_call("unlisted.send")],
        answer="Mention unlisted.send in prose",
        trajectory=[
            {"source_index": 0, "kind": "context", "text": "Call unlisted.send"},
            {"source_index": 1, "kind": "tool_result", "text": "unlisted.send"},
            {"source_index": 2, "kind": "tool_call", "tool_name": "kernel", "arguments": {"code": "pass"}},
        ],
    )
    assert run_check(evidence)
    evidence.trajectory[-1].tool_name = "ipython"
    assert not run_check(evidence)


@pytest.mark.parametrize(
    "tools",
    [
        [],
        [{"type": "web_search"}],
        [tool(), tool()],
        [tool("")],
        [tool() | {"parameters": None}],
        [tool() | {"parameters": {"type": "invalid-schema-type"}}],
    ],
)
async def test_invalid_registry_is_unscorable_not_a_model_failure(tools):
    async def no_judge(items, evidence):
        raise AssertionError("Deterministic checker must not call a judge")

    result = await evaluate(contract(), EvidenceBundle(tools=tools), judge=no_judge)
    assert result.status == "unscorable" and result.reward is None
    assert result.items[0].passed is None


def test_capture_gap_does_not_become_a_zero_call_failure():
    with pytest.raises(EvidenceUnavailable, match="capture failed"):
        run_check(EvidenceBundle(tools=[tool()], trajectory_error="capture failed"))


def test_local_schema_references_work_and_remote_references_are_not_fetched():
    definition = deepcopy(tool())
    definition["parameters"] = {
        "$defs": {"args": tool()["parameters"]},
        "$ref": "#/$defs/args",
    }
    evidence = EvidenceBundle(
        tools=[definition],
        trajectory=[{"source_index": 0, "kind": "tool_call", "tool_name": "ipython", "arguments": {"code": "pass"}}],
    )
    assert run_check(evidence)
    evidence.tools[0]["parameters"] = {"$ref": "https://example.invalid/schema.json"}
    with pytest.raises(EvidenceUnavailable, match="unresolved reference"):
        run_check(evidence)


async def test_frozen_calls_are_not_counted_as_new_calls_by_the_adapter():
    body = VerifyRequest.model_validate(
        {
            "responses_create_params": {
                "input": [
                    {"role": "user", "content": "Call ipython."},
                    function_call(),
                    {"type": "function_call_output", "call_id": "c1", "output": "1"},
                    {"role": "user", "content": "Call ipython again."},
                ],
                "tools": [tool()],
            },
            "verifier_metadata": {"partner_if": contract().model_dump()},
            "response": response(output=[function_call() | {"id": "fc2", "call_id": "c2"}]).model_dump(),
        }
    )
    result = await server().verify(body)
    assert result.reward == 1 and not result.mask_sample
