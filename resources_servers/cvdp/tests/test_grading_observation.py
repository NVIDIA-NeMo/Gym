# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The CVDP verify records its calculations as grading operations and the docker-compose harness as its sandbox."""

from unittest.mock import AsyncMock, patch

import pytest

from nemo_gym.server_runtime import verifier_observability
from nemo_gym.trajectory import VerifierInvocation, grading_result_binding

from .test_app import _NO_SESSION, SAMPLE_RTL, _make_body, _make_request, _make_server


async def _verify(server, body_dict):
    invocation = VerifierInvocation(invocation_id="verify-cvdp")
    token = verifier_observability._active_verifier.set(invocation)
    try:
        result = await server.verify(_NO_SESSION, _make_request(body_dict))
    finally:
        verifier_observability._active_verifier.reset(token)
    return result, invocation


def _assert_bound_programmatic(invocation, result):
    (operation,) = invocation.grading_operations
    assert (operation.kind, operation.outcome, operation.dependencies) == ("programmatic_calculation", "completed", [])
    assert operation.result_binding_sha256 == grading_result_binding(result)


@pytest.mark.asyncio
async def test_a_harness_verdict_is_a_sandbox_observation_and_a_bound_grading():
    server = _make_server()
    body = _make_body(output_text=f"```systemverilog\n{SAMPLE_RTL}\n```")
    with patch.object(server._harness, "run", new_callable=AsyncMock, return_value=(0, "", [])):
        result, invocation = await _verify(server, body)
    assert result.reward == 1.0 and result.container_exit_code == 0
    _assert_bound_programmatic(invocation, result)
    (sandbox,) = invocation.sandbox_observations
    assert (sandbox.role, sandbox.provider, sandbox.outcome, sandbox.exit_code) == (
        "verifier",
        "docker-compose",
        "completed",
        0,
    )
    assert sandbox.wall_time_s is not None and sandbox.wall_time_s >= 0


@pytest.mark.asyncio
async def test_a_failing_testbench_is_still_a_completed_sandbox_and_a_bound_zero():
    server = _make_server()
    body = _make_body(output_text=f"```systemverilog\n{SAMPLE_RTL}\n```")
    with patch.object(server._harness, "run", new_callable=AsyncMock, return_value=(1, "FAILED", [])):
        result, invocation = await _verify(server, body)
    assert result.reward == 0.0 and result.container_exit_code == 1
    _assert_bound_programmatic(invocation, result)
    (sandbox,) = invocation.sandbox_observations
    assert (sandbox.outcome, sandbox.exit_code) == ("completed", 1)


@pytest.mark.asyncio
async def test_a_subjective_score_is_a_bound_programmatic_grading_without_a_sandbox():
    server = _make_server()
    body = _make_body(output_text="The module adds two numbers and registers the sum.")
    body["verifier_metadata"]["categories"] = ["cid006", "easy"]
    body["verifier_metadata"]["subjective_reference"] = "The module adds two numbers and registers the sum."
    result, invocation = await _verify(server, body)
    assert result.bleu_score is not None and 0.0 <= result.reward <= 1.0
    _assert_bound_programmatic(invocation, result)
    assert not invocation.sandbox_observations


@pytest.mark.asyncio
async def test_an_empty_output_is_a_bound_protocol_zero():
    server = _make_server()
    result, invocation = await _verify(server, _make_body(output_text=""))
    assert result.reward == 0.0 and result.container_exit_code is None
    _assert_bound_programmatic(invocation, result)
    assert not invocation.sandbox_observations
