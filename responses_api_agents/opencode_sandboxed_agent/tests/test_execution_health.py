# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from responses_api_agents.opencode_sandboxed_agent.execution_health import execution_health


def classify(**changes):
    values = dict(
        return_code=0,
        error_type=None,
        exception_type=None,
        elapsed_s=10,
        budget_s=28800,
        finished=True,
        export={"messages": [{"info": {"role": "assistant", "finish": "stop"}}]},
    )
    values.update(changes)
    return execution_health(**values)


def test_early_timeout_is_infrastructure_but_full_agent_budget_is_not():
    assert classify(error_type="timeout", return_code=124, elapsed_s=120)["retryable"]
    full = classify(error_type="timeout", return_code=124, elapsed_s=28801, finished=False)
    assert full["outcome"] == "agent_timeout"
    assert not full["retryable"]


def test_exception_preserved_even_when_partial_export_exists():
    result = classify(exception_type="ConnectionError", return_code=None, finished=False)
    assert result["outcome"] == "infrastructure_error"
    assert result["reason"] == "agent_execution_error"


def test_missing_terminal_evidence_is_retryable():
    assert classify(export={})["retryable"]
    assert classify(finished=False)["retryable"]


def test_terminal_endpoint_error_overrides_success_marker_and_elapsed_budget():
    result = classify(
        export={
            "messages": [
                {
                    "info": {
                        "role": "assistant",
                        "error": {"name": "APIError", "data": {"statusCode": 503, "message": "unavailable"}},
                    }
                }
            ]
        },
        error_type="timeout",
        return_code=124,
        elapsed_s=28801,
    )
    assert result["reason"] == "model_endpoint_error"


def test_context_exhaustion_is_a_benchmark_outcome():
    result = classify(export={"messages": [{"info": {"role": "assistant", "finish": "length"}}]})
    assert result["outcome"] == "context_limit"
    assert not result["retryable"]


def test_recovered_error_and_tool_output_do_not_create_false_infrastructure_failure():
    result = classify(
        export={
            "messages": [
                {"info": {"role": "assistant", "error": {"name": "APIError", "data": {"statusCode": 503}}}},
                {"info": {"role": "assistant", "finish": "stop"}, "parts": [{"text": "ECONNRESET"}]},
            ]
        }
    )
    assert result["outcome"] == "completed"
