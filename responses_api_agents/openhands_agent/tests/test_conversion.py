# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from nemo_gym.config import ModelServerRef
from nemo_gym.core import MODEL_CONTEXT_LIMIT, MODEL_REQUEST_ERROR, MODEL_TRANSPORT_ERROR
from nemo_gym.trajectory import AgentInvocation, ContextCompactionObservation, ToolCallObservation
from responses_api_agents.openhands_agent.conversion import (
    classify_error,
    observation_records,
    output_items,
    parse_events,
    response_usage,
    run_outcome,
)


MODEL = ModelServerRef(type="responses_api_models", name="policy_model")


# Shapes copied from OpenHands SDK 1.59.0 event JSON.
def action(call_id, response_id, *, tool="terminal", command="ls", thought="", reasoning=None):
    return {
        "kind": "ActionEvent",
        "source": "agent",
        "thought": [{"type": "text", "text": thought}] if thought else [],
        "reasoning_content": reasoning,
        "action": {"command": command, "kind": "TerminalAction"},
        "tool_name": tool,
        "tool_call_id": call_id,
        "tool_call": {
            "id": call_id,
            "name": tool,
            "arguments": json.dumps({"command": command}),
            "origin": "completion",
        },
        "llm_response_id": response_id,
    }


def observation(call_id, text, *, tool="terminal", exit_code=0, is_error=False, model_read=None):
    # The runner adds the tool message the model read; OpenHands' terminal appends its exit-code line.
    if model_read is None:
        model_read = f"{text}\n[Command finished with exit code {exit_code}]" if tool == "terminal" else text
    return {
        "kind": "ObservationEvent",
        "source": "environment",
        "tool_name": tool,
        "tool_call_id": call_id,
        "observation": {
            "content": [{"type": "text", "text": text}],
            "is_error": is_error,
            "exit_code": exit_code,
            "kind": "TerminalObservation" if tool == "terminal" else "FinishObservation",
        },
        "tool_message_text": model_read,
    }


def usage(response_id, prompt=100, completion=10):
    return {
        "response_id": response_id,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cache_read_tokens": 4,
    }


def result(*responses, status="returned", execution_status="finished", error=None, condenser=()):
    return {
        "version": 1,
        "status": status,
        "execution_status": execution_status,
        "error": error,
        "started_at": 10.0,
        "ended_at": 12.5,
        "stats": {
            "usage_to_metrics": {
                "agent": {"token_usages": [usage(r) for r in responses]},
                "condenser": {"token_usages": [usage(r, 50, 5) for r in condenser]},
            }
        },
    }


def recorded(*events):
    return [(100.0 + index, event) for index, event in enumerate(events)]


def test_parse_events_keeps_valid_lines_and_counts_the_rest():
    text = "\n".join(
        [json.dumps([1.5, {"kind": "MessageEvent"}]), "{broken", json.dumps({"kind": "x"}), "", json.dumps([2, {}])]
    )
    events, malformed = parse_events(text)
    assert events == [(1.5, {"kind": "MessageEvent"})]
    assert malformed == 3


def test_output_items_keep_order_and_turn_the_finish_message_into_text():
    events = recorded(
        {"kind": "SystemPromptEvent", "source": "agent"},
        {"kind": "MessageEvent", "source": "user", "llm_message": {"content": [{"type": "text", "text": "task"}]}},
        action("c1", "r1", thought="Look first", reasoning="hidden"),
        observation("c1", "file.txt\n"),
        {
            "kind": "AgentErrorEvent",
            "source": "agent",
            "tool_name": "terminal",
            "tool_call_id": "c2",
            "error": "bad arguments",
            "tool_message_text": "bad arguments",
        },
        action("c3", "r2", tool="finish"),
        observation("c3", "done", tool="finish"),
    )
    items = output_items(events)
    assert [item.type for item in items] == [
        "reasoning",
        "message",
        "function_call",
        "function_call_output",
        "function_call_output",
        "function_call",
        "function_call_output",
        "message",
    ]
    assert items[2].call_id == "c1" and json.loads(items[2].arguments) == {"command": "ls"}
    assert items[3].output == "file.txt\n\n[Command finished with exit code 0]"
    assert items[4].output == "bad arguments"
    assert items[-1].content[0].text == "done"


def test_tool_results_hold_the_text_the_model_read():
    # OpenHands 1.59.0 TerminalObservation content and to_llm_message text for a command that was still running.
    notice = (
        "[Below is the output of the previous command.]\n\n[The command has no new output after 30 seconds.]"
        "\n[Current working directory: /app]"
    )
    events = recorded(
        action("c1", "r1", command="apt-get install -y python3"),
        observation("c1", "", exit_code=-1, model_read=notice),
        {
            "kind": "UserRejectObservation",
            "tool_name": "terminal",
            "tool_call_id": "c2",
            "rejection_reason": "not allowed",
            "tool_message_text": "Action rejected: not allowed",
        },
        action("c3", "r2", tool="finish"),
        observation("c3", "All done", tool="finish", model_read="All done (as a tool message)"),
    )
    items = output_items(events)
    outputs = [item.output for item in items if item.type == "function_call_output"]
    assert outputs == [notice, "Action rejected: not allowed", "All done (as a tool message)"]
    # The final answer is the agent's own finish message, not the tool message built from it.
    assert items[-1].content[0].text == "All done"


def test_agent_message_becomes_assistant_text():
    message = {
        "kind": "MessageEvent",
        "source": "agent",
        "llm_message": {"role": "assistant", "content": [{"type": "text", "text": "All set."}]},
        "llm_response_id": "r1",
    }
    (item,) = output_items(recorded(message))
    assert item.type == "message" and item.content[0].text == "All set."


def test_usage_is_reported_only_when_every_used_response_has_a_record():
    events = recorded(action("c1", "r1"), observation("c1", "ok"), action("c2", "r2", tool="finish"))
    total = response_usage(events, result("r1", "r2"))
    assert (total.input_tokens, total.output_tokens, total.total_tokens) == (200, 20, 220)
    assert total.input_tokens_details.cached_tokens is None
    assert total.output_tokens_details.reasoning_tokens is None
    assert response_usage(events, result("r1")) is None
    assert response_usage(events, None) is None


def test_condenser_calls_count_toward_usage():
    events = recorded(action("c1", "r1"), {"kind": "Condensation", "llm_response_id": "s1", "summary": "sum"})
    assert response_usage(events, result("r1", condenser=["s1"])).input_tokens == 150
    assert response_usage(events, result("r1")) is None


@pytest.mark.parametrize(
    "agent_details,condenser_details",
    [
        ({}, {}),
        ({"cache_read_tokens": 0, "reasoning_tokens": 0}, {"cache_read_tokens": 0, "reasoning_tokens": 0}),
        ({"cache_read_tokens": 4, "reasoning_tokens": 2}, {"cache_read_tokens": 3, "reasoning_tokens": 1}),
        ({"cache_read_tokens": 4, "reasoning_tokens": 2}, {}),
    ],
    ids=["missing", "normalized-zero", "positive", "mixed"],
)
def test_normalized_sdk_details_remain_unknown_and_raw_stats_are_preserved(agent_details, condenser_details):
    events = recorded(action("c1", "r1"), {"kind": "Condensation", "llm_response_id": "s1", "summary": "sum"})
    run_result = result("r1", condenser=["s1"])
    metrics = run_result["stats"]["usage_to_metrics"]
    for name, details in (("agent", agent_details), ("condenser", condenser_details)):
        record = metrics[name]["token_usages"][0]
        record.pop("cache_read_tokens")
        record.update(details)
    raw_result = json.dumps(run_result)

    total = response_usage(events, run_result)
    assert (total.input_tokens, total.output_tokens, total.total_tokens) == (150, 15, 165)
    assert total.input_tokens_details.cached_tokens is None
    assert total.output_tokens_details.reasoning_tokens is None
    evidence = observation_records(
        events,
        run_result,
        invocation_id="rollout",
        model_ref=MODEL,
        conversation=[],
        outcome=run_outcome(events, run_result, timed_out=False),
    )
    gaps = {gap.code: gap for gap in evidence.gaps}
    for code in ("cached_token_usage_unavailable", "reasoning_token_usage_unavailable"):
        assert gaps[code].invocation_id == "rollout"
        assert "do not preserve" in gaps[code].detail
    assert json.dumps(run_result) == raw_result


@pytest.mark.parametrize(
    "outcome_result,events,expected",
    [
        (result(), [], ("completed", "finished")),
        (result(execution_status="stuck"), [], ("incomplete", "stuck")),
        (
            result(execution_status="error"),
            recorded({"kind": "ConversationErrorEvent", "code": "MaxIterationsReached", "detail": "limit"}),
            ("incomplete", "max_iterations"),
        ),
        (
            result(execution_status="error"),
            recorded({"kind": "ConversationErrorEvent", "code": "MaxBudgetReached", "detail": "spent"}),
            ("incomplete", "max_budget"),
        ),
        (result(execution_status="error"), [], ("failed", "error")),
        (result(status="terminated", execution_status="running"), [], ("incomplete", "cancelled")),
        (None, [], ("failed", "runner_error")),
    ],
)
def test_run_outcome_follows_openhands_terminal_status(outcome_result, events, expected):
    outcome = run_outcome(events, outcome_result, timed_out=False)
    assert (outcome.status, outcome.stop_reason) == expected


# What OpenHands records before its first model call returns.
OPENING = (
    {"kind": "SystemPromptEvent", "source": "agent"},
    {"kind": "MessageEvent", "source": "user", "llm_response_id": None},
)


def test_the_deadline_is_graded_after_a_model_reply_and_a_failure_of_unknown_cause_before_one():
    replied = run_outcome(recorded(action("c1", "r1")), result("r1", status="terminated"), timed_out=True)
    assert (replied.status, replied.stop_reason, replied.failure_kind) == ("incomplete", "timeout", None)
    # A usage record also shows a reply, such as one OpenHands answered with an error event.
    assert run_outcome(recorded(*OPENING), result("r1", status="terminated"), timed_out=True).status == "incomplete"
    # Without a reply the deadline is all that was observed: no model call needs to have started.
    for events, runner_result in ((recorded(*OPENING), result(status="terminated")), ([], None)):
        silent = run_outcome(events, runner_result, timed_out=True)
        assert (silent.status, silent.stop_reason, silent.failure_kind) == ("failed", "timeout", "unknown")
        assert silent.message == "Gym's deadline passed before any model call returned"
    # A rollout stop is not a failure, so the stopped run is graded.
    assert run_outcome([], None, timed_out=True, rollout_stop=True).status == "incomplete"


def test_a_model_timeout_openhands_recorded_is_a_transport_failure():
    chain = [
        {"type": "ConversationRunError", "module": "openhands.sdk.conversation.exceptions", "status_code": None},
        {"type": "LLMTimeoutError", "module": "openhands.sdk.llm.exceptions.types", "status_code": None},
        {"type": "Timeout", "module": "litellm.exceptions", "status_code": 408},
    ]
    outcome = run_outcome(recorded(*OPENING), result(status="error", error=chain), timed_out=False)
    assert (outcome.status, outcome.stop_reason, outcome.failure_kind, outcome.status_code) == (
        "failed",
        "error",
        MODEL_TRANSPORT_ERROR,
        None,
    )
    assert outcome.message == (
        "The OpenHands run failed: ConversationRunError caused by LLMTimeoutError caused by Timeout (HTTP 408)"
    )


@pytest.mark.parametrize(
    "chain,expected",
    [
        ([{"type": "ConversationRunError"}, {"type": "LLMContextWindowExceedError"}], (MODEL_CONTEXT_LIMIT, None)),
        (
            [{"type": "LLMRateLimitError"}, {"type": "RateLimitError", "status_code": 429}],
            (MODEL_REQUEST_ERROR, 429),
        ),
        (
            [{"type": "LLMServiceUnavailableError"}, {"type": "InternalServerError", "status_code": 502}],
            (MODEL_REQUEST_ERROR, 502),
        ),
        (
            [{"type": "LLMContentPolicyViolationError"}, {"type": "BadRequestError", "status_code": 400}],
            (MODEL_REQUEST_ERROR, 400),
        ),
        # An error class does not establish an HTTP status; one no error recorded stays unknown.
        ([{"type": "LLMContentPolicyViolationError"}], ("unknown", None)),
        ([{"type": "ConversationRunError"}, {"type": "LLMRateLimitError"}], ("unknown", None)),
        ([{"type": "ServiceUnavailableError"}], ("unknown", None)),
        ([{"type": "LLMTimeoutError"}], (MODEL_TRANSPORT_ERROR, None)),
        # LiteLLM sets status codes on its timeout and connection errors, though no response carried them.
        (
            [
                {"type": "ConversationRunError", "status_code": None},
                {"type": "LLMTimeoutError", "status_code": None},
                {"type": "Timeout", "status_code": 408},
            ],
            (MODEL_TRANSPORT_ERROR, None),
        ),
        (
            [
                {"type": "ConversationRunError", "status_code": None},
                {"type": "LLMServiceUnavailableError", "status_code": None},
                {"type": "APIConnectionError", "status_code": 500},
            ],
            (MODEL_TRANSPORT_ERROR, None),
        ),
        ([{"type": "BadRequestError", "status_code": 400}], (MODEL_REQUEST_ERROR, 400)),
        ([{"type": "ValueError"}], ("unknown", None)),
    ],
)
def test_classify_error_reads_the_whole_exception_chain(chain, expected):
    assert classify_error(chain) == expected


def test_failed_run_and_context_limit_outcomes():
    chain = [{"type": "LLMRateLimitError"}, {"type": "RateLimitError", "status_code": 429}]
    rate_limited = result(status="error", execution_status="error", error=chain)
    outcome = run_outcome([], rate_limited, timed_out=False)
    assert (outcome.status, outcome.failure_kind, outcome.status_code) == ("failed", MODEL_REQUEST_ERROR, 429)
    overflow = result(status="error", error=[{"type": "LLMContextWindowExceedError"}])
    outcome = run_outcome([], overflow, timed_out=False)
    assert (outcome.status, outcome.stop_reason) == ("incomplete", "context_limit")


def test_observations_time_each_tool_call_and_fail_nonzero_exits():
    events = recorded(
        action("c1", "r1"),
        observation("c1", "ok"),
        action("c2", "r2", command="false"),
        observation("c2", "", exit_code=1),
        action("c3", "r3"),
    )
    evidence = observation_records(
        events,
        result("r1", "r2", "r3"),
        invocation_id="rollout",
        model_ref=MODEL,
        conversation=[],
        outcome=run_outcome(events, None, timed_out=True),
    )
    invocation = evidence.records[0]
    assert isinstance(invocation, AgentInvocation)
    assert (invocation.status, invocation.stop_reason, invocation.duration_ms) == ("incomplete", "timeout", 2500)
    assert [call.response_id for call in invocation.model_calls] == ["r1", "r2", "r3"]
    tools = {record.tool_call_id: record for record in evidence.records if isinstance(record, ToolCallObservation)}
    assert tools["c1"].status == "completed" and tools["c1"].duration_ms == 1000
    assert tools["c2"].status == "failed"
    assert tools["c3"].status == "incomplete" and tools["c3"].completed_at is None
    assert [gap.code for gap in evidence.gaps] == [
        "cached_token_usage_unavailable",
        "reasoning_token_usage_unavailable",
        "tool_timing_unavailable",
    ]


def test_condensation_records_its_summary_call_and_neighbors():
    events = recorded(
        action("c1", "r1"),
        observation("c1", "ok"),
        {"kind": "Condensation", "llm_response_id": "s1", "summary": "earlier work", "forgotten_event_ids": ["e1"]},
        action("c2", "r2", tool="finish"),
    )
    evidence = observation_records(
        events,
        result("r1", "r2", condenser=["s1"]),
        invocation_id="rollout",
        model_ref=MODEL,
        conversation=[],
        outcome=run_outcome(events, result(), timed_out=False),
    )
    (compaction,) = [record for record in evidence.records if isinstance(record, ContextCompactionObservation)]
    assert compaction.summary == "earlier work"
    assert [call.response_id for call in compaction.model_calls] == ["s1"]
    assert compaction.before_model_call.response_id == "r1"
    assert compaction.after_model_call.response_id == "r2"
    # The summary call belongs to the compaction, not to a policy step of the invocation.
    assert [call.response_id for call in evidence.records[0].model_calls] == ["r1", "r2"]


@pytest.mark.parametrize("separator", ["\u0085", "\u2028", "\u2029"])
@pytest.mark.parametrize("ensure_ascii", [False, True])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_jsonl_keeps_unicode_message_and_model_call_identity(separator, ensure_ascii, line_ending):
    value = f"before{separator}after"
    message = {
        "kind": "MessageEvent",
        "source": "agent",
        "llm_response_id": "response-unicode",
        "llm_message": {"content": [{"type": "text", "text": value}]},
    }
    action_event = action("unicode-call", "response-action", thought=value, command=value)
    stdout = line_ending.join(json.dumps([1.5, event], ensure_ascii=ensure_ascii) for event in (message, action_event))
    events, malformed = parse_events(stdout + line_ending)
    assert malformed == 0
    assert len(events) == 2
    assert [event["llm_response_id"] for _, event in events] == ["response-unicode", "response-action"]
    items = output_items(events)
    assert [item.content[0].text for item in items if item.type == "message"] == [value, value]
    call = next(item for item in items if item.type == "function_call")
    assert call.call_id == "unicode-call" and json.loads(call.arguments)["command"] == value
