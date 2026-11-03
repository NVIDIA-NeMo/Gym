# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert the OpenHands SDK 1.59.0 events and run result that the sandbox runner writes into Gym records.

The runner writes each event as ``[seen_at, event]``, where ``event`` is the SDK's own JSON for one event
(``kind`` names its class). Nothing here imports OpenHands: the runtime exists only inside the task sandbox.
"""

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    MODEL_CONTEXT_LIMIT,
    MODEL_REQUEST_ERROR,
    MODEL_TRANSPORT_ERROR,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
)
from nemo_gym.trajectory import (
    AgentInvocation,
    ContextCompactionObservation,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
)


# OpenHands codes for the run limits it enforces itself; the run ends with partial work that can be graded.
RUN_LIMIT_CODES = {"MaxIterationsReached": "max_iterations", "MaxBudgetReached": "max_budget"}
# Exception classes in a failed run's chain, from openhands.sdk.llm.exceptions (with LiteLLM's originals).
_CONTEXT_LIMIT_ERRORS = {"LLMContextWindowExceedError", "ContextWindowExceededError"}
_TRANSPORT_ERRORS = {"LLMTimeoutError", "Timeout", "APIConnectionError", "APITimeoutError", "LLMNoResponseError"}
_STATUS_BY_ERROR = {
    "LLMAuthenticationError": 401,
    "AuthenticationError": 401,
    "LLMRateLimitError": 429,
    "RateLimitError": 429,
    "LLMServiceUnavailableError": 503,
    "ServiceUnavailableError": 503,
    "InternalServerError": 500,
    "LLMBadRequestError": 400,
    "BadRequestError": 400,
    "LLMContentPolicyViolationError": 400,
}


def parse_events(text: str) -> tuple[list[tuple[float, dict[str, Any]]], int]:
    """Return the recorded ``(seen_at, event)`` pairs and the number of malformed lines skipped."""
    events: list[tuple[float, dict[str, Any]]] = []
    malformed = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            seen_at, event = json.loads(line)
        except (ValueError, TypeError):
            malformed += 1
            continue
        if isinstance(seen_at, (int, float)) and isinstance(event, dict) and isinstance(event.get("kind"), str):
            events.append((float(seen_at), event))
        else:
            malformed += 1
    return events, malformed


def text_of(content: Any) -> str:
    """Join the text parts of an OpenHands content list."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str))


def _arguments(event: dict[str, Any]) -> str:
    arguments = (event.get("tool_call") or {}).get("arguments")
    if isinstance(arguments, str):
        return arguments
    action = event.get("action")
    if isinstance(action, dict):
        return json.dumps({key: value for key, value in action.items() if key != "kind"})
    return "{}"


def _tool_result(event: dict[str, Any]) -> tuple[str, bool]:
    """The text OpenHands returned to the model for a tool call, and whether the call failed."""
    kind = event["kind"]
    if kind == "AgentErrorEvent":
        return str(event.get("error") or ""), True
    if kind == "UserRejectObservation":
        return str(event.get("rejection_reason") or ""), True
    observation = event.get("observation") or {}
    exit_code = observation.get("exit_code")
    # A shell command that exits nonzero failed, although OpenHands reports the call itself as successful.
    failed = observation.get("is_error") is True or (type(exit_code) is int and exit_code != 0)
    return text_of(observation.get("content")), failed


def output_items(events: Iterable[tuple[float, dict[str, Any]]]) -> list[Any]:
    """Responses API items for the agent's messages, tool calls, and tool results, in order."""
    items: list[Any] = []

    def add_reasoning(text: Any) -> None:
        if isinstance(text, str) and text:
            items.append(
                NeMoGymResponseReasoningItem(id=f"rs_{len(items)}", summary=[{"type": "summary_text", "text": text}])
            )

    def add_message(text: str) -> None:
        if text:
            items.append(
                NeMoGymResponseOutputMessage(
                    id=f"msg_{len(items)}", content=[NeMoGymResponseOutputText(text=text, annotations=[])]
                )
            )

    for _, event in events:
        kind = event["kind"]
        if kind == "ActionEvent":
            add_reasoning(event.get("reasoning_content"))
            add_message(text_of(event.get("thought")))
            items.append(
                NeMoGymResponseFunctionToolCall(
                    id=f"fc_{len(items)}",
                    call_id=str(event.get("tool_call_id")),
                    name=str(event.get("tool_name")),
                    arguments=_arguments(event),
                    status="completed",
                )
            )
        elif kind in ("ObservationEvent", "AgentErrorEvent", "UserRejectObservation") and event.get("tool_call_id"):
            output, _ = _tool_result(event)
            items.append(NeMoGymFunctionCallOutput(call_id=str(event["tool_call_id"]), output=output))
            if kind == "ObservationEvent" and event.get("tool_name") == "finish":
                # OpenHands delivers its final answer through the finish tool.
                add_message(output)
        elif kind == "MessageEvent" and event.get("source") == "agent":
            message = event.get("llm_message") or {}
            add_reasoning(message.get("reasoning_content"))
            add_message(text_of(message.get("content")))
    return items


def _response_ids(events: Iterable[tuple[float, dict[str, Any]]]) -> list[str]:
    """IDs of the model responses the conversation used, in order, without repeats."""
    seen: dict[str, None] = {}
    for _, event in events:
        response_id = event.get("llm_response_id")
        if event["kind"] in ("ActionEvent", "MessageEvent", "Condensation") and isinstance(response_id, str):
            seen.setdefault(response_id, None)
    return list(seen)


def token_usages(result: Optional[dict[str, Any]], usage_id: Optional[str] = None) -> list[dict[str, Any]]:
    """Per-call usage records OpenHands kept, for one LLM (``agent`` or ``condenser``) or for all of them."""
    stats = (result or {}).get("stats") or {}
    usages = []
    for name, metrics in (stats.get("usage_to_metrics") or {}).items():
        if usage_id is not None and name != usage_id:
            continue
        for usage in (metrics or {}).get("token_usages") or []:
            if isinstance(usage, dict) and isinstance(usage.get("response_id"), str) and usage["response_id"]:
                usages.append(usage)
    return usages


def response_usage(
    events: list[tuple[float, dict[str, Any]]], result: Optional[dict[str, Any]]
) -> Optional[NeMoGymResponseUsage]:
    """Total usage, only when every model response the conversation used has a usage record."""
    usages = token_usages(result)
    recorded = {usage["response_id"] for usage in usages}
    used = _response_ids(events)
    if not usages or any(response_id not in recorded for response_id in used):
        return None

    def total(name: str) -> int:
        return sum(value for usage in usages if type(value := usage.get(name)) is int and value >= 0)

    input_tokens, output_tokens = total("prompt_tokens"), total("completion_tokens")
    return NeMoGymResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=total("cache_read_tokens")),
        output_tokens=output_tokens,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=total("reasoning_tokens")),
        total_tokens=input_tokens + output_tokens,
    )


def error_chain(result: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    chain = (result or {}).get("error")
    return [entry for entry in chain if isinstance(entry, dict)] if isinstance(chain, list) else []


def error_status_code(chain: list[dict[str, Any]]) -> Optional[int]:
    """The HTTP status a provider error carried, or the status its OpenHands error class stands for."""
    for entry in chain:
        status = entry.get("status_code")
        if type(status) is int and 400 <= status <= 599:
            return status
    for entry in chain:
        if entry.get("type") in _STATUS_BY_ERROR:
            return _STATUS_BY_ERROR[entry["type"]]
    return None


def classify_error(chain: list[dict[str, Any]]) -> tuple[str, Optional[int]]:
    """A registered failure kind (or ``unknown``) and HTTP status for the error that ended an OpenHands run."""
    types = {entry.get("type") for entry in chain}
    if types & _CONTEXT_LIMIT_ERRORS:
        return MODEL_CONTEXT_LIMIT, None
    status = error_status_code(chain)
    if status is not None:
        return MODEL_REQUEST_ERROR, status
    if types & _TRANSPORT_ERRORS:
        return MODEL_TRANSPORT_ERROR, None
    return "unknown", None


@dataclass
class RunOutcome:
    """How an activation ended, by OpenHands' own rule at the pinned version."""

    status: Literal["completed", "failed", "incomplete"]
    stop_reason: str
    message: Optional[str] = None
    failure_kind: Optional[str] = None
    status_code: Optional[int] = None


def run_outcome(
    events: list[tuple[float, dict[str, Any]]], result: Optional[dict[str, Any]], *, timed_out: bool
) -> RunOutcome:
    """Map the runner result to a response status.

    ``conversation.run()`` raises only for errors that end the run, such as a model call that still fails after
    OpenHands' own retries. Tool errors go back to the model and do not end the run. A run limit (iterations or
    budget) and stuck detection end the run with partial work, as does the deadline.
    """
    if timed_out:
        return RunOutcome("incomplete", "timeout")
    if result is None:
        return RunOutcome("failed", "runner_error", "The OpenHands runner wrote no result")
    if result.get("status") == "terminated":
        return RunOutcome("incomplete", "cancelled")
    chain = error_chain(result)
    if result.get("status") == "error" or chain:
        kind, status_code = classify_error(chain)
        message = "; ".join(f"{entry.get('type')}: {entry.get('message')}" for entry in chain[:2])
        if kind == MODEL_CONTEXT_LIMIT:
            return RunOutcome("incomplete", "context_limit", message, kind)
        return RunOutcome("failed", "error", message or "The OpenHands run failed", kind, status_code)
    execution_status = result.get("execution_status")
    if execution_status == "finished":
        return RunOutcome("completed", "finished")
    if execution_status == "stuck":
        return RunOutcome("incomplete", "stuck")
    codes = [event.get("code") for _, event in events if event["kind"] == "ConversationErrorEvent"]
    if execution_status == "error" and codes and codes[-1] in RUN_LIMIT_CODES:
        return RunOutcome("incomplete", RUN_LIMIT_CODES[codes[-1]])
    return RunOutcome("failed", "error", f"The OpenHands run ended with status {execution_status!r}")


@dataclass
class EvidenceRecords:
    records: list[Any] = field(default_factory=list)
    gaps: list[ObservationGap] = field(default_factory=list)


def observation_records(
    events: list[tuple[float, dict[str, Any]]],
    result: Optional[dict[str, Any]],
    *,
    invocation_id: str,
    model_ref: Optional[ModelServerRef],
    conversation: list[Any],
    outcome: RunOutcome,
    sandbox_id: Optional[str] = None,
) -> EvidenceRecords:
    """The invocation, its tool calls and context compactions, and gaps for what the events do not show."""
    evidence = EvidenceRecords()

    def gap(code: str, detail: Optional[str] = None) -> None:
        evidence.gaps.append(ObservationGap(code=code, invocation_id=invocation_id, detail=detail))

    # Policy calls are the agent's own; condenser summaries belong to their compaction records, not to steps.
    compaction_ids = {
        event["llm_response_id"]
        for _, event in events
        if event["kind"] == "Condensation" and isinstance(event.get("llm_response_id"), str)
    }
    agent_ids = [usage["response_id"] for usage in token_usages(result, "agent")]
    response_ids = [
        response_id
        for response_id in dict.fromkeys([*_response_ids(events), *agent_ids])
        if response_id not in compaction_ids
    ]
    model_calls = [ModelCallRef(model_ref=model_ref, response_id=response_id) for response_id in response_ids]
    if result is None:
        gap("run_result_unavailable")
    elif not token_usages(result):
        gap("model_call_usage_unavailable")
    started, ended = (result or {}).get("started_at"), (result or {}).get("ended_at")
    duration_ms = (
        (ended - started) * 1000
        if isinstance(started, (int, float)) and isinstance(ended, (int, float)) and ended >= started
        else None
    )
    evidence.records.append(
        AgentInvocation(
            invocation_id=invocation_id,
            status=outcome.status,
            stop_reason=outcome.stop_reason,
            duration_ms=duration_ms,
            error_type=outcome.failure_kind if outcome.status != "completed" else None,
            model_calls=model_calls,
            conversation=conversation,
        )
    )

    actions: dict[str, tuple[float, str]] = {}
    results: dict[str, tuple[float, bool]] = {}
    for seen_at, event in events:
        call_id = event.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        if event["kind"] == "ActionEvent":
            actions.setdefault(call_id, (seen_at, str(event.get("tool_name"))))
        elif event["kind"] in ("ObservationEvent", "AgentErrorEvent", "UserRejectObservation"):
            results[call_id] = (seen_at, _tool_result(event)[1])
    for call_id, (started_at, tool_name) in actions.items():
        if call_id in results:
            completed_at, failed = results[call_id]
            evidence.records.append(
                ToolCallObservation(
                    invocation_id=invocation_id,
                    tool_call_id=call_id,
                    sandbox_id=sandbox_id,
                    tool_name=tool_name,
                    started_at=started_at,
                    completed_at=completed_at,
                    duration_ms=max(completed_at - started_at, 0.0) * 1000,
                    timing_source="harness",
                    status="failed" if failed else "completed",
                )
            )
        else:
            evidence.records.append(
                ToolCallObservation(
                    invocation_id=invocation_id,
                    tool_call_id=call_id,
                    sandbox_id=sandbox_id,
                    tool_name=tool_name,
                    started_at=started_at,
                    timing_source="harness",
                    status="incomplete",
                )
            )
            gap("tool_timing_unavailable", call_id)
    # An AgentErrorEvent can answer a tool call that never became an action, such as one with invalid arguments.
    for call_id in results.keys() - actions.keys():
        gap("tool_call_action_unavailable", call_id)

    last_call: Optional[ModelCallRef] = None
    pending: list[ContextCompactionObservation] = []
    for seen_at, event in events:
        response_id = event.get("llm_response_id")
        if event["kind"] in ("ActionEvent", "MessageEvent") and isinstance(response_id, str):
            if last_call is None or last_call.response_id != response_id:
                last_call = ModelCallRef(model_ref=model_ref, response_id=response_id)
                for compaction in pending:
                    compaction.after_model_call = last_call
                pending.clear()
        if event["kind"] != "Condensation":
            continue
        summary = event.get("summary")
        compaction = ContextCompactionObservation(
            invocation_id=invocation_id,
            observed_at=seen_at,
            trigger="condenser",
            outcome="completed",
            summary=summary if isinstance(summary, str) else None,
            before_model_call=last_call,
            # The condenser's summarization call; a condensation that only drops events makes none.
            model_calls=[ModelCallRef(model_ref=model_ref, response_id=response_id)]
            if isinstance(response_id, str)
            else [],
        )
        evidence.records.append(compaction)
        pending.append(compaction)
        gap("compaction_tokens_unavailable")
    for _ in pending:
        gap("compaction_after_model_call_unavailable")
    return evidence
