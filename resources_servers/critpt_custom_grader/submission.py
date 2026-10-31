# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract acknowledged terminal code submissions from saved agent trajectories."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.core import AGENT_LIMIT


_PROTOCOL = "critpt-finish-v1"
_FENCE = re.compile(r"```([^\r\n]*)\r?\n(.*?)\r?\n```", re.DOTALL)


class FinishRequest(BaseModel):
    """The terminal tool's arguments; an empty answer is an explicit unanswered attempt."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)

    reason: str
    derivation: str
    answer: str

    @field_validator("reason", "derivation", "answer")
    @classmethod
    def utf8_text(cls, value: str) -> str:
        """Reject lone surrogates before acknowledgement hashing or transcript serialization."""
        value.encode("utf-8")
        return value


def finish_acknowledgement(body: FinishRequest) -> str:
    """Bind a successful tool result to the exact submitted answer without repeating its contents."""
    return json.dumps(
        {"protocol": _PROTOCOL, "answer_sha256": hashlib.sha256(body.answer.encode("utf-8")).hexdigest()},
        separators=(",", ":"),
        sort_keys=True,
    )


@dataclass(frozen=True)
class Submission:
    """An extracted source awaiting grading, a scored no-answer, or unusable trajectory evidence."""

    source: str | None
    category: str
    scored: bool


_NO_ANSWER = Submission(None, "candidate_no_answer", True)
_INVALID = Submission(None, "submission_invalid", False)
_INCOMPLETE = Submission(None, "response_incomplete", False)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> Any:
    raise ValueError("nonfinite JSON constant")


def _object(text: str) -> dict[str, Any]:
    if type(text) is not str:
        raise ValueError("expected JSON text")
    value = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if type(value) is not dict:
        raise ValueError("expected JSON object")
    return value


def _termination(response: dict[str, Any]) -> Submission | None:
    """Preserve failure evidence before considering a source or a scored no-answer.

    Declared response-budget exhaustion or an explicit content-filter outcome establishes a no-answer attempt.
    A wall deadline, cancellation, provider failure or undifferentiated limit does not. HTTP model rejection codes
    alone likewise do not distinguish a model refusal from a request/configuration fault.
    """
    metadata = response.get("metadata")
    if metadata is None:
        metadata = {}
    if type(metadata) is not dict:
        return _INVALID
    outcome = None
    if AGENT_OUTCOME_METADATA_KEY in metadata:
        try:
            outcome = AgentOutcome.model_validate(_object(metadata[AGENT_OUTCOME_METADATA_KEY]))
        except (ValueError, TypeError, RecursionError):
            return _INVALID
        if outcome.failure_kind != AGENT_LIMIT:
            return _INCOMPLETE
    if response.get("error") or metadata.get("stop_reason") in ("cancelled", "wall_time"):
        return _INCOMPLETE
    status = response.get("status")
    details = response.get("incomplete_details")
    if status == "completed":
        return _INCOMPLETE if outcome is not None or details else None
    if status == "incomplete" or (status == "failed" and outcome is not None):
        if metadata.get("stop_reason") == "max_turns" or (
            type(details) is dict and details.get("reason") in ("max_output_tokens", "content_filter")
        ):
            return _NO_ANSWER
    return _INCOMPLETE


def extract_finish_source(response: dict[str, Any]) -> Submission:
    """Extract one uniquely paired, acknowledged finish from a completed trajectory.

    A valid-looking call alone is not evidence that the tool accepted it. Acknowledgements belong to the tool
    results, must follow their calls, and bind the exact answer bytes. Duplicate IDs or conflicting terminal
    records cannot be resolved by taking whichever value occurred last. These checks assume a harness-produced
    trajectory; the acknowledgement is an integrity binding, not authentication of a caller-supplied transcript.
    """
    termination = _termination(response)
    if termination is not None and termination != _NO_ANSWER:
        return termination
    output = response.get("output", [])
    if type(output) is not list:
        return _INVALID
    calls: dict[str, tuple[int, dict[str, Any]]] = {}
    results: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, item in enumerate(output):
        if type(item) is not dict:
            return _INVALID
        kind = item.get("type")
        if kind not in ("function_call", "function_call_output"):
            continue
        call_id = item.get("call_id")
        if type(call_id) is not str or not call_id:
            return _INVALID
        if kind == "function_call":
            if call_id in calls:
                return _INVALID
            calls[call_id] = (index, item)
        else:
            if call_id in results or call_id not in calls:
                return _INVALID
            results[call_id] = (index, item)

    accepted: list[tuple[int, FinishRequest]] = []
    for call_id, (_, call) in calls.items():
        if call.get("name") != "finish" or call_id not in results:
            continue
        result_index, result = results[call_id]
        reply = result.get("output")
        # Rejected calls contain ordinary tool-error text, not the acknowledgement protocol.
        if type(reply) is not str or not reply.lstrip().startswith("{"):
            continue
        try:
            acknowledgement = _object(reply)
        except (ValueError, TypeError, RecursionError):
            return _INVALID
        if acknowledgement.get("protocol") != _PROTOCOL:
            continue
        if call.get("status") not in (None, "completed") or result.get("status") not in (None, "completed"):
            return _INVALID
        try:
            body = FinishRequest.model_validate(_object(call.get("arguments")))
        except (ValueError, TypeError, RecursionError):
            return _INVALID
        if acknowledgement != json.loads(finish_acknowledgement(body)):
            return _INVALID
        accepted.append((result_index, body))

    if not accepted:
        return _NO_ANSWER
    if len(accepted) != 1 or termination is not None:
        return _INVALID
    result_index, body = accepted[0]
    # Stirrup can execute other calls from the same finishing response. Their results may follow the finish
    # acknowledgement, but another model turn issuing a new call after it contradicts terminal completion.
    if any(index > result_index for index, _ in calls.values()):
        return _INVALID
    # Consume unsupported blocks too, so their closing fences cannot open a later bare block.
    matches = [match for match in _FENCE.finditer(body.answer) if match.group(1).rstrip(" \t") in ("", "python")]
    source = (matches[-1].group(2) if matches else body.answer).strip()
    return Submission(source, "candidate_ready", False) if source else _NO_ANSWER
