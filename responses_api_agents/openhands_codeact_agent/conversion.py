# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Turn the OpenHands CodeAct runner's logs into a Gym response and agent observations.

The runner keeps every policy call in ``completions/*.json`` (the fork's ``NemoGymClient``
completion log: request ``messages``, the served chat completion in ``response``, ``kwargs`` with
the tools, ``timestamp``), one timing record per HTTP attempt in ``model-requests.jsonl``, and one
record per context compaction in ``compactions.jsonl``. The condenser marks its summary call with
``call_kind = "compaction_summary"``.

These files are written inside the task sandbox, where the policy has a shell, so they are evidence
only. Training tokens come from Gym's model-call capture, never from these logs: the response built
here is text-only, and a compaction is reported as a ``ContextCompactionObservation`` naming its
summary call, which rollout collection uses to keep that call out of the rebuilt turns.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from nemo_gym.config import ModelServerRef
from nemo_gym.core import (
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.trajectory import ContextCompactionObservation, ModelCallRef


CALL_KIND_KEY = "call_kind"
SUMMARY_CALL_KIND = "compaction_summary"
# Fallback marker: the first line of the OpenHands LLMSummarizingCondenser prompt, which the
# compaction condenser reuses verbatim. Only consulted when the log carries no explicit kind.
SUMMARY_PROMPT_PREFIX = "You are maintaining a context-aware state summary for an interactive agent."
# Policy-limit errors OpenHands 0.62 records as its last error, and the stop reasons Gym and the GDPVal
# Resources Server read (``max_iterations`` and ``context_limit`` are judged like any other ending).
_CONTEXT_LIMIT_TEXT = "context length limit has been exceeded"
_ITERATION_LIMIT_TEXT = "Agent reached maximum iteration"
_STUCK_TEXT = "Agent got stuck in a loop"
_TRANSPORT_ERROR_TEXT = ("SocketTimeoutError", "APITimeoutError", "ClientConnectorError", "ServerDisconnectedError")


def _text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return str(content)


def _reply_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    choices = (payload.get("response") or {}).get("choices") or []
    if len(choices) != 1:
        raise ValueError(f"completion log needs exactly one choice, found {len(choices)}")
    message = dict(choices[0].get("message") or {})
    message.setdefault("role", "assistant")
    return message


def call_kind(payload: dict[str, Any]) -> str:
    """``"compaction_summary"`` for the condenser's summarization call, ``"agent"`` otherwise."""
    kind = payload.get(CALL_KIND_KEY)
    if kind:
        return str(kind)
    messages = payload.get("messages") or []
    if (
        len(messages) == 1
        and messages[0].get("role") == "user"
        and _text(messages[0].get("content")).lstrip().startswith(SUMMARY_PROMPT_PREFIX)
    ):
        return SUMMARY_CALL_KIND
    return "agent"


def load_calls(completions: Path) -> list[dict[str, Any]]:
    """Completed policy calls in dispatch order; an empty list when the run made none."""
    payloads = sorted(
        (json.loads(path.read_text()) for path in Path(completions).glob("*.json")), key=lambda x: x["timestamp"]
    )
    return [
        {
            "kind": call_kind(payload),
            "response_id": (payload.get("response") or {}).get("id"),
            "messages": [dict(m) for m in payload.get("messages") or []],
            "tools": (payload.get("kwargs") or {}).get("tools") or [],
            "reply": _reply_from_payload(payload),
            "timestamp": payload["timestamp"],
        }
        for payload in payloads
    ]


def _plain_text_content(message: dict[str, Any]) -> dict[str, Any]:
    """OpenHands sends text as a list of ``{"type": "text"}`` parts; Gym's converter reads it as a string."""
    content = message.get("content")
    if isinstance(content, list) and all(isinstance(part, dict) and part.get("type") == "text" for part in content):
        return {**message, "content": "".join(str(part.get("text") or "") for part in content)}
    return message


def output_items(calls: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """The last agent call's context after the initial prompt plus its reply, as text-only output items and tools."""
    from openai.types.responses import FunctionTool

    from nemo_gym.models import VLLMConverter, split_responses_input_output_items

    agent_calls = [call for call in calls if call["kind"] != SUMMARY_CALL_KIND]
    if not agent_calls:
        return [], []
    last = agent_calls[-1]
    messages = [_plain_text_content(message) for message in (*last["messages"], last["reply"])]
    items = VLLMConverter(return_token_id_information=False).chat_completions_messages_to_responses_items(messages)
    _, output = split_responses_input_output_items(items)
    tools = [FunctionTool.model_validate(tool["function"] | {"type": "function"}) for tool in last["tools"]]
    return output, tools


def response_usage(model_requests: list[dict[str, Any]]) -> Optional[NeMoGymResponseUsage]:
    """Total usage, or None unless every answered model call reported it."""
    answered = [record for record in model_requests if record.get("http_status") == 200]
    if not answered or any(not record.get("usage") for record in answered):
        return None
    prompt = sum(int(record["usage"].get("prompt_tokens") or 0) for record in answered)
    completion = sum(int(record["usage"].get("completion_tokens") or 0) for record in answered)
    return NeMoGymResponseUsage(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=prompt + completion,
        # The chat completions usage the runner logs carries no cached or reasoning token counts.
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=None),
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=None),
    )


def compaction_records(
    calls: list[dict[str, Any]],
    compactions: list[dict[str, Any]],
    *,
    invocation_id: str,
    model_ref: ModelServerRef,
) -> list[ContextCompactionObservation]:
    """One observation per summary call, joined to the agent calls on either side by response id."""
    notes = {note.get("response_id"): note for note in compactions if note.get("response_id")}
    records = []
    for position, call in enumerate(calls):
        if call["kind"] != SUMMARY_CALL_KIND or not call["response_id"]:
            continue
        before = next((c for c in reversed(calls[:position]) if c["kind"] != SUMMARY_CALL_KIND), None)
        after = next((c for c in calls[position + 1 :] if c["kind"] != SUMMARY_CALL_KIND), None)
        note = notes.get(call["response_id"], {})
        usage = note.get("trigger_usage") or {}
        before_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0) if usage else None

        def ref(c: Optional[dict[str, Any]]) -> Optional[ModelCallRef]:
            if c is None or not c["response_id"]:
                return None
            return ModelCallRef(model_ref=model_ref, response_id=c["response_id"])

        records.append(
            ContextCompactionObservation(
                invocation_id=invocation_id,
                observed_at=call["timestamp"],
                trigger="token_cutoff",
                tokens_before=before_tokens,
                outcome="completed",
                summary=_text(call["reply"].get("content")) or None,
                before_model_call=ref(before),
                model_calls=[ref(call)],
                after_model_call=ref(after),
            )
        )
    return records


@dataclass(frozen=True)
class RunOutcome:
    """How a run ended: the response status, the stop reason Gym reads, and the error message if it failed."""

    status: Literal["completed", "incomplete", "failed"]
    stop_reason: str
    message: str = ""


def run_outcome(result: Optional[dict[str, Any]], *, timed_out: bool, rollout_stop: bool) -> RunOutcome:
    """Classify the runner's ``result.json``; a missing result after the deadline is a timed-out run."""
    if rollout_stop:
        return RunOutcome("incomplete", "rollout_stop")
    if timed_out:
        return RunOutcome("incomplete", "wall_time")
    if result is None:
        return RunOutcome("failed", "runner_error", "OpenHands runner wrote no result")
    if result.get("finished"):
        return RunOutcome("completed", "completed")
    error = str(result.get("last_error") or "")
    if _CONTEXT_LIMIT_TEXT in error:
        return RunOutcome("incomplete", "context_limit")
    if _ITERATION_LIMIT_TEXT in error:
        return RunOutcome("incomplete", "max_iterations")
    if _STUCK_TEXT in error:
        # The controller's loop detector ended a policy that repeated itself; its work is judged.
        return RunOutcome("incomplete", "stuck_in_loop")
    return RunOutcome("failed", "error", error or f"OpenHands stopped in state {result.get('agent_state')}")


def classify_error(result: dict[str, Any], model_requests: list[dict[str, Any]]) -> tuple[str, Optional[int]]:
    """The failure kind of a failed run, from the last model call's record or the controller's error text."""
    failed = [record for record in model_requests if record.get("http_status") not in (None, 200)]
    if failed and (failed[-1] is model_requests[-1]):
        return "model_request_error", int(failed[-1]["http_status"])
    error = str(result.get("last_error") or "")
    if (model_requests and model_requests[-1].get("error_type")) or any(
        text in error for text in _TRANSPORT_ERROR_TEXT
    ):
        return "model_transport_error", None
    return "unknown", None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Records of a JSONL log, skipping lines a killed writer left incomplete."""
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records
