# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How one episode ended, and which failure, if any, its verify result reports.

Execution health stays separate from the reward: the reward is the task backend's verdict, and the failure fields say
whether that verdict is a valid measurement (``mask_sample``) and why not.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import JsonValue, ValidationError

from nemo_gym.agent_outcome import AGENT_OUTCOME_METADATA_KEY, AgentOutcome
from nemo_gym.failure_kinds import (
    AGENT_RUN_ERROR,
    AGENT_TIMEOUT,
    FAILURE_KINDS,
    MODEL_CONTEXT_LIMIT,
    SESSION_LOST,
    VERIFIER_ERROR,
)
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence


ExecutionStatus = Literal["completed", "length", "timeout", "error"]

EVIDENCE_INCOMPLETE = "agentic_if:evidence_incomplete"
# The profile plugin fails closed: without its load receipt, nothing shows the actor saw the task's tool setup.
MISSING_LOAD_RECEIPT = "hook_receipt_error:missing live hook load receipt"
# Issues that leave the actor's setup unshown: the plugin bundle's gaps about the plugins' load and configuration
# receipts (opencode_bundle.collect_bundle), a model call without the row's system prompt
# (opencode_bundle.presentation_gaps), and a root session that did not start from the row's task text.
_SETUP_ISSUES = (
    "bundle_gap:profile_",
    "bundle_gap:boundary_",
    "boundary_system_mismatch",
    "system_prompt_not_presented:",
    "literal_task_not_presented:",
)
# OpenCode 1.17.11's error name for a context-window overflow on the last turn (as the OpenCode agent reads it).
CONTEXT_OVERFLOW_ERROR = "ContextOverflowError"


def setup_unverified(evidence: SessionEvidence) -> bool:
    """Whether nothing shows the actor ran with the task's plugins, system prompt and task: then nothing is measured."""
    return any(issue == MISSING_LOAD_RECEIPT or issue.startswith(_SETUP_ISSUES) for issue in evidence.issues)


def _assistants(root_export: Mapping[str, JsonValue] | None) -> list[Mapping[str, JsonValue]]:
    messages = (root_export or {}).get("messages")
    return [
        message["info"]
        for message in (messages if isinstance(messages, list) else [])
        if isinstance(message, Mapping)
        and isinstance(message.get("info"), Mapping)
        and message["info"].get("role") == "assistant"
    ]


def execution_status(
    response: NeMoGymResponse, root_export: Mapping[str, JsonValue] | None, evidence: SessionEvidence
) -> ExecutionStatus:
    """How the harness run ended, from the harness's response and the root session's OpenCode export.

    The OpenCode agent reports a harness timeout as an incomplete response. The root export tells a timeout apart from
    a stop at the output-token limit (``length``), an error the last turn recorded, and a content filter, which counts
    as completed, as it did for the episode's own driver. A run that ended by itself keeps that driver's checks: no
    assistant turn, a turn error or incomplete evidence is an ``error``.
    """
    assistants = _assistants(root_export)
    last = assistants[-1] if assistants else {}
    if response.status not in ("completed", "incomplete"):
        return "error"
    if response.status == "incomplete" and not (
        last.get("error") or last.get("finish") in ("length", "content-filter")
    ):
        return "timeout"
    if not assistants or last.get("error") or evidence.status != "complete":
        return "error"
    if last.get("finish") == "length":
        return "length"
    return "completed"


@dataclass(frozen=True)
class Verdict:
    """The task reward and the failure fields of one verify result."""

    reward: float
    mask_sample: bool
    evaluation_completed: bool
    failure_kind: str | None = None
    failure_reason: str | None = None

    def fields(self) -> dict[str, JsonValue]:
        """The verify-response fields, with the failure fields only when there is a failure."""
        result: dict[str, JsonValue] = {
            "reward": self.reward,
            "mask_sample": self.mask_sample,
            "evaluation_completed": self.evaluation_completed,
        }
        if self.failure_kind is not None:
            result |= {"failure_kind": self.failure_kind, "failure_reason": self.failure_reason}
        return result


def _reward(result: Mapping[str, JsonValue] | None) -> float:
    reward = (result or {}).get("reward")
    return float(reward) if isinstance(reward, (int, float)) and not isinstance(reward, bool) else 0.0


def _agent_failure_kind(response: NeMoGymResponse) -> str:
    raw = (response.metadata or {}).get(AGENT_OUTCOME_METADATA_KEY)
    if raw is not None:
        try:
            kind = AgentOutcome.model_validate_json(raw).failure_kind
        except ValidationError:
            kind = None
        if kind in FAILURE_KINDS:
            return kind
    return AGENT_RUN_ERROR


def _agent_failure_reason(response: NeMoGymResponse, last: Mapping[str, JsonValue]) -> str:
    if response.error is not None and response.error.message:
        return response.error.message
    error = last.get("error")
    if isinstance(error, Mapping):
        data = error.get("data") if isinstance(error.get("data"), Mapping) else {}
        return str(data.get("message") or error.get("name") or "OpenCode recorded an error")
    return f"OpenCode run ended with status {response.status!r}"


def classify(
    *,
    response: NeMoGymResponse,
    root_export: Mapping[str, JsonValue] | None,
    evidence: SessionEvidence,
    execution: ExecutionStatus,
    backend_result: Mapping[str, JsonValue] | None,
    backend_error: str | None,
    collection_error: str | None,
) -> Verdict:
    """The reward and failure fields, by the first rule that applies.

    1. The bundle could not be read from the borrowed sandbox: ``session_lost``, masked.
    2. The task backend's verification failed, was masked by the backend or did not complete: ``verifier_error`` (or
       the backend's own failure kind), masked.
    3. The harness hit its timeout, or the model its context window: scored, as ``agent_timeout`` or
       ``model_context_limit``. The task verdict measures the final state; IF is graded only from complete evidence,
       which an interrupted tail usually is not. A setup the evidence does not show (the plugins' receipts, the row's
       system prompt on every model call, the row's task text) still masks the row, through rule 4.
    4. The evidence is incomplete: ``agentic_if:evidence_incomplete``, masked.
    5. The harness or model ended in another error: masked, with the agent's reported kind (``agent_run_error`` when
       it reported none).

    A masked row's reward is still a number, and it carries ``evaluation_completed: false`` when the measurement is
    missing (rules 1, 2 and 4).
    """
    reward = _reward(backend_result)
    if collection_error is not None:
        return Verdict(reward, True, False, SESSION_LOST, collection_error[:2000])
    if backend_error is not None or backend_result is None:
        return Verdict(reward, True, False, VERIFIER_ERROR, (backend_error or "Task verification did not run")[:2000])
    measured = backend_result.get("reward")
    if isinstance(measured, bool) or not isinstance(measured, (int, float)):
        return Verdict(reward, True, False, VERIFIER_ERROR, "The task verifier returned no numeric reward")
    if backend_result.get("mask_sample") is True:
        kind = backend_result.get("failure_kind")
        reason = backend_result.get("failure_reason") or backend_result.get("error") or "The task verifier masked it"
        return Verdict(
            reward,
            True,
            backend_result.get("evaluation_completed") is True,
            kind if isinstance(kind, str) and kind else VERIFIER_ERROR,
            str(reason)[:2000],
        )
    if backend_result.get("evaluation_completed") is False:
        reason = (
            backend_result.get("failure_reason") or backend_result.get("error") or "Task verification did not complete"
        )
        return Verdict(reward, True, False, VERIFIER_ERROR, str(reason)[:2000])
    assistants = _assistants(root_export)
    last = assistants[-1] if assistants else {}
    error = last.get("error")
    context_limit = isinstance(error, Mapping) and error.get("name") == CONTEXT_OVERFLOW_ERROR
    if (execution == "timeout" or context_limit) and not setup_unverified(evidence):
        if execution == "timeout":
            return Verdict(
                reward,
                False,
                True,
                AGENT_TIMEOUT,
                "OpenCode stopped at its harness timeout; the final state was verified",
            )
        return Verdict(reward, False, True, MODEL_CONTEXT_LIMIT, _agent_failure_reason(response, last)[:2000])
    if evidence.status != "complete":
        return Verdict(reward, True, False, EVIDENCE_INCOMPLETE, "; ".join(evidence.issues)[:2000])
    if execution == "error":
        return Verdict(reward, True, True, _agent_failure_kind(response), _agent_failure_reason(response, last)[:2000])
    return Verdict(reward, False, True)
