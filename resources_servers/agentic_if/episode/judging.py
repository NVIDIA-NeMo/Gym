# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep each judge call's reply ID, model and token usage, and tell a judge outage apart from a judge that answered.

Both stay outside ``if_grading``, whose sources make up the scorer's identity: it keeps only a reply's text and turns a
vote without a verdict into an ``error`` decision, whatever the cause.

The Resources Server asks the judge outside the rollout's context, so the judge's calls are not in the rollout's
model-call capture: the ledger is their only record. Per grade, it keeps every call's reply ID, the model that
answered (a model server may serve another model than the judge configuration names) and token usage, or why
the call got no reply, and which judge questions any request reached. A check that errored on a question no request
reached is a judge outage: verify raises ``JudgeError`` so ``judge_failsafe`` records ``judge_failed``, which
re-verification recovers. An unparseable reply stays a graded check outcome.
"""

import hashlib
import json
import threading
from collections.abc import Mapping
from typing import Any

from aiohttp import ClientError, ClientResponseError
from pydantic import BaseModel, ConfigDict

from nemo_gym.judge import JudgeError
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.agentic_if.if_grading.contracts import IFResult


def vote_key(judge_hash: str, prompt: str, vote: object) -> str:
    """The cache key ``if_grading.judge.Judge`` records on each vote."""
    return hashlib.sha256(json.dumps([judge_hash, prompt, vote], ensure_ascii=False).encode()).hexdigest()


def judge_prompt(body: Mapping[str, Any]) -> str:
    """The question of one judge request: its single user message."""
    return str(body["input"][0]["content"])


def judge_unreachable(error: JudgeError) -> bool:
    """A failed call (transport, timeout, HTTP status), not a reply that could not be parsed."""
    return isinstance(error.__cause__, (ClientError, TimeoutError, OSError))


def _failure(error: JudgeError) -> str:
    """Why a call got no reply: the exception type, and the HTTP status of a failed call; never an error body."""
    cause = error.__cause__ if error.__cause__ is not None else error
    if isinstance(cause, ClientResponseError):
        return f"{type(cause).__name__} (HTTP {cause.status})"
    return type(cause).__name__


class JudgeCallSummary(BaseModel):
    """One judge call as a verify result reports it: the reply's ID, model and token usage (None where missing)."""

    model_config = ConfigDict(extra="forbid")

    response_id: str | None = None
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None


class JudgeCall(JudgeCallSummary):
    """One judge call as the episode record keeps it: also its question and, without a usable reply, why."""

    # The judge question's key: the SHA-256 of its prompt. A grade asks one question at a time, so the calls of one
    # question are in the order it was asked; each of its votes in ``if_result`` says in ``metadata.attempts`` how
    # many calls it took.
    prompt_sha256: str
    error: str | None = None

    def summary(self) -> JudgeCallSummary:
        """The fields a verify result reports."""
        return JudgeCallSummary.model_validate(self.model_dump(include=set(JudgeCallSummary.model_fields)))


class JudgeUsage(BaseModel):
    """Totals over the judge calls of one grade."""

    model_config = ConfigDict(extra="forbid")

    calls: int = 0
    # Calls that got no usable reply: no reply ID and no tokens.
    failed_calls: int = 0
    # Replies that carried no token usage.
    calls_without_usage: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0


def judge_usage(calls: list[JudgeCall]) -> JudgeUsage:
    """The totals of ``calls``; a count a reply did not report adds nothing."""
    return JudgeUsage(
        calls=len(calls),
        failed_calls=sum(call.error is not None for call in calls),
        calls_without_usage=sum(call.error is None and call.total_tokens is None for call in calls),
        input_tokens=sum(call.input_tokens or 0 for call in calls),
        output_tokens=sum(call.output_tokens or 0 for call in calls),
        reasoning_tokens=sum(call.reasoning_tokens or 0 for call in calls),
        total_tokens=sum(call.total_tokens or 0 for call in calls),
    )


class JudgeLedger:
    """The judge calls of one grade, and which questions reached the judge; calls may come from several threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: list[JudgeCall] = []
        self._sent: set[str] = set()
        self._answered: set[str] = set()

    def replied(self, prompt: str, reply: NeMoGymResponse) -> None:
        """Record a call for ``prompt`` that got ``reply``."""
        usage = reply.usage
        call = JudgeCall(
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            response_id=reply.id or None,
            model=reply.model or None,
            input_tokens=usage.input_tokens if usage is not None else None,
            output_tokens=usage.output_tokens if usage is not None else None,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens if usage is not None else None,
            total_tokens=usage.total_tokens if usage is not None else None,
        )
        self._record(prompt, call, reached=True)

    def failed(self, prompt: str, error: JudgeError) -> None:
        """Record a call for ``prompt`` that got no usable reply."""
        call = JudgeCall(prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), error=_failure(error))
        self._record(prompt, call, reached=not judge_unreachable(error))

    def _record(self, prompt: str, call: JudgeCall, *, reached: bool) -> None:
        with self._lock:
            self._calls.append(call)
            self._sent.add(prompt)
            if reached:
                self._answered.add(prompt)

    def calls(self) -> list[JudgeCall]:
        """The calls so far, in the order they were made."""
        with self._lock:
            return [call.model_copy() for call in self._calls]

    def outage(self, result: IFResult, judge_hash: str) -> str | None:
        """Why a check errored because the judge could not be reached, or None if no check did."""
        with self._lock:
            unanswered = self._sent - self._answered
        if not unanswered:
            return None
        for check in result.checks:
            if check.status != "error":
                continue
            # The votes of the graded units and of a prefix judgment that got no verdict (``prefix_errors``).
            votes = [
                vote
                for section in ("units", "prefix_errors")
                for unit in check.details.get(section) or []
                if isinstance(unit, dict)
                for vote in (unit.get("metadata") or {}).get("votes") or []
                if isinstance(vote, dict)
            ]
            for vote in votes:
                metadata = vote.get("metadata") or {}
                if vote.get("status") == "error" and any(
                    vote_key(judge_hash, prompt, metadata.get("vote")) == metadata.get("key") for prompt in unanswered
                ):
                    return f"check {check.id}: {vote.get('explanation')}"
        return None
