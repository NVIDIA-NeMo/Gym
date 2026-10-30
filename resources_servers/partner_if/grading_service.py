# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Gym model transport and receipts; no task-specific dispatch."""

import asyncio
import hashlib
import json

from pydantic import Field

from nemo_gym.core import JUDGE_UNPARSEABLE, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, rollout_context
from nemo_gym.judge import call_judge
from nemo_gym.server_utils import ServerClient
from resources_servers.partner_if.engine import aggregate, evaluate, reward_checks
from resources_servers.partner_if.evidence import EvidenceBundle, EvidenceTooLarge
from resources_servers.partner_if.grading_contract import (
    Evaluation,
    GradingContract,
    GradingItem,
    RawJudgment,
    StrictModel,
)
from resources_servers.partner_if.llm import parse_verdict, request_payload


class JudgeRejection(StrictModel):
    """A judge reply that failed validation, and why. The judge was asked again."""

    reason: str
    response: NeMoGymResponse


class GradeResponse(StrictModel):
    grading_result: Evaluation
    judge_request: dict | None = None
    judge_response: NeMoGymResponse | None = None
    # Replies before `judge_response` that failed validation, in order.
    judge_rejected_responses: list[JudgeRejection] = Field(default_factory=list, exclude_if=lambda value: not value)


def contract_hash(contract: GradingContract) -> str:
    return hashlib.sha256(
        json.dumps(contract.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


async def grade(
    contract: GradingContract,
    evidence: EvidenceBundle,
    *,
    server_client: ServerClient,
    server_name: str,
    semaphore: asyncio.Semaphore,
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming,
    max_input_chars: int,
) -> GradeResponse:
    """All evidence adapters share this judge transport and score engine."""
    receipt_request = None
    receipt_response = None
    rejected: list[JudgeRejection] = []
    gating = reward_checks(contract)

    def unparseable(items: list[GradingItem], error: ValueError) -> list[RawJudgment]:
        # These items are unscorable, as Gym's judge_unparseable, and the row keeps the actual requests and replies.
        # A failed judge call raises JudgeError instead, which propagates to Gym's judge_failsafe.
        return [
            RawJudgment(
                id=item.id,
                passed=None,
                reason=f"Invalid shared IF judge response: {error}",
                evidence="Judge validation failed; original receipts retained for review",
                failure_kind=JUDGE_UNPARSEABLE,
            )
            for item in items
        ]

    def validated(response: NeMoGymResponse, items: list[GradingItem], evidence: EvidenceBundle) -> list[RawJudgment]:
        if response.status != "completed" or response.error or response.incomplete_details:
            raise ValueError("Judge response was incomplete")
        if any(item.type not in {"message", "reasoning"} for item in response.output):
            raise ValueError("Judge returned unexpected tool output")
        return parse_verdict(response.output_text, items, evidence, gating=gating)

    async def ask(request: dict) -> NeMoGymResponse:
        nonlocal receipt_response
        # The judge grades the rollout and is not part of it: outside the rollout context its call stays out of
        # the rollout's model-call capture, which gym-p0/v1 expects to hold only agent-owned calls. The
        # request and replies stay on the row as receipts.
        async with semaphore:
            with rollout_context(None):
                receipt_response = await call_judge(
                    server_client,
                    server_name=server_name,
                    url_path="/v1/responses",
                    json=request,
                    response_model=NeMoGymResponse,
                )
        return receipt_response

    async def judge(items: list[GradingItem], evidence: EvidenceBundle) -> list[RawJudgment]:
        nonlocal receipt_request
        try:
            receipt_request = request_payload(
                items, evidence, responses_create_params, max_input_chars=max_input_chars
            )
        except EvidenceTooLarge:
            # Nothing was sent; the engine decides which items the judge grades instead.
            raise
        except ValueError as error:
            return unparseable(items, error)
        # A reply that fails the protocol or validation checks gets one more try, and the second reply decides.
        first = await ask(receipt_request)
        try:
            return validated(first, items, evidence)
        except ValueError as error:
            rejected.append(JudgeRejection(reason=str(error), response=first))
        try:
            return validated(await ask(receipt_request), items, evidence)
        except ValueError as error:
            return unparseable(items, error)

    result = await evaluate(contract, evidence, judge=judge)
    return GradeResponse(
        grading_result=result,
        judge_request=receipt_request,
        judge_response=receipt_response,
        judge_rejected_responses=rejected,
    )


def unanswered(contract: GradingContract, failure_kind: str, reason: str) -> GradeResponse:
    """With no answer to grade, every check fails without the judge, so a prohibition cannot pass by omission."""
    judgments = [
        RawJudgment(id=item.id, passed=False, reason=reason, evidence="No answer to grade", failure_kind=failure_kind)
        for item in contract.checks
    ]
    return GradeResponse(grading_result=aggregate(contract, judgments))
