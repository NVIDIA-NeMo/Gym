# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Gym model transport and receipts; no task-specific dispatch."""

import asyncio
import hashlib
import json

from nemo_gym.judge import call_judge
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from resources_servers.partner_if.engine import aggregate, evaluate
from resources_servers.partner_if.evidence import EvidenceBundle, EvidenceTooLarge
from resources_servers.partner_if.grading_contract import (
    EVIDENCE_TOO_LARGE,
    Evaluation,
    GradingContract,
    GradingItem,
    RawJudgment,
    StrictModel,
)
from resources_servers.partner_if.llm import parse_verdict, request_payload


class GradeResponse(StrictModel):
    grading_result: Evaluation
    judge_request: dict | None = None
    judge_response: NeMoGymResponse | None = None


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

    async def judge(items: list[GradingItem], evidence: EvidenceBundle) -> list[RawJudgment]:
        nonlocal receipt_request, receipt_response
        try:
            receipt_request = request_payload(
                items, evidence, responses_create_params, max_input_chars=max_input_chars
            )
            async with semaphore:
                receipt_response = await call_judge(
                    server_client,
                    server_name=server_name,
                    url_path="/v1/responses",
                    json=receipt_request,
                    response_model=NeMoGymResponse,
                )
            if receipt_response.status != "completed" or receipt_response.error or receipt_response.incomplete_details:
                raise ValueError("Judge response was incomplete")
            if any(item.type not in {"message", "reasoning"} for item in receipt_response.output):
                raise ValueError("Judge returned unexpected tool output")
            return parse_verdict(receipt_response.output_text, items, evidence)
        except EvidenceTooLarge as error:
            # Nothing is sent: these items are unscorable rather than judged on truncated evidence.
            return [
                RawJudgment(
                    id=item.id,
                    passed=None,
                    reason=str(error),
                    evidence="Evidence too large for the judge",
                    failure_kind=EVIDENCE_TOO_LARGE,
                )
                for item in items
            ]
        except ValueError as error:
            # A reply that fails protocol or validation checks leaves these items
            # unscorable and keeps the actual request/reply. A failed judge call
            # raises JudgeError, which propagates to Gym's judge_failsafe.
            return [
                RawJudgment(
                    id=item.id,
                    passed=None,
                    reason=f"Invalid shared IF judge response: {error}",
                    evidence="Judge validation failed; original receipts retained for review",
                )
                for item in items
            ]

    result = await evaluate(contract, evidence, judge=judge)
    return GradeResponse(grading_result=result, judge_request=receipt_request, judge_response=receipt_response)


def unanswered(contract: GradingContract, failure_kind: str, reason: str) -> GradeResponse:
    """With no answer to grade, every check fails without the judge, so a prohibition cannot pass by omission."""
    judgments = [
        RawJudgment(id=item.id, passed=False, reason=reason, evidence="No answer to grade", failure_kind=failure_kind)
        for item in contract.checks
    ]
    return GradeResponse(grading_result=aggregate(contract, judgments))
