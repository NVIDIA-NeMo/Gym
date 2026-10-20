# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Gym model transport and receipts; no task-specific dispatch."""

import asyncio
import hashlib
import json

from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.partner_if.engine import evaluate
from resources_servers.partner_if.evidence import EvidenceBundle
from resources_servers.partner_if.grading_contract import (
    Evaluation,
    GradingContract,
    GradingItem,
    RawJudgment,
    StrictModel,
)
from resources_servers.partner_if.llm import PROTOCOL, parse_verdict, request_payload


class GradeRequest(StrictModel):
    contract: GradingContract
    evidence: EvidenceBundle


class GradeResponse(StrictModel):
    task_id: str
    rubric_hash: str
    grading_result: Evaluation
    judge_protocol: str = PROTOCOL
    judge_request: dict | None = None
    judge_response: NeMoGymResponse | None = None


def contract_hash(contract: GradingContract) -> str:
    return hashlib.sha256(
        json.dumps(contract.model_dump(), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


async def grade(
    body: GradeRequest,
    *,
    server_client: ServerClient,
    server_name: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
) -> GradeResponse:
    """All evidence adapters share this judge transport and score engine."""
    receipt_request = None
    receipt_response = None

    async def judge(items: list[GradingItem], evidence: EvidenceBundle) -> list[RawJudgment]:
        nonlocal receipt_request, receipt_response
        receipt_request = request_payload(items, evidence, max_tokens=max_tokens, temperature=temperature)
        try:
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
        except (ValueError, JudgeError) as error:
            # Preserve the actual request/reply even when validation fails. These
            # are technical/protocol errors, never ordinary model failures or
            # valid judge abstentions.
            return [
                RawJudgment(
                    id=item.id,
                    passed=None,
                    reason=f"Invalid shared IF judge response: {error}",
                    evidence="Judge validation failed; original receipts retained for review",
                )
                for item in items
            ]

    result = await evaluate(body.contract, body.evidence, judge=judge)
    return GradeResponse(
        task_id=body.contract.task_id,
        rubric_hash=contract_hash(body.contract),
        grading_result=result,
        judge_request=receipt_request,
        judge_response=receipt_response,
    )
