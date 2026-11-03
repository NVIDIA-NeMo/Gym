# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym verify fields shared by every partner_if adapter."""

import logging

from pydantic import JsonValue

from resources_servers.partner_if.engine import reward_checks
from resources_servers.partner_if.grading_contract import UNSCORABLE, GradingContract
from resources_servers.partner_if.grading_service import GradeResponse, contract_hash
from resources_servers.partner_if.schema import IFResult


logger = logging.getLogger(__name__)


def scoring_fields(contract: GradingContract, receipt: GradeResponse) -> dict[str, JsonValue]:
    """Expose binary reward and both score dimensions without mixing completion into IF."""
    evaluation = receipt.grading_result
    items = [item for item in evaluation.items if item.dimension == "if"]
    scored = evaluation.status == "scored"
    result = IFResult(
        task_id=contract.task_id,
        rubric_version=contract.rubric_version,
        rubric_hash=contract_hash(contract),
        status=("scored" if contract.audit_signoff else "provisional") if scored else "unscorable",
        passed=sum(item.passed is True for item in items),
        total=len(items),
        items=[
            item.model_dump(include={"id", "passed", "reason", "evidence"})
            for item in items
            if item.passed is not None
        ],
    )
    # Report the first check that gates reward and has no verdict of its own: an unscorable
    # one, or one failed for a named reason such as a missing answer.
    gating = reward_checks(contract)
    checks = {item.id: item for item in evaluation.prerequisites + evaluation.items}
    failure = next(
        (
            checks[item.id]
            for item in contract.checks
            if item.id in gating and (checks[item.id].failure_kind if scored else checks[item.id].raw_passed is None)
        ),
        None,
    )
    failure_kind = (failure.failure_kind or UNSCORABLE) if failure else None
    if not scored:
        logger.warning("partner_if masked task %s (%s): %s", contract.task_id, failure_kind, failure.reason)
    return {
        "reward": evaluation.reward or 0.0,
        "mask_sample": not scored,
        "failure_kind": failure_kind,
        "failure_reason": failure.reason if failure else None,
        "if_result": result.model_dump(mode="json"),
        **receipt.model_dump(mode="json"),
    }


def masked_fields(contract: GradingContract, failure_kind: str, reason: str) -> dict[str, JsonValue]:
    """Gym verify fields for a rollout that was not graded, so metrics leave it out."""
    logger.warning("partner_if masked task %s (%s): %s", contract.task_id, failure_kind, reason)
    result = IFResult(
        task_id=contract.task_id,
        rubric_version=contract.rubric_version,
        rubric_hash=contract_hash(contract),
        status="unscorable",
        passed=0,
        total=sum(r.dimension == "if" for r in contract.rubrics),
        items=[],
        reason=reason,
    )
    return {
        "reward": 0.0,
        "mask_sample": True,
        "failure_kind": failure_kind,
        "failure_reason": reason,
        "if_result": result.model_dump(mode="json"),
    }
