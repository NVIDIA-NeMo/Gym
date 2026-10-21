# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym verify fields shared by every partner_if adapter."""

from pydantic import JsonValue

from resources_servers.partner_if.grading_contract import GradingContract
from resources_servers.partner_if.grading_service import GradeResponse, contract_hash
from resources_servers.partner_if.schema import IFResult


def scoring_fields(contract: GradingContract, receipt: GradeResponse) -> dict[str, JsonValue]:
    """Expose binary reward and both score dimensions without mixing completion into IF."""
    if receipt.task_id != contract.task_id or receipt.rubric_hash != contract_hash(contract):
        raise ValueError("Grading receipt does not match the task contract")
    evaluation = receipt.grading_result
    items = [item for item in evaluation.items if item.dimension == "if"]
    scored = evaluation.status == "scored"
    result = IFResult(
        task_id=contract.task_id,
        rubric_version=contract.rubric_version,
        rubric_hash=receipt.rubric_hash,
        status=("scored" if contract.audit_signoff else "provisional") if scored else "unscorable",
        passed=sum(item.passed is True for item in items),
        total=len(items),
        items=[
            item.model_dump(include={"id", "passed", "reason", "evidence"})
            for item in items
            if item.passed is not None
        ],
    )
    return {
        "reward": evaluation.reward or 0.0,
        "mask_sample": not scored,
        "failure_kind": None if scored else "partner_if:unscorable",
        "failure_reason": None if scored else "Incomplete evidence or judge protocol failure prevents a score",
        "if_result": result.model_dump(mode="json"),
        **receipt.model_dump(mode="json", exclude={"task_id", "rubric_hash"}),
    }
