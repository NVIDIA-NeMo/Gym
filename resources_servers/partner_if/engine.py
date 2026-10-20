# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared verifier dispatch, dependency handling and score aggregation for all adapters."""

import asyncio
from collections.abc import Awaitable, Callable

from resources_servers.partner_if.checks import CheckContext, LLMParams, resolve
from resources_servers.partner_if.evidence import EvidenceBundle, EvidenceUnavailable
from resources_servers.partner_if.grading_contract import (
    DimensionScore,
    Evaluation,
    GradingContract,
    GradingItem,
    ItemResult,
    RawJudgment,
)


Judge = Callable[[list[GradingItem], EvidenceBundle], Awaitable[list[RawJudgment]]]


async def evaluate(contract: GradingContract, evidence: EvidenceBundle, *, judge: Judge) -> Evaluation:
    """Score each item exactly once; judge errors propagate to Gym's infrastructure masking."""
    implementations = {r.id: resolve(r.verifier) for r in contract.checks}
    context = CheckContext(evidence=evidence, parsed={})
    raw: dict[str, RawJudgment] = {}
    llm_items = []
    for item in contract.checks:
        if evidence.trajectory_error:
            raw[item.id] = RawJudgment(
                id=item.id,
                passed=None,
                reason=evidence.trajectory_error,
                evidence="Actor execution or trajectory capture is incomplete",
            )
            continue
        checker, params = implementations[item.id]
        try:
            if checker.run is None:
                assert isinstance(params, LLMParams)
                # A completion-only capture gap must not suppress an independent
                # IF check. Missing receipts are infrastructure errors, whereas
                # an explicit absent receipt is ordinary gradable model behavior.
                required = evidence.artifacts if params.artifacts is None else params.artifacts
                for name in required:
                    evidence.artifact_text(name)
                llm_items.append(item)
                continue
            passed, detail = await asyncio.to_thread(checker.run, params, context)
            raw[item.id] = RawJudgment(id=item.id, passed=passed, reason=detail, evidence=detail)
        except EvidenceUnavailable as error:
            raw[item.id] = RawJudgment(id=item.id, passed=None, reason=str(error), evidence=str(error))

    if llm_items:
        judgments = await judge(llm_items, evidence)
        identifiers = [j.id for j in judgments]
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != {r.id for r in llm_items}:
            raise ValueError("Judge must return every rubric ID exactly once")
        raw.update({j.id: j for j in judgments})

    return aggregate(contract, list(raw.values()))


def aggregate(contract: GradingContract, judgments: list[RawJudgment]) -> Evaluation:
    """Recompute both dimensions from raw verdicts without mixing their availability."""
    raw = {item.id: item for item in judgments}
    if len(raw) != len(judgments) or set(raw) != {item.id for item in contract.checks}:
        raise ValueError("Judgments must cover the contract exactly once")
    effective: dict[str, ItemResult] = {}
    items = {r.id: r for r in contract.checks}

    def apply(identifier: str) -> ItemResult:
        if identifier in effective:
            return effective[identifier]
        rubric, judgment = items[identifier], raw[identifier]
        dependencies = [apply(dep) for dep in rubric.depends_on]
        blocked = [dep.id for dep in dependencies if dep.passed is False]
        unavailable = [dep.id for dep in dependencies if dep.passed is None]
        passed = None if unavailable else False if blocked else judgment.passed
        result = ItemResult(
            **(judgment.model_dump() | {"passed": passed}),
            raw_passed=judgment.passed,
            verifier=rubric.verifier.id,
            dimension=rubric.dimension,
            blocked_by=blocked + unavailable,
        )
        effective[identifier] = result
        return result

    results = [apply(r.id) for r in contract.rubrics]
    prerequisites = [apply(r.id) for r in contract.prerequisites]
    dimensions = {}
    for dimension in dict.fromkeys(r.dimension for r in contract.rubrics):
        group = [r for r in results if r.dimension == dimension]
        earned = None if any(r.passed is None or r.raw_passed is None for r in group) else sum(r.passed for r in group)
        dimensions[dimension] = DimensionScore(
            points_earned=earned,
            points_possible=len(group),
            partial_score=earned / len(group) if earned is not None else None,
            binary_score=int(earned == len(group)) if earned is not None else None,
        )
    # Completion is diagnostic, including when it cannot be scored. The contract
    # rejects IF dependencies on completion, so only IF availability gates reward.
    complete = all(
        r.passed is not None and r.raw_passed is not None for r in results + prerequisites if r.dimension == "if"
    )
    return Evaluation(
        status="scored" if complete else "unscorable",
        items=results,
        prerequisites=prerequisites,
        dimensions=dimensions,
        reward=float(dimensions["if"].binary_score) if complete else None,
    )
