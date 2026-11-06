# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared verifier dispatch, dependency handling and score aggregation for all adapters."""

import asyncio
from collections.abc import Awaitable, Callable

from resources_servers.partner_if.checks import CheckContext, LLMParams, resolve
from resources_servers.partner_if.evidence import EvidenceBundle, EvidenceTooLarge, EvidenceUnavailable
from resources_servers.partner_if.grading_contract import (
    EVIDENCE_TOO_LARGE,
    DimensionScore,
    Evaluation,
    GradingContract,
    GradingItem,
    ItemResult,
    RawJudgment,
)


Judge = Callable[[list[GradingItem], EvidenceBundle], Awaitable[list[RawJudgment]]]


def too_large(items: list[GradingItem], error: EvidenceTooLarge) -> list[RawJudgment]:
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


async def evaluate(contract: GradingContract, evidence: EvidenceBundle, *, judge: Judge) -> Evaluation:
    """Score each item exactly once; errors raised by `judge`, such as a failed judge call, propagate.
    The judge is sent only the artifacts its items need. If `judge` refuses that input as too large
    (EvidenceTooLarge, before sending it), the checks that gate reward are judged alone with their own
    artifacts, and any item left out is evidence_too_large."""
    implementations = {r.id: resolve(r.verifier) for r in contract.checks}
    context = CheckContext(evidence=evidence)
    raw: dict[str, RawJudgment] = {}
    llm_items, needed_artifacts = [], {}
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
                needed_artifacts[item.id] = set(required)
                continue
            passed, detail = await asyncio.to_thread(checker.run, params, context)
            raw[item.id] = RawJudgment(id=item.id, passed=passed, reason=detail, evidence=detail)
        except EvidenceUnavailable as error:
            raw[item.id] = RawJudgment(
                id=item.id,
                passed=None,
                reason=str(error),
                evidence=str(error),
                failure_kind=EVIDENCE_TOO_LARGE if isinstance(error, EvidenceTooLarge) else None,
            )

    async def judge_items(items: list[GradingItem]) -> list[RawJudgment]:
        needed = set().union(*(needed_artifacts[r.id] for r in items))
        artifacts = {name: artifact for name, artifact in evidence.artifacts.items() if name in needed}
        judgments = await judge(items, evidence.model_copy(update={"artifacts": artifacts}))
        identifiers = [j.id for j in judgments]
        if len(set(identifiers)) != len(identifiers) or set(identifiers) != {r.id for r in items}:
            raise ValueError("Judge must return every rubric ID exactly once")
        return judgments

    if llm_items:
        try:
            judgments = await judge_items(llm_items)
        except EvidenceTooLarge as error:
            # Too large together: the checks that gate reward are judged alone, so the evidence of a
            # completion item cannot leave them unscored. The other items are too large.
            gating = reward_checks(contract)
            gating_items = [item for item in llm_items if item.id in gating]
            other_items = [item for item in llm_items if item.id not in gating]
            judgments = too_large(other_items, error)
            if gating_items and other_items:
                try:
                    judgments += await judge_items(gating_items)
                except EvidenceTooLarge as still_too_large:
                    judgments += too_large(gating_items, still_too_large)
            else:
                judgments += too_large(gating_items, error)
        raw.update({j.id: j for j in judgments})

    return aggregate(contract, list(raw.values()))


def reward_checks(contract: GradingContract) -> set[str]:
    """The checks that gate reward. The contract rejects IF dependencies on completion,
    so these are the IF rubrics and the checks they depend on, directly or transitively."""
    checks = {r.id: r for r in contract.checks}
    gating = {r.id for r in contract.rubrics if r.dimension == "if"}
    pending = list(gating)
    while pending:
        for dependency in checks[pending.pop()].depends_on:
            if dependency not in gating:
                gating.add(dependency)
                pending.append(dependency)
    return gating


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
    # Completion is diagnostic, including when it cannot be scored.
    complete = all(
        effective[i].passed is not None and effective[i].raw_passed is not None for i in reward_checks(contract)
    )
    return Evaluation(
        status="scored" if complete else "unscorable",
        items=results,
        prerequisites=prerequisites,
        dimensions=dimensions,
        reward=float(dimensions["if"].binary_score) if complete else None,
    )
