# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate resolved checks against the session evidence an Environment Server records."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

from pydantic import JsonValue

from .checkers import calls, language, text
from .checkers.run import Facts, Step, call_matches, evidence_steps, facts_from_steps, select
from .contracts import CheckResult, IFResult, JudgeDecision, ResolvedCheck, SessionEvidence
from .instructions import captured_instruction
from .judge import Judge
from .provenance import fingerprint, scorer_identity
from .rendering import changed_files, render, render_options


TEXT_CHECKERS = {
    "length": text.check_length,
    "element_absent": text.check_element_absent,
    "element_present": text.check_element_present,
    "ending": text.check_ending,
    "exact": text.check_exact,
    "json": text.check_json,
}
RUN_CHECKERS = {
    "calls": calls.check_calls,
    "order": calls.check_order,
    "args": calls.check_args,
    "after_event": calls.check_after_event,
}
SETTLED_CALL_MODES = {"never", "only", "max", "per_message_max"}
PREFIX_QUESTION = (
    "{question}\n\nThe run shown is only the first part of the session: it goes on after the last step "
    "shown, and whatever the rule asks of later steps or of the final message is still to come. Judge "
    "this part alone. Answer NO only if something in it already breaks the rule in a way that no later "
    "step could repair; otherwise answer YES."
)


@dataclass(frozen=True)
class Unit:
    """One graded unit of a check: a place, the step it grades (None for a whole run) and its decision."""

    where: str
    index: int | None
    decision: JudgeDecision


def _decision(ok: bool, why: str) -> JudgeDecision:
    return JudgeDecision(status="pass" if ok else "fail", explanation=why)


def _activation(check: ResolvedCheck, evidence: SessionEvidence) -> tuple[bool, int | None]:
    if check.scope.activation == "event":
        event = next((event for event in evidence.events if event.id == check.scope.event_id), None)
        return event is not None, event.step_index if event else None
    if check.scope.activation == "continuation_start":
        if evidence.continuation_start is None:
            raise ValueError("required continuation boundary is missing from evidence")
        return True, evidence.continuation_start
    return True, None


def _judge_unit(
    check: ResolvedCheck,
    steps: list[Step],
    context: str,
    index: int | None,
    judge: Judge,
    *,
    question: str | None = None,
) -> JudgeDecision:
    shown = render(steps, context, index, judge.config, **render_options(check.parameters))
    timing = {"n_steps": len(steps) - index} if context == "after_event" and index is not None else {}
    if check.instruction is not None:
        timing["instruction"] = check.instruction
    return judge.evaluate(
        rubric=question if question is not None else check.rubric, context=context, evidence=shown, **timing
    )


def _evaluate(
    check: ResolvedCheck, steps: list[Step], facts: Facts, judge: Judge, start: int | None = None
) -> list[Unit]:
    """Evaluate the check's units; an activation step limits what is graded, never which checker runs."""
    parameters = check.parameters
    value = parameters.get("value")
    context = parameters.get("context")
    run_mode = check.kind in RUN_CHECKERS or (
        check.kind == "judge"
        and (
            context in {"run", "after_event"}
            or (context is None and set(check.scope.places) <= {"run", "after_event"})
        )
    )
    units: list[Unit] = []
    for where in check.scope.places:
        selected = select(steps, where, parameters)
        if start is not None:
            if where == "first_turn":
                selected = [
                    ("message", index)
                    for index, step in enumerate(steps)
                    if index > start and step["role"] == "assistant"
                ][:1]
            else:
                selected = [
                    (kind, index)
                    for kind, index in selected
                    if index is None or index > start or where == "after_event"
                ]
        for _, index in selected:
            if run_mode:
                if check.kind == "judge":
                    # A rule that takes effect mid-run is shown the run from the step it arrived at.
                    kind = "after_event" if start is not None else context or "run"
                    units.append(
                        Unit(
                            where, None, _judge_unit(check, steps, kind, start if start is not None else index, judge)
                        )
                    )
                    continue
                if check.kind == "order" and start is not None:
                    preceding = sum(
                        1
                        for _, _, call in calls.assistant_calls(steps[: start + 1], rejected=False)
                        if call_matches(call, value["then"], facts)
                    )
                    outcomes = calls.check_order(steps, value, parameters, facts, 0)[preceding:]
                else:
                    boundary = start if start is not None else index
                    if check.kind != "after_event":
                        boundary = (boundary + 1) if boundary is not None else 0
                    outcomes = RUN_CHECKERS[check.kind](steps, value, parameters, facts, boundary)
                units.extend(Unit(where, None, _decision(ok, why)) for ok, why in outcomes)
                continue
            if index is None:
                raise ValueError(f"{check.kind} requires a text place, not {where}")
            step = steps[index]
            content = step.get("text") or ""
            if not content.strip():
                if parameters.get("if_no_text") == "fail":
                    units.append(Unit(where, index, _decision(False, "no text")))
                elif parameters.get("if_no_text") == "pass":
                    units.append(Unit(where, index, _decision(True, "no text: the message only calls a tool")))
                continue
            if check.kind == "language":
                expected = value
                if str(value).lower() == "user":
                    user = next(
                        (
                            prior
                            for prior in reversed(steps[:index])
                            if prior["role"] == "user" and prior.get("session_id") == step.get("session_id")
                        ),
                        {},
                    )
                    expected = language.detect(user.get("text", ""))[0]
                    if not expected:
                        continue
                assessment = language.assess(expected, content)
                if assessment["ok"] is None:
                    continue
                if assessment["clear"]:
                    decision = _decision(assessment["ok"], assessment["detail"])
                else:
                    question = judge.config["checker_questions"]["language"].replace(
                        "{language}", str(expected).capitalize()
                    )
                    decision = judge.evaluate(rubric=question, context="message", evidence=content)
            elif check.kind == "judge":
                changed = changed_files(steps, index - 1) if context == "message_with_changed_files" else None
                if context == "message_with_changed_files" and not changed:
                    continue
                # The marks around a changed file's path are checked by code before the judge sees the message.
                marks = parameters.get("mark_parts") or {}
                if parameters.get("marks") == "changed_file" and marks:
                    bad = text.changed_file_marks(content, changed or [], marks["path"])
                    if bad:
                        units.append(
                            Unit(where, index, _decision(False, "marks (checked by code): " + "; ".join(bad[:3])))
                        )
                        continue
                decision = _judge_unit(check, steps, context or "message", index, judge)
            else:
                ok, why = TEXT_CHECKERS[check.kind](content, value, parameters, step, facts)
                decision = _decision(ok, why)
            units.append(Unit(where, index, decision))
    return units


def _prefix_failures(check: ResolvedCheck, evidence: SessionEvidence, judge: Judge) -> list[dict[str, JsonValue]]:
    if (
        not check.scope.check_prefix
        or evidence.continuation_start is None
        or check.scope.activation != "session_start"
    ):
        return []
    steps = evidence_steps(evidence)[: evidence.continuation_start]
    if not steps:
        return []
    if check.kind == "judge" and check.scope.places == ["run"]:
        answer = _judge_unit(check, steps, "run", None, judge, question=PREFIX_QUESTION.format(question=check.rubric))
        if answer.status == "error":
            raise ValueError(f"prefix judgment failed: {answer.explanation}")
        return [{"where": "run", "step": None, "why": answer.explanation}] if answer.status == "fail" else []
    places = [
        where
        for where in check.scope.places
        if where != "final"
        and (
            where not in {"run", "after_event"}
            or check.kind in {"order", "args", "after_event"}
            or (check.kind == "calls" and check.parameters["value"].get("mode") in SETTLED_CALL_MODES)
        )
    ]
    if not places:
        return []
    scoped = check.model_copy(update={"scope": check.scope.model_copy(update={"places": places})})
    units = _evaluate(
        scoped, steps, facts_from_steps(steps, dict(check.parameters.get("facts") or {}) | evidence.facts), judge
    )
    errors = [unit.decision.explanation for unit in units if unit.decision.status == "error"]
    if errors:
        raise ValueError("prefix judgment failed: " + "; ".join(errors))
    return [
        {"where": unit.where, "step": unit.index, "why": unit.decision.explanation}
        for unit in units
        if unit.decision.status == "fail"
    ]


def _grade_check(check: ResolvedCheck, evidence: SessionEvidence, judge: Judge) -> CheckResult:
    check = check.model_copy(update={"instruction": captured_instruction(check, evidence)})
    applicable, start = _activation(check, evidence)
    if not applicable:
        return CheckResult(id=check.id, status="not_applicable", explanation="The instruction event did not fire")
    steps = evidence_steps(evidence)
    if start is not None:
        # Each criterion uses its own activation receipt, not the first planted step.
        for index, step in enumerate(steps):
            step["planted"] = index == start
    units = _evaluate(
        check, steps, facts_from_steps(steps, dict(check.parameters.get("facts") or {}) | evidence.facts), judge, start
    )
    applicable_units = [unit for unit in units if unit.decision.status != "not_applicable"]
    statuses = {unit.decision.status for unit in applicable_units}
    status = (
        "error"
        if "error" in statuses
        else "fail"
        if "fail" in statuses
        else "pass"
        if applicable_units
        else "not_applicable"
    )
    explanations = [unit.decision.explanation for unit in units if unit.decision.status == status]
    return CheckResult(
        id=check.id,
        status=status,
        explanation="; ".join(explanations) or "No applicable units in the captured session",
        details={
            "units": [
                {"where": unit.where, "step": unit.index, **unit.decision.model_dump(mode="json")} for unit in units
            ],
            "prefix_failures": _prefix_failures(check, evidence, judge),
        },
    )


def grade(checks: Sequence[ResolvedCheck], evidence: SessionEvidence, *, judge: Judge) -> IFResult:
    """Grade every check once; a check that cannot be graded is an error, never a failure.

    The task reward is not changed: it belongs to the task's own verifier.
    """
    if len({check.id for check in checks}) != len(checks):
        raise ValueError("check IDs must be unique")
    results = []
    for check in checks:
        if evidence.status != "complete":
            result = CheckResult(id=check.id, status="error", explanation="; ".join(evidence.issues))
        else:
            try:
                result = _grade_check(check, evidence, judge)
            except Exception as error:
                # The per-check boundary preserves other evidence without inventing a score.
                result = CheckResult(id=check.id, status="error", explanation=f"{type(error).__name__}: {error}")
        result.details.update(
            provenance=deepcopy(check.provenance), check_sha256=fingerprint(check.model_dump(mode="json"))
        )
        results.append(result)
    applicable = [result for result in results if result.status != "not_applicable"]
    valid = bool(applicable) and all(result.status != "error" for result in applicable)
    return IFResult(
        checks=results,
        if_all=int(all(result.status == "pass" for result in applicable)) if valid else None,
        if_frac=round(sum(result.status == "pass" for result in applicable) / len(applicable), 3) if valid else None,
        metadata={
            "evidence_status": evidence.status,
            "scorer": deepcopy(scorer_identity()),
            "judge": {"version": judge.version, "config_hash": judge.hash},
        },
    )
