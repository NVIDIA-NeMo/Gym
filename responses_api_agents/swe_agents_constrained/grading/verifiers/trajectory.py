"""Trajectory parsing and constraint grading — framework-free core.

This is the testable heart of constraint grading: it depends only on the
constraint registries and verifier implementations (all inside this grading
package), never on the rest of nemo_gym, so the grading semantics (scope
filtering, injection-turn awareness, N/A handling, reward aggregation) can be
unit-tested in any venv and the agent wrapper stays a thin shell around it.

Accepts either attribute-style items (pydantic Responses objects) or plain
dicts with the same keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional

from ..if_format.constraints import (
    AGENTIC_CONSTRAINT_REGISTRY,
    CONVERSATIONAL_CONSTRAINT_REGISTRY,
    AgenticConstraintType,
    ConstraintScope,
    ConversationalConstraintType,
    InjectionMode,
)
from .if_format import (
    AGENTIC_VERIFIER_REGISTRY,
    CONVERSATIONAL_VERIFIER_REGISTRY,
)


def registry_params(ctype_str: str) -> dict:
    """Registry-default parameters for a constraint id (either registry)."""
    try:
        entry = AGENTIC_CONSTRAINT_REGISTRY.get(AgenticConstraintType(ctype_str))
    except ValueError:
        try:
            entry = CONVERSATIONAL_CONSTRAINT_REGISTRY.get(ConversationalConstraintType(ctype_str))
        except ValueError:
            return {}
    return dict(getattr(entry, "parameters", None) or {})


def resolve_constraint(ctype_str: str):
    """(verifier, scope) for a constraint of either family, or (None, None).

    Conversational constraints govern the model's text response, so they are
    scoped to the final answer. Dispatching only to the agentic registry
    silently scored every conversational constraint 0.0 as 'unknown'.
    """
    try:
        ctype = AgenticConstraintType(ctype_str)
    except ValueError:
        pass
    else:
        entry = AGENTIC_CONSTRAINT_REGISTRY.get(ctype)
        return (AGENTIC_VERIFIER_REGISTRY.get(ctype),
                entry.scope if entry else ConstraintScope.ALL_STEPS)
    try:
        ctype_c = ConversationalConstraintType(ctype_str)
    except ValueError:
        return None, None
    return CONVERSATIONAL_VERIFIER_REGISTRY.get(ctype_c), ConstraintScope.FINAL_OUTPUT


def _anchor_text_required(ctype_str: str) -> bool:
    """Registry flag: does this constraint obligate text at every anchor?

    2026-08-15: consult the conversational registry too — deliverable finals
    (json_required_fields, fenced_final_answer) are owed, so a trajectory
    with no final message violates them instead of abstaining.
    """
    try:
        entry = AGENTIC_CONSTRAINT_REGISTRY.get(AgenticConstraintType(ctype_str))
    except ValueError:
        try:
            entry = CONVERSATIONAL_CONSTRAINT_REGISTRY.get(ConversationalConstraintType(ctype_str))
        except ValueError:
            return False
    return bool(entry and getattr(entry, "anchor_text_required", False))


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


@dataclass
class Step:
    text: str
    step_index: int
    step_type: str  # "thinking" | "tool_call" | "observation" | "final_answer"
    tool_name: Optional[str] = None
    is_first_step: bool = False
    is_final_step: bool = False
    # True for assistant steps that directly follow a tool observation —
    # AFTER_TOOL_CALL constraints govern these, never the observation itself
    # (observations are environment-generated; the model cannot comply there).
    follows_observation: bool = False
    # True for assistant steps immediately followed by a tool call — the only
    # place a "declare intent before calling" constraint can be satisfied.
    precedes_tool_call: bool = False
    # 1-based assistant-turn ordinal. A turn is one model response: it opens
    # with the first assistant-authored step (message or tool call) after an
    # observation (or at trajectory start) and spans until the next
    # observation. Observations belong to the turn that caused them. This
    # reconstructs API-round boundaries from the flat item list, including
    # bare-tool-call rounds that contain no message text.
    turn: int = 0


@dataclass
class StepVerdict:
    """One constraint checked at one concrete point in the conversation.

    The unit callers need for "at exactly what turn did the constraint fail":
    machine-readable, per (constraint, step), with the turn attached. An
    uncovered tool-call anchor (the model called a tool with no qualifying
    text) yields a verdict at that anchor's turn — previously those failures
    were counted but located nowhere.
    """

    constraint: str
    step_index: int
    turn: int
    passed: bool
    violation: str | None = None
    # "text" = a model-authored step was checked; "anchor" = a tool-call
    # anchor had no model-authored text to check (violation by omission).
    kind: str = "text"


@dataclass
class GradingResult:
    reward: float
    # False when no constraint had a single gradeable step (all steps were
    # out of scope, pre-injection, empty, or judge-only). Callers must treat
    # this as "format not measured" — NOT as perfect compliance, or a pair
    # whose constraint never fires collects a free 1.0.
    any_graded: bool = True
    constraint_results: dict[str, bool] = field(default_factory=dict)
    # Per-constraint fraction of gradeable steps that complied. Strict
    # all-or-nothing scoring collapses very different behaviours to the same
    # 0.0 (measured: 40/41 compliant steps and 1/13 both scored 0.00), which
    # destroys the gradient. Partial credit preserves it.
    constraint_scores: dict[str, float] = field(default_factory=dict)
    constraint_applicable: dict[str, bool] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    # Every (constraint, step) check performed, passes and failures alike,
    # with turn attribution. Deterministic — same trajectory, same verdicts.
    step_verdicts: list[StepVerdict] = field(default_factory=list)

    def first_violation_turn(self, constraint: str | None = None) -> int | None:
        """Earliest turn at which a (or any) constraint was violated."""
        turns = [v.turn for v in self.step_verdicts
                 if not v.passed and (constraint is None or v.constraint == constraint)]
        return min(turns) if turns else None

    def violations_by_turn(self) -> dict[int, list["StepVerdict"]]:
        out: dict[int, list[StepVerdict]] = {}
        for v in self.step_verdicts:
            if not v.passed:
                out.setdefault(v.turn, []).append(v)
        return dict(sorted(out.items()))


def extract_message_text(item: Any) -> str:
    content = _get(item, "content") or []
    if isinstance(content, str):
        return content
    parts = []
    for c in content:
        t = _get(c, "type")
        if t in ("output_text", "text"):
            parts.append(_get(c, "text", ""))
        elif _get(c, "text") is not None:
            parts.append(_get(c, "text"))
    return " ".join(parts)


def parse_trajectory(output_items: Any) -> List[Step]:
    """Responses-API output items -> typed steps.

    Native function calls are rendered as 'Action: <name>\\nAction Input: <args>'
    text so text-level verifiers (forbidden tools, intent tags) see a uniform
    format regardless of harness.
    """
    steps: List[Step] = []
    for i, item in enumerate(output_items or []):
        item_type = _get(item, "type")
        if item_type == "message":
            steps.append(Step(
                text=extract_message_text(item),
                step_index=i, step_type="thinking", is_first_step=(i == 0),
            ))
        elif item_type == "function_call":
            name = _get(item, "name", "")
            args = _get(item, "arguments", "")
            steps.append(Step(
                text=f"Action: {name}\nAction Input: {args}",
                step_index=i, step_type="tool_call", tool_name=name,
                is_first_step=(i == 0),
            ))
        elif item_type == "function_call_output":
            steps.append(Step(
                text=_get(item, "output", ""),
                step_index=i, step_type="observation", is_first_step=(i == 0),
            ))
        elif item_type == "reasoning":
            summary = _get(item, "summary") or []
            text = " ".join(_get(s, "text", "") for s in summary)
            steps.append(Step(
                text=text, step_index=i, step_type="thinking", is_first_step=(i == 0),
            ))

    for step in reversed(steps):
        # 2026-08-12 trace-QA: stop at the last tool call — a message with
        # tool calls AFTER it is mid-work narration, not a final answer.
        # Promoting it graded FINAL_OUTPUT constraints at a point before the
        # work (and its obligations, e.g. edited-files manifests) existed.
        # Trajectories that end on silent tool calls now have NO final_answer,
        # which routes owed finals to the synthetic-final check instead.
        #
        # 2026-08-14 trace-QA: EXCEPT the agent-framework `finish` call. On
        # OpenHands, episodes end via finish(message=...) and the message
        # argument IS the final answer the user sees — grading "final message"
        # obligations without it zeroed every compliant trajectory (verified:
        # 8/8 opus impact_assessment_final_line rows carried the required line
        # inside finish args yet scored 0). Promote its message text instead.
        if step.step_type == "tool_call":
            if step.tool_name == "finish":
                raw = step.text.split("Action Input:", 1)[-1].strip()
                try:
                    parsed = json.loads(raw)
                    message = str(parsed.get("message") or "")
                except (ValueError, TypeError):
                    message = raw
                if message:
                    step.step_type = "final_answer"
                    step.text = message
                    step.is_final_step = True
            break
        if step.step_type == "thinking":
            step.step_type = "final_answer"
            step.is_final_step = True
            break

    prev_type = None
    for i, step in enumerate(steps):
        # 2026-08-12 trace-QA audit: tool_call steps count too. A model that
        # answers an observation with a silent next command IS the model's
        # next output — excluding it made every AFTER_TOOL_CALL constraint
        # vacuous exactly on the silent-chained-call style frontier models
        # use (the violating step was never in scope).
        if step.step_type in ("thinking", "final_answer", "tool_call") \
                and prev_type == "observation":
            step.follows_observation = True
        nxt = steps[i + 1] if i + 1 < len(steps) else None
        if step.step_type in ("thinking", "final_answer") and nxt is not None \
                and nxt.step_type == "tool_call":
            step.precedes_tool_call = True
        prev_type = step.step_type

    # Assistant-turn attribution. A new turn opens at the first
    # assistant-authored step (message or tool call) after an observation or
    # at the start; observations inherit the turn of the call that caused
    # them. This recovers API-round boundaries from the flat item list —
    # including bare-tool-call rounds, which have no message text and were
    # previously unlocatable in turn terms.
    turn = 0
    after_observation = True
    for step in steps:
        if step.step_type == "observation":
            after_observation = True
        else:
            if after_observation:
                turn += 1
                after_observation = False
        step.turn = max(turn, 1)
    return steps


#: Scopes that are anchored to a tool call rather than to a text step. For
#: these, the constraint is universally quantified over the ANCHORS ("before
#: every tool call, ...", "after every observation, ..."), not over whatever
#: text the model happened to write. Quantifying over text steps let a model
#: escape the constraint entirely by emitting bare tool calls: with no text
#: step in scope the constraint came back not-applicable, the compound reward
#: dropped the format term, and the worst trajectories left no trace in the IF
#: statistic. Measured on EnvFactory: 22 of 114 Nemotron episodes.
_ANCHORED_SCOPES = {
    ConstraintScope.BEFORE_TOOL_CALL: ("tool_call", "precedes_tool_call"),
    ConstraintScope.AFTER_TOOL_CALL: ("observation", "follows_observation"),
}


def count_scope_anchors(steps: List[Step], scope: ConstraintScope,
                        effective_from: int = 0) -> int:
    """How many tool calls / observations the constraint must be satisfied for."""
    anchor = _ANCHORED_SCOPES.get(scope)
    if anchor is None:
        return 0
    step_type, _ = anchor
    return sum(1 for s in steps
               if s.step_type == step_type and s.step_index >= effective_from)


def matches_scope(step: Step, scope: ConstraintScope) -> bool:
    if scope == ConstraintScope.ALL_STEPS:
        return True
    if scope == ConstraintScope.REASONING_STEPS:
        return step.step_type in ("thinking", "final_answer")
    if scope == ConstraintScope.CODE_STEPS:
        # 2026-08-12 trace-QA audit: tool_call steps ARE code steps. Excluding
        # them made every command/edit-discipline constraint vacuous under
        # native tool calling — verifiers only ever saw fenced prose, so the
        # actual commands went ungraded (agreement run: 341 scope-miss
        # disagreements across 22 constraints; artifacts/verifier_qa/).
        return (step.step_type == "tool_call"
                or (step.step_type in ("thinking", "final_answer") and "```" in step.text))
    if scope == ConstraintScope.AFTER_TOOL_CALL:
        return step.follows_observation
    if scope == ConstraintScope.BEFORE_TOOL_CALL:
        return step.precedes_tool_call
    if scope == ConstraintScope.FIRST_STEP_ONLY:
        return step.is_first_step
    if scope == ConstraintScope.FINAL_OUTPUT:
        return step.is_final_step
    return False


def grade_constraints(
    steps: List[Step],
    constraints: list[dict],
    *,
    injection_mode: InjectionMode = InjectionMode.SYSTEM_PROMPT,
    injection_step: int = 0,
    grading_mode: str = "binary",
    step_aggregation: str = "all",
) -> GradingResult:
    """Grade a trajectory against constraint declarations.

    Semantics:
      - scope filters which steps a constraint governs;
      - steps before the injection turn are never penalised (MID_CONVERSATION);
      - a constraint with no in-scope steps is applicable=False and EXCLUDED
        from the reward denominator (vacuous-pass invites trigger-avoidance
        hacking; vacuous-fail punishes tasks where the trigger cannot occur);
      - binary: all graded constraints must pass; fraction: mean of graded;
      - step_aggregation "all": a constraint passes only if every gradeable
        step complied; "mean": the constraint scores the fraction of compliant
        steps (partial credit — preferred for multi-step agentic constraints).
    """
    effective_from = injection_step if injection_mode == InjectionMode.MID_CONVERSATION else 0

    results: dict[str, bool] = {}
    scores: dict[str, float] = {}
    applicable: dict[str, bool] = {}
    violations: list[str] = []
    step_verdicts: list[StepVerdict] = []

    for raw in constraints:
        ctype_str = raw.get("type", "")
        # 2026-08-15: dataset declarations usually carry params={} — grading
        # must then use the REGISTRY defaults (what the injected description
        # rendered), not each verifier's hardcoded fallback. Divergence here
        # made conditional_required_sentence grade against the retired scan
        # pattern while the agent was instructed with the /tmp condition.
        params = {**registry_params(ctype_str), **(raw.get("params") or {})}
        verifier, scope = resolve_constraint(ctype_str)
        if verifier is None:
            results[ctype_str] = False
            scores[ctype_str] = 0.0
            applicable[ctype_str] = True
            violations.append(f"Unknown constraint type: {ctype_str!r}")
            continue

        step_passed: list[bool] = []
        graded_steps: list[Step] = []
        for step in steps:
            if step.step_index < effective_from:
                continue
            if not matches_scope(step, scope):
                continue
            # Empty assistant turns are an artifact of the tool-calling channel
            # (the model's output went into tool_calls, not text). There is no
            # model-authored text to grade, so the step is NOT APPLICABLE —
            # grading it would make every text-format constraint structurally
            # unsatisfiable under native tool calling.
            if not step.text.strip():
                continue
            # Observations are environment-authored. The model cannot comply
            # inside them, so they are never gradeable for any constraint —
            # grading them made ALL_STEPS constraints unsatisfiable even when
            # the model complied in its own messages.
            if step.step_type == "observation":
                continue
            ctx = {
                "step_index": step.step_index,
                "step_type": step.step_type,
                "tool_name": step.tool_name,
                "prior_steps": [s for s in steps if s.step_index < step.step_index],
                "is_first_step": step.is_first_step,
                "is_final_step": step.is_final_step,
                "constraint_params": params,
            }
            result = verifier.check(step.text, ctx)
            # A verifier that cannot decide statically (needs_llm_judge) must
            # not contribute a vacuous pass — that is how scope_constraint
            # scored a free 1.00. Treat as not-applicable until a judge path
            # exists (design/constraint_audit.md open item 3).
            if result.needs_llm_judge:
                continue
            step_passed.append(result.passed)
            graded_steps.append(step)
            step_verdicts.append(StepVerdict(
                constraint=ctype_str, step_index=step.step_index,
                turn=step.turn, passed=result.passed,
                violation=None if result.passed else (result.violation or "violation"),
            ))
            if not result.passed and result.violation:
                violations.append(f"{ctype_str}[step {step.step_index}]: {result.violation}")

        # FINAL_OUTPUT with no final message: a trajectory that ends on a bare
        # tool call (max turns exhausted, silent chains) still owes its
        # final-message obligations — abstaining here let policies dodge every
        # final-output constraint by never writing a final message
        # (2026-08-12 trace-QA: 69 ABSTAIN-vs-VIOLATED disagreements across 4
        # constraints). Grade the verifier once on an empty synthetic final
        # step: conditional finals (manifest with no edits) still pass; owed
        # ones fail with a located verdict.
        # Gated on anchor_text_required: judges read generic formatting
        # finals (output_sections) as conditional on a final message existing,
        # but deliverable finals (manifest, impact, ledgers) as owed
        # regardless — the flag distinguishes them, same as tool anchors.
        if scope == ConstraintScope.FINAL_OUTPUT and not graded_steps and steps \
                and _anchor_text_required(ctype_str):
            last = steps[-1]
            result = verifier.check("", {
                "step_index": last.step_index + 1,
                "step_type": "final_answer",
                "tool_name": None,
                "prior_steps": steps,
                "is_first_step": False,
                "is_final_step": True,
                "constraint_params": params,
            })
            if not result.needs_llm_judge:
                step_passed.append(result.passed)
                if not result.passed:
                    step_verdicts.append(StepVerdict(
                        constraint=ctype_str, step_index=last.step_index + 1,
                        turn=last.turn, passed=False,
                        violation=(result.violation or "") + " (trajectory ended without a final message)",
                        kind="anchor",
                    ))
                    violations.append(
                        f"{ctype_str}: trajectory ended without a final message satisfying "
                        f"the final-output obligation")

        # Tool-anchored scopes with anchor_text_required: every anchor needs a
        # complying step, so anchors with no gradable text are violations, not
        # absences — see _ANCHORED_SCOPES. Without this, "tag your intent
        # before every tool call" was vacuous for a model that never writes
        # text. Each uncovered anchor gets a located verdict: the failure
        # happened AT that tool call's turn, not "somewhere".
        # 2026-08-12 trace-QA audit: coverage applies ONLY to constraints that
        # obligate text at every anchor (anchor_text_required=True in the
        # registry). For conditional constraints (only-on-error, only-on-test
        # -run, only-restricts-existing-text) a silent tool call is vacuous —
        # auto-failing it mass-punished silent chained tool calls.
        if scope in _ANCHORED_SCOPES and _anchor_text_required(ctype_str):
            anchor_type, companion_flag = _ANCHORED_SCOPES[scope]
            anchor_steps = [s for s in steps
                            if s.step_type == anchor_type and s.step_index >= effective_from]
            uncovered = len(anchor_steps) - len(step_passed)
            if uncovered > 0:
                # The graded text steps cover anchors in order; the anchors
                # beyond that coverage are the ones with no qualifying text.
                covered_turns = {s.turn for s in graded_steps}
                located = [a for a in anchor_steps if a.turn not in covered_turns]
                # Length can disagree with `uncovered` when one text step
                # served a turn with several anchors; fall back to the tail.
                if len(located) != uncovered:
                    located = anchor_steps[-uncovered:]
                step_passed.extend([False] * uncovered)
                for a in located:
                    step_verdicts.append(StepVerdict(
                        constraint=ctype_str, step_index=a.step_index,
                        turn=a.turn, passed=False,
                        violation="no model-authored text at this tool-call anchor",
                        kind="anchor",
                    ))
                violations.append(
                    f"{ctype_str}: {uncovered} of {len(anchor_steps)} tool-call anchors had no "
                    f"model-authored text to satisfy the constraint "
                    f"(turns {sorted({a.turn for a in located})})"
                )

        applicable[ctype_str] = bool(step_passed)
        results[ctype_str] = all(step_passed) if step_passed else True
        if not step_passed:
            scores[ctype_str] = 1.0
        elif step_aggregation == "mean":
            scores[ctype_str] = sum(1.0 for p in step_passed if p) / len(step_passed)
        else:
            scores[ctype_str] = float(all(step_passed))

    graded = [scores[name] for name in results if applicable.get(name)]
    if grading_mode == "fraction":
        reward = float(sum(graded) / len(graded)) if graded else 1.0
    else:
        reward = float(all(s == 1.0 for s in graded)) if graded else 1.0

    return GradingResult(
        reward=reward,
        any_graded=bool(graded),
        constraint_results=results,
        constraint_scores=scores,
        constraint_applicable=applicable,
        violations=violations,
        step_verdicts=step_verdicts,
    )
