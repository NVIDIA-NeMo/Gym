"""Trajectory grading semantics — the RL-reward core of the constraint system.

Covers what the RL-testing literature says silently breaks reward pipelines:
scope filtering, injection-turn awareness, N/A (vacuity) handling, and
aggregation math — all against synthetic Responses-style trajectories.
"""

from __future__ import annotations

from responses_api_agents.swe_agents_constrained.grading.if_format.constraints import InjectionMode
from responses_api_agents.swe_agents_constrained.grading.verifiers.trajectory import (
    grade_constraints,
    matches_scope,
    parse_trajectory,
)


def _msg(text):
    return {"type": "message", "content": [{"type": "output_text", "text": text}]}


def _call(name, args="{}"):
    return {"type": "function_call", "name": name, "arguments": args}


def _obs(output):
    return {"type": "function_call_output", "output": output}


# ── Parser ────────────────────────────────────────────────────────────────────


def test_parse_trajectory_types_and_final_answer_promotion():
    steps = parse_trajectory([
        _msg("planning"),
        _call("VehicleControl-get_fuel_level"),
        _obs('{"fuel_level": 8.5}'),
        _msg("the fuel is 8.5 gallons"),
    ])
    assert [s.step_type for s in steps] == ["thinking", "tool_call", "observation", "final_answer"]
    assert steps[1].tool_name == "VehicleControl-get_fuel_level"
    assert steps[1].text.startswith("Action: VehicleControl-get_fuel_level")
    assert steps[0].is_first_step and steps[-1].is_final_step


def test_parse_trajectory_accepts_plain_string_content():
    steps = parse_trajectory([{"type": "message", "content": "hello"}])
    assert steps[0].text == "hello"


def test_parse_trajectory_empty():
    assert parse_trajectory([]) == []
    assert parse_trajectory(None) == []


# ── Grading: applicability / vacuity ─────────────────────────────────────────


def test_constraint_with_no_inscope_steps_is_not_applicable_and_excluded():
    # step_summary_prefix scopes AFTER_TOOL_CALL; trajectory has no tool calls.
    steps = parse_trajectory([_msg("just thinking, no tools")])
    g = grade_constraints(steps, [{"type": "step_summary_prefix",
                                   "params": {"prefix": "Found:"}}])
    assert g.constraint_applicable["step_summary_prefix"] is False
    # N/A excluded from denominator -> neutral reward, not a free pass signal
    assert g.reward == 1.0
    assert g.constraint_results["step_summary_prefix"] is True  # reported, not graded


def test_na_constraint_does_not_dilute_fraction_reward():
    # One applicable failing constraint + one N/A constraint: fraction must be
    # 0.0 (1 graded, failed), not 0.5 (N/A counted as pass).
    steps = parse_trajectory([_msg("no ticket id here"), _msg("final")])
    g = grade_constraints(
        steps,
        [{"type": "ticket_id_in_all_steps"},
         {"type": "step_summary_prefix", "params": {"prefix": "Found:"}}],
        grading_mode="fraction",
    )
    assert g.constraint_applicable["ticket_id_in_all_steps"] is True
    assert g.constraint_applicable["step_summary_prefix"] is False
    assert g.reward == 0.0


def test_unknown_constraint_type_counts_as_applicable_failure():
    steps = parse_trajectory([_msg("anything")])
    g = grade_constraints(steps, [{"type": "not_a_real_constraint"}])
    assert g.reward == 0.0
    assert g.constraint_applicable["not_a_real_constraint"] is True


# ── Grading: injection awareness ──────────────────────────────────────────────


def test_mid_conversation_injection_ignores_earlier_violations():
    steps = parse_trajectory([
        _msg("no ticket id before the instruction arrived"),   # index 0 — pre-injection
        _msg("TKT-004411 now complying"),  # index 1 — post-injection
    ])
    g = grade_constraints(
        steps, [{"type": "ticket_id_in_all_steps"}],
        injection_mode=InjectionMode.MID_CONVERSATION, injection_step=1,
    )
    assert g.reward == 1.0, g.violations

    g_system = grade_constraints(steps, [{"type": "ticket_id_in_all_steps"}])
    assert g_system.reward == 0.0  # same trajectory, system-prompt injection


# ── Grading: universal quantification ─────────────────────────────────────────


def test_one_violating_step_fails_the_constraint():
    steps = parse_trajectory([
        _msg("TKT-004411 step one"),
        _msg("forgot the ticket id this time"),
        _msg("TKT-004411 final"),
    ])
    g = grade_constraints(steps, [{"type": "ticket_id_in_all_steps"}])
    assert g.reward == 0.0
    assert any("step 1" in v for v in g.violations)


def test_forbidden_tool_detected_in_native_call_synthesis():
    steps = parse_trajectory([
        _msg("I'll take a shortcut"),
        _call("bulk_delete", "{}"),
        _obs("deleted"),
        _msg("done"),
    ])
    g = grade_constraints(steps, [{"type": "forbidden_tool_abstention",
                                   "params": {"forbidden_tool": "bulk_delete"}}])
    assert g.reward == 0.0


def test_compliant_trajectory_full_reward():
    steps = parse_trajectory([
        _msg("TKT-004411 Step 1: inspect"),
        _call("VehicleControl-get_fuel_level", '{"ticket": "TKT-004411"}'),
        _obs('{"fuel_level": 8.5}'),
        _msg("TKT-004411 final answer"),
    ])
    g = grade_constraints(steps, [{"type": "ticket_id_in_all_steps"}], grading_mode="fraction")
    assert g.reward == 1.0, g.violations


# ── Scope matrix sanity ───────────────────────────────────────────────────────


def test_scope_matrix():
    from responses_api_agents.swe_agents_constrained.grading.if_format.constraints import ConstraintScope
    from responses_api_agents.swe_agents_constrained.grading.verifiers.trajectory import Step

    tool = Step(text="Action: x", step_index=1, step_type="tool_call")
    obs = Step(text="result", step_index=2, step_type="observation")
    after = Step(text="Found: it", step_index=3, step_type="thinking", follows_observation=True)
    final = Step(text="answer", step_index=4, step_type="final_answer", is_final_step=True)

    assert matches_scope(tool, ConstraintScope.ALL_STEPS)
    assert not matches_scope(tool, ConstraintScope.REASONING_STEPS)
    # AFTER_TOOL_CALL governs the assistant message following an observation,
    # never the observation itself (env-generated text can't comply).
    assert not matches_scope(obs, ConstraintScope.AFTER_TOOL_CALL)
    assert matches_scope(after, ConstraintScope.AFTER_TOOL_CALL)
    assert matches_scope(final, ConstraintScope.FINAL_OUTPUT)
    assert not matches_scope(obs, ConstraintScope.FINAL_OUTPUT)


def test_after_tool_call_grades_assistant_followup_not_observation():
    steps = parse_trajectory([
        _msg("planning"),
        _call("A-x"),
        _obs("raw tool output without any prefix"),
        _msg("Found: the fuel level is 8.5 gallons"),
    ])
    g = grade_constraints(
        steps,
        [{"type": "step_summary_prefix", "params": {"prefix": "Found:"}}],
        grading_mode="fraction",
    )
    assert g.constraint_applicable["step_summary_prefix"] is True
    assert g.reward == 1.0, g.violations

    steps_bad = parse_trajectory([
        _msg("planning"),
        _call("A-x"),
        _obs("raw tool output"),
        _msg("the fuel level is 8.5 gallons"),  # missing Found: prefix
    ])
    g_bad = grade_constraints(
        steps_bad,
        [{"type": "step_summary_prefix", "params": {"prefix": "Found:"}}],
        grading_mode="fraction",
    )
    assert g_bad.reward == 0.0


def test_observations_are_never_graded():
    # Environment-authored text: the model cannot comply inside it, so an
    # ALL_STEPS constraint must not fail because an observation lacks the marker.
    steps = parse_trajectory([
        _msg("TKT-004411 calling now"),
        _call("A-x", '{"ticket": "TKT-004411"}'),
        _obs("raw environment output with no ticket id"),
        _msg("TKT-004411 final"),
    ])
    g = grade_constraints(steps, [{"type": "ticket_id_in_all_steps"}], grading_mode="fraction")
    assert g.reward == 1.0, g.violations


def test_before_tool_call_scope_grades_the_preceding_message_only():
    # tool_call_intent_tag: compliance lives in the assistant message that
    # precedes the call, never in the synthesized "Action:" step.
    steps = parse_trajectory([
        _msg("[INTENT:READ] Looking up the contact record."),
        _call("A-get_contacts"),
        _obs('{"contacts": []}'),
        _msg("No contacts found; stopping."),
    ])
    g = grade_constraints(steps, [{"type": "tool_call_intent_tag"}], grading_mode="fraction")
    assert g.constraint_applicable["tool_call_intent_tag"] is True
    assert g.reward == 1.0, g.violations

    bad = parse_trajectory([
        _msg("Looking up the contact record."),   # no [INTENT:...] tag
        _call("A-get_contacts"),
        _obs('{"contacts": []}'),
    ])
    assert grade_constraints(bad, [{"type": "tool_call_intent_tag"}],
                             grading_mode="fraction").reward == 0.0


def test_conversational_constraints_are_dispatched():
    """Regression: grade_constraints only resolved AgenticConstraintType, so
    every conversational constraint scored 0.0 as 'Unknown constraint type'
    regardless of the response. It silently zeroed ~2/3 of single-shot pairs —
    a compliant "TL;DR: ..." answer scored 0.0 for both models."""
    compliant = parse_trajectory([_msg("TL;DR: the answer is 35.\n\nWorking below.\n\n#### 35")])
    g = grade_constraints(compliant, [{"type": "tldr_prefix"}],
                          grading_mode="fraction", step_aggregation="mean")
    assert g.reward == 1.0, g.violations
    assert not any("Unknown constraint" in v for v in g.violations)

    violating = parse_trajectory([_msg("In summary, the answer is 35.\n\n#### 35")])
    assert grade_constraints(violating, [{"type": "tldr_prefix"}],
                             grading_mode="fraction").reward == 0.0


def test_conversational_constraint_reads_its_parameters():
    long_text = " ".join(["word"] * 200)
    over = parse_trajectory([_msg(long_text)])
    under = parse_trajectory([_msg("short answer")])
    params = {"max_words": 150}
    assert grade_constraints(under, [{"type": "word_count_max", "params": params}],
                             grading_mode="fraction").reward == 1.0
    assert grade_constraints(over, [{"type": "word_count_max", "params": params}],
                             grading_mode="fraction").reward == 0.0


def test_unknown_type_still_rejected_after_dual_dispatch():
    steps = parse_trajectory([_msg("anything")])
    g = grade_constraints(steps, [{"type": "definitely_not_a_constraint"}])
    assert g.reward == 0.0
    assert any("Unknown constraint" in v for v in g.violations)


# ── Regression: bare tool calls must not escape tool-anchored constraints ─────


def test_bare_tool_calls_violate_before_tool_call_instead_of_being_na():
    """A model that emits tool calls with no text at all must FAIL a
    'tag your intent before each tool call' constraint, not score N/A.

    Quantifying over text steps let the worst trajectories drop out of the IF
    statistic entirely (22 of 114 EnvFactory episodes), and because the
    compound reward drops the constraint term when nothing is gradable, RL
    would have learned that emitting no text removes the format penalty.
    """
    steps = parse_trajectory([
        _call("A-x"), _obs("r1"),
        _call("A-y"), _obs("r2"),
        _msg("all done"),
    ])
    g = grade_constraints(steps, [{"type": "tool_call_intent_tag"}],
                          grading_mode="fraction", step_aggregation="mean")
    assert g.constraint_applicable["tool_call_intent_tag"] is True
    assert g.reward == 0.0, g.violations
    assert any("anchors had no model-authored text" in v for v in g.violations)


def test_partially_covered_tool_calls_score_the_covered_fraction():
    """One of two tool calls carries an intent tag -> 0.5, not 1.0."""
    steps = parse_trajectory([
        _msg("[INTENT:READ] look up the record"), _call("A-x"), _obs("r1"),
        _call("A-y"), _obs("r2"),          # bare — no preceding text
        _msg("done"),
    ])
    g = grade_constraints(steps, [{"type": "tool_call_intent_tag"}],
                          grading_mode="fraction", step_aggregation="mean")
    assert g.reward == 0.5, (g.reward, g.violations)


def test_after_tool_call_anchors_on_observations_too():
    """Same rule for AFTER_TOOL_CALL: an observation with no follow-up text
    is a violation of 'summarise after each tool observation'."""
    steps = parse_trajectory([
        _msg("start"), _call("A-x"), _obs("r1"),
        _msg("Found: the first result"),
        _call("A-y"), _obs("r2"),          # no follow-up message at all
    ])
    g = grade_constraints(
        steps, [{"type": "step_summary_prefix", "params": {"prefix": "Found:"}}],
        grading_mode="fraction", step_aggregation="mean")
    assert g.constraint_applicable["step_summary_prefix"] is True
    assert g.reward == 0.5, (g.reward, g.violations)


def test_no_tool_calls_still_means_not_applicable():
    """The fix must not turn a genuinely vacuous case into a failure: a
    trajectory with zero tool calls has zero anchors, so a tool-anchored
    constraint stays N/A."""
    steps = parse_trajectory([_msg("just an answer, no tools needed")])
    g = grade_constraints(steps, [{"type": "tool_call_intent_tag"}],
                          grading_mode="fraction", step_aggregation="mean")
    assert g.constraint_applicable["tool_call_intent_tag"] is False
    assert g.any_graded is False


# ── Turn attribution: exactly where in the conversation a constraint failed ──


def test_turns_reconstruct_api_rounds_including_bare_tool_calls():
    steps = parse_trajectory([
        _msg("TKT-004411 plan"),          # turn 1
        _call("A-x"), _obs("r1"),          # turn 1 (call), obs belongs to it
        _call("A-y"), _obs("r2"),          # turn 2 — bare round, no text
        _msg("TKT-004411 done"),           # turn 3
    ])
    assert [s.turn for s in steps] == [1, 1, 1, 2, 2, 3]


def test_first_violation_turn_locates_the_slip():
    steps = parse_trajectory([
        _msg("TKT-004411 step one"), _call("A-x", '{"t": "TKT-004411"}'), _obs("r1"),
        _msg("forgot the ticket id here"),                     # turn 2 — the slip
        _call("A-y", '{"t": "TKT-004411"}'), _obs("r2"),
        _msg("TKT-004411 final"),                              # turn 4
    ])
    g = grade_constraints(steps, [{"type": "ticket_id_in_all_steps"}],
                          grading_mode="fraction", step_aggregation="mean")
    assert g.first_violation_turn() == 2
    bad = [v for v in g.step_verdicts if not v.passed]
    assert len(bad) == 1 and bad[0].turn == 2 and bad[0].kind == "text"
    # passes are recorded too — the verdict list is the full audit trail
    assert sum(1 for v in g.step_verdicts if v.passed) >= 3


def test_uncovered_anchor_verdicts_carry_the_anchor_turn():
    """Bare tool calls: the violation is located AT the offending call's turn,
    not merely counted."""
    steps = parse_trajectory([
        _msg("[INTENT:READ] look it up"), _call("A-x"), _obs("r1"),   # turn 1 ok
        _call("A-y"), _obs("r2"),                                     # turn 2 bare
        _call("A-z"), _obs("r3"),                                     # turn 3 bare
        _msg("done"),
    ])
    g = grade_constraints(steps, [{"type": "tool_call_intent_tag"}],
                          grading_mode="fraction", step_aggregation="mean")
    anchor_fails = [v for v in g.step_verdicts if not v.passed and v.kind == "anchor"]
    assert sorted(v.turn for v in anchor_fails) == [2, 3], anchor_fails
    assert g.first_violation_turn("tool_call_intent_tag") == 2
    assert g.violations_by_turn().keys() == {2, 3}


def test_verdicts_are_deterministic():
    items = [_msg("no ticket"), _call("A-x"), _obs("r"), _msg("still none")]
    a = grade_constraints(parse_trajectory(items), [{"type": "ticket_id_in_all_steps"}])
    b = grade_constraints(parse_trajectory(items), [{"type": "ticket_id_in_all_steps"}])
    assert [(v.constraint, v.turn, v.passed) for v in a.step_verdicts] == \
           [(v.constraint, v.turn, v.passed) for v in b.step_verdicts]


def test_finish_call_message_is_the_final_answer():
    """OpenHands ends episodes with finish(message=...): that message is the
    final answer the user sees, so FINAL_OUTPUT obligations grade against it
    (2026-08-14 fix — previously such trajectories had no final_answer and
    compliant finals scored 0)."""
    import json as _json
    steps = parse_trajectory([
        _msg("working on it"), _call("A-x"), _obs("r1"),
        _call("finish", _json.dumps({"message": "All done.\nIMPACT: files=1 | ok"})),
    ])
    finals = [s for s in steps if s.step_type == "final_answer"]
    assert len(finals) == 1 and finals[0].is_final_step
    assert "IMPACT: files=1" in finals[0].text


def test_non_finish_terminal_tool_call_still_has_no_final_answer():
    steps = parse_trajectory([
        _msg("working"), _call("A-x"), _obs("r1"), _call("A-y"),
    ])
    assert not [s for s in steps if s.step_type == "final_answer"]
