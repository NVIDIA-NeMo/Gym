# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Observable grading semantics, independent of the task author and harness."""

import pytest

from resources_servers.agentic_if.if_grading import engine
from resources_servers.agentic_if.if_grading.configuration import load_config
from resources_servers.agentic_if.if_grading.contracts import JudgeDecision, ResolvedCheck, SessionEvidence


class Judge:
    """A judge with fixed decisions, recording what it is asked."""

    config = load_config()
    version, hash = "test-judge", "test-hash"

    def __init__(self, status="pass", error=None):
        self.status, self.error, self.seen = status, error, []

    def evaluate(self, *, rubric, context, evidence, n_steps=None, instruction=None):
        self.seen.append((rubric, context, evidence))
        if self.error:
            raise self.error
        return JudgeDecision(status=self.status, explanation="Controlled judgment", metadata={"model": "test-judge"})


def trace(*steps, **fields):
    return SessionEvidence(
        status="complete",
        steps=[{"id": str(index), "session_id": "main", **step} for index, step in enumerate(steps)],
        **fields,
    )


def call(tool, *, ident="c", args=None, rejected=False, model_tool=None, model_args=None):
    return {
        "call_id": ident,
        "invocation_id": "run",
        "tool": tool,
        "args": args or {},
        "model_tool": model_tool or tool,
        "model_args": model_args if model_args is not None else args or {},
        "rejected": rejected,
    }


def criterion(kind, value=None, *, ident="check", places=None, **parameters):
    return ResolvedCheck(
        id=ident, kind=kind, parameters={"value": value, **parameters}, scope={"places": places or ["final"]}
    )


@pytest.mark.parametrize(
    "check,text,status",
    [
        (
            criterion("length", {"n": 2, "unit": "sentences", "dir": "max"}, part="prose"),
            "Fixed it. Tests pass.",
            "pass",
        ),
        (criterion("length", {"n": 2, "unit": "sentences", "dir": "max"}), "One. Two. Three.", "fail"),
        (criterion("element_absent", "emoji"), "Done ✅", "fail"),
        (criterion("element_present", {"regex": "STATUS:", "part": "all"}, position="start"), "STATUS: done", "pass"),
        (criterion("ending", "no_question"), "Done. Want more?", "fail"),
        (criterion("exact", "DONE"), " DONE\n", "pass"),
        (criterion("json", {"keys": ["status"]}), '{"status":"ok"}', "pass"),
        (criterion("language", "Chinese"), "I fixed the bug and the tests pass now.", "fail"),
    ],
)
def test_deterministic_verdicts_preserve_checker_semantics(check, text, status):
    judge = Judge(error=AssertionError("This deterministic case must not call a judge"))
    result = engine.grade([check], trace({"role": "assistant", "text": text}), judge=judge)
    assert result.checks[0].status == status
    assert result.if_all == (1 if status == "pass" else 0)


def test_mixed_checks_have_one_result_order_and_fraction():
    checks = [
        criterion("exact", "DONE", ident="a"),
        ResolvedCheck(id="b", kind="judge", rubric="Confirm the reply ends the work."),
        criterion("exact", "OTHER", ident="c"),
    ]
    result = engine.grade(checks, trace({"role": "assistant", "text": "DONE"}), judge=Judge())
    assert [(c.id, c.status) for c in result.checks] == [("a", "pass"), ("b", "pass"), ("c", "fail")]
    assert result.if_all == 0
    assert result.if_frac == 0.667


def test_arbitrary_rubric_is_literal_and_receives_model_visible_tools():
    judge = Judge()
    evidence = trace(
        {
            "role": "assistant",
            "text": "I will test it.",
            "calls": [
                call("bash", args={"command": "pytest"}, model_tool="RunCommand", model_args={"cmd": "pytest"}),
            ],
        }
    )
    rule = ResolvedCheck(id="free", kind="judge", rubric="Use {RunCommand}; do not interpolate ${anything}.")
    assert engine.grade([rule], evidence, judge=judge).checks[0].status == "pass"
    question, context, shown = judge.seen[0]
    assert question == "Use {RunCommand}; do not interpolate ${anything}."
    assert context == "run"
    assert "RunCommand" in shown and '"cmd": "pytest"' in shown


def test_absent_place_and_judge_not_applicable_do_not_become_passes():
    e = trace({"role": "assistant", "calls": [call("bash")]})
    result = engine.grade([criterion("exact", "DONE")], e, judge=Judge())
    assert result.checks[0].status == "not_applicable"
    assert result.if_all is None and result.if_frac is None
    result = engine.grade(
        [ResolvedCheck(id="r", kind="judge", rubric="If relevant, explain.")], e, judge=Judge(status="not_applicable")
    )
    assert result.checks[0].status == "not_applicable"
    assert result.if_all is None and result.if_frac is None


def test_judge_failure_preserves_other_verdicts_but_invalidates_aggregate():
    checks = [
        criterion("exact", "DONE", ident="deterministic"),
        ResolvedCheck(id="judge", kind="judge", rubric="Be honest."),
    ]
    result = engine.grade(
        checks, trace({"role": "assistant", "text": "DONE"}), judge=Judge(error=TimeoutError("timeout"))
    )
    assert [check.status for check in result.checks] == ["pass", "error"]
    assert result.if_all is None and result.if_frac is None
    assert "TimeoutError" in result.checks[1].explanation


def test_prefix_judge_error_cannot_be_reported_as_a_clean_prefix():
    class PrefixFailure(Judge):
        def __init__(self):
            self.calls = 0

        def evaluate(self, **kwargs):
            self.calls += 1
            return JudgeDecision(status="error" if self.calls == 3 else "pass", explanation="prefix timeout")

    evidence = trace(
        {"role": "assistant", "text": "First"},
        {"role": "user", "text": "Continue"},
        {"role": "assistant", "text": "Last"},
        continuation_start=1,
    )
    check = ResolvedCheck(
        id="prefix", kind="judge", rubric="Be clear.", scope={"places": ["any_turn"], "check_prefix": True}
    )
    result = engine.grade([check], evidence, judge=PrefixFailure())
    assert result.checks[0].status == "error" and result.if_all is None


def test_incomplete_evidence_is_error_without_judge_or_false_pass():
    e = SessionEvidence(status="incomplete", steps=[], issues=["OpenCode export missing"])
    result = engine.grade([criterion("exact", "DONE")], e, judge=Judge(error=AssertionError("No grading")))
    assert result.checks[0].status == "error"
    assert "OpenCode export missing" in result.checks[0].explanation
    assert result.if_all is None and result.if_frac is None


def test_event_check_never_fires_is_not_applicable():
    check = ResolvedCheck(
        id="c", kind="judge", rubric="Stop after the notice.", scope={"activation": "event", "event_id": "notice"}
    )
    result = engine.grade([check], trace({"role": "assistant", "text": "DONE"}), judge=Judge(error=AssertionError()))
    assert result.checks[0].status == "not_applicable"


def test_missing_continuation_boundary_is_error_not_whole_run_grading():
    check = ResolvedCheck(
        id="c",
        kind="exact",
        parameters={"value": "DONE"},
        scope={"activation": "continuation_start", "places": ["final"]},
    )
    result = engine.grade([check], trace({"role": "assistant", "text": "DONE"}), judge=Judge())
    assert result.checks[0].status == "error"


def test_new_instruction_does_not_grade_earlier_calls():
    e = trace(
        {"role": "assistant", "calls": [call("bash")]},
        {"role": "user", "text": "Do not call bash again."},
        {"role": "assistant", "text": "DONE"},
        continuation_start=1,
    )
    check = ResolvedCheck(
        id="c",
        kind="calls",
        parameters={"value": {"mode": "never", "tool": "bash"}},
        scope={"activation": "continuation_start", "places": ["run"]},
    )
    assert engine.grade([check], e, judge=Judge()).checks[0].status == "pass"


def test_order_can_use_earlier_context_but_only_grades_new_then_calls():
    e = trace(
        {"role": "assistant", "calls": [call("read", args={"path": "a.py"})]},
        {"role": "user", "text": "Read before editing."},
        {"role": "assistant", "calls": [call("edit", ident="c2", args={"path": "a.py"})]},
        continuation_start=1,
    )
    check = ResolvedCheck(
        id="c",
        kind="order",
        parameters={
            "value": {
                "first": {"tool": "read"},
                "then": {"tool": "edit"},
                "same": "path",
            }
        },
        scope={"activation": "continuation_start"},
    )
    assert engine.grade([check], e, judge=Judge()).checks[0].status == "pass"


@pytest.mark.parametrize(
    "messages",
    [
        # A call the harness rejected is not a completed step.
        [[call("read", rejected=True)], [call("edit", ident="c2")]],
        # Parallel calls grouped in one message are not sequential actions.
        [[call("read"), call("edit", ident="c2")]],
    ],
)
def test_rejected_or_grouped_calls_do_not_satisfy_an_order(messages):
    e = trace(*({"role": "assistant", "calls": calls} for calls in messages))
    check = criterion("order", {"first": {"tool": "read"}, "then": {"tool": "edit"}}, places=["run"])
    assert engine.grade([check], e, judge=Judge()).checks[0].status == "fail"


def test_event_scope_uses_its_named_receipt_not_the_first_planted_step():
    e = trace(
        {"role": "user", "text": "First notice", "planted": True},
        {"role": "assistant", "calls": [call("bash")]},
        {"role": "user", "text": "Now stop", "planted": True},
        {"role": "assistant", "text": "DONE"},
        events=[{"id": "first", "step_index": 0}, {"id": "stop", "step_index": 2}],
    )
    check = ResolvedCheck(
        id="c",
        kind="after_event",
        parameters={"value": {"then": "no_calls_with_text"}, "event": "user_message"},
        scope={"activation": "event", "event_id": "stop", "places": ["after_event"]},
    )
    assert engine.grade([check], e, judge=Judge()).checks[0].status == "pass"
    assert e.steps[0].planted is True  # scope preparation never mutates the captured session


def test_setup_check_reports_irreparable_prefix_failure_separately():
    e = trace(
        {"role": "assistant", "calls": [call("bash")]},
        {"role": "user", "text": "Continue."},
        {"role": "assistant", "text": "DONE"},
        continuation_start=1,
    )
    check = ResolvedCheck(
        id="c", kind="calls", parameters={"value": {"mode": "never", "tool": "bash"}}, scope={"check_prefix": True}
    )
    result = engine.grade([check], e, judge=Judge())
    assert result.checks[0].status == "fail"
    assert result.checks[0].details["prefix_failures"]


def test_duplicate_ids_are_rejected_in_direct_package_calls():
    check = criterion("exact", "DONE")
    with pytest.raises(ValueError, match="unique"):
        engine.grade([check, check], trace({"role": "assistant", "text": "DONE"}), judge=Judge())


def test_bad_checker_parameters_return_error_not_a_default_verdict():
    check = criterion("calls", {"mode": "not_a_mode"}, places=["run"])
    result = engine.grade([check], trace({"role": "assistant", "text": "DONE"}), judge=Judge())
    assert result.checks[0].status == "error"
    assert result.if_all is None and result.if_frac is None


def test_uncertain_language_uses_the_same_judge_interface():
    judge = Judge(status="fail")
    result = engine.grade(
        [criterion("language", "Spanish")],
        trace({"role": "assistant", "text": "I fixed the bug and the tests pass now."}),
        judge=judge,
    )
    assert result.checks[0].status == "fail"
    assert len(judge.seen) == 1
    assert "Spanish" in judge.seen[0][0]


def test_judge_view_preserves_system_text_and_associates_results_by_call_identity():
    judge = Judge()
    e = trace(
        {"role": "system", "text": "Read before editing."},
        {
            "role": "assistant",
            "calls": [
                call("read", ident="first", rejected=True, model_tool="ReadSource"),
                call("read", ident="second", model_tool="ReadSource"),
            ],
        },
        {"role": "tool", "tool": "read", "call_id": "second", "invocation_id": "run", "output": "OK_SECOND"},
        {
            "role": "tool",
            "tool": "read",
            "call_id": "first",
            "invocation_id": "run",
            "output": "BAD_FIRST",
            "error": True,
        },
    )
    rule = ResolvedCheck(id="r", kind="judge", rubric="Use the tools correctly.")
    engine.grade([rule], e, judge=judge)
    shown = judge.seen[0][2]
    assert "SYSTEM: Read before editing." in shown
    assert "RESULT (ReadSource): OK_SECOND" in shown
    assert "RESULT (ReadSource, error, refused before it ran): BAD_FIRST" in shown
