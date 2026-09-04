"""Behavioural tests of the constraint verifier (responses_api_agents/swe_if_agents/if_constraints/verifier/).

Every matcher x pass/fail; every trigger x fires/does-not-fire; empty-list (not applicable) semantics; NO_TOOL/ANY_TOOL;
arg_predicate; prev_message on turn 0; the robustness rules of the verifier spec; resolver fallback to literal names; the
no-answer policy; the matcher registry. Moved here from the design recipe (agentic-if/recipes/if-constraint-design) on
2026-09-04; the tests that need the recipe's real-trace adapter stayed there.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from responses_api_agents.swe_if_agents.if_constraints import verifier as tv  # noqa: E402
from responses_api_agents.swe_if_agents.if_constraints.verifier import ToolCall, Turn, grade  # noqa: E402



def mkturn(index, text="", calls=None, is_final=False, preceding=None):
    return Turn(index=index, visible_text=text,
                tool_calls=[ToolCall(name=n, args=a or {}) for n, a in (calls or [])],
                is_final=is_final, preceding_messages=preceding or [])


def one_turn_grade(text, match, value):
    """Grade a single always-firing constraint over one turn carrying `text`."""
    c = {"template": "turn_output", "trigger": {"position": "any_turn"},
         "obligation": {"match": match, "value": value}}
    steps = grade([mkturn(0, text, is_final=True)], c)
    assert len(steps) == 1
    return steps[0].reward


# =========================================================================== #
# Matchers: every matcher x pass/fail
# =========================================================================== #


class TestMatchers(unittest.TestCase):
    def test_exact_pass_and_fail(self):
        self.assertEqual(one_turn_grade("  DONE \n", "exact", "DONE"), 1)  # end-ws tolerant
        self.assertEqual(one_turn_grade("DONE!", "exact", "DONE"), 0)
        # strict INSIDE: internal whitespace differences fail
        self.assertEqual(one_turn_grade("D ONE", "exact", "DONE"), 0)

    def test_prefix_pass_and_fail(self):
        self.assertEqual(one_turn_grade("<block> stop", "prefix", "<block>"), 1)
        self.assertEqual(one_turn_grade("sure, <block>", "prefix", "<block>"), 0)

    def test_suffix_pass_and_fail(self):
        self.assertEqual(one_turn_grade("verdict: LGTM!", "suffix", "LGTM!"), 1)
        self.assertEqual(one_turn_grade("LGTM! thanks", "suffix", "LGTM!"), 0)

    def test_regex_pass_and_fail(self):
        self.assertEqual(one_turn_grade("<severity>3</severity>", "regex",
                                        r"^<severity>\d</severity>$"), 1)
        self.assertEqual(one_turn_grade("severity 3", "regex",
                                        r"^<severity>\d</severity>$"), 0)

    def test_forbidden_pass_and_fail(self):
        self.assertEqual(one_turn_grade("all plain text", "forbidden", r"[\U0001F600-\U0001F64F]"), 1)
        self.assertEqual(one_turn_grade("done \U0001F600", "forbidden", r"[\U0001F600-\U0001F64F]"), 0)

    def test_json_schema_pass_and_fail(self):
        self.assertEqual(one_turn_grade('{"a": 1}', "json_schema", {"required": ["a"]}), 1)
        self.assertEqual(one_turn_grade('{"b": 1}', "json_schema", {"required": ["a"]}), 0)

    def test_fenced_pass_and_fail(self):
        self.assertEqual(one_turn_grade("intro\n```cpp\nint x;\n```\n", "fenced", "cpp"), 1)
        self.assertEqual(one_turn_grade("no fence at all", "fenced", "cpp"), 0)
        # paired fence with WRONG info-string also fails
        self.assertEqual(one_turn_grade("```python\nx=1\n```", "fenced", "^cpp$"), 0)

    def test_length_bound_lines(self):
        text = "a\n\nb\nc"  # 3 non-empty lines
        self.assertEqual(one_turn_grade(text, "length_bound",
                                        {"n": 3, "unit": "lines", "dir": "max"}), 1)
        self.assertEqual(one_turn_grade(text, "length_bound",
                                        {"n": 2, "unit": "lines", "dir": "max"}), 0)

    def test_length_bound_words(self):
        self.assertEqual(one_turn_grade("one two three", "length_bound",
                                        {"n": 3, "unit": "words", "dir": "min"}), 1)
        self.assertEqual(one_turn_grade("one two", "length_bound",
                                        {"n": 3, "unit": "words", "dir": "min"}), 0)

    def test_length_bound_sentences(self):
        text = "First. Second! Third?"
        self.assertEqual(one_turn_grade(text, "length_bound",
                                        {"n": 3, "unit": "sentences", "dir": "max"}), 1)
        self.assertEqual(one_turn_grade(text, "length_bound",
                                        {"n": 2, "unit": "sentences", "dir": "max"}), 0)
        # documented naivety: no terminal punctuation == one sentence
        self.assertEqual(one_turn_grade("no punctuation here", "length_bound",
                                        {"n": 1, "unit": "sentences", "dir": "max"}), 1)

    def test_length_bound_chars(self):
        self.assertEqual(one_turn_grade("abcd", "length_bound",
                                        {"n": 4, "unit": "chars", "dir": "max"}), 1)
        self.assertEqual(one_turn_grade("abcde", "length_bound",
                                        {"n": 4, "unit": "chars", "dir": "max"}), 0)

    def test_language_pass_and_fail(self):
        self.assertEqual(one_turn_grade("你好世界 ok", "language", "han"), 1)
        self.assertEqual(one_turn_grade("hello world entirely latin", "language", "han"), 0)
        self.assertEqual(one_turn_grade("hello world", "language", "latin"), 1)
        self.assertEqual(one_turn_grade("Привет мир", "language", "cyrillic"), 1)
        self.assertEqual(one_turn_grade("こんにちは", "language", "kana"), 1)
        self.assertEqual(one_turn_grade("안녕하세요", "language", "hangul"), 1)
        # no alphabetic chars -> no strict majority -> fail
        self.assertEqual(one_turn_grade("1234 !!", "language", "latin"), 0)

    def test_sentinel_exclusive_pass_and_fail(self):
        self.assertEqual(one_turn_grade("HEARTBEAT_OK", "sentinel_exclusive", "HEARTBEAT_OK"), 1)
        self.assertEqual(one_turn_grade("all good, HEARTBEAT_OK", "sentinel_exclusive",
                                        "HEARTBEAT_OK"), 0)


# =========================================================================== #
# turn_output triggers
# =========================================================================== #


class TestTurnOutputTriggers(unittest.TestCase):
    def setUp(self):
        self.turns = [
            mkturn(0, "looking", [("bash", {"command": "ls"})]),
            mkturn(1, "", [("edit", {"file": "a.py"})]),
            mkturn(2, "done", is_final=True),
        ]

    def test_tool_trigger_fires_only_on_matching_turns(self):
        c = {"template": "turn_output", "trigger": {"tool": "BASH_TOOL_NAME"},
             "obligation": {"match": "exact", "value": "looking"}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(0, 1)])

    def test_tool_trigger_never_fires_empty_list(self):
        c = {"template": "turn_output", "trigger": {"tool": "BROWSER_TOOL_NAME"},
             "obligation": {"match": "regex", "value": "."}}
        self.assertEqual(grade(self.turns, c), [])

    def test_no_tool_pseudo(self):
        c = {"template": "turn_output", "trigger": {"tool": "NO_TOOL"},
             "obligation": {"match": "exact", "value": "done"}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(2, 1)])

    def test_any_tool_pseudo(self):
        c = {"template": "turn_output", "trigger": {"tool": "ANY_TOOL"},
             "obligation": {"match": "regex", "value": "."}}
        steps = grade(self.turns, c)
        # fires on turns 0 and 1; only the silent tool turn passes
        self.assertEqual([(s.turn, s.reward) for s in steps], [(0, 1), (1, 0)])

    def test_arg_predicate_gates_firing(self):
        c = {"template": "turn_output",
             "trigger": {"tool": "BASH_TOOL_NAME",
                         "arg_predicate": {"field": "command", "regex": r"^ls"}},
             "obligation": {"match": "exact", "value": "looking"}}
        self.assertEqual(len(grade(self.turns, c)), 1)
        c["trigger"]["arg_predicate"] = {"field": "command", "regex": r"^rm"}
        self.assertEqual(grade(self.turns, c), [])
        # missing field never satisfies the predicate
        c["trigger"]["arg_predicate"] = {"field": "nope", "regex": r"."}
        self.assertEqual(grade(self.turns, c), [])

    def test_position_any_turn(self):
        c = {"template": "turn_output", "trigger": {"position": "any_turn"},
             "obligation": {"match": "forbidden", "value": r"!!!"}}
        # three turns, one of them silent: a ban is no-answer-compliant, so the silent turn is not a step (counted instead)
        steps, n_silent = tv.grade_ext(self.turns, c)
        self.assertEqual((len(steps), n_silent), (2, 1))
        self.assertEqual(grade([], c), [])  # no turns -> empty list

    def test_position_first_turn(self):
        c = {"template": "turn_output", "trigger": {"position": "first_turn"},
             "obligation": {"match": "prefix", "value": "look"}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(0, 1)])

    def test_position_final(self):
        c = {"template": "turn_output", "trigger": {"position": "final"},
             "obligation": {"match": "exact", "value": "done"}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(2, 1)])

    def test_resolver_fallback_to_literal_name(self):
        turns = [mkturn(0, "", [("think", {"thought": "hmm"})], is_final=True)]
        c = {"template": "turn_output", "trigger": {"tool": "think"},
             "obligation": {"match": "regex", "value": "."}}
        steps = grade(turns, c)  # 'think' is not in DEFAULT_RESOLVER -> literal
        self.assertEqual([(s.turn, s.reward) for s in steps], [(0, 0)])

    def test_custom_resolver_overrides_default(self):
        turns = [mkturn(0, "x", [("shell_run", {})], is_final=True)]
        c = {"template": "turn_output", "trigger": {"tool": "EXECUTE_BASH_TOOL_NAME"},
             "obligation": {"match": "exact", "value": "x"}}
        self.assertEqual(grade(turns, c), [])  # default resolver: execute_bash
        steps = grade(turns, c, resolver={"EXECUTE_BASH_TOOL_NAME": "shell_run"})
        self.assertEqual([(s.turn, s.reward) for s in steps], [(0, 1)])


# =========================================================================== #
# reply_output triggers
# =========================================================================== #


class TestReplyOutputTriggers(unittest.TestCase):
    def setUp(self):
        self.turns = [
            mkturn(0, "starting", [("execute_bash", {"command": "pytest"})],
                   preceding=[("system", "you are an agent"),
                              ("user", "please run the tests")]),
            mkturn(1, "tests: 3 passed",
                   preceding=[("tool", "3 passed in 0.2s")]),
            mkturn(2, "HEARTBEAT_OK", is_final=True,
                   preceding=[("user", "[OpenClaw heartbeat poll]")]),
        ]

    def test_prev_tool_fires_on_following_turn(self):
        c = {"template": "reply_output", "trigger": {"prev_tool": "EXECUTE_BASH_TOOL_NAME"},
             "obligation": {"match": "prefix", "value": "tests:"}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(1, 1)])

    def test_prev_tool_never_fires_on_turn_0(self):
        # turn 0 itself calls the tool; nothing precedes it -> no firing on 0
        turns = [mkturn(0, "x", [("execute_bash", {})], is_final=True)]
        c = {"template": "reply_output", "trigger": {"prev_tool": "EXECUTE_BASH_TOOL_NAME"},
             "obligation": {"match": "regex", "value": "."}}
        self.assertEqual(grade(turns, c), [])

    def test_prev_tool_no_tool_pseudo(self):
        c = {"template": "reply_output", "trigger": {"prev_tool": "NO_TOOL"},
             "obligation": {"match": "exact", "value": "HEARTBEAT_OK"}}
        steps = grade(self.turns, c)
        # turn 1 has no calls -> fires on turn 2
        self.assertEqual([(s.turn, s.reward) for s in steps], [(2, 1)])

    def test_prev_tool_any_tool_pseudo(self):
        c = {"template": "reply_output", "trigger": {"prev_tool": "ANY_TOOL"},
             "obligation": {"match": "prefix", "value": "tests:"}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(1, 1)])

    def test_prev_message_fires(self):
        c = {"template": "reply_output",
             "trigger": {"prev_message": r"heartbeat poll"},
             "obligation": {"match": "exact", "value": "HEARTBEAT_OK"}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(2, 1)])

    def test_prev_message_fires_on_turn_0_problem_statement(self):
        c = {"template": "reply_output",
             "trigger": {"prev_message": r"run the tests"},
             "obligation": {"match": "prefix", "value": "starting"}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(0, 1)])

    def test_prev_message_ignores_tool_role(self):
        c = {"template": "reply_output",
             "trigger": {"prev_message": r"3 passed"},
             "obligation": {"match": "regex", "value": "."}}
        self.assertEqual(grade(self.turns, c), [])  # only user/system count

    def test_prev_message_never_fires_empty_list(self):
        c = {"template": "reply_output",
             "trigger": {"prev_message": r"no such marker"},
             "obligation": {"match": "regex", "value": "."}}
        self.assertEqual(grade(self.turns, c), [])


# =========================================================================== #
# tool_args / tool_choice
# =========================================================================== #


class TestToolStream(unittest.TestCase):
    def setUp(self):
        self.turns = [
            mkturn(0, "", [("grep", {"pattern": "foo"})]),
            mkturn(1, "", [("read", {"path": "a.py"}),
                           ("approve_code_changes", {"comment": "LGTM!"})]),
            mkturn(2, "", [("approve_code_changes", {"comment": "looks good"})],
                   is_final=True),
        ]

    def test_per_call_one_step_per_call(self):
        c = {"template": "tool_args",
             "trigger": {"tool": "approve_code_changes"},
             "obligation": {"target": {"tool_arg": "comment"},
                            "match": "exact", "value": "LGTM!"}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(1, 1), (2, 0)])

    def test_per_call_missing_field_rewards_0_and_names_field(self):
        turns = [mkturn(0, "", [("approve_code_changes", {})], is_final=True)]
        c = {"template": "tool_args",
             "trigger": {"tool": "approve_code_changes"},
             "obligation": {"target": {"tool_arg": "comment"},
                            "match": "exact", "value": "LGTM!"}}
        steps = grade(turns, c)
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].reward, 0)
        self.assertIn("comment", steps[0].detail)

    def test_per_call_empty_list_when_never_called(self):
        c = {"template": "tool_args",
             "trigger": {"tool": "browser"},
             "obligation": {"target": {"tool_arg": "url"},
                            "match": "regex", "value": r"^https://"}}
        self.assertEqual(grade(self.turns, c), [])

    def test_stream_must_call_pass_and_fail(self):
        c = {"template": "tool_choice",
             "trigger": {"mode": "must_call", "tool": "GREP_TOOL_NAME"},
             "obligation": {"match": "exact", "value": ""}}
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(-1, 1)])
        c["trigger"]["tool"] = "BROWSER_TOOL_NAME"
        steps = grade(self.turns, c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(-1, 0)])

    def test_stream_never_call_pass_and_fail(self):
        c = {"template": "tool_choice",
             "trigger": {"mode": "never_call", "tool": "BROWSER_TOOL_NAME"},
             "obligation": {}}
        self.assertEqual(grade(self.turns, c)[0].reward, 1)
        c["trigger"]["tool"] = "GREP_TOOL_NAME"
        self.assertEqual(grade(self.turns, c)[0].reward, 0)

    def test_stream_exactly_n_pass_and_fail(self):
        c = {"template": "tool_choice",
             "trigger": {"mode": "exactly_n", "n": 2,
                         "tool": "approve_code_changes"},
             "obligation": {}}
        self.assertEqual(grade(self.turns, c)[0].reward, 1)
        c["trigger"]["n"] = 1
        self.assertEqual(grade(self.turns, c)[0].reward, 0)

    def test_order_pass_fail_and_vacuous(self):
        c = {"template": "tool_choice",
             "trigger": {"mode": "order", "first": "GREP_TOOL_NAME",
                         "then": "READ_TOOL_NAME"},
             "obligation": {}}
        steps = grade(self.turns, c)  # grep (flat #0) precedes read (flat #1)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(-1, 1)])
        # reversed: every grep call needs an earlier read call -> fail
        c["trigger"] = {"mode": "order", "first": "READ_TOOL_NAME",
                        "then": "GREP_TOOL_NAME"}
        self.assertEqual(grade(self.turns, c)[0].reward, 0)
        # vacuous pass when `then` never called; detail notes vacuity
        c["trigger"] = {"mode": "order", "first": "GREP_TOOL_NAME",
                        "then": "BROWSER_TOOL_NAME"}
        step = grade(self.turns, c)[0]
        self.assertEqual((step.turn, step.reward), (-1, 1))
        self.assertIn("vacuous", step.detail.lower())


# =========================================================================== #
# The 7 robustness rules of VERIFIER_SPEC.md
# =========================================================================== #


class TestConfigValidation(unittest.TestCase):
    def test_unknown_template_and_matcher_raise(self):
        with self.assertRaises(ValueError):
            grade([], {"template": "nope", "trigger": {}, "obligation": {}})
        with self.assertRaises(ValueError):
            one_turn_grade("x", "unknown_matcher", "v")

    def test_turn_output_needs_exactly_one_trigger_arm(self):
        with self.assertRaises(ValueError):
            grade([], {"template": "turn_output",
                       "trigger": {"tool": "BASH_TOOL_NAME", "position": "final"},
                       "obligation": {"match": "regex", "value": "."}})
        with self.assertRaises(ValueError):
            grade([], {"template": "turn_output", "trigger": {},
                       "obligation": {"match": "regex", "value": "."}})

    def test_tool_arg_target_only_under_per_call(self):
        with self.assertRaises(ValueError):
            grade([mkturn(0, "x", is_final=True)],
                  {"template": "turn_output", "trigger": {"position": "final"},
                   "obligation": {"target": {"tool_arg": "f"},
                                  "match": "exact", "value": "v"}})
        with self.assertRaises(ValueError):
            grade([], {"template": "tool_args",
                       "trigger": {"tool": "bash"},
                       "obligation": {"match": "exact", "value": "v"}})


# =========================================================================== #
# Multi-constraint smoke test over one synthetic trajectory
# =========================================================================== #


class TestValidationFollowups(unittest.TestCase):
    """Regression tests for the two additions after real-trace validation
    (VALIDATION.md findings F2 and F3)."""

    def _stream_turns(self):
        return [
            mkturn(0, "", calls=[("exec", {"command": "ls -la"})]),
            mkturn(1, "", calls=[("exec", {"command": "echo hi > notes/2026-08-31-summary.md"}),
                                 ("memory", {"op": "get"})]),
            mkturn(2, "done", is_final=True),
        ]

    # ---- F3: arg_predicate on tool_args ----
    def test_per_call_arg_predicate_filters_gradable_calls(self):
        c = {"template": "tool_args",
             "trigger": {"tool": "exec",
                         "arg_predicate": {"field": "command",
                                           "regex": r"summary\.md"}},
             "obligation": {"target": {"tool_arg": "command"},
                            "match": "regex",
                            "value": r"notes/2026-[0-9]{2}-[0-9]{2}-summary\.md"}}
        steps = grade(self._stream_turns(), c, resolver={})
        # only the summary-writing exec call is gradable; the plain ls is not
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].turn, 1)
        self.assertEqual(steps[0].reward, 1)

    def test_per_call_arg_predicate_missing_field_does_not_fire(self):
        turns = [mkturn(0, "", calls=[("exec", {})])]
        c = {"template": "tool_args",
             "trigger": {"tool": "exec",
                         "arg_predicate": {"field": "command", "regex": "."}},
             "obligation": {"target": {"tool_arg": "command"},
                            "match": "regex", "value": "."}}
        self.assertEqual(grade(turns, c, resolver={}), [])

    def test_per_call_without_predicate_still_grades_all_calls(self):
        c = {"template": "tool_args",
             "trigger": {"tool": "exec"},
             "obligation": {"target": {"tool_arg": "command"},
                            "match": "regex", "value": r"summary\.md"}}
        steps = grade(self._stream_turns(), c, resolver={})
        self.assertEqual([s.reward for s in steps], [0, 1])

    # ---- F2: only_call allowlist mode ----
    def test_only_call_flags_out_of_allowlist_tool(self):
        c = {"template": "tool_choice",
             "trigger": {"mode": "only_call",
                         "tools": ["memory", "skill_manage"]}}
        steps = grade(self._stream_turns(), c, resolver={})
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].turn, -1)
        self.assertEqual(steps[0].reward, 0)
        self.assertIn("exec", steps[0].detail)

    def test_only_call_passes_when_all_calls_allowed(self):
        turns = [mkturn(0, "", calls=[("memory", {})]),
                 mkturn(1, "done", is_final=True)]
        c = {"template": "tool_choice",
             "trigger": {"mode": "only_call",
                         "tools": ["memory"]}}
        steps = grade(turns, c, resolver={})
        self.assertEqual([(s.turn, s.reward) for s in steps], [(-1, 1)])

    def test_only_call_vacuous_pass_and_resolver(self):
        turns = [mkturn(0, "no calls at all", is_final=True)]
        c = {"template": "tool_choice",
             "trigger": {"mode": "only_call",
                         "tools": ["BASH_TOOL_NAME"]}}
        steps = grade(turns, c)  # DEFAULT_RESOLVER: BASH_TOOL_NAME -> bash
        self.assertEqual([(s.turn, s.reward) for s in steps], [(-1, 1)])
        turns2 = [mkturn(0, "", calls=[("bash", {"command": "ls"})],
                         is_final=True)]
        self.assertEqual(grade(turns2, c)[0].reward, 1)


class TestNoAnswerPolicy(unittest.TestCase):
    """Owner ruling 2026-09-03, two kinds of constraints. Required shapes ('fail'): a silent in-scope turn is a step with
    reward 0 and an episode with no final message fails its final rules once. No-answer-compliant rules ('ungradable':
    bans, maximum bounds, sentinels): a silent turn is not a step; only turns with text are graded. `empty` is removed. grade_ext() also returns the number of silent in-scope turns for the no-answer rate."""

    def _turns(self, texts, final_is_message=True):
        turns = []
        for i, t in enumerate(texts):
            last = i == len(texts) - 1
            calls = [] if (last and final_is_message) else [ToolCall(name="bash", args={})]
            turns.append(Turn(index=i, visible_text=t, tool_calls=calls, is_final=(last and final_is_message)))
        return turns

    def test_policy_derivation(self):
        pol = lambda m, v=None: tv.no_answer_policy({"obligation": {"match": m, "value": v}})
        for m in ("prefix", "exact", "fenced", "json_schema", "regex", "language"):
            self.assertEqual(pol(m), "fail", m)
        self.assertEqual(pol("length_bound", {"n": 3, "unit": "lines", "dir": "min"}), "fail")
        self.assertEqual(pol("length_bound", {"n": 3, "unit": "lines", "dir": "max"}), "ungradable")
        for m in ("forbidden", "sentinel_exclusive"):
            self.assertEqual(pol(m), "ungradable", m)
        with self.assertRaises(ValueError):
            pol("empty")   # removed 2026-09-03 (decision D15)
        self.assertEqual(tv.no_answer_policy({"no_answer": "ungradable", "obligation": {"match": "forbidden"}}), "ungradable")   # a matching tag is accepted
        with self.assertRaises(ValueError):
            tv.no_answer_policy({"no_answer": "fail", "obligation": {"match": "forbidden"}})   # a tag cannot override the matcher's kind
        with self.assertRaises(ValueError):
            tv.no_answer_policy({"no_answer": "maybe", "obligation": {"match": "forbidden"}})

    def test_required_shape_silent_turn_is_a_failed_step(self):
        c = {"template": "turn_output", "trigger": {"position": "any_turn"}, "obligation": {"match": "prefix", "value": "[LOG]"}}
        steps, n_silent = tv.grade_ext(self._turns(["", "[LOG] done"]), c)
        self.assertEqual([s.reward for s in steps], [0, 1]); self.assertEqual(n_silent, 1)
        self.assertTrue(tv.is_silent_step(steps[0])); self.assertFalse(tv.is_silent_step(steps[1]))

    def test_ban_silent_turn_is_not_a_step_but_is_counted(self):
        c = {"template": "turn_output", "trigger": {"position": "any_turn"}, "obligation": {"match": "forbidden", "value": "!"}}
        steps, n_silent = tv.grade_ext(self._turns(["", "done!", "done"]), c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(1, 0), (2, 1)]); self.assertEqual(n_silent, 1)

    def test_maximum_silent_turn_is_not_a_step(self):
        c = {"template": "turn_output", "trigger": {"tool": "BASH_TOOL_NAME"}, "obligation": {"match": "length_bound", "value": {"n": 3, "unit": "lines", "dir": "max"}}}
        steps, n_silent = tv.grade_ext(self._turns(["", "x"]), c)
        self.assertEqual(steps, []); self.assertEqual(n_silent, 1)

    def test_minimum_silent_turn_fails(self):
        c = {"template": "turn_output", "trigger": {"tool": "BASH_TOOL_NAME"}, "obligation": {"match": "length_bound", "value": {"n": 1, "unit": "words", "dir": "min"}}}
        steps, n_silent = tv.grade_ext(self._turns(["", "x"]), c)
        self.assertEqual([s.reward for s in steps], [0]); self.assertEqual(n_silent, 1)

    def test_unfinished_episode_required_final_fails_once(self):
        c = {"template": "turn_output", "trigger": {"position": "final"}, "obligation": {"match": "prefix", "value": "DONE:"}}
        steps, n_silent = tv.grade_ext(self._turns(["a", "b"], final_is_message=False), c)
        self.assertEqual([(s.turn, s.reward) for s in steps], [(1, 0)]); self.assertEqual(n_silent, 1); self.assertTrue(tv.is_silent_step(steps[0]))

    def test_unfinished_episode_ban_final_not_gradable(self):
        c = {"template": "turn_output", "trigger": {"position": "final"}, "obligation": {"match": "forbidden", "value": "!"}}
        steps, n_silent = tv.grade_ext(self._turns(["a", "b"], final_is_message=False), c)
        self.assertEqual(steps, []); self.assertEqual(n_silent, 1)

    def test_finished_episode_unchanged(self):
        c = {"template": "turn_output", "trigger": {"position": "final"}, "obligation": {"match": "prefix", "value": "DONE:"}}
        self.assertEqual([s.reward for s in grade(self._turns(["a", "DONE: b"]), c)], [1])

    def test_grade_still_returns_steps_only(self):
        c = {"template": "turn_output", "trigger": {"position": "any_turn"}, "obligation": {"match": "forbidden", "value": "!"}}
        self.assertEqual(len(grade(self._turns(["", "ok"]), c)), 1)


class TestMatcherRegistry(unittest.TestCase):
    """Every matcher is declared once in MATCHERS with a mandatory silent_turn; a declaration without one is rejected."""

    def test_every_matcher_declares_silent_turn(self):
        for name, m in tv.MATCHERS.items():
            self.assertEqual(m.name, name)
            self.assertTrue(callable(m.check) and m.doc)
            values = ({"n": 3, "unit": "lines", "dir": "max"}, {"n": 3, "unit": "lines", "dir": "min"}) if name == "length_bound" else ("x",)
            for value in values:
                self.assertIn(m.silence_policy(value), tv.NO_ANSWER_POLICIES, name)

    def test_registry_rejects_a_matcher_without_a_valid_silent_turn(self):
        with self.assertRaises(ValueError):
            tv.Matcher("bogus", lambda v, s: (True, "ok"), "maybe", "a matcher that forgot to decide")
        with self.assertRaises(TypeError):
            tv.Matcher("bogus", lambda v, s: (True, "ok"))   # silent_turn and doc are not optional

    def test_empty_is_gone(self):
        self.assertNotIn("empty", tv.MATCHERS)
        with self.assertRaises(ValueError):
            tv._apply_matcher("empty", None, "")


if __name__ == "__main__":
    unittest.main()
