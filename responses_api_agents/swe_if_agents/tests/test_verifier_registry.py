"""Conformance of every registry entry of the constraint verifier (matchers, triggers, templates). Generated from the
registries: a contributor who adds an entry gets these checks for free, and an entry without examples fails here."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from responses_api_agents.swe_if_agents.if_constraints import verifier as tv  # noqa: E402


class MatcherConformance(unittest.TestCase):
    def test_every_matcher_declares_examples_and_a_policy(self):
        for name, m in tv.MATCHERS.items():
            self.assertEqual(name, m.name)
            self.assertTrue(m.examples, "matcher %r declares no examples" % name)
            self.assertTrue(m.doc.strip(), "matcher %r has no doc" % name)
            for value in m.examples:
                self.assertIn(m.silence_policy(value), tv.NO_ANSWER_POLICIES, "matcher %r: bad policy for %r" % (name, value))
                self.assertIsInstance(m.value_key(value), str)

    def test_witness_passes_and_violation_fails(self):
        for name, m in tv.MATCHERS.items():
            for value in m.examples:
                w, v = m.witness(value), m.violation(value)
                self.assertTrue(w is not None or v is not None, "matcher %r: no witness and no violation for %r" % (name, value))
                if w is not None:
                    ok, why = tv._apply_matcher(name, value, w)
                    self.assertTrue(ok, "matcher %r: witness %r for %r fails: %s" % (name, w, value, why))
                if v is not None:
                    ok, why = tv._apply_matcher(name, value, v)
                    self.assertFalse(ok, "matcher %r: violation %r for %r passes" % (name, v, value))

    def test_no_answer_policy_reads_the_registry(self):
        for name, m in tv.MATCHERS.items():
            value = m.examples[0]
            c = {"template": "turn_output", "trigger": {"position": "final"}, "obligation": {"match": name, "value": value}}
            self.assertEqual(tv.no_answer_policy(c), m.silence_policy(value))
            wrong = tv.SILENT_TURN_FAILS if m.silence_policy(value) == tv.SILENT_TURN_NOT_GRADABLE else tv.SILENT_TURN_NOT_GRADABLE
            with self.assertRaises(ValueError):
                tv.no_answer_policy(dict(c, no_answer=wrong))

    def test_silent_turn_follows_the_declared_policy(self):
        silent = [tv.Turn(0, "", [tv.ToolCall("bash", {})]), tv.Turn(1, "", [], is_final=True)]
        for name, m in tv.MATCHERS.items():
            value = m.examples[0]
            c = {"template": "turn_output", "trigger": {"position": "any_turn"}, "obligation": {"match": name, "value": value}}
            steps, n_silent = tv.grade_ext(silent, c)
            self.assertEqual(n_silent, 2, name)
            if m.silence_policy(value) == tv.SILENT_TURN_FAILS:
                self.assertEqual([s.reward for s in steps], [0, 0], name)
                self.assertTrue(all(tv.is_silent_step(s) for s in steps), name)
            else:
                self.assertEqual(steps, [], name)


class TriggerConformance(unittest.TestCase):
    def test_every_trigger_declares_examples_that_select_the_expected_turns(self):
        trace = tv.example_trace()
        for key, t in tv.TRIGGERS.items():
            self.assertEqual(key, t.key)
            self.assertTrue(t.examples, "trigger %r declares no examples" % key)
            self.assertTrue(t.doc.strip(), "trigger %r has no doc" % key)
            for trigger, expected in t.examples:
                self.assertEqual(tv.trigger_kind(trigger), key)
                got = tuple(turn.index for turn, _ in tv.select_turns(trace, trigger))
                self.assertEqual(got, tuple(expected), "trigger %r: %r selected %r, expected %r" % (key, trigger, got, expected))

    def test_two_kinds_or_an_unowned_modifier_are_rejected(self):
        with self.assertRaises(ValueError):
            tv.trigger_kind({"position": "final", "tool": "BASH_TOOL_NAME"})
        with self.assertRaises(ValueError):
            tv.trigger_kind({"position": "final", "arg_predicate": {"field": "x", "regex": "y"}})
        with self.assertRaises(ValueError):
            tv.trigger_kind({})

    def test_final_without_a_final_message_is_one_no_answer(self):
        turns = [tv.Turn(0, "x", [tv.ToolCall("bash", {})]), tv.Turn(1, "y", [tv.ToolCall("bash", {})])]
        self.assertIsNotNone(tv.missing_target(turns, {"position": "final"}, tv.DEFAULT_RESOLVER))
        self.assertIsNone(tv.missing_target(turns, {"position": "any_turn"}, tv.DEFAULT_RESOLVER))
        self.assertIsNone(tv.missing_target(turns, {"tool": "BASH_TOOL_NAME"}, tv.DEFAULT_RESOLVER))


class TemplateConformance(unittest.TestCase):
    def test_every_template_grades_the_example_trace(self):
        trace = tv.example_trace()
        params = {
            "turn_output": {"trigger": {"position": "final"}, "obligation": {"match": "exact", "value": "Done."}},
            "reply_output": {"trigger": {"prev_tool": "BASH_TOOL_NAME"}, "obligation": {"match": "prefix", "value": "R"}},
            "tool_args": {"trigger": {"tool": "BASH_TOOL_NAME"}, "obligation": {"target": {"tool_arg": "command"}, "match": "regex", "value": "."}},
            "tool_choice": {"trigger": {"mode": "must_call", "tool": "BASH_TOOL_NAME"}, "obligation": {}},
        }
        self.assertEqual(set(params), set(tv.TEMPLATES), "every template needs an example here")
        for name, t in tv.TEMPLATES.items():
            self.assertTrue(t.doc.strip(), name)
            steps, n_silent = tv.grade_ext(trace, dict(params[name], template=name))
            self.assertTrue(steps, "template %r graded nothing on the example trace" % name)
            self.assertTrue(all(s.reward in (0, 1) for s in steps), name)
        with self.assertRaises(ValueError):
            tv.grade_ext(trace, {"template": "no_such_template", "trigger": {}, "obligation": {}})


if __name__ == "__main__":
    unittest.main()
