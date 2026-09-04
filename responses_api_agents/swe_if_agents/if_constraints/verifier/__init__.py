"""Static constraint-template verifier, as a small package.

    core.py       data model (Turn, ToolCall, GradedStep), DEFAULT_RESOLVER, no-answer policies, text helpers
    matchers.py   MATCHERS registry (what an obligation checks) + value_key / no_answer_policy
    triggers.py   TRIGGERS registry (which turns a constraint applies to) + all_of composition
    templates.py  TEMPLATES registry (which surface is graded: turn_output, reply_output, tool_args, tool_choice)

Output contract (standing owner ruling)
---------------------------------------
`grade(turns, constraint, resolver=None)` returns a list of GradedStep, one per gradable step, each with reward 1 or 0.
If the trigger never fires the list is EMPTY (applicability == non-empty list; no abstain flag). Exception: `tool_choice`
is trajectory-scoped and ALWAYS returns exactly one GradedStep with turn=-1. `grade_ext` also returns the number of
silent in-scope turns (no visible text, or no final message) so the no-answer rate can be reported; under the `fail`
policy those turns are also among the returned steps (reward 0), under `ungradable` they are not steps at all.

Adding a matcher, a trigger (conditioner) or a template = one registry entry in the corresponding module, with examples;
`test_registry_conformance.py` checks every entry. Stdlib only; python 3.9 compatible.
"""
from typing import Any, Dict, List, Optional, Tuple

from .core import (ANY_TOOL, DEFAULT_RESOLVER, NO_ANSWER_POLICIES, NO_FINAL_DETAIL, NO_TOOL, SILENT_DETAIL,  # noqa: F401
                   SILENT_TURN_FAILS, SILENT_TURN_NOT_GRADABLE, GradedStep, ToolCall, Turn, is_silent_step,
                   _count_paired_fences, _count_sentences, _flatten_calls, _length_count, _matching_calls, _resolve,
                   _script_matches, _tool_trigger_fires, _SCRIPT_NAME_PREFIXES)
from .matchers import MATCHERS, Matcher, _apply_matcher, no_answer_policy, value_key  # noqa: F401
from .triggers import POSITIONS, TRIGGERS, Trigger, example_trace, missing_target, select_turns, trigger_kind  # noqa: F401
from .templates import TEMPLATES, Template, _grade_visible, _require_visible_target  # noqa: F401

__all__ = [
    "grade", "grade_ext", "Turn", "ToolCall", "GradedStep", "DEFAULT_RESOLVER", "NO_TOOL", "ANY_TOOL",
    "MATCHERS", "Matcher", "TRIGGERS", "Trigger", "TEMPLATES", "Template",
    "SILENT_TURN_FAILS", "SILENT_TURN_NOT_GRADABLE", "NO_ANSWER_POLICIES", "SILENT_DETAIL", "NO_FINAL_DETAIL",
    "no_answer_policy", "is_silent_step", "value_key", "select_turns", "missing_target", "trigger_kind", "example_trace", "POSITIONS",
]


def grade(turns, constraint, resolver=None):
    # type: (List[Turn], Dict[str, Any], Optional[Dict[str, str]]) -> List[GradedStep]
    """Grade one constraint over one trajectory. See the package docstring for the output contract."""
    return grade_ext(turns, constraint, resolver)[0]


def grade_ext(turns, constraint, resolver=None):
    # type: (List[Turn], Dict[str, Any], Optional[Dict[str, str]]) -> Tuple[List[GradedStep], int]
    """Like grade(), and also returns the number of silent in-scope turns (no visible text, or no final message)."""
    if resolver is None:
        resolver = DEFAULT_RESOLVER
    name = constraint.get("template")
    if name not in TEMPLATES:
        raise ValueError("unknown template %r" % (name,))
    template = TEMPLATES[name]
    trigger = constraint.get("trigger") or {}
    obligation = constraint.get("obligation") or {}
    policy = no_answer_policy(constraint) if template.applies_policy else SILENT_TURN_FAILS
    return template.grade(turns, trigger, obligation, resolver, policy)
