"""Triggers (conditioners) and their registry: which turns of a trajectory a constraint applies to.

A trigger is the part of a verifier parameter that selects turns; the obligation (matcher) then grades each selected turn.
The trigger dict names its kind by KEY: {"position": "final"}, {"tool": "GREP_TOOL_NAME", "arg_predicate": {...}},
{"prev_tool": ...}, {"prev_message": ...}, {"all_of": [trigger, trigger]}. EVERY kind is declared exactly once in TRIGGERS
as a Trigger(...): its `select` function, the extra keys it owns (modifiers), one line of documentation, and `examples`
that the conformance test runs against the standard synthetic trace (`example_trace()`). To add a conditioner: write its
select, add one Trigger(...) entry with examples, run `test_registry_conformance.py`. Composition is `all_of`
(intersection of the selected turns), so a new conditioner never needs a new branch in the template code.

`select(turns, trigger, resolver)` returns the in-scope turns in trajectory order as (turn, detail_prefix) pairs.
`missing(turns, trigger, resolver)` (optional) returns a detail string when the trigger's target NEVER exists in the
trajectory although it should — today only `position: final` uses it (an episode with no final message): the template
then counts one no-answer and, under the `fail` policy, appends one failed step on the last turn.
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core import DEFAULT_RESOLVER, NO_FINAL_DETAIL, ToolCall, Turn, _resolve, _tool_trigger_fires


POSITIONS = ("any_turn", "first_turn", "final")


# --------------------------------------------------------------------------- #
# select functions
# --------------------------------------------------------------------------- #
def _sel_position(turns, trigger, resolver):
    position = trigger["position"]
    if position not in POSITIONS:
        raise ValueError("unknown position %r" % (position,))
    out = []
    for turn in turns:
        fires = (
            position == "any_turn"
            or (position == "first_turn" and turn.index == 0)
            or (position == "final" and turn.is_final)
        )
        if fires:
            out.append((turn, "turn %d (position=%s): " % (turn.index, position)))
    return out


def _missing_position(turns, trigger, resolver):
    # Final scope with no genuine final message (loop, error, timeout: the last turn still calls a tool): a required
    # shape has FAILED once; a no-answer-compliant rule is not gradable; either way it counts as a no-answer.
    if trigger["position"] == "final" and turns and not any(t.is_final for t in turns):
        return "turn %d (position=final): %s" % (turns[-1].index, NO_FINAL_DETAIL)
    return None


def _sel_tool(turns, trigger, resolver):
    resolved = _resolve(trigger["tool"], resolver)
    arg_pred = trigger.get("arg_predicate")
    return [
        (turn, "turn %d triggered by tool %s: " % (turn.index, resolved))
        for turn in turns
        if _tool_trigger_fires(turn, resolved, arg_pred)
    ]


def _sel_prev_tool(turns, trigger, resolver):
    resolved = _resolve(trigger["prev_tool"], resolver)
    out = []
    for i, turn in enumerate(turns):
        if i == 0:
            continue  # never fires on turn 0
        if _tool_trigger_fires(turns[i - 1], resolved, None):
            out.append((turn, "turn %d after prev-turn tool %s: " % (turn.index, resolved)))
    return out


def _sel_prev_message(turns, trigger, resolver):
    pattern = trigger["prev_message"]
    out = []
    for turn in turns:
        hit = any(role in ("user", "system") and re.search(pattern, text) for role, text in turn.preceding_messages)
        if hit:
            out.append((turn, "turn %d after message matching %r: " % (turn.index, pattern)))
    return out


def _sel_all_of(turns, trigger, resolver):
    parts = trigger["all_of"]
    if not isinstance(parts, list) or len(parts) < 2:
        raise ValueError("all_of needs a list of at least two triggers")
    selected = None
    for part in parts:
        indices = {t.index for t, _ in select_turns(turns, part, resolver)}
        selected = indices if selected is None else (selected & indices)
    return [(turn, "turn %d (all_of): " % (turn.index,)) for turn in turns if turn.index in selected]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Trigger:
    """One trigger kind, keyed by the trigger-dict key that names it.
    `select(turns, trigger, resolver) -> [(turn, detail_prefix)]`; `owns` = modifier keys this kind accepts beside its own;
    `doc`, one line; `missing` (optional), see the module docstring; `examples` = ((trigger_dict, expected_turn_indices),
    ...) evaluated on `example_trace()` by the conformance test."""

    key: str
    select: Callable[[List[Turn], Dict[str, Any], Dict[str, str]], List[Tuple[Turn, str]]]
    doc: str
    owns: Tuple[str, ...] = ()
    missing: Optional[Callable[[List[Turn], Dict[str, Any], Dict[str, str]], Optional[str]]] = None
    examples: Tuple[Tuple[Dict[str, Any], Tuple[int, ...]], ...] = ()


TRIGGERS: Dict[str, Trigger] = {
    t.key: t
    for t in (
        Trigger(
            "position",
            _sel_position,
            "a turn by position: any_turn, first_turn (index 0) or final (the last tool-free turn)",
            missing=_missing_position,
            examples=(
                ({"position": "any_turn"}, (0, 1, 2, 3)),
                ({"position": "first_turn"}, (0,)),
                ({"position": "final"}, (3,)),
            ),
        ),
        Trigger(
            "tool",
            _sel_tool,
            "a turn that calls the tool (identifier, ANY_TOOL or NO_TOOL), optionally filtered by arg_predicate",
            owns=("arg_predicate",),
            examples=(
                ({"tool": "BASH_TOOL_NAME"}, (0, 2)),
                ({"tool": "ANY_TOOL"}, (0, 1, 2)),
                ({"tool": "NO_TOOL"}, (3,)),
                ({"tool": "BASH_TOOL_NAME", "arg_predicate": {"field": "command", "regex": "pytest"}}, (2,)),
            ),
        ),
        Trigger(
            "prev_tool",
            _sel_prev_tool,
            "a turn whose PREVIOUS turn called the tool (never turn 0)",
            examples=(({"prev_tool": "BASH_TOOL_NAME"}, (1, 3)),),
        ),
        Trigger(
            "prev_message",
            _sel_prev_message,
            "a turn preceded by a user/system message matching the regex",
            examples=(({"prev_message": "please"}, (2,)),),
        ),
        Trigger(
            "all_of",
            _sel_all_of,
            "the intersection of two or more triggers",
            examples=(
                ({"all_of": [{"tool": "ANY_TOOL"}, {"position": "any_turn"}]}, (0, 1, 2)),
                ({"all_of": [{"tool": "BASH_TOOL_NAME"}, {"prev_tool": "BASH_TOOL_NAME"}]}, ()),
            ),
        ),
    )
}


def trigger_kind(trigger, allowed=None):
    """The registered kind named by this trigger dict: exactly one registered key must be present, and every other key
    must be a modifier that kind owns. `allowed` restricts the kinds a template accepts (None = any)."""
    kinds = [k for k in trigger if k in TRIGGERS]
    if allowed is not None:
        kinds = [k for k in kinds if k in allowed]
    if len(kinds) != 1:
        wanted = "/".join("'%s'" % k for k in (allowed if allowed is not None else TRIGGERS))
        raise ValueError("trigger needs exactly one of %s; got %s" % (wanted, sorted(trigger)))
    kind = kinds[0]
    extra = [k for k in trigger if k != kind and k not in TRIGGERS[kind].owns]
    if extra:
        raise ValueError("%r is only valid with a trigger that owns it; %r trigger got %s" % (extra[0], kind, extra))
    return kind


def select_turns(turns, trigger, resolver=None, allowed=None):
    """In-scope turns for a trigger dict, as (turn, detail_prefix) pairs in trajectory order."""
    kind = trigger_kind(trigger, allowed)
    return TRIGGERS[kind].select(turns, trigger, DEFAULT_RESOLVER if resolver is None else resolver)


def missing_target(turns, trigger, resolver=None, allowed=None):
    """Detail string when the trigger's target never exists in the trajectory (see module docstring), else None."""
    kind = trigger_kind(trigger, allowed)
    fn = TRIGGERS[kind].missing
    return fn(turns, trigger, DEFAULT_RESOLVER if resolver is None else resolver) if fn is not None else None


def example_trace():
    """The standard four-turn trajectory the trigger examples are evaluated on (default tool names):
    0: text + bash call; 1: text + read call; 2: text + bash pytest call, preceded by a user message "please ...";
    3: final text, no tool call."""
    return [
        Turn(0, "Starting.", [ToolCall("bash", {"command": "ls"})]),
        Turn(1, "Reading.", [ToolCall("read", {"path": "a.py"})]),
        Turn(
            2,
            "Testing.",
            [ToolCall("bash", {"command": "pytest -q"})],
            preceding_messages=[("user", "please run the tests")],
        ),
        Turn(3, "Done.", [], is_final=True),
    ]
