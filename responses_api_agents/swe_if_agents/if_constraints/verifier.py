"""Static constraint-template verifiers: turn_output / reply_output / tool_args / tool_choice.

Reference implementation of the four static templates from
`constraints_trace_map/templates.md` (sections 3 and 5), per VERIFIER_SPEC.md in this
directory. Stdlib only; python 3.9 compatible (no match statements, no `X | None`).

Output contract (standing owner ruling)
---------------------------------------
`grade(turns, constraint, resolver=None)` returns a list of GradedStep, one per
gradable step, each with reward 1 or 0. If the trigger never fires the list is EMPTY
(applicability == non-empty list; no abstain flag). Exception: `tool_choice`
(which tools / how many / in what order) is trajectory-scoped and ALWAYS returns
exactly one GradedStep with turn=-1 -- for a "must call T" obligation the empty-list
convention would silently convert "never complied" into "not applicable".
`tool_args` (argument-format contract) follows the per-call empty-list rule like
turn_output: one step per gradable call, empty if the tool is never called.

Matcher semantics deliberately documented here
----------------------------------------------
- All matchers evaluate on ``s = text.strip()``.
- `exact` is whitespace-tolerant at the ENDS and strict inside:
  ``s == value.strip()``. This is a deliberate decision.
- there is no `empty` matcher (removed 2026-09-03, decision D15); reasoning-only turns count as silent (the adapter
  excludes the reasoning channel from visible_text).
- `length_bound` sentence counting is NAIVE: sentences are split on `.` `!` `?`
  followed by whitespace or end-of-string; abbreviations ("e.g. ") and decimal
  points followed by space over-count, text with no terminal punctuation counts
  as one sentence.
- `language` is SCRIPT-LEVEL detection only: pass iff a strict majority of
  alphabetic characters fall in the expected script's unicode ranges.
  Latin-language distinctions (Spanish vs English) are NOT attempted.
- `fenced` counts OPENING fences only, and only properly PAIRED ones: opening =
  line starting with three backticks plus a non-empty info-string; closing = line
  of exactly three backticks. Closing fences and unpaired openers never count.
- `json_schema` parses the WHOLE message: a valid object followed by trailing
  garbage fails; non-dict JSON fails.
"""
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Pseudo tool identifiers (never resolved through the resolver)
# --------------------------------------------------------------------------- #
NO_TOOL = "NO_TOOL"    # trigger: a turn with zero tool calls
ANY_TOOL = "ANY_TOOL"  # trigger: a turn with >= 1 tool call

# --------------------------------------------------------------------------- #
# DEFAULT_RESOLVER: tool IDENTIFIER -> default concrete emitted name.
#
# Extracted verbatim from the PINNED clone
#   /lustre/fsw/portfolios/llmservice/users/charlwang/cluster/gym_workdir/
#     nv-OpenHands/openhands/llm/tool_names.py
# at pinned commit 7466868e2, with diversification OFF (DIVERSIFY_TOOL_NAMES
# unset -> every identifier takes its default, i.e. the first `_pick` argument)
# and camel-casing OFF. Covers the CodeAct section and the two OpenCode
# sections ("OpenCode-inspired" + "OpenCode additional") per the spec's
# "opencode and CodeAct identifiers"; the Codex-inspired and readonly-agent
# identifiers in that file are deliberately not included. A trigger value not
# present here is treated as a literal concrete name (covers never-diversified
# tools like `think`).
# --------------------------------------------------------------------------- #
DEFAULT_RESOLVER = {
    # ---------- CodeAct tools ----------
    "EXECUTE_BASH_TOOL_NAME": "execute_bash",
    "STR_REPLACE_EDITOR_TOOL_NAME": "str_replace_editor",
    "BROWSER_TOOL_NAME": "browser",
    "FINISH_TOOL_NAME": "finish",
    "LLM_BASED_EDIT_TOOL_NAME": "edit_file",
    "TASK_TRACKER_TOOL_NAME": "task_tracker",
    # ---------- OpenCode-inspired tools ----------
    "BASH_TOOL_NAME": "bash",
    "GLOB_TOOL_NAME": "glob",
    "GREP_TOOL_NAME": "grep",
    "LIST_DIR_TOOL_NAME": "list_dir",
    "READ_TOOL_NAME": "read",
    "WRITE_TOOL_NAME": "write",
    "EDIT_TOOL_NAME": "edit",
    # ---------- OpenCode additional tools ----------
    "OPENCODE_APPLY_PATCH_TOOL_NAME": "apply_patch",
    "QUESTION_TOOL_NAME": "question",
    "TODO_READ_TOOL_NAME": "todo_read",
    "TODO_WRITE_TOOL_NAME": "todo_write",
}


# --------------------------------------------------------------------------- #
# Data model (per VERIFIER_SPEC.md)
# --------------------------------------------------------------------------- #
@dataclass
class ToolCall:
    name: str                       # concrete emitted tool name (post-diversification)
    args: Dict[str, Any] = field(default_factory=dict)  # parsed args; {} if unparseable


@dataclass
class Turn:
    index: int                      # 0-based assistant-turn index
    visible_text: str = ""          # model-authored visible text; reasoning EXCLUDED
    tool_calls: List[ToolCall] = field(default_factory=list)
    is_final: bool = False          # last assistant turn of the episode
    # (role, text) messages between the previous assistant turn and this one
    preceding_messages: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class GradedStep:
    turn: int                       # assistant-turn index; -1 for trajectory-scoped
    reward: int                     # 1 or 0
    detail: str                     # short reason, human-readable


# --------------------------------------------------------------------------- #
# Matchers
# --------------------------------------------------------------------------- #
# Width-variant forms (FULLWIDTH LATIN ..., HALFWIDTH KATAKANA ...,
# HALFWIDTH HANGUL ...) belong to the same SCRIPT; their unicodedata names
# start with FULLWIDTH/HALFWIDTH, so those prefixes are listed explicitly.
_SCRIPT_NAME_PREFIXES = {
    "latin": ("LATIN", "FULLWIDTH LATIN"),
    "han": ("CJK",),                    # CJK UNIFIED IDEOGRAPH, CJK COMPATIBILITY ...
    "kana": ("HIRAGANA", "KATAKANA", "HALFWIDTH KATAKANA"),
    "hangul": ("HANGUL", "HALFWIDTH HANGUL"),
    "cyrillic": ("CYRILLIC",),
}


def _script_matches(ch, prefixes):
    name = unicodedata.name(ch, "")
    for p in prefixes:
        if name.startswith(p):
            return True
    return False


def _count_paired_fences(s, info_pattern):
    """Number of properly PAIRED fences whose info-string matches info_pattern."""
    open_info = None       # info-string of the currently open fence, else None
    matched = 0
    for line in s.splitlines():
        stripped = line.strip()
        if open_info is None:
            if stripped.startswith("```"):
                info = stripped[3:].strip()
                if info:                 # opener requires a NON-EMPTY info-string
                    open_info = info
                # bare ``` outside a fence: neither opener nor a countable match
        else:
            if stripped == "```":        # closing = line of exactly three backticks
                if re.search(info_pattern, open_info):
                    matched += 1
                open_info = None
            # any other line (including a ```lang line) is fence CONTENT
    # an unpaired opener left at EOF never counts
    return matched


def _count_sentences(s):
    if not s.strip():
        return 0
    parts = re.split(r"[.!?](?:\s+|$)", s)
    return len([p for p in parts if p.strip()])


def _length_count(s, unit):
    if unit == "lines":
        return len([ln for ln in s.splitlines() if ln.strip()])
    if unit == "words":
        return len(s.split())
    if unit == "sentences":
        return _count_sentences(s)
    if unit == "chars":
        return len(s)
    raise ValueError("length_bound: unknown unit %r" % (unit,))


# --------------------------------------------------------------------------- #
# Matcher registry
# --------------------------------------------------------------------------- #
# EVERY matcher is declared exactly once below, as a Matcher(...) with THREE mandatory parts: its check, its `silent_turn`
# property, and one line of documentation. `silent_turn` says what a silent in-scope turn (a bare tool call with no visible
# text; also an episode that never reaches a final message, for final-scope rules) means for the matcher. It has no default
# on purpose: whoever adds a matcher must decide it (owner ruling 2026-09-03).
SILENT_TURN_FAILS = "fail"                # the rule needs an answer: a silent turn is a graded step with reward 0
SILENT_TURN_NOT_GRADABLE = "ungradable"   # silence cannot violate the rule: a silent turn is not a step; only turns with text are graded
NO_ANSWER_POLICIES = (SILENT_TURN_FAILS, SILENT_TURN_NOT_GRADABLE)


def _m_exact(value, s):
    want = str(value).strip()
    ok = (s == want)
    return ok, ("ok" if ok else "expected exactly %r, got %r" % (want, s[:80]))


def _m_prefix(value, s):
    ok = s.startswith(str(value))
    return ok, ("ok" if ok else "must start with %r, got %r" % (value, s[:80]))


def _m_suffix(value, s):
    ok = s.endswith(str(value))
    return ok, ("ok" if ok else "must end with %r, got %r" % (value, s[-80:]))


def _m_regex(value, s):
    ok = re.search(str(value), s) is not None
    return ok, ("ok" if ok else "regex %r not found" % (value,))


def _m_forbidden(value, s):
    m = re.search(str(value), s)
    ok = m is None
    return ok, ("ok" if ok else "forbidden pattern %r matched %r" % (value, m.group(0)[:40]))


def _m_json_schema(value, s):
    try:
        obj = json.loads(s)
    except (ValueError, RecursionError):
        # RecursionError: json.loads on adversarially deep nesting (e.g. "[" * 200000) raises it instead of ValueError
        return False, "message is not a single valid JSON document"
    if not isinstance(obj, dict):
        return False, "JSON parses but is not an object (got %s)" % type(obj).__name__
    required = list((value or {}).get("required", []))
    missing = [k for k in required if k not in obj]
    if missing:
        return False, "JSON object missing required keys: %s" % (missing,)
    return True, "ok"


def _m_fenced(value, s):
    n = _count_paired_fences(s, str(value))
    ok = n >= 1
    return ok, ("ok (%d paired fence(s))" % n if ok
                else "no paired fence with info-string matching %r" % (value,))


def _m_length_bound(value, s):
    n = int(value["n"])
    unit = value["unit"]
    direction = value["dir"]
    count = _length_count(s, unit)
    if direction == "max":
        ok = count <= n
    elif direction == "min":
        ok = count >= n
    else:
        raise ValueError("length_bound: unknown dir %r" % (direction,))
    return ok, ("ok (%d %s)" % (count, unit) if ok
                else "%d %s violates %s %d" % (count, unit, direction, n))


def _length_bound_silent_turn(value):
    # a maximum cannot be violated by silence; a minimum needs an answer
    return SILENT_TURN_NOT_GRADABLE if (value or {}).get("dir") == "max" else SILENT_TURN_FAILS


def _m_language(value, s):
    prefixes = _SCRIPT_NAME_PREFIXES.get(str(value))
    if prefixes is None:
        raise ValueError("language: unknown script %r" % (value,))
    alpha = [ch for ch in s if ch.isalpha()]
    in_script = sum(1 for ch in alpha if _script_matches(ch, prefixes))
    ok = len(alpha) > 0 and in_script * 2 > len(alpha)  # STRICT majority
    return ok, ("ok (%d/%d %s)" % (in_script, len(alpha), value) if ok
                else "no strict %s majority (%d of %d alphabetic chars)"
                % (value, in_script, len(alpha)))


def _m_sentinel_exclusive(value, s):
    token = str(value)
    ok = (token not in s) or (s == token)
    return ok, ("ok" if ok
                else "contains sentinel %r but message is not exactly it" % (token,))


@dataclass(frozen=True)
class Matcher:
    """One obligation matcher. All three fields are mandatory.
    check(value, stripped_text) -> (ok, detail).
    silent_turn: SILENT_TURN_FAILS or SILENT_TURN_NOT_GRADABLE, or a function of the obligation value returning one of them
    (only length_bound needs that: a maximum is not gradable on silence, a minimum fails).
    doc: what the matcher checks, one line."""
    name: str
    check: Callable[[Any, str], Tuple[bool, str]]
    silent_turn: Any
    doc: str

    def __post_init__(self):
        if not (callable(self.silent_turn) or self.silent_turn in NO_ANSWER_POLICIES):
            raise ValueError("matcher %r: silent_turn must be SILENT_TURN_FAILS, SILENT_TURN_NOT_GRADABLE or a callable" % (self.name,))

    def silence_policy(self, value):
        return self.silent_turn(value) if callable(self.silent_turn) else self.silent_turn


MATCHERS = {m.name: m for m in (
    Matcher("exact", _m_exact, SILENT_TURN_FAILS, "the whole visible text equals the value (after strip)"),
    Matcher("prefix", _m_prefix, SILENT_TURN_FAILS, "the visible text starts with the value"),
    Matcher("suffix", _m_suffix, SILENT_TURN_FAILS, "the visible text ends with the value"),
    Matcher("regex", _m_regex, SILENT_TURN_FAILS, "re.search(value) finds a match in the visible text"),
    Matcher("forbidden", _m_forbidden, SILENT_TURN_NOT_GRADABLE, "re.search(value) finds NO match in the visible text"),
    Matcher("json_schema", _m_json_schema, SILENT_TURN_FAILS, "the visible text is one JSON object containing value['required'] keys"),
    Matcher("fenced", _m_fenced, SILENT_TURN_FAILS, "at least one paired code fence whose info string matches the value regex"),
    Matcher("length_bound", _m_length_bound, _length_bound_silent_turn, "count of value['unit'] is <= n (dir max) or >= n (dir min)"),
    Matcher("language", _m_language, SILENT_TURN_FAILS, "a strict majority of alphabetic characters belongs to the named script"),
    Matcher("sentinel_exclusive", _m_sentinel_exclusive, SILENT_TURN_NOT_GRADABLE, "if the sentinel token appears, the text is exactly the token"),
)}
# `empty` ("say nothing") was REMOVED on 2026-09-03 (decision D15): a silent turn cannot be told apart from a deliberate
# silence; the legitimate silence rule is an explicit token graded by `exact` (e.g. [SILENT]).


def _apply_matcher(match, value, text):
    """Returns (ok: bool, detail: str). `text` is the raw target text."""
    if match not in MATCHERS:
        raise ValueError("unknown matcher %r" % (match,))
    return MATCHERS[match].check(value, text.strip())


def no_answer_policy(constraint):
    """The constraint's no-answer kind, from the matcher registry (`Matcher.silent_turn`). A `no_answer` tag on the
    constraint is accepted only when it agrees with the registry."""
    ob = constraint.get("obligation") or {}
    m = ob.get("match")
    if m not in MATCHERS:
        raise ValueError("unknown matcher %r" % (m,))
    derived = MATCHERS[m].silence_policy(ob.get("value"))
    tag = constraint.get("no_answer")
    if tag is not None:
        if tag not in NO_ANSWER_POLICIES:
            raise ValueError("unknown no_answer policy %r" % (tag,))
        if tag != derived:
            raise ValueError("no_answer tag %r contradicts the matcher's silent_turn %r for %r" % (tag, derived, m))
    return derived


# --------------------------------------------------------------------------- #
# Trigger helpers
# --------------------------------------------------------------------------- #
def _resolve(identifier, resolver):
    """Identifier -> concrete name; pseudo-identifiers and unknown identifiers
    pass through (unknown == literal concrete name, e.g. `think`)."""
    if identifier in (NO_TOOL, ANY_TOOL):
        return identifier
    return resolver.get(identifier, identifier)


def _matching_calls(turn, resolved):
    if resolved == ANY_TOOL:
        return list(turn.tool_calls)
    return [c for c in turn.tool_calls if c.name == resolved]


def _tool_trigger_fires(turn, resolved, arg_predicate):
    if resolved == NO_TOOL:
        # no call exists, so an arg_predicate can never be satisfied
        return (len(turn.tool_calls) == 0) and (arg_predicate is None)
    calls = _matching_calls(turn, resolved)
    if not calls:
        return False
    if arg_predicate is None:
        return True
    fld = arg_predicate["field"]
    pattern = arg_predicate["regex"]
    for c in calls:
        if fld in c.args and re.search(pattern, str(c.args[fld])):
            return True
    return False


def _require_visible_target(obligation, template):
    target = obligation.get("target", "visible_message")
    if target != "visible_message":
        raise ValueError(
            "%s obligations only support target 'visible_message' (tool_arg is "
            "tool_args only); got %r" % (template, target))


SILENT_DETAIL = "silent turn: no visible text (a required shape needs an answer)"
NO_FINAL_DETAIL = "no final message: the episode ended with a tool call or an error"
def _grade_visible(turn, obligation, prefix, policy="fail"):
    """Grade one in-scope turn. Returns a GradedStep, or None when the turn is silent and the policy is 'ungradable'."""
    if not turn.visible_text.strip():
        if policy == "ungradable":
            return None
        return GradedStep(turn=turn.index, reward=0, detail=prefix + SILENT_DETAIL)
    ok, why = _apply_matcher(obligation["match"], obligation.get("value"),
                             turn.visible_text)
    return GradedStep(turn=turn.index, reward=1 if ok else 0,
                      detail=prefix + why)


def is_silent_step(step):
    """True when a graded step failed only because the turn had no visible text (or the episode had no final message)."""
    return step.detail.endswith(SILENT_DETAIL) or step.detail.endswith(NO_FINAL_DETAIL)


# --------------------------------------------------------------------------- #
# Template graders
# --------------------------------------------------------------------------- #
def _grade_turn_output(turns, trigger, obligation, resolver, policy="fail", silent=None):
    """`silent`, when given, is a one-element list that receives the number of silent in-scope turns (both kinds), so
    that callers can report the no-answer rate; the returned steps follow the no-answer policy."""
    _require_visible_target(obligation, "turn_output")
    has_tool = "tool" in trigger
    has_pos = "position" in trigger
    if has_tool == has_pos:
        raise ValueError("turn_output trigger needs exactly one of 'tool'/'position'")
    out = []
    n_silent = 0

    def add(turn, prefix):
        nonlocal n_silent
        if not turn.visible_text.strip():
            n_silent += 1
        step = _grade_visible(turn, obligation, prefix, policy)
        if step is not None:
            out.append(step)
    if has_tool:
        resolved = _resolve(trigger["tool"], resolver)
        arg_pred = trigger.get("arg_predicate")
        for turn in turns:
            if _tool_trigger_fires(turn, resolved, arg_pred):
                add(turn, "turn %d triggered by tool %s: " % (turn.index, resolved))
    else:
        if trigger.get("arg_predicate") is not None:
            raise ValueError("arg_predicate is only valid with a 'tool' trigger")
        position = trigger["position"]
        if position not in ("any_turn", "first_turn", "final"):
            raise ValueError("unknown position %r" % (position,))
        for turn in turns:
            fires = (position == "any_turn"
                     or (position == "first_turn" and turn.index == 0)
                     or (position == "final" and turn.is_final))
            if fires:
                add(turn, "turn %d (position=%s): " % (turn.index, position))
        # Final scope with no genuine final message (loop, error, timeout: the last turn still calls a tool): a required
        # shape has FAILED once; a no-answer-compliant rule is not gradable; either way it counts as a no-answer.
        if position == "final" and turns and not any(t.is_final for t in turns):
            n_silent += 1
            if policy == "fail":
                out.append(GradedStep(turn=turns[-1].index, reward=0,
                                      detail="turn %d (position=final): %s" % (turns[-1].index, NO_FINAL_DETAIL)))
    if silent is not None:
        silent[0] = n_silent
    return out


def _grade_reply_output(turns, trigger, obligation, resolver):
    _require_visible_target(obligation, "reply_output")
    has_pt = "prev_tool" in trigger
    has_pm = "prev_message" in trigger
    if has_pt == has_pm:
        raise ValueError(
            "reply_output trigger needs exactly one of 'prev_tool'/'prev_message'")
    out = []
    if has_pt:
        resolved = _resolve(trigger["prev_tool"], resolver)
        for i, turn in enumerate(turns):
            if i == 0:
                continue  # never fires on turn 0
            if _tool_trigger_fires(turns[i - 1], resolved, None):
                out.append(_grade_visible(turn, obligation,
                                          "turn %d after prev-turn tool %s: "
                                          % (turn.index, resolved)))
    else:
        pattern = trigger["prev_message"]
        for turn in turns:
            hit = any(role in ("user", "system") and re.search(pattern, text)
                      for role, text in turn.preceding_messages)
            if hit:
                out.append(_grade_visible(turn, obligation,
                                          "turn %d after message matching %r: "
                                          % (turn.index, pattern)))
    return out


def _flatten_calls(turns):
    flat = []
    for turn in turns:
        for c in turn.tool_calls:
            flat.append((turn.index, c))
    return flat


def _grade_tool_args(turns, trigger, obligation, resolver):
    """tool_args: argument-format contract. One GradedStep per (gradable) call of
    the triggering tool; empty list if it is never called."""
    resolved = _resolve(trigger["tool"], resolver)
    if resolved == NO_TOOL:
        raise ValueError("tool_args cannot target NO_TOOL")
    target = obligation.get("target")
    if not (isinstance(target, dict) and "tool_arg" in target):
        raise ValueError("tool_args requires obligation target {'tool_arg': <field>}")
    fld = target["tool_arg"]
    # Optional trigger filter: only calls whose predicate arg matches are
    # gradable (added after real-trace validation finding F3 -- without it a
    # path constraint on summary-writing exec calls over-fires on every exec
    # call in the episode). A call whose predicate field is MISSING does not
    # fire (unlike the obligation field, whose absence grades 0).
    arg_pred = trigger.get("arg_predicate")
    out = []
    for turn_index, call in _flatten_calls(turns):
        if resolved != ANY_TOOL and call.name != resolved:
            continue
        if arg_pred is not None:
            pf = arg_pred["field"]
            if pf not in call.args or not re.search(arg_pred["regex"],
                                                    str(call.args[pf])):
                continue
        if fld not in call.args:
            out.append(GradedStep(
                turn=turn_index, reward=0,
                detail="call of %s has no argument %r" % (call.name, fld)))
            continue
        ok, why = _apply_matcher(obligation["match"], obligation.get("value"),
                                 str(call.args[fld]))
        out.append(GradedStep(
            turn=turn_index, reward=1 if ok else 0,
            detail="call of %s arg %r: %s" % (call.name, fld, why)))
    return out  # empty list if T never called


def _grade_tool_choice(turns, trigger, obligation, resolver):
    """tool_choice: which tools, how many times, in what order. ALWAYS exactly
    one trajectory-scoped GradedStep (turn=-1) -- never an empty list."""
    mode = trigger.get("mode")
    if mode == "only_call":
        # Allowlist over the whole call stream (added after real-trace
        # validation finding F2 -- "you may only call these tools").
        allowed = {_resolve(t, resolver) for t in trigger["tools"]}
        offenders = sorted({c.name for _, c in _flatten_calls(turns)
                            if c.name not in allowed})
        ok = not offenders
        why = ("ok (all calls within allowlist %s)" % sorted(allowed) if ok
               else "calls outside allowlist %s: %s"
               % (sorted(allowed), offenders))
        return [GradedStep(turn=-1, reward=1 if ok else 0, detail=why)]

    if mode == "order":
        t1 = _resolve(trigger["first"], resolver)
        t2 = _resolve(trigger["then"], resolver)
        names = [c.name for _, c in _flatten_calls(turns)]
        t2_positions = [i for i, n in enumerate(names) if n == t2]
        if not t2_positions:
            return [GradedStep(turn=-1, reward=1,
                               detail="vacuous pass: %s never called" % (t2,))]
        for pos in t2_positions:
            if not any(names[i] == t1 for i in range(pos)):
                return [GradedStep(
                    turn=-1, reward=0,
                    detail="call #%d of %s has no earlier call of %s"
                    % (pos, t2, t1))]
        return [GradedStep(turn=-1, reward=1,
                           detail="ok: every %s call preceded by a %s call"
                           % (t2, t1))]

    if mode in ("must_call", "never_call", "exactly_n"):
        resolved = _resolve(trigger["tool"], resolver)
        count = len([1 for _, c in _flatten_calls(turns)
                     if resolved == ANY_TOOL or c.name == resolved])
        if mode == "must_call":
            ok = count >= 1
            why = ("ok (%d call(s) of %s)" % (count, resolved) if ok
                   else "%s never called (must_call)" % (resolved,))
        elif mode == "never_call":
            ok = count == 0
            why = ("ok (%s never called)" % (resolved,) if ok
                   else "%s called %d time(s) (never_call)" % (resolved, count))
        else:
            n = int(trigger["n"])
            ok = count == n
            why = ("ok (exactly %d call(s) of %s)" % (n, resolved) if ok
                   else "%s called %d time(s), required exactly %d"
                   % (resolved, count, n))
        return [GradedStep(turn=-1, reward=1 if ok else 0, detail=why)]

    raise ValueError("unknown tool_choice mode %r" % (mode,))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def grade(turns, constraint, resolver=None):
    # type: (List[Turn], Dict[str, Any], Optional[Dict[str, str]]) -> List[GradedStep]
    """Grade one constraint over one trajectory. See module docstring for the
    output contract."""
    return grade_ext(turns, constraint, resolver)[0]


def grade_ext(turns, constraint, resolver=None):
    # type: (List[Turn], Dict[str, Any], Optional[Dict[str, str]]) -> Tuple[List[GradedStep], int]
    """Like grade(), and also returns the number of silent in-scope turns (no visible text, or no final message) so that
    the no-answer rate can be reported. Under policy 'fail' those turns are also among the returned steps (reward 0);
    under 'ungradable' they are not steps at all."""
    if resolver is None:
        resolver = DEFAULT_RESOLVER
    template = constraint.get("template")
    trigger = constraint.get("trigger") or {}
    obligation = constraint.get("obligation") or {}
    if template == "turn_output":
        silent = [0]
        steps = _grade_turn_output(turns, trigger, obligation, resolver, no_answer_policy(constraint), silent)
        return steps, silent[0]
    return _grade_other(template, turns, trigger, obligation, resolver), 0


def _grade_other(template, turns, trigger, obligation, resolver):
    if template == "reply_output":
        return _grade_reply_output(turns, trigger, obligation, resolver)
    if template == "tool_args":
        return _grade_tool_args(turns, trigger, obligation, resolver)
    if template == "tool_choice":
        return _grade_tool_choice(turns, trigger, obligation, resolver)
    raise ValueError("unknown template %r" % (template,))
