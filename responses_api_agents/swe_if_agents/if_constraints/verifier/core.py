"""Core of the constraint verifier: data model, resolver, text helpers, no-answer policies.

No registry lives here. `matchers.py`, `triggers.py` and `templates.py` build on this module; `__init__.py` exposes the
public API (`grade`, `grade_ext`, ...). Stdlib only; python 3.9 compatible (no match statements, no `X | None`).
"""
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

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
# No-answer policies (what a silent in-scope turn means for a matcher)
# --------------------------------------------------------------------------- #
SILENT_TURN_FAILS = "fail"                # the rule needs an answer: a silent turn is a graded step with reward 0
SILENT_TURN_NOT_GRADABLE = "ungradable"   # silence cannot violate the rule: a silent turn is not a step; only turns with text are graded
NO_ANSWER_POLICIES = (SILENT_TURN_FAILS, SILENT_TURN_NOT_GRADABLE)

SILENT_DETAIL = "silent turn: no visible text (a required shape needs an answer)"
NO_FINAL_DETAIL = "no final message: the episode ended with a tool call or an error"


def is_silent_step(step):
    """True when a graded step failed only because the turn had no visible text (or the episode had no final message)."""
    return step.detail.endswith(SILENT_DETAIL) or step.detail.endswith(NO_FINAL_DETAIL)


# --------------------------------------------------------------------------- #
# Text helpers shared by matchers
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
# Tool-call helpers shared by triggers and templates
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


def _flatten_calls(turns):
    flat = []
    for turn in turns:
        for c in turn.tool_calls:
            flat.append((turn.index, c))
    return flat
