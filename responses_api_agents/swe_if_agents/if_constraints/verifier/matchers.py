"""Obligation matchers and their registry.

EVERY matcher is declared exactly once in MATCHERS, as a Matcher(...) whose mandatory parts are its check, its
`silent_turn` policy and one line of documentation (owner ruling 2026-09-03: whoever adds a matcher must decide what a
silent in-scope turn means for it; there is no default). The optional parts let the rest of the pipeline and the
conformance tests discover the matcher instead of hard-coding it: `value_key` (the key the value pools and the no-op gate
use), `witness` (one compliant text for a value), `violation` (one non-compliant text), `examples` (values the conformance
test exercises) and `instruction_kind` (the key the phrasing layer uses). To add a matcher: write its check, add one
Matcher(...) entry with examples, run `test_registry_conformance.py`.

Semantics deliberately documented here
- All matchers evaluate on ``s = text.strip()``.
- `exact` is whitespace-tolerant at the ENDS and strict inside: ``s == value.strip()``. This is a deliberate decision.
- there is no `empty` matcher (removed 2026-09-03, decision D15); reasoning-only turns count as silent (the adapter
  excludes the reasoning channel from visible_text).
- `length_bound` sentence counting is NAIVE: sentences are split on `.` `!` `?` followed by whitespace or end-of-string;
  abbreviations ("e.g. ") and decimal points followed by space over-count, text with no terminal punctuation counts as one.
- `language` is SCRIPT-LEVEL detection only: pass iff a strict majority of alphabetic characters fall in the expected
  script's unicode ranges. Latin-language distinctions (Spanish vs English) are NOT attempted.
- `fenced` counts OPENING fences only, and only properly PAIRED ones: opening = line starting with three backticks plus a
  non-empty info-string; closing = line of exactly three backticks. Closing fences and unpaired openers never count.
- `json_schema` parses the WHOLE message: a valid object followed by trailing garbage fails; non-dict JSON fails.
"""
import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .core import (NO_ANSWER_POLICIES, SILENT_TURN_FAILS, SILENT_TURN_NOT_GRADABLE, _SCRIPT_NAME_PREFIXES,
                   _count_paired_fences, _length_count, _script_matches)

_REGEX_META = re.compile(r"[\\^$.|?*+()\[\]{}]")


def _literal(pattern):
    """The pattern itself when it is a plain literal (no regex metacharacters), else None."""
    return None if _REGEX_META.search(str(pattern)) else str(pattern)


# --------------------------------------------------------------------------- #
# checks: (value, stripped_text) -> (ok, detail)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# witnesses / violations (one compliant and one non-compliant text per value; None = none can be built generically)
# --------------------------------------------------------------------------- #
_SCRIPT_SAMPLE = {"latin": "hello world", "han": "你好世界", "kana": "こんにちは", "hangul": "안녕하세요", "cyrillic": "привет мир"}


def _units(n, unit):
    """A text with exactly n units of the given kind."""
    if unit == "words":
        return " ".join(["word"] * n)
    if unit == "lines":
        return "\n".join(["line"] * n)
    if unit == "sentences":
        return " ".join(["Done."] * n)
    if unit == "chars":
        return "x" * n
    raise ValueError("length_bound: unknown unit %r" % (unit,))


def _w_length_bound(value):
    return _units(int(value["n"]), value["unit"])


def _v_length_bound(value):
    n = int(value["n"])
    if value["dir"] == "max":
        return _units(n + 1, value["unit"])
    return _units(n - 1, value["unit"]) if n > 0 else None


def _w_regex(value):
    return _literal(value)


def _v_regex(value):
    return None if re.search(str(value), "zzz") else "zzz"


def _v_forbidden(value):
    lit = _literal(value)
    return None if lit is None else "text with %s inside" % lit


def _w_fenced(value):
    lit = _literal(value)
    return None if lit is None else "```%s\ncontent\n```" % lit


def _w_json_schema(value):
    return json.dumps({k: "x" for k in (value or {}).get("required", [])})


def _v_language(value):
    return _SCRIPT_SAMPLE["cyrillic"] if str(value) == "latin" else _SCRIPT_SAMPLE["latin"]


def _vk_length_bound(value):
    return "%s:%s:%s" % (value["unit"], value["dir"], value["n"])


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Matcher:
    """One obligation matcher.

    Mandatory: `name`; `check(value, stripped_text) -> (ok, detail)`; `silent_turn` = SILENT_TURN_FAILS or
    SILENT_TURN_NOT_GRADABLE, or a function of the obligation value returning one of them (only length_bound needs that:
    a maximum is not gradable on silence, a minimum fails); `doc`, one line.
    Optional (used by the pools, the gate, the phrasing layer and the conformance tests): `value_key(value) -> str`,
    `witness(value) -> Optional[str]` (a text that passes), `violation(value) -> Optional[str]` (a text that fails),
    `examples` (values the conformance test runs), `instruction_kind` (phrasing key; defaults to the name)."""
    name: str
    check: Callable[[Any, str], Tuple[bool, str]]
    silent_turn: Any
    doc: str
    value_key: Callable[[Any], str] = lambda value: str(value)
    witness: Callable[[Any], Optional[str]] = lambda value: None
    violation: Callable[[Any], Optional[str]] = lambda value: None
    examples: Tuple[Any, ...] = ()
    instruction_kind: str = ""

    def __post_init__(self):
        if not (callable(self.silent_turn) or self.silent_turn in NO_ANSWER_POLICIES):
            raise ValueError("matcher %r: silent_turn must be SILENT_TURN_FAILS, SILENT_TURN_NOT_GRADABLE or a callable" % (self.name,))
        if not self.instruction_kind:
            object.__setattr__(self, "instruction_kind", self.name)

    def silence_policy(self, value):
        return self.silent_turn(value) if callable(self.silent_turn) else self.silent_turn


MATCHERS: Dict[str, Matcher] = {m.name: m for m in (
    Matcher("exact", _m_exact, SILENT_TURN_FAILS, "the whole visible text equals the value (after strip)",
            witness=lambda v: str(v), violation=lambda v: str(v) + " and more", examples=("DONE", "[SILENT]")),
    Matcher("prefix", _m_prefix, SILENT_TURN_FAILS, "the visible text starts with the value",
            witness=lambda v: str(v) + " done", violation=lambda v: "x " + str(v), examples=("OK:", "[LOG]")),
    Matcher("suffix", _m_suffix, SILENT_TURN_FAILS, "the visible text ends with the value",
            witness=lambda v: "done " + str(v), violation=lambda v: str(v) + " x", examples=("END", ".")),
    Matcher("regex", _m_regex, SILENT_TURN_FAILS, "re.search(value) finds a match in the visible text",
            witness=_w_regex, violation=_v_regex, examples=("hello", r"^TL;DR:")),
    Matcher("forbidden", _m_forbidden, SILENT_TURN_NOT_GRADABLE, "re.search(value) finds NO match in the visible text",
            witness=lambda v: None if re.search(str(v), "done") else "done", violation=_v_forbidden, examples=(";", r"(?i)\bin summary\b")),
    Matcher("json_schema", _m_json_schema, SILENT_TURN_FAILS, "the visible text is one JSON object containing value['required'] keys",
            value_key=lambda v: "any", witness=_w_json_schema, violation=lambda v: "not json", examples=({"required": ["status", "files"]},)),
    Matcher("fenced", _m_fenced, SILENT_TURN_FAILS, "at least one paired code fence whose info string matches the value regex",
            witness=_w_fenced, violation=lambda v: "no fence here", examples=("json", "diff")),
    Matcher("length_bound", _m_length_bound, _length_bound_silent_turn, "count of value['unit'] is <= n (dir max) or >= n (dir min)",
            value_key=_vk_length_bound, witness=_w_length_bound, violation=_v_length_bound,
            examples=({"n": 3, "unit": "words", "dir": "max"}, {"n": 2, "unit": "sentences", "dir": "min"}, {"n": 1, "unit": "lines", "dir": "max"})),
    Matcher("language", _m_language, SILENT_TURN_FAILS, "a strict majority of alphabetic characters belongs to the named script",
            witness=lambda v: _SCRIPT_SAMPLE.get(str(v)), violation=_v_language, examples=tuple(_SCRIPT_SAMPLE)),
    Matcher("sentinel_exclusive", _m_sentinel_exclusive, SILENT_TURN_NOT_GRADABLE, "if the sentinel token appears, the text is exactly the token",
            witness=lambda v: str(v), violation=lambda v: str(v) + " and more", examples=("HEARTBEAT_OK",)),
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


def value_key(matcher, value):
    """The key the value pools and the no-op gate use for a (matcher, value) pair."""
    if matcher not in MATCHERS:
        raise ValueError("unknown matcher %r" % (matcher,))
    return MATCHERS[matcher].value_key(value)
