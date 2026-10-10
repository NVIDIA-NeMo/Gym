# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protected, single-case comparison API. Run only in an independent Daytona sandbox.

Roles, expectations, policy, and outcome kinds MUST be assembled by trusted orchestration, never merged
from a returned candidate dict. Exception kinds attest an actual call outcome, not stdout or returned text.
A reference mismatch is invalid_reference and must stop candidate allocation. This function cannot enforce
its own hard deadline or authenticate the runner's exception observations.
"""

import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any

from pydantic import ValidationError

from .codec import (
    REAL_TYPES,
    ComplexValue,
    LegacyText,
    RepresentationError,
    SetValue,
    SymbolicText,
    decode_value,
    pointer_child,
)
from .task_data import (
    DECIMAL_PATTERN,
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    MAX_DECIMAL_EXPONENT,
    MAX_INTEGER_DIGITS,
    MAX_POLICY_BYTES,
    MAX_STATEMENT_CHARS,
    MAX_TEXT_BYTES,
    SILENT_DEFAULT_ATOL,
    SILENT_DEFAULT_RTOL,
    CasePolicy,
    bounded_json,
    decimal_text,
)


_INTEGER_TEXT = re.compile(r"[+-]?[0-9]+\Z")
_FRACTION_TEXT = re.compile(r"([+-]?[0-9]+)/([0-9]+)\Z")
_MATH_ATOMS = {"i", "I", "pi", "E", "oo", "zoo", "GoldenRatio", "EulerGamma", "Catalan"}
_ENCODING_ERRORS = {
    "invalid_type",
    "integer_limit",
    "invalid_integer",
    "invalid_decimal",
    "text_limit",
    "invalid_text",
    "invalid_unicode",
    "invalid_complex",
    "structure_limit",
    "json_bounds",
    "unsupported_type",
}


@dataclass(frozen=True)
class _Expression:
    value: Any


def _result(status: str, side: str, code: str, path: str = "") -> dict:
    return {
        "version": 1,
        "status": status,
        "equal": True if status == "equal" else False if status == "mismatch" else None,
        "side": side,
        "code": code,
        "path": path[:1024],
    }


def _expression_shaped(text: str) -> bool:
    stripped = text.strip()
    return stripped.lstrip("+-") in _MATH_ATOMS or any(char in stripped for char in "*/()^+") or " - " in stripped


# --- categorical fallback for stored expectations the restricted grammar does not parse ------------ #
#
# Narrow on purpose. It admits an equation or assignment and a bare infinity atom as exact-string
# comparisons, but not Python-shaped inputs, which the grammar refuses as defense-in-depth.
_MATH_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ .,'+-*/^!=<>()[]{}|:")
_CLOSE_TO_OPEN = {")": "(", "]": "[", "}": "{"}
_BINARY_OPS = "+*/^"
_LEGAL_OP_PAIRS = frozenset({"**", "//"})  # Python power and floor division are legal adjacent operators


def _dangling_operator(text: str) -> bool:
    """A binary operator with no operand: a truncated or mangled expression such as "2*x +" or "*2"."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[-1] in _BINARY_OPS + "-":
        return True  # nothing follows the trailing operator
    if stripped[0] in "*/^":
        return True  # nothing precedes the leading operator ("-x" is a legal unary sign)
    dense = "".join(stripped.split())
    return any(
        first in _BINARY_OPS + "-" and second in _BINARY_OPS and first + second not in _LEGAL_OP_PAIRS
        for first, second in zip(dense, dense[1:])
    )


def _corrupt_expression(text: str) -> bool:
    """An expression-shaped string that no notation could produce.

    Corruption is a character outside the mathematical alphabet, an unbalanced bracket, or a dangling
    operator. It is NOT "the grammar cannot parse it": legitimate physics notation is unparseable yet not
    corrupt.
    """
    if not _expression_shaped(text):
        return False
    if set(text) - _MATH_CHARS:
        return True
    if _dangling_operator(text):
        return True
    stack: list[str] = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in _CLOSE_TO_OPEN and (not stack or stack.pop() != _CLOSE_TO_OPEN[char]):
            return True
    return bool(stack)


def _top_level_relation(text: str) -> bool:
    """Whether a relational operator sits at bracket depth zero — an equation or assignment shape.

    A keyword-argument "=" inside a call is at a positive depth and does NOT count. Comparison spellings
    count the same as a bare "=".
    """
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif depth == 0 and char == "=" and (index == 0 or text[index - 1] not in "=<>!"):
            return True
        elif depth == 0 and char in "<>":
            return True
    return False


def _categorical_reference_form(text: str) -> bool:
    """A stored expectation our grammar refuses but compares as an exact string.

    Two shapes qualify: an equation or assignment, and a bare infinity atom. A corrupt string never
    qualifies, so it still reaches the parser and is refused as an invalid expectation.
    """
    if _corrupt_expression(text):
        return False
    return _top_level_relation(text) or _infinity_atom(text)


def _infinity_atom(text: str) -> bool:
    """A bare SymPy infinity atom (oo, zoo). It denotes a nonfinite value, not a category."""
    return text.strip().lstrip("+-") in ("oo", "zoo")


def _numeric_text(text: str) -> Any:
    text = text.strip()
    if _INTEGER_TEXT.fullmatch(text):
        if len(text.lstrip("+-")) > MAX_INTEGER_DIGITS:
            raise RepresentationError("integer_limit")
        return int(text)
    fraction = _FRACTION_TEXT.fullmatch(text)
    if fraction:
        numerator, denominator = fraction.groups()
        if max(len(numerator.lstrip("+-")), len(denominator)) > MAX_INTEGER_DIGITS:
            raise RepresentationError("integer_limit")
        if int(denominator) == 0:
            raise RepresentationError("invalid_denominator")
        return Fraction(int(numerator), int(denominator))
    if DECIMAL_PATTERN.fullmatch(text):
        try:
            return Decimal(decimal_text(text))
        except ValueError as exc:
            raise RepresentationError("invalid_decimal") from exc
    if text.lower().lstrip("+-") in ("nan", "inf", "infinity"):
        return math.nan if "nan" in text.lower() else -math.inf if text.startswith("-") else math.inf
    return None


def _nonfinite(value: Any) -> bool:
    return (type(value) is float and not math.isfinite(value)) or (type(value) is Decimal and not value.is_finite())


def _parse(text: str, path: str) -> _Expression:
    # A fixed import, deferred so schema/codec consumers need no SymPy installation.
    from .symbolic import parse_expression

    try:
        return _Expression(parse_expression(text))
    except RepresentationError as exc:
        raise RepresentationError(exc.code, path) from exc


def _prepare(value: Any, *, expected: bool, path: str = "") -> Any:
    kind = type(value)
    if kind is LegacyText:
        text = value.text.strip()
        if text.startswith(("[", "{", "'", '"')) or text in ("True", "False", "None"):
            raise RepresentationError("ambiguous_legacy_literal", path)
        if re.fullmatch(r"[+-]?[0-9]+(?:_[0-9]+)+", text):
            raise RepresentationError("ambiguous_legacy_literal", path)
        try:
            number = _numeric_text(value.text)
        except RepresentationError as exc:
            raise RepresentationError(exc.code, path) from exc
        if number is not None:
            return _prepare(number, expected=expected, path=path)
        if expected and _infinity_atom(value.text):
            raise RepresentationError("nonfinite_expected", path)
        if _categorical_reference_form(value.text):
            # An equation or assignment kept as a categorical string, graded by exact equality.
            return value.text
        return _parse(value.text, path) if _expression_shaped(value.text) else value.text
    if kind is SymbolicText:
        if expected and _infinity_atom(value.text):
            raise RepresentationError("nonfinite_expected", path)
        if _categorical_reference_form(value.text):
            return value.text
        return _parse(value.text, path)
    if expected and _nonfinite(value):
        raise RepresentationError("nonfinite_expected", path)
    if kind is ComplexValue:
        return ComplexValue(
            _prepare(value.real, expected=expected, path=path), _prepare(value.imag, expected=expected, path=path)
        )
    if kind is list or kind is tuple:
        return [
            _prepare(child, expected=expected, path=pointer_child(path, index)) for index, child in enumerate(value)
        ]
    if kind is SetValue:
        # Members keep a positional path, a slot the unordered match reassigns, not a specific member.
        return SetValue(
            tuple(
                _prepare(child, expected=expected, path=pointer_child(path, index))
                for index, child in enumerate(value.items)
            )
        )
    if kind is dict:
        return {key: _prepare(child, expected=expected, path=pointer_child(path, key)) for key, child in value.items()}
    return value


def _leaves(value: Any, path: str = ""):
    if type(value) is list:
        for index, child in enumerate(value):
            yield from _leaves(child, pointer_child(path, index))
    elif type(value) is SetValue:
        # A set expectation addresses its members positionally, so a per-member tolerance path resolves.
        for index, child in enumerate(value.items):
            yield from _leaves(child, pointer_child(path, index))
    elif type(value) is dict:
        for key, child in value.items():
            yield from _leaves(child, pointer_child(path, key))
    else:
        yield path, value


# --- statement policy: bounded extraction, run only inside the comparator ------------------------ #
#
# A clause is recognized only by these fixed patterns near accuracy vocabulary, never by reading prose.
# Everything else stays unrecognized and visible through statement_promises() rather than guessed at.
_PROMISE_KEYWORD = re.compile(
    r"(error|accura\w+|toleran\w+|precision|within|correct to|agree\w* to|decimal places|"
    r"significant (?:decimal )?(?:fig|digit)\w*)",
    re.IGNORECASE,
)
_ABSOLUTE = re.compile("absolut", re.IGNORECASE)
_RELATIVE = re.compile("relativ", re.IGNORECASE)
_NUMBER_WORD = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
_WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_SCIENTIFIC = re.compile(r"\b\d+(?:\.\d+)?[eE]-\d+\b")
_POWER_OF_TEN = re.compile(
    r"(\d+(?:\.\d+)?)?\s*(?:[×x*·]|\\times|\\cdot)?\s*10\s*(?:\^|\*\*)?\s*[({\[]?\s*(-\d+)\s*[)}\]]?"
)
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_SIGNIFICANT = re.compile(rf"({_NUMBER_WORD})\s*significant\s+(?:decimal\s+)?(?:fig|digit)\w*", re.IGNORECASE)
_DECIMAL_PLACES = re.compile(rf"({_NUMBER_WORD})\s*decimal\s+places", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"[.;:]\s|\n")
_FENCE = re.compile(r"```.*?(?:```|\Z)", re.S)
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_QUANTITY_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
# Leaf names that are the extractor's own vocabulary are never claimed by a clause.
_PROMISE_VOCABULARY = frozenset(
    {"absolute", "decimal", "digit", "digits", "figure", "figures", "places", "relative", "significant"}
)
_PROMISE_FLOOR = Fraction(1, 10**15)  # at or below binary64 resolution no float computation can honour it
_PROMISE_CEILING = Fraction(1, 10)  # above ten percent it is not an accuracy claim
_KEYWORD_LEAD, _KEYWORD_TAIL = 60, 160  # characters around a keyword that may hold its number
_ATTRIBUTION_WINDOW = 90  # characters a clause looks across, inside its own sentence, for a name


def _decimal_value(text: str) -> Fraction | None:
    """An exact promise value from decimal text, or None when the text is outside the decimal bounds."""
    try:
        return Fraction(Decimal(decimal_text(text)))
    except ValueError:
        return None


def _count_value(token: str, extra_places: int) -> Fraction | None:
    """5 x 10^-(N + extra_places) for a numeric or spelled-out count N, or None when N is unsupported."""
    if token.isdigit():
        if len(token) > 4:
            return None
        count = int(token)
    else:
        count = _WORD_NUMBERS[token.lower()]
    count += extra_places
    return None if count > MAX_DECIMAL_EXPONENT else Fraction(5, 10**count)


def _normalized(statement: str) -> str:
    """Offset-preserving normalization: unicode minus and superscripts to ASCII, fenced code blanked."""
    text = statement.replace("−", "-").translate(_SUPERSCRIPTS)
    return _FENCE.sub(lambda match: " " * len(match.group()), text)


def _clauses(text: str) -> list[dict]:
    """Every accuracy clause the fixed patterns recognize, honoured or not, in order of appearance.

    Each keyword opens one fixed character window. One component kind covers the whole window: absolute,
    else relative, else unknown. A percentage, significant-figure count and decimal-place count keep their
    own inherent kind. The window is wide on purpose, and the loosest honoured clause becomes the blanket.
    """
    found: dict[tuple, str] = {}
    breaks = list(_SENTENCE_BREAK.finditer(text))
    break_ends = [match.end() for match in breaks]
    break_starts = [match.start() for match in breaks]
    keywords = list(_PROMISE_KEYWORD.finditer(text))
    spans = [(keyword.start(), keyword.end()) for keyword in keywords]

    def owner(at: int) -> int:
        # The keyword nearest the number owns it, ties to the earliest, so a number a wide window pulls
        # across a sentence break is scored under one keyword's kind, not both.
        best, best_distance = 0, None
        for index, (span_start, span_end) in enumerate(spans):
            distance = span_start - at if at < span_start else at - span_end if at > span_end else 0
            if best_distance is None or distance < best_distance:
                best, best_distance = index, distance
        return best

    def note(at: int, value: Fraction | None, kind: str, matched: str, index: int) -> None:
        if owner(at) != index:
            return  # a number belongs to its nearest keyword, and only that keyword records it
        found.setdefault((at, value, kind), matched.strip())  # overlapping windows re-read one clause

    for index, keyword in enumerate(keywords):
        start = max(0, keyword.start() - _KEYWORD_LEAD)
        window = text[start : keyword.end() + _KEYWORD_TAIL]
        # The kind read stays inside the keyword's own sentence, the same bounds name attribution uses, so
        # "absolute" in a neighbour sentence cannot flip a "relative" clause. The number search still spans
        # the full window: a keyword may pull a number stated after a colon or in the next sentence.
        before = bisect_right(break_ends, keyword.start())
        sentence_start = break_ends[before - 1] if before else 0
        after = bisect_left(break_starts, keyword.end())
        sentence_end = break_starts[after] if after < len(break_starts) else len(text)
        kind_region = text[max(start, sentence_start) : min(keyword.end() + _KEYWORD_TAIL, sentence_end)]
        kind = "abs" if _ABSOLUTE.search(kind_region) else "rel" if _RELATIVE.search(kind_region) else "unknown"
        for match in _SCIENTIFIC.finditer(window):
            note(start + match.start(), _decimal_value(match.group()), kind, match.group(), index)
        for match in _POWER_OF_TEN.finditer(window):
            # Read through the decimal literal, never multiplied through floats: 2.5 x 10^-6 is exactly 2.5e-6.
            value = _decimal_value(f"{match.group(1) or '1'}e{match.group(2)}")
            note(start + match.start(), value, kind, match.group(), index)
        for match in _PERCENT.finditer(window):
            value = _decimal_value(match.group(1))
            note(start + match.start(), None if value is None else value / 100, "rel", match.group(), index)
        for match in _SIGNIFICANT.finditer(window):
            note(start + match.start(), _count_value(match.group(1), 0), "rel", match.group(), index)
        for match in _DECIMAL_PLACES.finditer(window):
            note(start + match.start(), _count_value(match.group(1), 1), "abs", match.group(), index)

    def order(key: tuple) -> tuple:
        at, value, kind = key
        return at, Fraction(-1) if value is None else value, kind

    clauses = []
    for at, value, kind in sorted(found, key=order):
        reason = (
            "unsupported_number"
            if value is None
            else "below_float64_resolution"
            if value <= _PROMISE_FLOOR
            else "too_loose_to_be_an_accuracy_claim"
            if value > _PROMISE_CEILING
            else ""
        )
        clauses.append(
            {
                "value": value,
                "kind": kind,
                "at": at,
                "text": found[(at, value, kind)],
                "honoured": not reason,
                "reason": reason,
            }
        )
    return clauses


def _word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


def _names_in(text: str, names: frozenset, start: int, end: int) -> list[str]:
    """Candidate names occurring as whole tokens inside text[start:end], in order of first appearance."""
    claimed: list[str] = []
    for match in _TOKEN.finditer(text, start, end):
        cut_lead = match.start() == start and start > 0 and _word_char(text[start - 1])
        cut_tail = match.end() == end and end < len(text) and _word_char(text[end])
        if cut_lead or cut_tail:
            continue  # a token the window sliced through is not evidence of a name
        if match.group() in names and match.group() not in claimed:
            claimed.append(match.group())
    return claimed


def _attribute(text: str, clauses: list[dict], names: frozenset) -> dict[str, tuple[Fraction, str]]:
    """A clause claims every name on one side of it inside its own sentence: the preceding names, else the
    following ones. A clause that names several quantities binds its margin to each of them.

    Each clause is bounded by its neighbours and by the sentence. The first clause to claim a name keeps it,
    so an explicit per-quantity margin is never replaced by a looser one. A blanket clause is a catch-all for
    the rest and does not widen a leaf the author gave its own tolerance entry.
    """
    named: dict[str, tuple[Fraction, str]] = {}
    if not names:
        return named
    breaks = list(_SENTENCE_BREAK.finditer(text))
    starts = [match.start() for match in breaks]
    ends = [match.end() for match in breaks]
    positions = [clause["at"] for clause in clauses]
    previous_end = 0
    for clause in clauses:
        if not clause["honoured"]:
            continue
        at = clause["at"]
        before = bisect_right(ends, at)
        sentence_start = ends[before - 1] if before else 0
        claimed = _names_in(text, names, max(previous_end, at - _ATTRIBUTION_WINDOW, sentence_start), at)
        if not claimed:
            after = bisect_left(starts, at)
            sentence_end = starts[after] if after < len(starts) else len(text)
            following = bisect_right(positions, at)
            next_clause = positions[following] if following < len(positions) else len(text)
            reach = min(at + _ATTRIBUTION_WINDOW, sentence_end, next_clause)
            claimed = _names_in(text, names, at, reach)
        for name in claimed:
            named.setdefault(name, (clause["value"], clause["kind"]))
        previous_end = at + len(clause["text"])
    return named


def statement_promises(statement: str, quantities: Iterable[str] = ()) -> dict:
    """What the comparator harvests from a statement, for inspection. No comparison is performed.

    Returns {"clauses": [...], "blanket": (value, kind) | None, "named": {name: (value, kind)}}. Values are
    exact Fractions and kind is "abs", "rel" or "unknown". A clause is honoured only inside (1e-15, 0.1], and
    the loosest honoured clause is the blanket. Names come from `quantities` only.
    """
    if type(statement) is not str or len(statement) > MAX_STATEMENT_CHARS:
        raise RepresentationError("statement_limit")
    text = _normalized(statement)
    clauses = _clauses(text)
    honoured = [clause for clause in clauses if clause["honoured"]]
    names = frozenset(
        name
        for name in quantities
        if type(name) is str
        and _QUANTITY_NAME.fullmatch(name)
        and name.lower() not in _PROMISE_VOCABULARY
        and not _PROMISE_KEYWORD.fullmatch(name)
    )
    blanket = max(honoured, key=lambda clause: clause["value"]) if honoured else None
    return {
        "clauses": clauses,
        "blanket": None if blanket is None else (blanket["value"], blanket["kind"]),
        "named": _attribute(text, clauses, names),
    }


def _leaf_name(path: str) -> str:
    """The key a leaf path ends in, or "" for the root or a positional index, which no statement can name."""
    tail = path.rsplit("/", 1)[-1].replace("~1", "/").replace("~0", "~")
    return "" if not tail or tail.isdigit() else tail


def _pair_text(rtol: str | None, atol: str | None) -> tuple[Fraction, Fraction]:
    """Exact (rtol, atol) from decimal text. A missing rtol is 5e-12, never implicit exactness."""
    return (
        Fraction(Decimal(DEFAULT_RTOL if rtol is None else rtol)),
        Fraction(Decimal(DEFAULT_ATOL if atol is None else atol)),
    )


def _magnitude(expected: Any) -> Fraction:
    if type(expected) is ComplexValue:
        return max(abs(Fraction(expected.real)), abs(Fraction(expected.imag)))
    return abs(Fraction(expected))


def _margin(pair: tuple[Fraction, Fraction], magnitude: Fraction) -> tuple[Fraction, Fraction]:
    """A leaf's admitted width, then rtol as the tiebreak, so two pairs order consistently.

    The width is ``atol + rtol * |expected|``, compared exactly. At magnitude 0 an emptied ``(0, 0)`` promise
    and the strict relative default both admit width 0, and the default's larger rtol makes it the looser, so
    such a leaf is floored to the default rather than held exact.
    """
    return (pair[1] + pair[0] * magnitude, pair[0])


class _Policy:
    def __init__(self, raw: dict, expected: Any, *, has_value: bool = True):
        try:
            bounded_json(raw, max_bytes=MAX_POLICY_BYTES)
            statement = raw.get("statement") if type(raw) is dict else None
            if type(statement) is str and len(statement) > MAX_STATEMENT_CHARS:
                raise RepresentationError("statement_limit")
            self.policy = CasePolicy.model_validate(raw)
        except RepresentationError:
            raise
        except (ValidationError, ValueError, UnicodeError) as exc:
            raise RepresentationError("invalid_policy") from exc
        self.by_path = {entry.path: entry for entry in self.policy.tolerances}
        # The server default for a genuinely silent numeric leaf: the operator knob, else SILENT_DEFAULT.
        # It never floors a promise or fills a partial authored entry, which keep the strict DEFAULT anchor.
        self.silent = _pair_text(
            SILENT_DEFAULT_RTOL if self.policy.default_rtol is None else self.policy.default_rtol,
            SILENT_DEFAULT_ATOL if self.policy.default_atol is None else self.policy.default_atol,
        )
        self.blanket: tuple[Fraction, str] | None = None
        self.named: dict[str, tuple[Fraction, str]] = {}
        if not has_value:
            return  # Neither numeric tolerances nor statement promises apply to an exception outcome.
        leaves = dict(_leaves(expected))
        # A tolerance entry whose path addresses no numeric or complex leaf is IGNORED, not a defect.
        # A consumed entry (a real or complex leaf) stays in by_path and still governs its leaf through pair().
        self.by_path = {
            path: entry
            for path, entry in self.by_path.items()
            if path in leaves and type(leaves[path]) in (*REAL_TYPES, ComplexValue)
        }
        if self.policy.statement is not None:
            # Candidate names are the expectation's own leaf keys, never words scraped from the prose.
            promises = statement_promises(self.policy.statement, (_leaf_name(path) for path in leaves))
            self.blanket, self.named = promises["blanket"], promises["named"]

    def _promise(self, path: str, magnitude: Fraction, authored_entry: bool) -> tuple[Fraction, Fraction] | None:
        """The statement promise that GOVERNS this leaf, scale-guarded, as an exact pair. None when no clause
        governs it.

        A promise named for this leaf governs it. A blanket clause reaches a leaf the author gave its own
        tolerance only when the statement named no other quantity. The scale guard drops the absolute
        component for a leaf whose magnitude sits below it. A leaf a clause governs whose promise the guard
        empties returns ``(0, 0)``, NOT None, so pair() floors it at the strict default rather than reading it
        as silent. Only a leaf no clause governs returns None.
        """
        promise = self.named.get(_leaf_name(path))
        if promise is None and not (authored_entry and self.named):
            promise = self.blanket
        if promise is None:
            return None
        value, kind = promise
        rtol = value if kind in ("rel", "unknown") else Fraction(0)
        atol = value if kind in ("abs", "unknown") else Fraction(0)
        if atol and magnitude < atol:
            atol = Fraction(0)  # Scale guard: an absolute promise says nothing about a leaf smaller than itself.
        return rtol, atol

    def pair(self, path: str, expected: Any) -> tuple[Fraction, Fraction]:
        """Resolve one leaf: the statement promise that governs it, else an authored entry or task setting,
        else the server default. An integer expectation no clause governs is exact.

        A leaf no clause governs and no author spoke for resolves to ``self.silent``, except an integer
        expectation, which stays exact by construction. Once a clause governs a leaf, integer exactness no
        longer applies. Where no author spoke the promise is floored at the strict default. Where an author
        also spoke, the promise governs unless the scale guard emptied it, and then the author's entry governs.
        """
        entry = self.by_path.get(path)
        setting = self.policy if entry is None else entry
        authored = None
        if entry is not None or setting.rtol is not None or setting.atol is not None:
            authored = _pair_text(setting.rtol, setting.atol)
        magnitude = _magnitude(expected)
        promise = self._promise(path, magnitude, entry is not None)
        if promise is None:
            # No clause governs this leaf.
            if authored is not None:
                return authored
            if type(expected) is int:
                return (Fraction(0), Fraction(0))  # an integer expectation is exact by construction
            return self.silent
        if authored is not None:
            # The promise governs unless the scale guard emptied it, in which case the author's entry governs.
            return authored if promise == (Fraction(0), Fraction(0)) else promise
        default = _pair_text(None, None)  # the strict 5e-12 / 0 floor
        if _margin(promise, magnitude) < _margin(default, magnitude):
            return default
        return promise


def _symbolic_equal(left: Any, right: Any, path: str) -> tuple[bool | None, str, str]:
    from .symbolic import equivalent, rational_expression

    try:
        a = left.value if type(left) is _Expression else rational_expression(Fraction(left))
        b = right.value if type(right) is _Expression else rational_expression(Fraction(right))
        equal = equivalent(a, b)
    except RepresentationError as exc:
        return None, exc.code, path
    return equal, "symbolic_undecided" if equal is None else "symbolic_comparison", path


def _close(left: Any, right: Any, pair: tuple[Fraction, Fraction]) -> bool:
    if _nonfinite(left) or _nonfinite(right):
        return False
    rtol, atol = pair
    a, b = Fraction(left), Fraction(right)
    # The relative bound scales by |expected|, the reference side.
    return abs(a - b) <= atol + rtol * abs(b)


class _MatchBudgetExhausted(Exception):
    """Raised when the unordered match spends its step budget, so the caller reports undecided."""


# Precompute and matching search both charge this shared budget so a large set cannot run without a bound.
_UNORDERED_STEP_BUDGET = 50_000


def _match_unordered(observed: list, expected: list, policy: _Policy, path: str) -> tuple[bool | None, str, str]:
    """Compare two sets UNORDERED: equal size and a one-to-one matching under _compare.

    A pair is CERTAIN when _compare decides equal, POSSIBLE when equal or undecided. A perfect matching over
    certain pairs is equal. Otherwise a perfect matching over possible pairs is undecided, because an
    undecided pair cannot be ruled out. No perfect matching even over possible pairs is a mismatch. Each
    expected slot is graded at its own positional path."""
    if len(observed) != len(expected):
        return False, "structure_mismatch", path
    size = len(expected)
    if size == 0:
        return True, "unordered_comparison", path
    steps = 0
    certain: list[list[int]] = [[] for _ in range(size)]
    possible: list[list[int]] = [[] for _ in range(size)]
    for i in range(size):
        for j in range(size):
            steps += 1
            if steps > _UNORDERED_STEP_BUDGET:
                return None, "unordered_budget", path
            try:
                verdict = _compare(observed[j], expected[i], policy, pointer_child(path, i))[0]
            except RepresentationError:
                continue  # an unparseable cross-pair is not a match, never a failure of the whole set
            if verdict is True:
                certain[i].append(j)
                possible[i].append(j)
            elif verdict is None:
                possible[i].append(j)

    budget = [steps]

    def has_perfect_matching(adjacency: list[list[int]]) -> bool:
        assigned = [-1] * size  # observed member -> the expected slot that took it

        def augment(slot: int, seen: list[bool]) -> bool:
            for member in adjacency[slot]:
                budget[0] += 1
                if budget[0] > _UNORDERED_STEP_BUDGET:
                    raise _MatchBudgetExhausted
                if seen[member]:
                    continue
                seen[member] = True
                if assigned[member] == -1 or augment(assigned[member], seen):
                    assigned[member] = slot
                    return True
            return False

        for slot in range(size):
            if not augment(slot, [False] * size):
                return False
        return True

    try:
        if has_perfect_matching(certain):
            return True, "unordered_comparison", path
        if has_perfect_matching(possible):
            return None, "unordered_undecided", path
        return False, "unordered_comparison", path
    except _MatchBudgetExhausted:
        return None, "unordered_budget", path


def _compare(observed: Any, expected: Any, policy: _Policy, path: str = "") -> tuple[bool | None, str, str]:
    left, right = type(observed), type(expected)
    if left is bool or right is bool:
        return left is right and observed == expected, "boolean_comparison", path
    if right is str:
        return left is str and observed == expected, "categorical_comparison", path
    if right in REAL_TYPES or right is _Expression:
        if left is str:
            try:
                number = _numeric_text(observed)
            except RepresentationError as exc:
                raise RepresentationError(exc.code, path) from exc
            if number is not None:
                observed = number
            elif right is _Expression or _expression_shaped(observed):
                observed = _parse(observed, path)
            left = type(observed)
        if left not in (*REAL_TYPES, _Expression):
            return False, "type_mismatch", path
        if _nonfinite(observed):
            return False, "nonfinite_observed", path
        if left is _Expression or right is _Expression:
            equal, code, spath = _symbolic_equal(observed, expected, path)
            if equal is None and code == "symbolic_undecided" and right is not _Expression:
                # A candidate expression against a concrete numeric expectation. An undecided algebra verdict
                # means the candidate does not reduce to the number, so it is wrong, never a task defect.
                return False, "symbolic_number_mismatch", spath
            return equal, code, spath
        return _close(observed, expected, policy.pair(path, expected)), "numeric_comparison", path
    if right is ComplexValue:
        if left is not ComplexValue:
            return False, "type_mismatch", path
        # Each component is scaled by its OWN reference part. policy.pair resolves against the modulus
        # max(|Re|, |Im|) for the scale guard alone.
        pair = policy.pair(path, expected)
        equal = _close(observed.real, expected.real, pair) and _close(observed.imag, expected.imag, pair)
        return equal, "complex_comparison", path
    if right is SetValue or (left is SetValue and right is list):
        # A set on either side is matched UNORDERED. A set carrier or a plain list is accepted, and anything
        # else against a set expectation is a type mismatch.
        if left is not SetValue and left is not list:
            return False, "type_mismatch", path
        observed_items = list(observed.items) if left is SetValue else list(observed)
        expected_items = list(expected.items) if right is SetValue else list(expected)
        return _match_unordered(observed_items, expected_items, policy, path)
    if right in (list, dict):
        if left is not right:
            return False, "type_mismatch", path
        if (right is list and len(observed) != len(expected)) or (right is dict and set(observed) != set(expected)):
            return False, "structure_mismatch", path
        undecided = None
        for key in range(len(expected)) if right is list else expected:
            result = _compare(observed[key], expected[key], policy, pointer_child(path, key))
            if result[0] is False:
                return result
            if result[0] is None:
                undecided = result
        return undecided if undecided is not None else (True, "composite_comparison", path)
    return left is right and observed == expected, "value_comparison", path


def _outcome(raw: Any, *, expected: bool) -> tuple[str, Any]:
    if type(raw) is not dict or len(raw) > 3 or type(raw.get("kind")) is not str:
        raise RepresentationError("invalid_outcome")
    kind = raw["kind"]
    if kind == "value" and set(raw) == {"kind", "value"}:
        return kind, _prepare(decode_value(raw["value"]), expected=expected)
    if kind == "exception" and set(raw) <= {"kind", "message", "type"}:
        for field in ("message", "type"):
            value = raw.get(field)
            if value is not None:
                if type(value) is not str or len(value) > MAX_TEXT_BYTES:
                    raise RepresentationError("invalid_exception_diagnostic")
                try:
                    if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
                        raise RepresentationError("invalid_exception_diagnostic")
                except UnicodeError as exc:
                    raise RepresentationError("invalid_exception_diagnostic") from exc
        return kind, None
    if not expected and kind == "encoding_error" and set(raw) == {"kind", "code"}:
        if type(raw["code"]) is str and raw["code"] in _ENCODING_ERRORS:
            return kind, raw["code"]
    raise RepresentationError("invalid_outcome")


def _exhausted(role: str, expected_ready: bool, code: str) -> dict:
    """A time or memory cap hit during a comparison.

    Uncertain by default. Once the expectation has parsed within the caps, a candidate exhaustion is
    attributed to the candidate and scored as an invalid observation. A reference observation, and any
    exhaustion on the expectation itself, stay uncertain.
    """
    if expected_ready and role == "candidate":
        return _result("invalid_candidate", "candidate", code)
    return _result("uncertain", "comparator", code)


def compare_request(request: dict) -> dict:
    """No IO, execution, or reference synthesis. The caller supplies protected outcomes."""
    role = "comparator"
    expected_ready = False
    try:
        if (
            type(request) is not dict
            or len(request) > 5
            or set(request) - {"version", "observed_role", "observed", "expected", "policy"}
            or not {"version", "observed_role", "observed", "expected"} <= set(request)
            or type(request["version"]) is not int
            or request["version"] != 1
            or type(request["observed_role"]) is not str
            or request["observed_role"] not in ("candidate", "reference")
        ):
            return _result("uncertain", "request", "invalid_request")
        role = request["observed_role"]
        # Validate the entire expectation before inspecting even the observed shape.
        try:
            expected_kind, expected = _outcome(request["expected"], expected=True)
            policy = _Policy(request.get("policy", {}), expected, has_value=expected_kind == "value")
        except RepresentationError as exc:
            return _result("invalid_expected", "expected", exc.code, exc.path)
        expected_ready = True  # The expectation parsed and resolved within the caps: it is tractable.
        try:
            observed_kind, observed = _outcome(request["observed"], expected=False)
        except RepresentationError as exc:
            if exc.code in ("invalid_outcome", "invalid_exception_diagnostic"):
                return _result("uncertain", role, "invalid_observation_protocol", exc.path)
            return _result("invalid_" + role, role, exc.code, exc.path)
        if observed_kind == "encoding_error":
            if observed == "unsupported_type":
                return _result("uncertain", role, "native_type_not_supported")
            return _result("invalid_" + role, role, observed)
        if expected_kind == "exception" or observed_kind == "exception":
            equal = observed_kind == expected_kind
            code = "expected_exception" if equal else "did_not_raise" if expected_kind == "exception" else "raised"
            path = ""
        else:
            try:
                equal, code, path = _compare(observed, expected, policy)
            except RepresentationError as exc:
                return _result("invalid_" + role, role, exc.code, exc.path)
        if equal is None:
            return _result("uncertain", "comparator", code, path)
        if equal:
            return _result("equal", role, code, path)
        return _result("invalid_reference" if role == "reference" else "mismatch", role, code, path)
    except TimeoutError:
        return _exhausted(role, expected_ready, "comparator_timeout")
    except MemoryError:
        return _exhausted(role, expected_ready, "comparator_memory")
    except Exception:
        # No exception text, values, source, or diagnostic payloads leave this boundary.
        return _result("uncertain", "comparator", "comparator_failure")
