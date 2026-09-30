# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recover the model's JSON object and apply upstream's field coercions.

Prompts ask for a bare JSON object; replies arrive with think blocks, prose and
fences. Coercions mirror ``predict.py`` so a reply upstream would have scored is
scored identically here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from slot_canonicalization import canonicalize_slots


# Thinking models emit these; the JSON we want is always after them.
_THINK_BLOCK_RE = re.compile(r"<(think|thinking)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
# An unterminated block (truncated by the output budget) leaves no closing tag.
_OPEN_THINK_RE = re.compile(r"<(think|thinking)\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Upstream's threshold for turning the requested score into a discrete label
# (predict.py, `label = bool(score >= threshold)`).
BINARY_LABEL_THRESHOLD = 0.5

# Bound anything model-controlled before it reaches a parser, and bound the
# rationale before it reaches the O(n*m) token comparison downstream.
MAX_RESPONSE_CHARS = 200_000
MAX_RATIONALE_CHARS = 20_000


def _iter_json_objects(text: str):
    """Candidate JSON objects, leftmost-first. Callers take the LAST that parses:
    a later object is the model's correction of an earlier draft.
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        # Only inside an object: a prose quote before any "{" would otherwise
        # swallow the rest of the reply and hide a valid trailing object.
        if ch == '"' and depth > 0:
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start : i + 1]


def clean_text(raw: Optional[str]) -> str:
    """Bounded text with reasoning blocks removed.

    DEPARTURE, deliberate: upstream's ``_raw`` is the whole answer, think blocks
    included. Every raw-text fallback here uses this instead, because scoring a
    reasoning model's trace measures the trace and not the answer. One rule for
    all three fallbacks -- ordering, contrastive choice and rationalization.
    """
    if not raw:
        return ""
    text = raw[:MAX_RESPONSE_CHARS]
    text = _THINK_BLOCK_RE.sub(" ", text)
    return _OPEN_THINK_RE.sub(" ", text)


def extract_json(raw: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return ``(object, status)``; ``object`` is None when nothing parsed."""
    if not raw or not raw.strip():
        return None, "empty_output"
    text = clean_text(raw)
    if not text.strip():
        # The whole reply was reasoning: the budget ran out before any answer.
        return None, "no_json_found"

    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates.extend(_iter_json_objects(text))

    parsed = None
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        if isinstance(obj, dict):
            parsed = obj  # keep going; the rightmost valid object wins
    if parsed is None:
        return None, "no_json_found"
    return parsed, "ok"


def _as_float(value: Any, default: float) -> float:
    """Coerce to float, treating every conversion limit as malformed output.

    A JSON number with hundreds of digits raises OverflowError, not ValueError,
    and an unhandled one escapes verify() as a 500 that aborts the run.
    """
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return default
    return out


_ID_TOKEN_RE = re.compile(r"id(\d+)")
_DIGITS_RE = re.compile(r"\d+")
_RAW_ID_RE = re.compile(r"[\"\']?([A-Za-z0-9_\-]+)[\"\']?")


def canonicalize_step_token(token: Any) -> str:
    """Upstream's ``_canonicalize_step_token``: 'id2' and 'step_2' both mean '2'."""
    text = str(token).strip()
    if text.isdigit():
        return text
    match = _ID_TOKEN_RE.fullmatch(text)
    if match:
        return match.group(1)
    match = _DIGITS_RE.search(text)
    return match.group(0) if match else text


def post_ordering(obj: Optional[Dict[str, Any]], expected: List[str], raw: str = "") -> List[str]:
    """Port of upstream ``post_ordering``: canonicalize ids, drop repeats, and --
    once one legal id has matched -- append the unmentioned ids in presentation
    order. An answer matching nothing yields [], never a fabricated order.
    """
    obj = obj if isinstance(obj, dict) else {}
    got = obj.get("predicted_order")
    if got is None:
        got = obj.get("order")
    if got is None:
        got = []
    if not isinstance(got, list) or not all(isinstance(x, (str, int)) for x in got):
        ids = _RAW_ID_RE.findall(raw or "")
        got = [str(x) for x in ids] if ids else []

    seen = set()
    clean: List[str] = []
    for item in map(str, got):
        canonical = canonicalize_step_token(item)
        if canonical in expected and canonical not in seen:
            clean.append(canonical)
            seen.add(canonical)

    if not clean:
        return []
    for expected_id in expected:
        if expected_id not in seen:
            clean.append(expected_id)
    return clean


def post_contrastive(obj: Optional[Dict[str, Any]], options: List[Any], raw: str = "") -> int:
    """Port of upstream ``post_contrastive``, index only. Recovers the choice from
    raw text and range-checks; an invalid index becomes -1, never option 0.
    """
    obj = obj if isinstance(obj, dict) else {}
    choice = obj.get("predicted_choice") or obj.get("choice")
    index = obj.get("predicted_option_idx", obj.get("pred_idx"))

    if choice not in options:
        for option in options:
            if isinstance(option, str) and option in (raw or ""):
                choice = option
                break

    if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < len(options)):
        index = options.index(choice) if choice in options else -1
    return int(index)


def to_prediction(
    task_type: str,
    obj: Optional[Dict[str, Any]],
    expected_step_ids: Optional[List[str]] = None,
    options: Optional[List[Any]] = None,
    raw: str = "",
    legend: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Coerce a parsed object into the shape ``metrics.score_row`` expects.

    UPSTREAM'S PARSE-FAILURE CONTRACT, which is easy to get backwards.
    ``_extract_json_from_answer`` returns ``{"_raw": answer}`` on EVERY failure
    path -- bad JSON, a JSON list, an exception. So the post-processors never
    receive a non-dict, ``post_binary``'s non-dict branch is unreachable in the
    real pipeline, and an unparseable binary reply becomes ``score=0.5`` and
    therefore ``label=True``. An earlier version of this file ported that dead
    branch and assigned ``label=False``, which moved both validation metrics.

    The same fact scopes the raw-text fallbacks: ``_raw`` exists only when
    parsing failed, so a successfully parsed object must never be re-scanned as
    text. ``{"predicted_order": "1 2 0"}`` scores empty upstream, not full
    credit.
    """
    no_object = not isinstance(obj, dict)
    obj = obj if isinstance(obj, dict) else {}
    # Upstream only has _raw when parsing failed, so recovery text is available
    # only then. Cleaned first; see clean_text on why that departs.
    recovery = clean_text(raw).strip() if no_object else ""

    if task_type == "ordering":
        return {"predicted_order": post_ordering(None if no_object else obj, expected_step_ids or [], recovery)}

    if task_type == "contrastive_choice":
        return {"predicted_option_idx": post_contrastive(None if no_object else obj, options or [], recovery)}

    if task_type in ("step_validation", "condition_validation"):
        # The prompt asks for `score` only; the label is derived, never requested.
        # No special case for a failed parse: upstream hands post_binary
        # {"_raw": ...}, a dict with no score, which is exactly the path below.
        score = obj.get("score", obj.get("prob_positive"))
        score = min(1.0, max(0.0, _as_float(score, 0.5)))
        return {"score": score, "label": bool(score >= BINARY_LABEL_THRESHOLD)}

    if task_type == "step_completion":
        slots = obj.get("slots")
        if not isinstance(slots, dict):
            slots = {}
        # Upstream canonicalizes against the row's legend before scoring; without
        # it a matched reagent reads as an unrecognised key, or an unnormalised
        # unit trips the fatal flag and zeroes the task.
        slots = canonicalize_slots({str(k): v for k, v in slots.items()}, legend)
        return {"action": str(obj.get("action", "")), "slots": slots}

    if task_type == "rationalization":
        for key in ("gold_rationale", "rationale", "predicted_rationale", "answer"):
            value = obj.get(key)
            if isinstance(value, list):
                value = " ".join(str(item) for item in value)
            if isinstance(value, str) and value.strip():
                return {"gold_rationale": value[:MAX_RATIONALE_CHARS]}
        # Upstream sets _raw ONLY when parsing fails (predict.py:303), so a dict
        # that parsed but lacks every rationale key scores "" there -- not its own
        # JSON text. Mirror that: fall back to the reply only when nothing parsed.
        return {"gold_rationale": recovery[:MAX_RATIONALE_CHARS]}

    raise ValueError(f"unknown task_type: {task_type!r}")


# --------------------------------------------------------------------------
# lm protocol
# --------------------------------------------------------------------------

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
# An lm option index is a bare small integer. Bounded and delimiter-guarded so a
# `$5$` reagent placeholder is not read as option 5, and a 5,000-digit run does
# not reach int(), whose 4,300-digit limit raises ValueError.
_INDEX_RE = re.compile(r"(?<![\w$.])-?\d{1,6}(?![\w$.])")


def _norm_token(token: str) -> str:
    """Upstream's provider-token normalizer: strip spaces and stray quotes."""
    return str(token).strip().strip('"').strip("'").strip()


def _first_token_alternatives(logprobs: Any) -> List[Tuple[str, float]]:
    """``(token, logprob)`` candidates at the first generated position only, which
    is where upstream compares the decision tokens. The chosen token is included
    because a provider may omit it from its own ``top_logprobs``.
    """
    if not isinstance(logprobs, list) or not logprobs:
        return []
    first = logprobs[0]
    if not isinstance(first, dict):
        return []
    out: List[Tuple[str, float]] = []
    token, logprob = first.get("token"), first.get("logprob")
    if isinstance(token, str) and isinstance(logprob, (int, float)):
        out.append((token, float(logprob)))
    for alternative in first.get("top_logprobs") or []:
        if not isinstance(alternative, dict):
            continue
        token, logprob = alternative.get("token"), alternative.get("logprob")
        if isinstance(token, str) and isinstance(logprob, (int, float)):
            out.append((token, float(logprob)))
    return out


def _restricted_argmax(alternatives: List[Tuple[str, float]], match) -> Optional[Any]:
    """Highest-logprob candidate whose token ``match``es, or None. Restricting to
    the decision tokens makes any surrounding scaffolding irrelevant.
    """
    best_value, best_logprob = None, float("-inf")
    for token, logprob in alternatives:
        value = match(_norm_token(token))
        if value is not None and logprob > best_logprob:
            best_value, best_logprob = value, logprob
    return best_value


def _match_yes_no(token: str) -> Optional[bool]:
    upper = token.upper()
    if upper in ("YES", "Y", "TRUE"):
        return True
    if upper in ("NO", "N", "FALSE"):
        return False
    return None


def _match_index(token: str) -> Optional[int]:
    return int(token) if token.isdigit() else None


def to_prediction_lm(task_type: str, raw: Optional[str], logprobs: Any = None) -> Dict[str, Any]:
    """Parse an lm-protocol reply, whose contract is one bare decision token.

    Upstream decides from token probabilities; this agrees whenever the reply
    opens with a decision token. Replies opening with neither take the
    conservative default rather than leaving the denominator. See README.
    """
    alternatives = _first_token_alternatives(logprobs)
    if task_type in ("step_validation", "condition_validation"):
        decided = _restricted_argmax(alternatives, _match_yes_no)
        if decided is not None:
            return {"score": 1.0 if decided else 0.0, "label": decided, "status": "ok_logprobs"}
    elif task_type == "contrastive_choice":
        decided = _restricted_argmax(alternatives, _match_index)
        if decided is not None:
            return {"predicted_option_idx": decided, "status": "ok_logprobs"}

    text = _norm_token(_THINK_BLOCK_RE.sub(" ", raw or ""))
    text = _OPEN_THINK_RE.sub(" ", text).strip()

    if task_type in ("step_validation", "condition_validation"):
        if not text:
            return {"score": 0.5, "label": False, "status": "empty_output"}
        yes, no = _YES_RE.search(text), _NO_RE.search(text)
        if yes and (not no or yes.start() < no.start()):
            return {"score": 1.0, "label": True, "status": "ok"}
        if no:
            return {"score": 0.0, "label": False, "status": "ok"}
        # Neither token present. Deliberately unlike the gen path, where 0.5 is
        # upstream's own >= 0.5 default and counts positive: here there is no
        # upstream default to match, since upstream reads probability mass and
        # abstains outright. Negative is the conservative reading. See README.
        return {"score": 0.5, "label": False, "status": "no_decision_token"}

    if task_type == "contrastive_choice":
        if not text:
            return {"predicted_option_idx": -1, "status": "empty_output"}
        match = _INDEX_RE.search(text)
        if match is None:
            return {"predicted_option_idx": -1, "status": "no_decision_token"}
        return {"predicted_option_idx": int(match.group()), "status": "ok"}

    raise ValueError(f"task_type {task_type!r} has no lm protocol")
