# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract the model's JSON object and coerce it into the scorer's contract.

Every ChemReason prompt ends with "Return JSON ONLY with EXACT keys", so a
well-behaved reply is a bare JSON object. Reasoning models are not well-behaved:
they emit think blocks, prose, and fenced code. This module recovers the object
and then applies the same field coercions upstream's ``predict.py`` applies, so
a reply that upstream would have scored is scored identically here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


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
    """Yield candidate JSON objects, LEFTMOST-first, by brace matching.

    Callers take the LAST successful parse: when a reply contains several
    objects, the rightmost is the model's conclusion, not a draft it corrected
    further down. Ordering candidates by position rather than by kind (fenced
    before bare) is what makes that true.
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
        if ch == '"':
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


def extract_json(raw: Optional[str]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Return ``(object, status)``; ``object`` is None when nothing parsed."""
    if not raw or not raw.strip():
        return None, "empty_output"
    text = raw[:MAX_RESPONSE_CHARS]
    text = _THINK_BLOCK_RE.sub(" ", text)
    text = _OPEN_THINK_RE.sub(" ", text)
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
    try:
        out = float(value)
    except (TypeError, ValueError):
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
    """Port of upstream ``post_ordering``.

    Three behaviours our first implementation lacked, in increasing order of how
    much they move a score: step ids are canonicalized, repeats are dropped, and
    -- once at least one legal id has matched -- the ids the model never
    mentioned are appended in presentation order. That last one is upstream's
    own comment's "critical behavior": a partial answer is completed rather than
    scored short, but a wholly unmatched answer is NOT fabricated from the
    expected order.
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
    """Port of upstream ``post_contrastive``, index only.

    Recovers the choice from the raw text when ``predicted_choice`` is absent or
    unknown, and range-checks the index. Upstream is explicit that an invalid
    index must NOT fall back to option 0, so it becomes -1 and scores wrong.
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
) -> Dict[str, Any]:
    """Coerce a parsed object into the shape ``metrics.score_row`` expects.

    A missing or malformed object is NOT excused -- it yields the conservative
    fallback upstream uses (score 0.5 -> label False for the binary tasks, empty
    order, index -1), so the row scores as a wrong answer rather than being
    dropped from the denominator.

    The "no object at all" and "object without the requested key" cases are
    deliberately NOT collapsed. Upstream forces ``label=False`` only when the
    reply was not a JSON dict; a dict whose ``score`` is missing falls through
    to ``0.5 >= 0.5`` and so counts as positive. Collapsing them would label
    every unparseable reply positive, and with 46-57% of gold labels positive
    that alone buys ~0.63-0.73 f1_positive -- the do-nothing floor, credited as
    if it were capability.
    """
    no_object = not isinstance(obj, dict)
    obj = obj if isinstance(obj, dict) else {}

    if task_type == "ordering":
        return {"predicted_order": post_ordering(None if no_object else obj, expected_step_ids or [], raw)}

    if task_type == "contrastive_choice":
        return {"predicted_option_idx": post_contrastive(None if no_object else obj, options or [], raw)}

    if task_type in ("step_validation", "condition_validation"):
        # The prompt asks for `score` only; the label is derived, never requested.
        if no_object:
            return {"score": 0.5, "label": False}
        score = obj.get("score", obj.get("prob_positive"))
        score = min(1.0, max(0.0, _as_float(score, 0.5)))
        return {"score": score, "label": bool(score >= BINARY_LABEL_THRESHOLD)}

    if task_type == "step_completion":
        slots = obj.get("slots")
        if not isinstance(slots, dict):
            slots = {}
        # Keys must be strings for the slot matcher's endswith/startswith checks.
        slots = {str(k): v for k, v in slots.items()}
        return {"action": str(obj.get("action", "")), "slots": slots}

    if task_type == "rationalization":
        for key in ("gold_rationale", "rationale", "predicted_rationale", "answer"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return {"gold_rationale": value[:MAX_RATIONALE_CHARS]}
        return {"gold_rationale": ""}

    raise ValueError(f"unknown task_type: {task_type!r}")


# --------------------------------------------------------------------------
# lm protocol
# --------------------------------------------------------------------------

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)
_INDEX_RE = re.compile(r"-?\d+")


def _norm_token(token: str) -> str:
    """Upstream's provider-token normalizer: strip spaces and stray quotes."""
    return str(token).strip().strip('"').strip("'").strip()


def _first_token_alternatives(logprobs: Any) -> List[Tuple[str, float]]:
    """Flatten the first generated position into ``(token, logprob)`` candidates.

    Upstream scores lm by comparing the decision tokens' logits at exactly this
    position, so only position 0 is read. The chosen token is included because a
    provider may report it outside its own ``top_logprobs`` list.
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
    """Highest-logprob candidate whose token ``match``es, or None.

    This is upstream's decision rule: an argmax restricted to the decision
    tokens, so whatever scaffolding the model would have gone on to emit is
    irrelevant. Tokenizers differ on leading spaces and case, hence the
    normalization inside each matcher.
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

    Upstream picks the label by argmax over the decision tokens' logits. With
    greedy decoding the first generated token IS that argmax over the whole
    vocabulary, which agrees whenever the top token is one of the decision
    tokens -- overwhelmingly the case given the prompt asks for exactly that.
    Where it can differ is a reply that opens with neither, and there upstream's
    restricted argmax still commits while generation does not. Those fall back
    to the conservative default rather than being dropped, so the denominator is
    unchanged either way.
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
        # Neither token present: upstream's conservative fallback.
        return {"score": 0.5, "label": False, "status": "no_decision_token"}

    if task_type == "contrastive_choice":
        if not text:
            return {"predicted_option_idx": -1, "status": "empty_output"}
        match = _INDEX_RE.search(text)
        if match is None:
            return {"predicted_option_idx": -1, "status": "no_decision_token"}
        return {"predicted_option_idx": int(match.group()), "status": "ok"}

    raise ValueError(f"task_type {task_type!r} has no lm protocol")
