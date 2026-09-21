# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-output handling for CombiBench, mirroring upstream one-stage Fine-Eval.

The rules below reproduce the behaviour of ``evaluation/util.py`` and
``evaluation/verifier/one_stage_verify.py`` in
https://github.com/MoonshotAI/CombiBench at revision
c67e4213597b1477351d9ef5ca37fb622084cc78 (MIT License, Copyright (c) 2025
Moonshot AI and Project Numina). Nothing here is copied verbatim; the regular
expressions and the split-on-blank-line statement check are re-derived from that
code so that a rollout scored here and a rollout scored by upstream agree, and
every deliberate difference is named at its site.

Upstream's pipeline for one model output is:

1. take the **last** fenced ``lean4`` block (falling back to ``lean``);
2. strip comments, so a theorem hidden inside a comment cannot satisfy step 4;
3. reject the code outright if it contains ``axiom`` or ``local_instance``;
4. require every non-header paragraph of the reference statement, with its
   ``sorry`` removed, to appear verbatim in the code — this is the guard
   against a model weakening the statement it proves;
5. for every ``abbrev <name>_solution`` paragraph, append
   ``example : <name>_solution = <ground truth> := by try rfl; try norm_num``;
6. compile; success iff no error message and no ``sorry`` warning.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional


# Upstream prepends this when the model's code does not begin with an import.
DEFAULT_HEADER = """import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

"""

# Upstream rejects any code containing these substrings, comments excluded.
FORBIDDEN_SUBSTRINGS = ("axiom", "local_instance")

# Statement paragraphs beginning with these keywords are not checked for
# presence in the model's code (upstream treats them as header).
HEADER_PREFIXES = ("import", "set_option", "open")

SORRY_WARNING = "declaration uses 'sorry'"

_LEAN4_BLOCK_RE = re.compile(r"```lean4\n(.*?)\n```", re.DOTALL)
_LEAN_BLOCK_RE = re.compile(r"```lean\n(.*?)\n```", re.DOTALL)
_BLOCK_COMMENT_RE = re.compile(r"/-[\s\S]*?-/")
_LINE_COMMENT_RE = re.compile(r"^\s*--.*\n", re.MULTILINE)
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def remove_comments(text: str) -> str:
    """Drop block and whole-line comments, exactly as upstream does.

    Upstream removes block comments first and then replaces each whole-line
    ``--`` comment with a bare newline. A trailing ``-- ...`` after code on the
    same line is kept, matching upstream.
    """
    text = _BLOCK_COMMENT_RE.sub("", text)
    return _LINE_COMMENT_RE.sub("\n", text)


def extract_lean_code(text: str) -> Optional[str]:
    """Return the model's Lean code, or None when no fenced Lean block exists.

    The last block is used, so a model that revises its proof is read at its
    conclusion. Comments are removed and the default header is prepended when
    the code does not start with an ``import`` line.
    """
    blocks = _LEAN4_BLOCK_RE.findall(text)
    if not blocks:
        blocks = _LEAN_BLOCK_RE.findall(text)
    if not blocks:
        return None
    code = remove_comments(blocks[-1]).strip()
    if not code.startswith("import"):
        code = DEFAULT_HEADER + code
    return code


def has_forbidden_substring(code: str) -> bool:
    """Upstream's ban on ``axiom``/``local_instance`` is a plain substring test."""
    return any(token in code for token in FORBIDDEN_SUBSTRINGS)


def statement_chunks(formal_statement: str) -> list[str]:
    """Split the reference statement into the paragraphs the code must contain.

    Mirrors upstream: comments removed, split on blank lines, header
    paragraphs dropped, every ``sorry`` deleted, each paragraph stripped. Empty
    paragraphs are dropped here because an empty string is a substring of
    everything and therefore checks nothing.
    """
    chunks = remove_comments(formal_statement).split("\n\n")
    chunks = [chunk.replace("\ntheorem", "\n\ntheorem") for chunk in chunks]
    result = []
    for chunk in chunks:
        if chunk.startswith(HEADER_PREFIXES):
            continue
        cleaned = chunk.replace("sorry", "").strip()
        if cleaned:
            result.append(cleaned)
    return result


def _normalize_trailing_whitespace(text: str) -> str:
    return _TRAILING_WS_RE.sub("", text)


def missing_chunks(code: str, chunks: list[str], normalize_trailing_whitespace: bool = True) -> list[str]:
    """Return the statement paragraphs that do not appear verbatim in ``code``.

    Deliberate departure from upstream: with ``normalize_trailing_whitespace``
    (the default) trailing spaces and tabs are stripped from every line on both
    sides before comparing. Thirteen of the hundred pinned statements contain
    lines consisting only of spaces, left behind when upstream deleted comments
    from the published dataset. Trailing whitespace is never significant to
    Lean, so a model that copies the statement without those invisible
    characters has not changed what it proves; upstream would reject it. Set
    the flag to ``False`` for upstream's byte-exact behaviour.
    """
    if normalize_trailing_whitespace:
        code = _normalize_trailing_whitespace(code)
        chunks = [_normalize_trailing_whitespace(chunk) for chunk in chunks]
    return [chunk for chunk in chunks if chunk not in code]


def answer_tags(chunks: list[str]) -> list[str]:
    """Names of the ``<name>_solution`` abbreviations, in statement order.

    Upstream derives the name from the text between ``abbrev`` and
    ``_solution`` of each paragraph containing both tokens. The order matters:
    tags are zipped positionally with the published ground-truth list.
    """
    tags = []
    for chunk in chunks:
        if "_solution" in chunk and "abbrev" in chunk:
            name = chunk.split("abbrev", 1)[1].split("_solution", 1)[0].strip()
            tags.append(f"{name}_solution")
    return tags


_ABBREV_TYPE_RE = re.compile(
    r"^(?:noncomputable\s+)?abbrev\s+\S+"
    r"(?P<binders>(?:\s*(?:\{[^}]*\}|\([^)]*\)|\[[^\]]*\]))*)"
    r"\s*:\s*(?P<type>.+?)\s*:=\s*$",
    re.DOTALL,
)


def abbrev_types(chunks: list[str]) -> list[Optional[str]]:
    """Declared type of each ``<name>_solution`` abbreviation, aligned with ``answer_tags``.

    ``abbrev name {k} : (Fin k → ℕ) → ℕ :=`` yields ``(Fin k → ℕ) → ℕ``. None
    when the paragraph does not parse, in which case the check falls back to
    upstream's unascribed form.
    """
    types: list[Optional[str]] = []
    for chunk in chunks:
        if "_solution" in chunk and "abbrev" in chunk:
            match = _ABBREV_TYPE_RE.match(chunk)
            types.append(match.group("type").strip() if match else None)
    return types


def answer_check(tag: str, ground_truth: str, type_ascription: Optional[str] = None) -> str:
    """The Lean snippet appended to compare a filled-in answer with gold.

    Without ``type_ascription`` this is byte-for-byte upstream's snippet. With
    it, the gold answer is elaborated at the abbreviation's declared type.

    Deliberate departure from upstream: ``=`` elaborates both sides before
    unifying them, so a gold answer such as ``fun n => ⌈√n⌉₊ - 1`` is read with
    ``n : ℝ`` and then fails to match an ``ℕ → ℕ`` abbreviation, and
    ``fun n => n * (n + 1) / 4`` fails to synthesize ``HDiv ℕ ℕ ℚ``. Measured
    on the pinned corpus, five of the 45 published answers never elaborate in
    upstream's form, so those problems cannot be solved under it even with the
    exact gold answer; ascribing the declared type makes all 45 elaborate.
    """
    rhs = ground_truth if type_ascription is None else f"({ground_truth} : {type_ascription})"
    return f"\n\nexample: {tag} = {rhs} := by\n  try rfl\n  try norm_num"


def build_submission(
    code: str,
    tags: list[str],
    ground_truths: Optional[list[str]],
    types: Optional[list[Optional[str]]] = None,
) -> str:
    """Append one answer check per (tag, ground truth) pair, positionally zipped.

    Upstream zips the two lists and silently ignores any surplus on either
    side; this does the same so the compiled text is identical. ``types``, when
    given, ascribes each gold answer with the abbreviation's declared type.
    """
    if not ground_truths:
        return code
    if types is None:
        types = [None] * len(tags)
    return code + "".join(answer_check(tag, gt, ty) for tag, gt, ty in zip(tags, ground_truths, types))


@dataclass
class LeanResult:
    """What the Lean server said about one submission."""

    error: Optional[str] = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    sorries: list[dict[str, Any]] = field(default_factory=list)
    time: Optional[float] = None
    # True when the failure happened before Lean ran (network, HTTP, malformed
    # reply). The model cannot cause these, so they are reported as harness
    # faults rather than as a failed proof.
    transport_failure: bool = False


def classify_lean_result(result: LeanResult) -> str:
    """Map a Lean server reply onto a status, mirroring upstream ``is_error``.

    Upstream fails a submission when the server reports an error string, when
    any message has severity ``error``, or (with sorry not accepted) when a
    warning says the declaration uses ``sorry``. The REPL's ``sorries`` list is
    consulted too: it is the structured form of the same warning.
    """
    if result.transport_failure:
        return "lean_server_error"
    if result.error:
        return "timeout" if "timed out" in result.error else "lean_error"
    if any(message.get("severity") == "error" for message in result.messages):
        return "proof_failed"
    has_sorry_warning = any(
        message.get("severity") == "warning" and SORRY_WARNING in str(message.get("data", ""))
        for message in result.messages
    )
    if has_sorry_warning or result.sorries:
        return "has_sorry"
    return "success"
