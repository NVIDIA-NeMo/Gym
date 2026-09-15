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

"""Turn Formal Conjectures source files into self-contained Lean 4 proof tasks.

Formal Conjectures (https://github.com/google-deepmind/formal-conjectures) is a library of
formalized mathematical statements, most of them **open** -- the headline conjectures carry a
``sorry`` that nobody on earth can fill. Those are useless as a benchmark: every model scores
zero and the number says nothing.

What is usable is the other half of the repo. Alongside each open conjecture sit theorems that
ship with **real Lean proofs**: ``@[category test]`` sanity checks, ``@[category API]``
supporting lemmas, textbook exercises, and the handful of ``research solved`` results that have
been formalized. Those have ground truth, so "did the model prove it?" is a decidable question.
This module strips the proof off such a theorem and hands back the statement with a hole.

Three things make an FC file harder to turn into a task than a LeanCat one:

1. **Files are not self-contained.** They ``import FormalConjecturesUtil`` and 760 of 1268 of
   them declare ``def``s that the target theorem depends on. So a task is a *whole file*, not a
   bare theorem, and the definitions have to travel with it.

2. **One file holds several declarations, and the others may legitimately contain ``sorry``.**
   A file typically pairs a proved ``test`` lemma with the open conjecture it sanity-checks.
   A LeanCat-style "no ``sorry`` anywhere" check would reject a correct answer. Rather than
   teach the verifier to scope its check to one declaration, we emit a task containing exactly
   one declaration -- everything else is dropped -- so the file has exactly one hole and the
   simple check is the correct check.

3. **``FormalConjecturesUtil`` would have to be in the sandbox.** It exists for FC's own
   metadata tooling: the ``@[category ...]``/``@[AMS ...]`` attributes and ``answer(...)``
   syntax. None of that is needed to state a proof obligation, so the attributes are stripped
   and the import is rewritten to plain ``import Mathlib``. Tasks then compile against a stock
   Mathlib sandbox and FC never has to be installed. Statements that genuinely reference
   FC-only definitions (``dominationNumber``, ``cvetkovic``, ``answer(...)``, ...) cannot
   survive that rewrite -- they are detected here and dropped, and anything this filter misses
   is caught by the compile-the-reference validation step (see ``prepare.py``).
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Set


# A declaration is introduced by FC's own attribute. Everything we care about hangs off it.
# `category` is not always the first item: `@[simp, category API, AMS 5]` is common, and a
# regex anchored on `@[category` misses those entirely.
CATEGORY_RE = re.compile(r"^@\[[^\]]*\bcategory\s+([^,\]]*)[^\]]*\]", re.M)
ATTR_LIST_RE = re.compile(r"@\[([^\]]*)\]", re.S)
# Attributes to strip from the emitted task: they need FormalConjecturesUtil to elaborate.


def strip_fc_attributes(text: str) -> str:
    """Remove FC's ``category``/``AMS`` attributes, preserving any others in the same list.

    ``@[simp, category API, AMS 5]`` must become ``@[simp]``, not vanish: dropping the whole
    list silently removes a ``simp`` tag that later proofs in the file rely on. Removing only
    the leading form leaves the FC attributes in place, which then fail to elaborate against a
    stock Mathlib sandbox with a bare `unexpected token; expected ']'`.
    """

    def rewrite(m: re.Match) -> str:
        items = [i.strip() for i in m.group(1).split(",")]
        kept = [i for i in items if i and not re.match(r"^(category|AMS)\b", i)]
        return f"@[{', '.join(kept)}]" if kept else ""

    out = ATTR_LIST_RE.sub(rewrite, text)
    # collapse the blank line left where a whole attribute was removed
    return re.sub(r"\n[ \t]*\n(?=[ \t]*(theorem|lemma|def|abbrev|instance)\s)", "\n", out)


def _absorb_docstring(text: str, start: int) -> int:
    """Start of the docstring directly above ``start``, else ``start`` itself.

    Deliberately not a regex: a lazy ``/--.*?-/\s*\Z`` will happily match a docstring from
    far earlier in the file by stretching across every declaration between, which silently
    turns one block into half the file.
    """
    head = text[:start].rstrip()
    if not head.endswith("-/"):
        return start
    open_idx = head.rfind("/--")
    if open_idx < 0:
        return start
    # The docstring must be the thing immediately above: its first close must be its last.
    if head.find("-/", open_idx + 3) != len(head) - 2:
        return start
    return open_idx


THEOREM_RE = re.compile(r"^(theorem|lemma)\s+([A-Za-z_][A-Za-z0-9_.'!?]*)", re.M)
IMPORT_RE = re.compile(r"^import\s+(\S+)\s*$", re.M)

OPEN_PAIRS = {"(": ")", "[": "]", "{": "}", "⟨": "⟩"}
CLOSE_PAIRS = {v: k for k, v in OPEN_PAIRS.items()}


@dataclass
class Task:
    """One extracted proof obligation."""

    task_id: str
    source_path: str
    declaration: str
    # Namespace-qualified name. The verifier appends `#print axioms <full_name>` to whatever
    # file the model returns, and that file may or may not still close the namespace, so the
    # short name cannot be relied on to resolve.
    full_name: str
    category: str
    ams: Optional[str]
    # The file we hand the model: imports + defs + the target statement, proof replaced by sorry.
    # The target's signature, from its keyword up to the `:=` that starts the proof. The
    # verifier checks THIS is unchanged, rather than splitting the file on `:= by sorry`:
    # the kept prefix routinely contains open conjectures written in exactly that form, so
    # the hole is not textually unique and a split-based check rejects honest answers.
    target_statement: str
    task_file: str
    # The same file with FC's original proof restored. Compiling this is how we prove the
    # extraction is sound -- if the reference does not compile, the task is unanswerable.
    reference_file: str
    reference_proof: str


def split_top_level_assign(text: str, start: int = 0) -> int:
    """Index of the ``:=`` that ends a declaration's signature, or -1.

    Bracket-depth aware, because statements routinely contain ``:=`` inside binders
    (``let x := ...``, structure instances) and a naive ``find`` cuts the statement in half.
    Comments are skipped: FC docstrings are full of LaTeX that contains ``:=``, and a scan
    that does not skip them cuts the "proof" out of the middle of a comment.

    ``start`` should point at the declaration keyword, so a docstring *above* the declaration
    is never scanned at all.
    """
    depth = 0
    i = start
    n = len(text)
    while i < n - 1:
        two = text[i : i + 2]
        if two == "/-":  # block comment (incl. /-- docstrings): skip to its matching -/
            close = text.find("-/", i + 2)
            i = n if close < 0 else close + 2
            continue
        if two == "--":  # line comment
            nl = text.find("\n", i)
            i = n if nl < 0 else nl + 1
            continue
        c = text[i]
        if c in OPEN_PAIRS:
            depth += 1
        elif c in CLOSE_PAIRS:
            depth -= 1
        elif c == ":" and text[i + 1] == "=" and depth == 0:
            return i
        i += 1
    return -1


def _block_bounds(text: str) -> List[tuple]:
    """(start, end) of every ``@[category ...]`` declaration block, docstring included.

    A block runs from its attribute (or the docstring immediately above it) to the next
    attribute or the next top-level ``end``, whichever comes first.
    """
    raw_starts = [m.start() for m in CATEGORY_RE.finditer(text)]

    # Absorb a docstring sitting directly above the attribute: it belongs to the declaration
    # and would otherwise be left dangling above an unrelated one.
    starts = [_absorb_docstring(text, s) for s in raw_starts]

    # Ends must respect the *adjusted* start of the next block. Using the raw attribute
    # position here makes consecutive ranges overlap by the next block's docstring, so
    # deleting one block silently eats the next one's documentation.
    bounds = []
    for i, start in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else len(text)
        end_kw = re.compile(r"^end\s", re.M).search(text, start, nxt)
        bounds.append((start, end_kw.start() if end_kw else nxt))
    return bounds


def extract_file(path: str, text: str, fc_only_names: Set[str]) -> List[Task]:
    """Extract every proved theorem in ``text`` as its own single-hole task."""
    imports = IMPORT_RE.findall(text)
    if imports != ["FormalConjecturesUtil"]:
        # Cross-imports of sibling FC modules would need those modules in the sandbox.
        return []

    bounds = _block_bounds(text)
    tasks: List[Task] = []

    for idx, (start, end) in enumerate(bounds):
        block = text[start:end]
        if re.search(r"\bsorry\b", block):
            continue  # already a hole: an open conjecture, no ground truth to check against
        thm = THEOREM_RE.search(block)
        if not thm:
            continue  # a def/example carrying the attribute, not a proof obligation

        cat_match = CATEGORY_RE.search(block)
        category = cat_match.group(1).split(",")[0].strip() if cat_match else "unknown"
        for prefix in ("research open", "research solved"):
            if category.startswith(prefix):
                category = prefix
        ams_match = re.search(r"AMS\s+([0-9 ]+)", block)

        # Namespaces opened before this declaration, so the target can be named from outside.
        ns = []
        for m in re.finditer(r"^(namespace|end)\s+(\S+)", text[: bounds[idx][0]], re.M):
            if m.group(1) == "namespace":
                ns.append(m.group(2))
            elif ns and ns[-1] == m.group(2):
                ns.pop()

        assign = split_top_level_assign(block, thm.start())
        if assign < 0:
            continue  # cannot locate the proof boundary; refuse to guess

        # Keep every declaration BEFORE the target and drop everything after it.
        #
        # Dropping the earlier ones too was the obvious simplification and it is wrong:
        # `@[category API]` declarations exist to "construct basic theory around a new
        # definition", and proofs lean on them. Measured on a 32-task sample, deleting them
        # broke 14 of 32 reference files -- `unsolved goals`, `Function expected at`,
        # `Unknown identifier`. Lean files are linear, so keeping the prefix restores every
        # dependency a proof can legally have.
        #
        # The cost is that the emitted file may still contain other `sorry`s (the open
        # conjectures this file is built around). That is why the verifier asks Lean
        # `#print axioms <target>` instead of grepping for `sorry`: the question is whether
        # THIS declaration is proved, not whether the file mentions the token anywhere.
        skeleton = text[: bounds[idx][1]]

        # FC's attributes and its Util import cannot elaborate against a stock Mathlib sandbox.
        skeleton = strip_fc_attributes(skeleton)
        skeleton = IMPORT_RE.sub("import Mathlib", skeleton, count=1)

        target = strip_fc_attributes(block)
        target_thm = THEOREM_RE.search(target)
        if not target_thm:
            continue
        target_assign = split_top_level_assign(target, target_thm.start())
        if target_assign < 0:
            continue
        target_statement = target[:target_assign]
        target_proof = target[target_assign + 2 :].strip()

        holed = target_statement + ":= by\n  sorry\n"
        if target not in skeleton:
            continue  # defensive: the block must appear verbatim for the swap to be sound
        task_file = skeleton.replace(target, holed)
        reference_file = skeleton

        blob = task_file
        if any(re.search(rf"\b{re.escape(n)}\b", blob) for n in fc_only_names):
            continue  # needs a definition only FormalConjecturesUtil/ForMathlib provides
        if re.search(r"\banswer\b\s*\(", blob):
            continue  # FC's `answer(...)` syntax comes from Util

        tasks.append(
            Task(
                task_id=f"{path}::{thm.group(2)}",
                source_path=path,
                declaration=thm.group(2),
                full_name=".".join(ns + [thm.group(2)]),
                category=category,
                ams=ams_match.group(1).strip() if ams_match else None,
                target_statement=target_statement[target_thm.start() :].strip(),
                task_file=task_file,
                reference_file=reference_file,
                reference_proof=target_proof,
            )
        )
    return tasks
