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

"""Text-level checks shared by the Lean benchmarks, run before the sandbox call.

These are free; a Mathlib compile is not. Anything that has already lost -- no code block, a
fabricated `axiom`, a weakened statement -- is rejected here rather than after paying for one.

Shared rather than copied per server because these are exactly the functions that accumulate
subtle bugs: comment-aware scanning, fence extraction, `<think>` stripping. Each was wrong at
least once during development, and a fix should land in one place.
"""

import re
from typing import List, Optional, Tuple


# Thinking models emit these; the code block inside must not be mistaken for the answer, and
# the reasoning text must not be scanned for banned tokens.
THINK_RE = re.compile(r"<(think|thinking)>.*?</\1>", re.S | re.I)
FENCE_RE = re.compile(r"```(?:lean4|lean)?\s*\n(.*?)```", re.S | re.I)

# Declarations that would let a model assert its way to a proof. `sorry`/`admit` are absent on
# purpose: the file is allowed to contain them elsewhere, and `#print axioms` is what decides
# whether the target itself is honest.
BANNED_DECL_RE = re.compile(r"^\s*(axiom|unsafe)\s", re.M)


def strip_thinking(text: str) -> str:
    """Remove ``<think>``/``<thinking>`` blocks, closed or dangling."""
    text = THINK_RE.sub("", text)
    # An unclosed <think> means the model ran out of budget mid-reasoning; everything after it
    # is reasoning, not an answer.
    opener = re.search(r"<(think|thinking)>", text, re.I)
    return text[: opener.start()] if opener else text


def has_unterminated_block_comment(code: str) -> bool:
    """Does a ``/-`` block comment run to end of file without closing?

    Worth asking separately: when a model mangles a closing delimiter (writing ``- /`` for
    ``-/``, which happened 48 times in a 12k-rollout run) the rest of the file is swallowed by
    the comment. Every later check then sees an empty file and reports "statement modified",
    which blames the model for cheating when it actually produced malformed output.
    """
    i, n, depth = 0, len(code), 0
    while i < n:
        two = code[i : i + 2]
        if two == "/-":
            depth += 1
            i += 2
        elif two == "-/" and depth:
            depth -= 1
            i += 2
        elif two == "--" and depth == 0:
            j = code.find("\n", i)
            i = n if j < 0 else j + 1
        else:
            i += 1
    return depth > 0


def strip_comments_and_strings(code: str) -> str:
    """Blank out comments and string literals, preserving length and line structure.

    Offsets are preserved so a match found here can be reported against the original text.
    An unterminated block comment blanks everything after it; callers that care about telling
    "malformed file" apart from "altered statement" should call
    :func:`has_unterminated_block_comment` first.
    """
    out = list(code)
    i, n = 0, len(code)
    while i < n:
        two = code[i : i + 2]
        if two == "/-":
            depth, j = 1, i + 2
            while j < n and depth:
                if code[j : j + 2] == "/-":
                    depth += 1
                    j += 2
                elif code[j : j + 2] == "-/":
                    depth -= 1
                    j += 2
                else:
                    j += 1
            for k in range(i, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif two == "--":
            j = code.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif code[i] == '"':
            j = i + 1
            while j < n and code[j] != '"':
                j += 2 if code[j] == "\\" else 1
            for k in range(i, min(j + 1, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
        else:
            i += 1
    return "".join(out)


def extract_lean_code(response_text: str) -> str:
    """Return the Lean file the model produced: the last fenced block, else the raw text."""
    text = strip_thinking(response_text)
    blocks = FENCE_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def find_banned_declarations(code: str) -> List[str]:
    """``axiom``/``unsafe`` declarations added by the model, ignoring comments and strings."""
    return sorted({m.group(1) for m in BANNED_DECL_RE.finditer(strip_comments_and_strings(code))})


def _normalise(text: str) -> str:
    """Collapse whitespace and unify `lemma`/`theorem` so cosmetic edits are not treated as edits.

    `lemma` is a macro for `theorem` in Lean 4 -- the two are interchangeable and neither
    weakens anything. Comparing them verbatim rejected 53 correct submissions in a 12k-rollout
    run, purely because the model wrote `theorem` where upstream wrote `lemma`.
    """
    stripped = strip_comments_and_strings(text)
    stripped = re.sub(r"\blemma\b", "theorem", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


def check_statement_preserved(target_statement: str, submission: str) -> Tuple[bool, Optional[str]]:
    """Is the theorem the model proved still the theorem it was asked to prove?

    Compares against the target's recorded signature rather than splitting the file on
    ``:= by sorry``. The task file keeps every declaration before the target, and those
    routinely include open conjectures written as exactly ``:= by sorry`` -- so the hole is not
    textually unique and a split-based check throws out honest answers (measured: 23 of 100).

    Whitespace and comments are normalised away, so reindenting or rewrapping is free; changing
    a hypothesis, a binder, the conclusion or the name is not.

    This exists because the whole-file format lets a model weaken the theorem and hand back
    something that compiles, which the compiler by definition cannot catch.
    """
    needle = _normalise(target_statement)
    if not needle:
        return False, "No target statement recorded for this task."
    if needle not in _normalise(submission):
        return False, f"The target statement was altered or removed: expected {needle[:100]!r}"
    return True, None
