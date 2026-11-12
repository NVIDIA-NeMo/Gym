# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foreign-script letters: letters in the turn's final reply of a script that the turn's input never shows.

A policy that writes a non-Latin language sometimes splices a few characters of another script into its reply, such as
Chinese characters or Cyrillic letters inside a Persian or Arabic sentence. This module counts such garbling with the
standard library only: a letter's script is the first word of its Unicode name (`SCRIPTS`). There is no language
identification and no word list.

The rule:

1. The reply is what the user reads of the turn's last model reply: its assistant messages and the strings in the
   arguments of its stop-tool calls (a question to the user), each object's strings on one line, as a question's
   option shows its label and description.
2. Code (fenced blocks, where a fence left open hides the rest, and inline spans), URLs and e-mail addresses, HTML and
   XML tags, block-quote lines and quoted spans ("...", “...”, «...», 「...」 and the like) are removed from it: any
   script may legitimately appear there.
3. A script of what is left is foreign when the turn's input holds none of its letters. The input is the row's
   instructions, every history item but reasoning (each with all its text fields), and the turn's tool results.
   Text counts as written: a JSON escape such as \\u4e2d is not a letter. (Results that echo a file the policy wrote
   often escape it, and decoding them would let the policy's own garbled writes excuse the same garbling.)
4. A foreign script with at least `MIN_LETTERS` letters counts, unless:
   - it is Latin: identifiers, commands, paths and product names are Latin in every agent conversation;
   - it is Greek and none of its words has three letters: lone Greek letters are math (α, β, π);
   - it is written mostly in lines of its own: at least half its letters sit on lines where it holds the majority
     of the letters (Han, kana and Bopomofo counted together, as Chinese and Japanese write them together). That is
     a deliberate passage, such as a translation or a reply in another language, not a fragment spliced into a
     sentence;
   - it is Han and the input has kana: Japanese writes kanji (Han) among its kana.

`foreign_script_letters` returns the letters of the scripts that count: 0 when none does.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any, Collection

from resources_servers.agentic_turn_simulation.items import Items, content_text, json_object


# A foreign script needs this many letters in the reply to count: fewer are a stray symbol.
MIN_LETTERS = 3
# The scripts told apart, by the first word of a letter's Unicode name (after FULLWIDTH or HALFWIDTH). Other letters
# belong to none: math alphanumerics, letterlike symbols such as µ and Ω, modifier letters.
SCRIPTS = {
    **{
        word: word.title()
        for word in (
            "LATIN GREEK CYRILLIC ARMENIAN HEBREW ARABIC SYRIAC THAANA DEVANAGARI BENGALI GURMUKHI GUJARATI ORIYA TAMIL "
            "TELUGU KANNADA MALAYALAM SINHALA THAI LAO TIBETAN MYANMAR GEORGIAN HANGUL ETHIOPIC CHEROKEE KHMER "
            "MONGOLIAN BOPOMOFO YI"
        ).split()
    },
    "CJK": "Han",
    "HIRAGANA": "Kana",
    "KATAKANA": "Kana",
    "KATAKANA-HIRAGANA": "Kana",
}
# Scripts that one language writes together, as one script for the line-majority test.
FAMILIES = {"Han": "CJK", "Kana": "CJK", "Bopomofo": "CJK"}
# What the reply loses before its letters are counted, in this order.
_REMOVED = (
    re.compile(r"(```|~~~).*?(?:\1|\Z)", re.S),  # fenced code; a fence left open hides the rest
    re.compile(r"`[^`\n]*`"),  # inline code
    re.compile(r"\b(?:https?|ftp|file)://\S+|\bwww\.\S+|\b[\w.+-]+@[\w-]+\.[\w.]+"),  # URLs, e-mail addresses
    re.compile(r"<[^>\n]{1,200}>"),  # HTML and XML tags
    re.compile(r"^[ \t]*>.*$", re.M),  # block-quote lines
    re.compile(  # quoted spans within a line
        r"\"[^\"\n]{1,400}\"|“[^”\n]{1,400}”|„[^“”\n]{1,400}[“”]|«[^»\n]{1,400}»|「[^」\n]{1,400}」|『[^』\n]{1,400}』"
        r"|‘[^’\n]{1,400}’"
    ),
)


@lru_cache(maxsize=65_536)
def script_of(char: str) -> str | None:
    """A letter's script (a `SCRIPTS` value), or None for a non-letter or a letter of none of them."""
    if not char.isalpha():
        return None
    words = unicodedata.name(char, "").split()
    if words[:1] in (["FULLWIDTH"], ["HALFWIDTH"]):
        words = words[1:]
    return SCRIPTS.get(words[0]) if words else None


def letter_counts(text: str) -> Counter[str]:
    """Letters per script."""
    counts: Counter[str] = Counter()
    for char, n in Counter(text).items():
        script = script_of(char)
        if script is not None:
            counts[script] += n
    return counts


def scripts_in(text: str) -> set[str]:
    return {script for script in map(script_of, set(text)) if script is not None}


def _longest_run(text: str, script: str) -> int:
    """The most letters of the script in a row (a combining mark continues the run)."""
    longest = run = 0
    for char in text:
        if script_of(char) == script:
            run += 1
            longest = max(longest, run)
        elif not unicodedata.category(char).startswith("M"):
            run = 0
    return longest


def _mostly_own_lines(text: str, script: str) -> bool:
    """Whether at least half the script's letters sit on lines where its family holds the majority of the letters."""
    family = FAMILIES.get(script, script)
    own = total = 0
    for line in text.split("\n"):
        counts = letter_counts(line)
        if not counts[script]:
            continue
        total += counts[script]
        in_family = sum(n for s, n in counts.items() if FAMILIES.get(s, s) == family)
        if 2 * in_family > sum(counts.values()):
            own += counts[script]
    return 2 * own >= total


def foreign_letters(reply: str, input_scripts: Collection[str]) -> int:
    """The reply's letters that count as foreign to an input written in `input_scripts` (rule steps 2 to 4)."""
    for pattern in _REMOVED:
        reply = pattern.sub(" ", reply)
    known = {"Latin", *input_scripts}
    if "Kana" in known:
        known.add("Han")
    total = 0
    for script, n in letter_counts(reply).items():
        if n < MIN_LETTERS or script in known:
            continue
        if script == "Greek" and _longest_run(reply, script) < 3:
            continue
        if not _mostly_own_lines(reply, script):
            total += n
    return total


def _lines(value: Any) -> list[str]:
    """The strings in a JSON value, in order: each object's own strings on one line, joined by " - "."""
    lines: list[str] = []
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            lines.append(node)
        elif isinstance(node, dict):
            if own := [v for v in node.values() if isinstance(v, str)]:
                lines.append(" - ".join(own))
            stack.extend(reversed([v for v in node.values() if isinstance(v, (dict, list))]))
        elif isinstance(node, list):
            stack.extend(reversed(node))
    return lines


def final_reply(turn: Items, stop_tools: Collection[str]) -> str:
    """What the user reads of the turn's last model reply (the items between the last two runs of tool results):
    its assistant messages and the strings in its stop-tool calls' arguments."""
    end = len(turn)
    while end and turn[end - 1].get("type") == "function_call_output":
        end -= 1  # the results of calls made beside a stop tool
    start = end
    while start and turn[start - 1].get("type") != "function_call_output":
        start -= 1
    parts: list[str] = []
    for item in turn[start:end]:
        if item.get("type") == "message" and item.get("role") == "assistant":
            parts.append(content_text(item.get("content")))
        elif item.get("type") == "function_call" and item.get("name") in stop_tools:
            parts.append("\n".join(_lines(json_object(item.get("arguments") or "{}"))))
    return "\n\n".join(part.strip() for part in parts if part.strip())


def foreign_script_letters(
    turn: Items, *, history: Items, stop_tools: Collection[str], instructions: str | None = None
) -> int:
    """Letters in the turn's final reply of scripts its input never shows (the module docstring has the rule).
    `turn` holds tool items (Cline's results as function_call_output items); a user message in it counts as input."""
    results = [item for item in turn if item.get("type") == "function_call_output" or item.get("role") == "user"]
    inputs = [instructions or ""] + [
        json.dumps(item, ensure_ascii=False, default=str)
        for item in [*history, *results]
        if item.get("type") != "reasoning"
    ]
    return foreign_letters(final_reply(turn, stop_tools), set().union(*map(scripts_in, inputs)))
