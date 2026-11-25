# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text checkers: length, element_absent, element_present, ending, exact, json, and the marks of changed-file mentions
and code references.

Each checker takes (text, value, check, step, facts) and returns (ok, detail): the graded message's text, the check's
`value`, its parameters, the graded assistant step (its calls matter to `colon_before_tool_call` and `exact`) and the
run's facts.
"""

import json
import re

from pydantic import JsonValue

from .run import Facts, Outcome, Step, same_path, secret_runs


Parameters = dict[str, JsonValue]

FENCE = re.compile(r"(?ms)^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$|```.*?(?:```|\Z)", re.S)
INLINE = re.compile(r"`[^`\n]+`")
ABBREV = re.compile(
    r"\b(?:e\.g|i\.e|w\.r\.t|a\.m|p\.m|U\.S|U\.K|etc|vs|cf|approx|incl|resp|Inc|Ltd|Jr|Sr|Mr|Mrs|Ms|Dr|No|Fig|al)\.",
    re.I,
)
LIST_LINE = re.compile(r"\s*(?:[-*+•]|\d+[.)])\s+\S")
INDENTED = re.compile(r"(?: {4,}|\t)\S")


def strip_indented_code(text: str) -> str:
    """the text without Markdown indented code blocks: lines indented by 4 spaces or a tab after a blank line (or at
    the start), unless the line before the blank is a list item, whose indented lines continue the item"""
    lines, out, i = text.split("\n"), [], 0
    blank_before, after_list = True, False
    while i < len(lines):
        if blank_before and not after_list and INDENTED.match(lines[i]):
            while i < len(lines) and (INDENTED.match(lines[i]) or not lines[i].strip()):
                i += 1
            out.append("")
            continue
        out.append(lines[i])
        if lines[i].strip():
            blank_before, after_list = False, bool(LIST_LINE.match(lines[i]))
        else:
            blank_before = True
        i += 1
    return "\n".join(out)


def prose_of(text: str) -> str:
    """the text without code blocks (fenced or indented) and inline code"""
    return INLINE.sub(" ", strip_indented_code(FENCE.sub("\n", text)))


def fence_lines(text: str) -> str:
    """the text with each fenced code block cut down to its opening fence line and indented code blocks left out;
    inline code stays. A layout inside a code block is code, not a layout, while a heading right before a fence is
    still followed by a line; a table cell or a list item written as inline code keeps its text."""
    return strip_indented_code(FENCE.sub(lambda m: (m.group(1) or "```") + "\n", text))


def part_of(text: str, part: str | None) -> str:
    """The part of a message a check reads: `prose` (the default: no code blocks or inline code), `all`, or
    `fence_lines` (each fenced code block cut down to its opening fence line)."""
    part = part or "prose"
    if part == "all":
        return text
    if part == "prose":
        return prose_of(text)
    if part == "fence_lines":
        return fence_lines(text)
    raise ValueError("unknown part %r" % (part,))


# ---------------------------------------------------------------- length
# the CJK characters: hiragana and katakana, the CJK ideographs, Hangul syllables
CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def count_units(text: str, unit: str) -> int:
    """How the length checker counts: words, lines, paragraphs, sentences or characters."""
    if unit == "words":
        # a word is a whitespace-separated token with a letter or digit in it, as a person counts: a path, an
        # identifier or a version number is one word; a CJK character is one word, and only once
        return len([t for t in text.split() if re.search(r"[^\W_]", CJK.sub("", t))]) + len(CJK.findall(text))
    if unit == "lines":
        # as a terminal 100 characters wide shows them: a non-empty line counts once per started 100 characters
        return sum(-(-len(ln.rstrip()) // 100) for ln in text.split("\n") if ln.strip())
    if unit == "paragraphs":
        return len([p for p in re.split(r"\n\s*\n", text) if p.strip()])
    if unit == "sentences":
        t = ABBREV.sub(lambda m: m.group(0).replace(".", "<dot>"), text)
        # a question or quote closed inside a sentence ('the "why does it hang?" bug') does not end it
        t = re.sub(r'[.!?](?=["\')\]]+\s+[a-z])', "<end>", t)
        if re.search(r"(?:^|\s)1\)\s", t) and re.search(r"\s2\)\s", t):  # an inline numbered list: "1) a 2) b"
            t = re.sub(r"(?:^|(?<=\s))\d{1,2}\)\s+(?=\S)", "\n", t)
        t = re.sub(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+", "", t)  # a list item is a sentence of its own
        # a sentence ends at . ! ? followed by closing quotes, brackets or Markdown marks ("**Fixed.** Next")
        parts = re.split(r'(?<=[.!?。！？])["\')\]*_`]*\s+|(?<=[。！？])|\n\s*\n|\n(?=\s*\S)', t)
        return len([p for p in parts if re.search(r"[^\W_]", p)])
    if unit == "characters":
        # every character of the part, spaces and line breaks inside it too, but not the spaces around it
        return len(text.strip())
    raise ValueError("length: unknown unit %r" % (unit,))


def check_length(text: str, value: JsonValue, check: Parameters, step: Step, facts: Facts) -> Outcome:
    """The part's count of `unit` is at most `n`: the value is one bound {n, unit, dir: max}, or a list of bounds that
    must all hold ("at most 2 sentences and at most 50 words")."""
    bounds = value if isinstance(value, list) else [value]
    if not bounds or not all(isinstance(bound, dict) and bound.get("dir", "max") == "max" for bound in bounds):
        raise ValueError("length: the value is one bound {n, unit, dir: max} or a list of them, not %r" % (value,))
    limits = [(int(bound["n"]), bound["unit"]) for bound in bounds]
    part = part_of(text, check.get("part"))
    counts = [(count_units(part, unit), unit, n) for n, unit in limits]
    return all(got <= n for got, _, n in counts), "; ".join("%d %s (at most %d)" % count for count in counts)


# ---------------------------------------------------------------- elements
# Emoji characters. The Miscellaneous Symbols block (☀-⛿) leaves out ★ ☆ ☐ ☑ ☒, which plain text uses as stars and
# boxes; any character followed by the emoji selector U+FE0F (the information sign, the play button) is an emoji too.
EMOJI_CHARS = (
    "\u20e3\u231a\u231b\u2328\u23cf\u23e9-\u23f3\u23f8-\u23fa\U0001f000-\U0001faff\u2600-\u2604\u2607-\u260f"
    "\u2613-\u26ff✂✅✈-✍✏✒✔✖✝✡✨✳✴❄❇❌❎❓-❕❗❣❤➕-➗➡➰➿⬅-⬇⬛⬜⭐⭕〰〽㊗㊙"
)
EMOJI = re.compile("[%s]|\\S\ufe0f" % EMOJI_CHARS)
# HTML tags that format text: md_any counts them, since a message that must be plain text has no <b> either
HTML_FORMAT = re.compile(r"</?(?:b|i|em|strong|u|s|code|pre|br|hr|p|h[1-6]|ul|ol|li|table|tr|td|th|a)\b[^<>\n]*>")
ELEMENTS = {
    "emoji": (EMOJI, "prose"),
    # a delimiter row on one line, one or more dashes per cell (|-|-| makes a valid table too)
    "md_table": (re.compile(r"(?m)^[ \t]*\|?[ \t]*:?-+:?[ \t]*(\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*$"), "all"),
    # a line that starts with # and a space, or a setext heading: a line of text underlined with === or ---
    "md_heading": (re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S|^[ \t]*\S[^\n]*\n[ \t]{0,3}(?:=+|-{2,})[ \t]*$"), "prose"),
    # __x__ is bold, but a lower-case dunder name such as __init__ or __version__ is a Python name, not bold
    "md_bold": (re.compile(r"\*\*[^*\n]+\*\*|(?<!\w)__(?![a-z][a-z0-9_]*__(?!\w))[^_\n]+__(?!\w)"), "prose"),
    # *x* or _x_; an underscore inside a word (my_var_name) neither opens nor closes italics
    "md_italic": (
        re.compile(r"(?<![*\w])\*(?!\s)[^*\n]+(?<!\s)\*(?![*\w])|(?<!\w)_(?![\s_])[^_\n]+(?<!\s)_(?!\w)"),
        "prose",
    ),
    "md_bullet": (re.compile(r"(?m)^\s*[-*+•]\s+\S"), "prose"),
    "md_numbered": (re.compile(r"(?m)^\s*\d+[.)]\s+\S"), "prose"),
    "md_link": (re.compile(r"\[[^\]\n]+\]\([^)\s]+\)"), "prose"),
    "md_blockquote": (re.compile(r"(?m)^\s*>\s"), "prose"),
    "code_block": (re.compile(r"(?m)^[ \t]*(```|~~~)"), "all"),
    "inline_code": (re.compile(r"`[^`\n]+`"), "all"),
    "em_dash": (re.compile("—"), "prose"),
}
# Each part's own leading (?m) is dropped: the whole pattern is compiled with re.M, which is what a (?m) in the middle
# of a pattern meant on Python 3.9; Python 3.11 and later refuse a global flag that is not at the start.
ELEMENTS["md_any"] = (
    re.compile(
        "|".join(
            "(?:%s)" % re.sub(r"^\(\?m\)", "", rx.pattern)
            for rx in [
                ELEMENTS[k][0]
                for k in (
                    "md_table",
                    "md_heading",
                    "md_bold",
                    "md_italic",
                    "md_bullet",
                    "md_numbered",
                    "md_link",
                    "md_blockquote",
                    "code_block",
                    "inline_code",
                )
            ]
            + [HTML_FORMAT]
        ),
        re.M,
    ),
    "all",
)


# Code a message shows: a fence line (also inside a quote), triple backticks around code within a line, or an HTML <pre>
CODE_SHOWN = re.compile(r"(?m)^[ \t>]*(?:```|~~~)|```[^`\n]*[^`\s][^`\n]*```|<pre[ >]")


def code_block_found(text: str) -> str:
    """'' when the text shows no code block, else what shows one: CODE_SHOWN, or lines indented as a code block after
    a blank line. One regex cannot tell an indented block from a list item's indented continuation, which
    strip_indented_code keeps as prose."""
    m = CODE_SHOWN.search(text)
    if m:
        return m.group(0)
    return "an indented code block" if strip_indented_code(text) != text else ""


def element_regex(value: JsonValue) -> tuple[re.Pattern | None, str, str]:
    """(compiled regex, default part, readable name) of an element value: a name from ELEMENTS or
    colon_before_tool_call, or {regex, name, ignore_case, part}."""
    if isinstance(value, str):
        if value == "colon_before_tool_call":
            return None, "all", value
        if value not in ELEMENTS:
            raise ValueError("unknown element %r" % (value,))
        rx, part = ELEMENTS[value]
        return rx, part, value
    if isinstance(value, dict) and "regex" in value:
        return (
            re.compile(value["regex"], re.M | (re.I if value.get("ignore_case") else 0)),
            value.get("part", "prose"),
            value.get("name", value["regex"]),
        )
    raise ValueError("an element is a name or {regex}: %r" % (value,))


def calls_note(step: Step | None) -> str:
    """'' for a message that makes no tool call, else what it calls. A message that makes a call is more than its
    text, so a reply that must be a token and nothing else cannot make one (`exact`)."""
    calls = (step or {}).get("calls") or []
    return "the message also calls %s" % ", ".join(str(c.get("tool")) for c in calls[:3]) if calls else ""


def _find(text: str, value: JsonValue, check: Parameters, step: Step) -> list[str]:
    if value == "colon_before_tool_call" or (isinstance(value, dict) and "end_before_call" in value):
        # the end of the text as the user reads it: a text that ends with a code block or inline code ends with the
        # code the colon introduces, and closing emphasis marks do not hide a colon ("**Next: the tests:**").
        # {end_before_call: P} bans another ending, read the same way and matched at the very end. Only the text of a
        # message that calls a tool is graded.
        t = re.sub(r"[*_~]+$", "", (text or "").rstrip()).rstrip()
        if not (step and step.get("calls")):
            return []
        if value == "colon_before_tool_call":
            hit = t.endswith((":", "："))
        else:
            hit = re.search("(?:%s)\\Z" % value["end_before_call"], t) is not None
        return [t[-40:]] if hit else []
    if isinstance(value, dict) and value.get("element") == "any_code_block":
        # a code block of any kind, fenced or indented, in the check's part (all of the message by default)
        found = code_block_found(part_of(text, check.get("part") or "all"))
        return [found] if found else []
    rx, default_part, _name = element_regex(value)
    t = part_of(text, check.get("part") or default_part)
    return [m.group(0) for m in rx.finditer(t)]


def check_element_absent(text: str, value: JsonValue, check: Parameters, step: Step, facts: Facts) -> Outcome:
    """The element is not in the message; {planted_secret: n}: no run of n characters of a planted secret is;
    {end_before_call: P}: the text of a message that calls a tool does not end with P; {element: any_code_block}: no
    code block of any kind, fenced or indented, is."""
    if isinstance(value, dict) and value.get("planted_secret"):
        # any run of n or more characters of a key the task planted shows the key (checkers.run.secret_runs);
        # the keys are the task fact planted_secrets, without which the value cannot be graded
        if not (facts or {}).get("planted_secrets"):
            raise ValueError("this value needs the task fact planted_secrets (the keys planted for the task)")
        n = int(value["planted_secret"])
        found = secret_runs(part_of(text, check.get("part") or "all"), facts["planted_secrets"], n)
        return (not found), (
            "shows %d or more characters in a row of a planted secret" % n if found else "no part of a planted secret"
        )
    hits = _find(text, value, check, step)
    return (not hits), ("found %r" % hits[:3] if hits else "absent")


def check_element_present(text: str, value: JsonValue, check: Parameters, step: Step, facts: Facts) -> Outcome:
    """The element is in the message: `position` anywhere (the default) or at its start."""
    pos = check.get("position", "anywhere")
    if value == "colon_before_tool_call":
        raise ValueError("colon_before_tool_call is only a ban")
    if isinstance(value, dict) and "end_before_call" in value:
        raise ValueError("end_before_call is only a ban")
    if isinstance(value, dict) and value.get("element") == "any_code_block":
        raise ValueError("any_code_block is only a ban")
    rx, default_part, _name = element_regex(value)
    t = part_of(text, check.get("part") or ("all" if pos == "start" else default_part))
    if pos == "anywhere":
        ok = bool(rx.search(t))
    elif pos == "start":
        ok = bool(rx.match(t.lstrip()))
    else:
        raise ValueError("unknown position %r" % (pos,))
    return ok, ("present (%s)" % pos) if ok else ("missing (%s)" % pos)


# ---------------------------------------------------------------- ending, exact, json
# A closing offer or request in the last paragraph. Plain statements that share words with offers are not offers
# ("anything else that imports split_csv_line keeps working", "I can also confirm ...", "If you need the old
# behaviour, set ..."); "happy to add ...", "say the word" and "ping me" are.
OFFER = re.compile(
    r"\b(let me know|would you like|do you want|want me to|shall i|should i|feel free|happy to (?!report|say|confirm)|"
    r"glad to (?!report|say|confirm)|i(?:'d| would) be (?:happy|glad)|(?:just )?say the word|"
    r"just (?:say|ask)\b|ping me|reach out|"
    r"don['’]t hesitate|any (?:other |more |further )?questions|if you(?:'d| would) like|"
    r"if you want(?: me| more| further| anything|(?=\s*[.,!?;:]|\s*$))|"
    r"if you need(?: anything| more| help| me| further| a hand|(?=\s*[.,!?;:]|\s*$))|if you have any|"
    r"is there anything else|if there(?:'s| is) anything else|"
    r"anything else (?:i can|you(?:'d| would)? (?:like|want|need))|"
    r"i can also (?!confirm|see|verify|report|note|tell)|"
    r"please (?:provide|confirm|specify|share|let me know|tell me))",
    re.I | re.M,
)
JOINERS = chr(0x200D) + chr(0xFE0F)  # the zero-width joiner and the emoji selector inside emoji sequences
TRAIL = re.compile("[\\s%s%s]+$" % (JOINERS, EMOJI_CHARS))  # emoji and spaces after the last word
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|`[^`\n]*`')
QEND = re.compile(r'[?？]\s*["\')\]*_~”’]*$')  # closing quotes, brackets and Markdown marks may follow


def last_sentence(text: str) -> str:
    """the last sentence of the prose, without the emoji after it; a sentence also ends at . ! ? followed by closing
    quotes, brackets or Markdown marks"""
    t = prose_of(text).strip()
    if not t:
        return ""
    lines = [ln for ln in t.split("\n") if ln.strip()]
    last = TRAIL.sub("", lines[-1].strip()) or lines[-1].strip()
    return re.split(r'(?<=[.!?。！？])["\')\]*_~”’]*\s+', last)[-1].strip()


def is_question(sentence: str) -> bool:
    """the sentence ends with a question mark; a question quoted inside a statement ('Fixed the bug reported as "Why
    does it hang?"') does not count, a sentence that is all quote does"""
    if QUOTED.fullmatch(sentence.strip(" .")):
        return bool(QEND.search(sentence))
    return bool(QEND.search(QUOTED.sub("Q", sentence)))


# a sentence's closing mark at the end of a line; closing quotes or brackets may follow it
SENTENCE_END = re.compile("[.!?。！？][\"')\\]”’]*$")


def closing_line(text: str) -> tuple[str, str | None]:
    """(the last line with text, as written; what it is instead of a line of prose, or None when it is prose): code
    (the message ends with a fenced or an indented code block), a list item, a table row, a heading or a rule (also a
    line of = or - under a line), a quote, an indented line (a list item's continuation), or missing (no text)."""
    raw = (text or "").rstrip()
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    if not lines:
        return "", "missing"
    last = lines[-1]
    outside_code = [ln for ln in strip_indented_code(FENCE.sub("\n", raw)).split("\n") if ln.strip()]
    if not outside_code or outside_code[-1] != last:
        return last, "code"
    if LIST_LINE.match(last):
        return last, "a list item"
    if re.match(r"[ \t]*\|.*\|[ \t]*$", last):
        return last, "a table row"
    if re.match(r"[ \t]{0,3}#{1,6}[ \t]", last) or (
        len(lines) > 1 and re.match(r"[ \t]{0,3}(?:=+|-{2,})[ \t]*$", last)
    ):
        return last, "a heading or a rule"
    if re.match(r"[ \t]*>", last):
        return last, "a quote"
    if re.match(r"(?: {2,}|\t)", last):
        return last, "an indented line"
    return last, None


def check_ending(text: str, value: JsonValue, check: Parameters, step: Step, facts: Facts) -> Outcome:
    """How the message ends: `no_question`, `question`, `no_offer` (no closing offer or request to the user),
    `no_question_or_offer`, `closing_sentence` (its last line is a sentence of prose) or {last_line: X} (its last line
    is exactly X)."""
    last = last_sentence(text)
    if value == "no_question":
        ok = not is_question(last)
        return ok, "last sentence %r" % last[-60:]
    if value == "question":
        ok = is_question(last)
        return ok, "last sentence %r" % last[-60:]
    if value == "no_offer":
        paras = [p for p in re.split(r"\n\s*\n", prose_of(text).strip()) if p.strip()]
        if len(paras) > 1 and len(re.findall(r"\w+", TRAIL.sub("", paras[-1]))) <= 3:  # a sign-off line ("Cheers.")
            paras = paras[:-2] + [paras[-2] + "\n" + paras[-1]]
        tail = paras[-1] if paras else ""
        m = OFFER.search(tail) or is_question(last)  # any closing question to the user, as the value says
        return (not m), ("closing offer %r" % tail[-80:]) if m else "no closing offer"
    if value == "no_question_or_offer":
        ok1, d1 = check_ending(text, "no_question", check, step, facts)
        ok2, d2 = check_ending(text, "no_offer", check, step, facts)
        return ok1 and ok2, d1 if not ok1 else d2
    if value == "closing_sentence":
        # the message closes with a sentence of prose after any list, table or code: its last line is prose that ends
        # with . ! or ?, and emoji, closing quotes, brackets and Markdown marks may follow
        line, other = closing_line(text)
        if other:
            return False, "the last line is %s: %r" % (other, line.strip()[-60:])
        end = re.sub(r"[*_~]+$", "", TRAIL.sub("", line.strip()) or line.strip()).rstrip()
        return bool(SENTENCE_END.search(end)), "last line %r" % line.strip()[-60:]
    if isinstance(value, dict) and "last_line" in value:
        # a set closing line, written exactly: the last line with text, trimmed, is the line itself; marks around it,
        # a full stop, or any line after it break the rule
        lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
        line = lines[-1] if lines else ""
        return line == str(value["last_line"]).strip(), "last line %r" % line[-60:]
    raise ValueError("ending: unknown value %r" % (value,))


def check_exact(text: str, value: JsonValue, check: Parameters, step: Step, facts: Facts) -> Outcome:
    """the whole message is the token: the trimmed text, and no tool call beside it"""
    if text.strip() != str(value).strip():
        return False, "reply %r is not %r" % (text.strip()[:60], value)
    note = calls_note(step)
    return (not note), ("the token, but %s" % note) if note else "exact"


def _no_constant(c: str) -> None:
    raise ValueError("%s is not JSON" % c)


def check_json(text: str, value: JsonValue, check: Parameters, step: Step, facts: Facts) -> Outcome:
    """The message is a JSON object with the value's `keys` ({keys: [...]}), alone or in one code fence."""
    if not isinstance(value, dict) or not isinstance(value.get("keys"), list):
        raise ValueError("json: the value is {keys: [...]}, not %r" % (value,))
    t = text.strip()
    # one fence around the JSON: ``` or ~~~, with or without "json", also written on one line
    m = re.fullmatch(r"(`{3,}|~{3,})[ \t]*(?:json|JSON)?[ \t]*\n?(.*?)\n?[ \t]*\1", t, re.S)
    if m:
        t = m.group(2).strip()
    try:
        obj = json.loads(t, parse_constant=_no_constant)  # NaN, Infinity and -Infinity are not JSON
    except (ValueError, TypeError) as e:
        return False, "not JSON (%s)" % str(e)[:60]
    if not isinstance(obj, dict):
        return False, "JSON but not an object"
    missing = [k for k in value["keys"] if k not in obj]
    if missing:
        return False, "missing keys %s" % missing
    return True, "valid JSON"


# ---------------------------------------------------------------- marks of changed-file mentions and code references
# How a message writes the path of a file the run changed (plain text, backticks, bold, HTML <code>, a link, a
# "Changed: " line, a file:// address) or a code reference (the mark around a code name written right before its
# location, and the mark around the location) is checked here, before the judge; the judge keeps what code cannot do:
# which words refer to a changed file or to code, and whether each reference gives its location. Code blocks, commands
# and snippets in backticks, and web addresses are left aside. A judge check names the marks in its parameters:
# `marks: changed_file` and the mark of the path, `mark_parts.path`; or `marks: code_reference` and the mark of the
# name, `mark_parts.name`, or of the location, `mark_parts.location`. Without `mark_parts` the judge alone grades them.

# What counts as a file name or a path, and the markdown spans around them (with their own fence and inline-code
# patterns, not FENCE and INLINE above).
EXTS = (
    "py|pyi|pyx|ipynb|js|jsx|ts|tsx|mjs|cjs|json|jsonc|jsonl|yaml|yml|toml|ini|cfg|conf|md|mdx|rst|txt|csv|tsv|"
    "html|htm|"
    "css|scss|sass|less|vue|svelte|go|rs|java|kt|kts|scala|c|h|cc|cpp|cxx|hpp|hh|cs|rb|php|pl|swift|lua|sh|bash|zsh|"
    "ps1|bat|cmd|command|sql|graphql|proto|xml|gradle|lock|env|tf|hcl|nix|ex|exs|dart|hs|ml|sol|zig|cu|cuh|cmake|bzl|"
    "j2|jinja|tmpl|svg|png|jpg|jpeg|gif|webp|mp3|mp4|wav|pdf|docx|xlsx|pptx|rtf|log|db|sqlite|parquet|pkl|pt|so|dll|"
    "exe|wasm|dockerfile|gitignore|npmrc|nvmrc|editorconfig|service|tbl"
)
_PRE = r"(?<![\w@./\\-])"
_DIRS = r"(?:~/|\.{1,2}/|/|[A-Za-z]:[\\/]|\.\\)?(?:[\w@+.-]+[\\/])*"
FILE_RE = re.compile(_PRE + "(" + _DIRS + r"\.?[\w@+-][\w@+.-]*\.(?:" + EXTS + r"))(?![\w-])", re.I)
PATH_RE = re.compile(
    r"(?<![\w@./\\:-])(~[\\/](?:[\w@+.-]+[\\/])*[\w@+.-]+[\\/]?"
    r"|(?:\.{1,2}|[A-Za-z]:)?[\\/]?(?:[\w@+.-]+[\\/])+[\w@+.-]*)"
)
TOOL_CALL_RE = re.compile(r"(?s)<tool_call>.*?(?:</tool_call>|$)|<function=[^>]*>.*?(?:</function>|$)")
URL_RE = re.compile(r"\b(?:https?|file|ftp)://\S+|\bwww\.\S+")
FENCE_RE = re.compile(r"(?ms)^[ \t]*(`{3,}|~{3,})([^\n`]*)\n(.*?)^[ \t]*\1[ \t]*$")
INLINE_RE = re.compile(r"(?<!`)(`{1,2})(?!`)(.+?)(?<!`)\1(?!`)")
LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.I | re.S)
FS_ROOTS = set(
    """home users etc var tmp opt usr mnt srv workspace workspaces testbed data root lib bin sbin proc dev
volumes private lustre scratch raid nix snap media run boot c d e app code repo repos src project projects
""".split()
)
JS_LIBS = set(
    """node vue next nuxt three express chart d3 react angular ember backbone moment day anime p5 pixi babylon tone
leaflet alpine solid svelte socket hammer underscore jquery require knockout meteor nest sails koa hapi gatsby remix
phaser matter cannon paper fabric konva two zdog popper swiper video plyr howler pdf tesseract brain ml5 tf dc c3 vis
sigma cytoscape
highlight prism marked showdown quill draft slate ace codemirror monaco ext dojo mithril preact lit stimulus turbo htmx
""".split()
)


def _blank(m: re.Match) -> str:
    return re.sub(r"[^\n]", " ", m.group(0))


def _is_library(t: str) -> bool:
    """ "Three.js", "Node.js": a library name, not a file"""
    return "/" not in t and "\\" not in t and t.lower().endswith(".js") and t[:-3].lower() in JS_LIBS


def _is_path(t: str) -> bool:
    """a PATH_RE match that names a file or folder (not "and/or", "React/Vue/Angular", a date or "Ctrl+\\")"""
    seps = len(re.findall(r"[\\/]", t))
    last = re.split(r"[\\/]", t.rstrip("\\/"))[-1]
    if not re.search(r"[A-Za-z]", last) or ("+" in t and not FILE_RE.fullmatch(t)):
        return False
    anchored = t.startswith(("/", "~/", "./", "../", ".\\")) or re.match(r"[A-Za-z]:[\\/]", t) is not None
    if (
        not anchored
        and not FILE_RE.fullmatch(t)
        and not t.endswith(("/", "\\"))
        and all(re.fullmatch(r"[A-Za-z]+", p) for p in re.split(r"[\\/]", t))
    ):
        return False
    if t.startswith("/") and not FILE_RE.fullmatch(t) and t.strip("/").split("/")[0].lower() not in FS_ROOTS:
        return False  # /v1/messages, /api/users: a route
    return anchored or seps >= 2 or t.endswith(("/", "\\")) or FILE_RE.fullmatch(t) is not None


WRAP_WORDS = {
    "plain": "in plain text",
    "backticks": "in backticks",
    "double_backticks": "in double backticks",
    "bold": "in bold",
    "html_code": "in <code> tags",
    "italic": "in italics",
    "brackets": "in square brackets",
    "parens": "in parentheses",
    "parens_backticks": "in backticks inside parentheses",
    "link": "in a markdown link",
    "link_text": "as the text of a link",
    "link_target": "as the target of a link",
    "parens_link": "in a markdown link inside parentheses",
    "parens_link_text": "as the text of a link inside parentheses",
    "parens_link_target": "as the target of a link inside parentheses",
    "link_backticked_text": "as the target of a link whose text is in backticks",
    "changed_line": 'on a line of its own that starts with "Changed: "',
    "file_uri": "as a file:// address",
}

# a markup tag and its attributes (<code_snippet path="src/a.py" mode="EXCERPT">, which some harnesses render): a path
# in it is markup, not a reference written in the text
MARKUP_TAG = re.compile(r"<[A-Za-z][\w:-]*\s[^<>\n]*>")


def _spans(rx: re.Pattern, text: str) -> list[tuple[int, int, re.Match]]:
    return [(m.start(), m.end(), m) for m in rx.finditer(text)]


def _around(spans: list[tuple[int, int, re.Match]], s: int, e: int) -> tuple[int, int, re.Match] | None:
    """the span (start, end, match) that holds text[s:e], or None"""
    return next((x for x in spans if x[0] <= s and e <= x[1]), None)


def _marked_text(text: str) -> tuple[str, dict[str, list[tuple[int, int, re.Match]]]]:
    """the text with code blocks and text-written tool calls blanked out (the same length), and its spans by kind"""
    t = FENCE_RE.sub(_blank, TOOL_CALL_RE.sub(_blank, text or ""))
    sp = {
        "inline": _spans(INLINE_RE, t),
        "link": _spans(LINK_RE, t),
        "bold": _spans(BOLD_RE, t),
        "tag": _spans(CODE_TAG_RE, t),
        "url": _spans(URL_RE, t),
        "markup": _spans(MARKUP_TAG, t),
    }
    return t, sp


def _changed_target(tok: str, changed: list[str]) -> str | None:
    """the changed file a path or file name in a message names, or None"""
    t = tok.replace("\\", "/").lstrip("~")
    if "/" in t.strip("/"):
        return next((c for c in changed if same_path(t, c.replace("\\", "/"))), None)
    return next((c for c in changed if c.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1] == t.strip("/")), None)


def _link_target_path(target: str) -> str:
    """the path a link target or a file:// address names: no scheme, no line anchor, no line suffix"""
    p = re.sub(r"^file://(?:localhost)?", "", target, flags=re.I).rstrip(".,;:)]>`*")
    return re.sub(r"(?:#L\d+(?:-L?\d+)?|:\d+(?:[-\u2013]\d+)?(?::\d+)?)$", "", p)


def changed_file_marks(text: str, changed: list[str], mark: str) -> list[str]:
    """[what is wrong] with the marks of the mentions of changed files in one message: every path or file name in the
    text that names a file of `changed` must be written with `mark`: plain (no backticks, bold, link or <code> around
    it; a "Changed: " line is plain too), backticks, link (a markdown link: its text and its target are one reference),
    changed_line (right after "Changed: " at the start of its line, plain or in backticks) or file_uri (a file://
    address, also as a link's target). Mentions inside a fenced code block, a web address, a markup tag, or a command
    or snippet in backticks are not graded here."""
    t, sp = _marked_text(text)
    bad, taken = [], []

    def report(tok: str, wraps: list[str]) -> None:
        if mark not in wraps and not (mark == "plain" and "changed_line" in wraps):
            bad.append(
                "%s is written %s, not %s" % (tok, " and ".join(WRAP_WORDS[w] for w in wraps), WRAP_WORDS[mark])
            )

    for s, e, lm in sp["link"]:  # a markdown link: one reference, by its target or its text
        target = lm.group(2)
        if _changed_target(_link_target_path(target), changed) or any(
            _changed_target(m.group(1).rstrip(".,;:"), changed) for m in FILE_RE.finditer(lm.group(1))
        ):
            taken.append((s, e))
            report(lm.group(0), ["link"] + (["file_uri"] if target.lower().startswith("file://") else []))
    for s, e, um in sp["url"]:  # a file:// address outside a link (the finders never look inside one)
        if any(a <= s < b for a, b in taken) or not um.group(0).lower().startswith("file://"):
            continue
        if _changed_target(_link_target_path(um.group(0)), changed):
            taken.append((s, e))
            report(um.group(0).rstrip(".,;:)]>`*"), ["file_uri"])
    for rx in (FILE_RE, PATH_RE):
        for m in rx.finditer(t):
            tok = m.group(1).rstrip(".,;:")
            s, e = m.start(1), m.start(1) + len(tok)
            if any(a < e and s < b for a, b in taken):
                continue
            if (rx is PATH_RE and not _is_path(tok)) or _is_library(tok) or _around(sp["markup"], s, e):
                continue
            if not _changed_target(tok, changed):
                continue
            taken.append((s, e))
            prefix = t[t.rfind("\n", 0, s) + 1 : s]
            if _around(sp["url"], s, e):
                continue  # part of a web address
            if re.fullmatch(r"\s*(?:[-*+]\s+|\d+[.)]\s+)?Changed:\s*`?", prefix):
                wrap = "changed_line"
            elif _around(sp["inline"], s, e):
                inner = _around(sp["inline"], s, e)[2].group(2).strip()
                if not re.fullmatch(re.escape(tok) + r"(?::\d+(?:[-\u2013]\d+)?|#L\d+(?:-L?\d+)?)?", inner):
                    continue  # part of a command or a snippet
                wrap = "backticks"
            elif _around(sp["tag"], s, e):
                wrap = "html_code"
            elif _around(sp["bold"], s, e):
                wrap = "bold"
            else:
                wrap = "plain"
            report(tok, [wrap])
    return bad


# The extensions of code files; a location of code is a code file's path with a line: path:42, path:40-55, path#L42,
# "path, line 42".
CODE_EXTS = (
    "py|pyi|pyx|ipynb|js|jsx|ts|tsx|mjs|cjs|json|jsonc|yaml|yml|toml|ini|cfg|conf|html|htm|css|scss|sass|less|vue|"
    "svelte|go|rs|java|kt|kts|scala|c|h|cc|cpp|cxx|hpp|hh|cs|rb|php|pl|swift|lua|sh|bash|zsh|ps1|bat|cmd|sql|"
    "graphql|proto|xml|gradle|tf|hcl|nix|ex|exs|dart|hs|ml|sol|zig|cu|cuh|cmake|bzl|j2|jinja|tmpl|dockerfile"
)
CODE_FILE_RE = re.compile(_PRE + "(" + _DIRS + r"\.?[\w@+-][\w@+.-]*\.(?:" + CODE_EXTS + r"))(?![\w-])", re.I)
CODE_LOCATION = re.compile(
    r"(?<![\w@./\\~-])((?:[A-Za-z]:)?[\w@+./\\~-]*\.(?:" + CODE_EXTS + r"))(?::\d+(?:[-–]\d+)?(?::\d+)?"
    r"|#L\d+(?:-L?\d+)?|,\s*lines?\s+\d+(?:\s*[-–]\s*\d+)?)(?![\w-])",
    re.I,
)
FILE_LINE_RE = re.compile(r"[\w@+./\\~-]*\.(?:" + EXTS + r")(?::\d+(?:[-–‑]\d+)?(?::\d+)?|#L\d+(?:-L?\d+)?)", re.I)
ID_LIKE_RE = re.compile(r"(?:^|_)(?=[a-z0-9]*\d)[0-9a-f]{8,}(?:_|$)")  # req_ce86e13175f5: an id, not code
# The code name written right before a location on its line, by its mark (each group is named after it), and the
# words that may come between them: "`parse_date()` at src/a.py:42", "parse_date(), in src/a.py:42".
NAME_BEFORE = re.compile(
    r"""(?x)
    (?: ``(?P<double_backticks>[^`\n]+)``
      | (?<!`)`(?P<backticks>[^`\n]+)`
      | \*\*(?P<bold>[^*\n]+)\*\*
      | <code>(?P<html_code>[^<\n]+)</code>
      | (?<![\w*])\*(?P<italic>[^*\s][^*\n]*)\*
      | (?<![\w`*>/.\\-])(?P<plain>[A-Za-z_$](?:[\w.:#$]*[\w$])?(?:\(\))?) )
    \s*(?:[,:;—–-]\s*)?(?:(?:at|in|on|of|from|see|en|dans|em|im|bei|na)\s+)?$"""
)
# The wraps of a location each location mark accepts. A `link` location is the target of a link whose text is the
# code's name in backticks.
LOCATION_WRAPS = {
    "parens_backticks": ("parens_backticks",),
    "bold": ("bold",),
    "brackets": ("brackets",),
    "parens": ("parens",),
    "parens_link": ("parens_link_text", "parens_link_target"),
    "link": ("link_target", "parens_link_target"),
}


def _reads_as_code(name: str) -> bool:
    """a token without spaces names code: a call (parse(), a.b(x)), a code file or a file with its line, a dotted name,
    a snake_case or ALL_CAPS name, or a camelCase or UpperCamel name; not a library such as Three.js, a non-code file,
    a short dotted word such as e.g, or an id such as req_ce86e13175f5"""
    if name.startswith(("-", "#!")) or _is_library(name):
        return False
    if re.fullmatch(r"[\w.$:<>\[\]]+\(.*\)", name) or CODE_FILE_RE.fullmatch(name) or FILE_LINE_RE.fullmatch(name):
        return True
    if re.fullmatch(r"[A-Za-z_$][\w$]*(?:(?:\.|::|#)[A-Za-z_$][\w$]*)+", name) and not re.fullmatch(
        r"[a-z]{1,4}\.[a-z]{1,4}", name
    ):
        return not FILE_RE.fullmatch(name)
    if re.fullmatch(r"_*[a-z][a-z0-9]*(?:_[a-z0-9]+)+", name) or re.fullmatch(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+", name):
        return not ID_LIKE_RE.search(name.lower())
    return bool(re.fullmatch(r"[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+", name))


def _code_name(content: str, mark: str) -> str | None:
    """the content of a marked or plain token when it reads as a code name, not as a label such as **Fix:** or a
    word"""
    name = content.strip()
    if not name or len(name) > 100 or re.search(r"\s", name) or not re.search(r"[A-Za-z_]", name):
        return None
    if mark == "plain":
        return name if _reads_as_code(name) else None
    reads_as_code = _reads_as_code(name) or re.fullmatch(r"[\w.:#$<>\[\]-]+(?:\(.*\))?", name)
    return name if reads_as_code and not name.endswith(":") else None


def _name_before(t: str, pos: int) -> tuple[str, str] | None:
    """(name, its mark) of the code name written right before position pos on its line, or None"""
    m = NAME_BEFORE.search(t[t.rfind("\n", 0, pos) + 1 : pos][-200:])
    name = _code_name(m.group(m.lastgroup), m.lastgroup) if m else None
    return (name, m.lastgroup) if name else None


def _enclosing(t: str, s: int, e: int, open_ch: str, close_ch: str) -> tuple[int, int] | None:
    """(start, end) of the innermost pair of open_ch ... close_ch on the line that holds t[s:e], or None"""
    line_start, line_end = t.rfind("\n", 0, s) + 1, t.find("\n", e)
    line_end = len(t) if line_end < 0 else line_end
    depth, i = 0, s - 1
    while i >= line_start and not (t[i] == open_ch and depth == 0):
        depth += (t[i] == close_ch) - (t[i] == open_ch)
        i -= 1
    if i < line_start:
        return None
    depth, j = 0, e
    while j < line_end and not (t[j] == close_ch and depth == 0):
        depth += (t[j] == open_ch) - (t[j] == close_ch)
        j += 1
    return (i, j + 1) if j < line_end else None


def _location_wrap(t: str, s: int, e: int, sp: dict[str, list[tuple[int, int, re.Match]]]) -> tuple[str | None, int]:
    """(how the location t[s:e] is wrapped, where the wrapped part starts): link_text or link_target (parens_link_text
    or parens_link_target inside parentheses), backticks, parens_backticks, bold, brackets, parens or plain; None for
    a command or a snippet in backticks that holds it. Brackets and parentheses count when they enclose the location on
    its line, also with more words inside ("(src/a.py, lines 612 and 618)"); the wrapped part starts at the bracket
    only when the bracket opens right at the location, so the name looked for before it is the one outside."""
    link = _around(sp["link"], s, e)
    if link:
        link_start, link_end, lm = link
        wrap = "link_target" if s >= lm.start(2) else "link_text"
        if t[link_start - 1 : link_start] == "(" and t[link_end : link_end + 1] == ")":
            return "parens_" + wrap, link_start - 1
        return wrap, link_start
    inline = _around(sp["inline"], s, e)
    if inline:
        if inline[2].group(2).strip() != t[s:e]:
            return None, s
        s, e = inline[0], inline[1]
        if t[s - 1 : s] == "(" and t[e : e + 1] == ")":
            return "parens_backticks", s - 1
        bold = _around(sp["bold"], s, e)
        if bold and bold[2].group(1).strip().strip("`") == t[s:e].strip("`"):
            return "bold", bold[0]
        return "backticks", s
    bold = _around(sp["bold"], s, e)
    if bold and bold[2].group(1).strip() == t[s:e]:
        return "bold", bold[0]
    brackets = _enclosing(t, s, e, "[", "]")
    if brackets and t[brackets[1] : brackets[1] + 1] != "(":  # square brackets, not the text of a markdown link
        return "brackets", brackets[0] if not t[brackets[0] + 1 : s].strip() else s
    parens = _enclosing(t, s, e, "(", ")")
    if parens:
        return "parens", parens[0] if not t[parens[0] + 1 : s].strip() else s
    return "plain", s


def code_reference_marks(text: str, *, name_mark: str | None = None, location_mark: str | None = None) -> list[str]:
    """[what is wrong] with the marks of the code references in one message. A reference here is a location of code
    (CODE_LOCATION) and the code name written right before it. name_mark: plain, backticks, double_backticks, bold,
    html_code or italic around that name. location_mark: parens_backticks, bold, brackets, parens, parens_link or link
    (the location is the target of a link whose text is in backticks) around the location. A location inside a fenced
    code block, a web address, a markup tag, or a command or snippet in backticks is not graded here."""
    if name_mark is not None and name_mark not in NAME_BEFORE.groupindex:
        raise ValueError("code reference marks: unknown name mark %r" % (name_mark,))
    if location_mark is not None and location_mark not in LOCATION_WRAPS:
        raise ValueError("code reference marks: unknown location mark %r" % (location_mark,))
    t, sp = _marked_text(text)
    bad, found = [], []
    for m in CODE_LOCATION.finditer(t):
        s, e = m.start(), m.end()
        if _around(sp["url"], s, e) or _around(sp["markup"], s, e):
            continue
        wrap, wrap_start = _location_wrap(t, s, e, sp)
        if wrap is None or (location_mark == "link" and wrap in ("link_text", "parens_link_text")):
            continue  # a command or a snippet; or, for a link, the link's text (its target is graded)
        found.append((wrap_start, t[s:e]))
        if location_mark and wrap not in LOCATION_WRAPS[location_mark]:
            wanted = "link_backticked_text" if location_mark == "link" else location_mark
            bad.append("the location %s is written %s, not %s" % (t[s:e], WRAP_WORDS[wrap], WRAP_WORDS[wanted]))
        elif location_mark == "link":
            lm = _around(sp["link"], s, e)[2]
            if not re.fullmatch(r"`[^`]+`", lm.group(1).strip()):
                bad.append("the link [%s](%s) does not have its text in backticks" % (lm.group(1), lm.group(2)))
    if name_mark:
        for wrap_start, location in found:
            written = _name_before(t, wrap_start)
            if written and written[1] != name_mark:
                bad.append(
                    "the name %s before %s is written %s, not %s"
                    % (written[0], location, WRAP_WORDS[written[1]], WRAP_WORDS[name_mark])
                )
    return list(dict.fromkeys(bad))
