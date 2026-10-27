# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed output checks: exact, deterministic contracts that need no judge.

None takes parameters, so a dataset row cannot loosen one. Markdown is parsed as CommonMark:
headings, code and comments inside code blocks, code spans or quotes are not the answer's own
structure. Meaning, tone and accuracy stay with the LLM judge.
"""

import json
import re
from typing import TYPE_CHECKING

from markdown_it import MarkdownIt
from markdown_it.token import Token

from resources_servers.partner_if.evidence import EvidenceUnavailable


if TYPE_CHECKING:
    from resources_servers.partner_if.checks import CheckContext

# The H2 sections a change review requires, in this order.
REVIEW_SECTIONS = (
    "Automated change summary",
    "Changes to exported or public declarations",
    "Dependency manifest review",
)
# The tool that drafts a plan for approval instead of acting.
PLAN_TOOL = "plan--draft"
# The artifact in which an environment records the agent's successful project edits.
EDITS_ARTIFACT = "project_edits"


def scratchpad_comments(context: "CheckContext") -> tuple[bool, str]:
    """Exact raw-comment scratchpad wrappers, without adding rendering requirements."""
    text = context.evidence.answer
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    markdown = MarkdownIt("commonmark").parse(text)
    code_blocks = [
        (offsets[token.map[0]], offsets[token.map[1]])
        for token in markdown
        if token.type in {"fence", "code_block"} and token.map
    ]
    inline_blocks = [
        (offsets[token.map[0]], offsets[token.map[1]]) for token in markdown if token.type == "inline" and token.map
    ]
    marker = re.compile(r"\[scratchpad_(?:start|end)\]")
    tokens = re.compile(r"<!--|`+|\[scratchpad_(?:start|end)\]")
    position = completed = 0
    while token := tokens.search(text, position):
        start, position = token.span()
        code_end = next((end for begin, end in code_blocks if begin <= start < end), None)
        if code_end is not None:
            position = code_end
            continue
        preceding = text[:start]
        escaped = (len(preceding) - len(preceding.rstrip("\\"))) % 2 == 1
        if token.group().startswith("`"):
            if not escaped:
                # Code spans cannot cross into another Markdown block. An
                # unmatched backtick must not swallow a later real scratchpad.
                limit = next((end for begin, end in inline_blocks if begin <= start < end), position)
                closing = re.compile(r"(?<!`)" + re.escape(token.group()) + r"(?!`)").search(text, position, limit)
                if closing:
                    position = closing.end()
            continue
        if token.group() != "<!--":
            return False, "A scratchpad marker appears outside a complete raw comment"
        if escaped:
            continue
        end = text.find("-->", position)
        if end < 0:
            if marker.search(text[start:]):
                return False, "Scratchpad comment is not closed"
            break
        position = end + 3
        comment = text[start:position]
        if not marker.search(comment):
            continue
        # Read raw comment contents as one unit: inline/list placement does not
        # change the literal contract, and code cannot hide nested delimiters.
        if (
            not comment.startswith("<!-- [scratchpad_start]")
            or not comment.endswith("[scratchpad_end] -->")
            or marker.findall(comment) != ["[scratchpad_start]", "[scratchpad_end]"]
            or "<!--" in comment[4:]
        ):
            return False, "Scratchpad must have the exact, paired, non-nested start and end wrappers"
        completed += 1
    if not completed:
        return False, "At least one complete scratchpad is required"
    return True, f"{completed} scratchpad(s) have exact, paired, non-nested wrappers"


def review_sections(context: "CheckContext") -> tuple[bool, str]:
    """The review's three rendered top-level H2 titles, in their specified order."""
    tokens = MarkdownIt("commonmark").parse(context.evidence.answer)
    headings = []
    for index, token in enumerate(tokens):
        # Quoted/list-nested examples are not the answer's top-level sections.
        if token.type == "heading_open" and token.tag == "h2" and token.level == 0:
            children = tokens[index + 1].children or []
            title = "".join(
                " " if child.type in {"softbreak", "hardbreak"} else child.content
                for child in children
                if child.type in {"text", "code_inline", "softbreak", "hardbreak"}
            )
            headings.append(" ".join(title.split()))
    if any(title not in headings for title in REVIEW_SECTIONS):
        return False, f"Missing required actual H2 section; observed titles: {headings!r}"
    positions = [headings.index(title) for title in REVIEW_SECTIONS]
    if positions != sorted(positions):
        return False, f"Required H2 sections are out of order: {headings!r}"
    return True, "All three required H2 sections appear in order"


def _word_count(text: str) -> int:
    """Whitespace-separated tokens with a letter or digit, after removing citation markers such as [web:1]."""
    text = re.sub(r"\[\w+:\d+\]", "", text)
    return sum(any(char.isalnum() for char in token) for token in text.split())


def max_100_words(context: "CheckContext") -> tuple[bool, str]:
    """At most 100 words in the answer; citation markers do not count."""
    count = _word_count(context.evidence.answer)
    return count <= 100, f"{count} words; at most 100 allowed, citation markers excluded"


def _visible_inline(children: list[Token]) -> str:
    parts = []
    for child in children:
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type == "image":
            parts.append(_visible_inline(child.children or []))
        elif child.type in {"softbreak", "hardbreak"} or (
            child.type == "html_inline" and re.fullmatch(r"<br\s*/?>", child.content, re.I)
        ):
            parts.append(" ")
    return "".join(parts)


def _headings(tokens: list[Token], *, exclude_quoted: bool = False) -> list[tuple[int, int, str]]:
    headings = []
    quote_depth = 0
    for index, token in enumerate(tokens):
        if token.type == "blockquote_open":
            quote_depth += 1
        elif token.type == "blockquote_close":
            quote_depth -= 1
        elif token.type == "heading_open" and not (exclude_quoted and quote_depth):
            headings.append((index, int(token.tag[1:]), _visible_inline(tokens[index + 1].children or [])))
    return headings


def _has_body(tokens: list[Token]) -> bool:
    for index, token in enumerate(tokens):
        if token.type == "inline" and not (index and tokens[index - 1].type == "heading_open"):
            if _visible_inline(token.children or []).strip() or any(
                child.type == "image" for child in token.children or []
            ):
                return True
        elif token.type in {"fence", "code_block"} and token.content.strip():
            return True
        elif token.type == "html_block":
            # Comments are not a visible introduction or a substantive body.
            visible = re.sub(r"<!--.*?-->", "", token.content, flags=re.S)
            if re.sub(r"<[^>]*>", "", visible).strip():
                return True
    return False


def short_headings(context: "CheckContext") -> tuple[bool, str]:
    """Every rendered heading has at most 5 words, counted as max_100_words counts them."""
    tokens = MarkdownIt("commonmark").parse(context.evidence.answer)
    counts = [_word_count(title) for _, _, title in _headings(tokens)]
    return bool(counts) and all(count <= 5 for count in counts), f"Heading word counts {counts}; limit 5 each"


def headed_section(context: "CheckContext") -> tuple[bool, str]:
    """At least one unquoted H2/H3 heading that introduces substantive content."""
    tokens = MarkdownIt("commonmark").parse(context.evidence.answer)
    # Ordered steps may contain real headings; only quoted examples are excluded.
    headings = _headings(tokens, exclude_quoted=True)
    for position, (index, level, title) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(tokens)
        if level in {2, 3} and title.strip() and _has_body(tokens[index + 3 : end]):
            return True, "An unquoted Markdown H2/H3 heading introduces substantive content"
    return False, "No substantive Markdown H2/H3 section (quoted/fenced examples do not satisfy the requirement)"


def max_five_sections(context: "CheckContext") -> tuple[bool, str]:
    """At most 5 top-level sections: Markdown outline roots plus one visible introductory block."""
    tokens = MarkdownIt("commonmark").parse(context.evidence.answer)
    headings = _headings(tokens, exclude_quoted=True)
    roots = []
    root_level = 7
    for _, level, title in headings:
        # Deeper headings are subsections until a same/higher-level heading.
        if level <= root_level:
            roots.append(title)
            root_level = level
    introduction = _has_body(tokens[: headings[0][0]] if headings else tokens)
    count = len(roots) + int(introduction)
    return 0 < count <= 5, f"{count} top-level sections; introduction={introduction}; outline roots={roots!r}; limit 5"


def no_plan_tool(context: "CheckContext") -> tuple[bool, str]:
    """No attempted call to the plan tool, whether or not its arguments were valid."""
    calls = [e for e in context.evidence.trajectory if e.kind == "tool_call" and e.tool_name == PLAN_TOOL]
    return not calls, f"{len(calls)} {PLAN_TOOL} call(s) in the episode"


def made_project_edits(context: "CheckContext") -> tuple[bool, str]:
    """At least one content-changing project edit, recorded by the environment itself."""
    text = context.evidence.artifact_text(EDITS_ARTIFACT)
    try:
        edits = json.loads(text) if text is not None else None
    except ValueError:
        edits = None
    if not isinstance(edits, list) or not all(isinstance(e, dict) and isinstance(e.get("path"), str) for e in edits):
        raise EvidenceUnavailable("Project edit receipts are missing or malformed")
    paths = sorted({edit["path"] for edit in edits})
    return bool(edits), f"{len(edits)} successful project edit(s) across {len(paths)} file(s): {', '.join(paths[:10])}"
