#!/usr/bin/env python3
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
"""Clean the training rows of one data source with named rules, one rule at a time: normalise what a redaction step
broke, drop rows that would teach the wrong thing, and cap the rows per session.

Reads one JSONL file of agentic_turn_simulation rows from one data source and writes the kept rows in input order.
A kept line keeps its exact bytes unless a normalise rule changed it. Rows are compared only with rows of the same
file, so clean each source separately and mix the results with mix_rows.py. The row id is
`verifier_metadata.row_id`.

Rules, in this order (each function's docstring says why; docs/data.md has the table):
  normalise  restore_sandbox_home     sandbox rows: the redacted home path (REDACTED_SANDBOX_HOME) becomes /home/user
  validity   flattened_tool_history   an assistant message is a text rendering of a tool step
  validity   absent_input             the last user message refers to an input the row does not carry
  validity   incomplete_tool_list     the harness offered tools the row's tool list lacks
  validity   last_user_placeholder    the last user message holds a redaction placeholder (--placeholders)
  validity   duplicate_same_state     the same last message in the same state as a kept row
  validity   duplicate_repeat         the same substantive request, re-sent in another effective session
  run        session_cap              at most --cap rows per effective session (clone sessions are merged)
Normalisations run first, so every drop rule sees the normalised row. The duplicate rules and the cap never keep a
row an earlier rule dropped, but every input row takes part in their grouping and in the effective sessions.
--levels picks the levels to run: a source file gets "normalise,validity"; a run's rows get every level in one
call, so the cap still sees the copies that link a conversation's clone sessions (a later "run"-only call on
cleaned rows would cap each clone separately). --plan writes, per rule, the ids it changed or dropped (no text).
--holdout reports held-out rows that share a session with, or are a same-state copy of, any input row; it never
changes the output. Without OUTPUT, only the summary and the plan are written.

Needs the env importable (the harness profile and tool names come from the env itself):

    PYTHONPATH=<Gym root> python clean_rows.py --reviewed reviewed.json --placeholders tags.json \\
        --plan plan.json rows.jsonl rows.clean.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from resources_servers.agentic_turn_simulation.harnesses import select_harness
from resources_servers.agentic_turn_simulation.harnesses.sandbox import HOME as SANDBOX_HOME
from resources_servers.agentic_turn_simulation.scripts.mix_rows import row_id


# Every rule with its level, in the order the rules run.
RULES: tuple[tuple[str, str], ...] = (
    ("restore_sandbox_home", "normalise"),
    ("flattened_tool_history", "validity"),
    ("absent_input", "validity"),
    ("incomplete_tool_list", "validity"),
    ("last_user_placeholder", "validity"),
    ("duplicate_same_state", "validity"),
    ("duplicate_repeat", "validity"),
    ("session_cap", "run"),
)
LEVELS = ("normalise", "validity", "run")


# ---------------------------------------------------------------------------------------------------- message text
def message_text(item: Mapping[str, Any]) -> str:
    """A message's text: the string content, or its text parts concatenated."""
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content or [] if isinstance(p, dict) and isinstance(p.get("text"), str))


def _part_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join((p.get("text") or "") if isinstance(p, dict) else str(p) for p in content)
    return json.dumps(content) if content is not None else ""


def system_text(items: Iterable[Mapping[str, Any]]) -> str:
    """Every system or developer message, one text part per line."""
    return "\n".join(
        _part_text(it.get("content"))
        for it in items
        if isinstance(it, dict) and it.get("type") == "message" and it.get("role") in ("system", "developer")
    )


@dataclass(frozen=True)
class TextPatterns:
    """What `typed_text` removes from a message before comparing it. `strip` removes every match; `cut` drops the
    text from the earliest match on (a match at the very start cuts nothing)."""

    strip: tuple[re.Pattern[str], ...]
    cut: tuple[re.Pattern[str], ...]


# Harness boilerplate only. Dataset-specific blocks go in a --patterns file.
HARNESS_PATTERNS = TextPatterns(
    strip=(
        re.compile(r"<system-reminder>.*?</system-reminder>", re.S),  # Claude Code
        re.compile(r"<local-command-caveat>.*?</local-command-caveat>", re.S),  # Claude Code
        re.compile(r"<ide_(selection|opened_file)>.*?</ide_(selection|opened_file)>", re.S),  # Claude Code IDE
        re.compile(r"(?i)(sender|conversation info) \(untrusted metadata\):\s*```json.*?```", re.S),  # OpenClaw
    ),
    cut=(
        re.compile(r"(?i)--- attached context ---"),  # Hermes: appended attachments
        re.compile(r"(?i)\[context compaction — reference only\]"),  # Hermes: appended compaction summary
    ),
)


def load_patterns(path: Path | None) -> TextPatterns:
    """The harness patterns plus a data-side file's {"strip": [regex, ...], "cut": [regex, ...]}. Strip patterns
    are compiled with re.S. The file's patterns run before the harness ones."""
    if path is None:
        return HARNESS_PATTERNS
    extra = json.loads(path.read_text())
    unknown = set(extra) - {"strip", "cut"}
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")
    return TextPatterns(
        strip=tuple(re.compile(p, re.S) for p in extra.get("strip", [])) + HARNESS_PATTERNS.strip,
        cut=tuple(re.compile(p) for p in extra.get("cut", [])) + HARNESS_PATTERNS.cut,
    )


_TIMESTAMP_PREFIX = re.compile(r"^\s*(\[[^\]\n]{0,40}\d{4}-\d\d-\d\d \d\d:\d\d[^\]\n]*\]\s*)+")


def typed_text(text: str, patterns: TextPatterns = HARNESS_PATTERNS) -> str:
    """The part of a message a person typed, normalised for comparison: harness-inserted blocks removed, appended
    boilerplate cut, timestamp prefixes dropped, NFKC, case-folded, whitespace collapsed, first 1,500 characters."""
    for pattern in patterns.strip:
        text = pattern.sub(" ", text)
    starts = [m.start() for m in (p.search(text) for p in patterns.cut) if m]
    if starts and min(starts) > 0:
        text = text[: min(starts)]
    text = _TIMESTAMP_PREFIX.sub("", text)
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()[:1500]


def shingles(text: str, k: int = 5) -> frozenset[str]:
    """Character k-grams of the text with every digit run masked to 0."""
    text = re.sub(r"\d+", "0", text)
    if len(text) <= k:
        return frozenset({text}) if text else frozenset()
    return frozenset(text[i : i + k] for i in range(len(text) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


# ---------------------------------------------------------------------------------------------------- normalise
# What a redaction step that mistook the sandbox account name for a person's name wrote in place of SANDBOX_HOME.
REDACTED_SANDBOX_HOME = "/home/[PERSON_NAME]"


def _replace_in_strings(value: Any, old: str, new: str) -> Any:
    """The decoded JSON value with `old` replaced by `new` in every string."""
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, list):
        return [_replace_in_strings(x, old, new) for x in value]
    if isinstance(value, dict):
        return {k: _replace_in_strings(x, old, new) for k, x in value.items()}
    return value


def _count_in_strings(value: Any, needle: str) -> int:
    if isinstance(value, str):
        return value.count(needle)
    if isinstance(value, list):
        return sum(_count_in_strings(x, needle) for x in value)
    if isinstance(value, dict):
        return sum(_count_in_strings(x, needle) for x in value.values())
    return 0


def _locations(row: Mapping[str, Any]) -> dict[str, Any]:
    """The parts of a row that `restore_sandbox_home` reports separately."""
    params = dict(row.get("responses_create_params") or {})
    metadata = dict(row.get("verifier_metadata") or {})
    parts = {
        "input": params.pop("input", None),
        "tools": params.pop("tools", None),
        "instructions": params.pop("instructions", None),
        "reference_turn": metadata.pop("reference_turn", None),
    }
    rest = {k: v for k, v in row.items() if k not in ("responses_create_params", "verifier_metadata")}
    parts["other"] = [params, metadata, rest]
    return parts


def restore_sandbox_home(line: bytes, row: dict[str, Any]) -> tuple[bytes, dict[str, Any], dict[str, int]]:
    """Normalise. In a row whose `verifier_metadata.harness` is `sandbox`, every REDACTED_SANDBOX_HOME becomes the
    sandbox's fixed home `/home/user`, in every string of the row: input, tools, instructions and
    `verifier_metadata.reference_turn` alike. /home/user is the sandbox's real account directory, not personal data,
    so a row must carry it for the policy, the simulator and the judge to see the paths the sandbox really has.

    The edit runs on the raw line and is checked: decoded, the new line must equal the same replacement applied to
    every string of the decoded row (ValueError otherwise, for example when the line escapes "/" as "\\/"). Returns
    the line and row (both unchanged when the row is not a sandbox row or holds no such path) and the occurrences
    replaced per location (input, tools, instructions, reference_turn, other)."""
    if (row.get("verifier_metadata") or {}).get("harness") != "sandbox":
        return line, row, {}
    old, new = REDACTED_SANDBOX_HOME.encode(), SANDBOX_HOME.encode()
    if old not in line and _count_in_strings(row, REDACTED_SANDBOX_HOME) == 0:
        return line, row, {}
    expected = _replace_in_strings(row, REDACTED_SANDBOX_HOME, SANDBOX_HOME)
    out = line.replace(old, new)
    if json.loads(out) != expected:
        raise ValueError(f"raw replacement of {REDACTED_SANDBOX_HOME} differs from the structural one")
    counts = {k: n for k, v in _locations(row).items() if (n := _count_in_strings(v, REDACTED_SANDBOX_HOME))}
    return out, expected, counts


# ---------------------------------------------------------------------------------------------------- one row
@dataclass
class Row:
    """What the rules read from one row."""

    row_id: str
    session_id: str
    system_prompt: str
    n_user: int
    n_items: int
    last_user: str
    typed_last: str
    typed_prev: str
    last_shingles: frozenset[str]
    prev_shingles: frozenset[str]
    flattened: bool = False
    tool_list_case: str | None = None
    placeholder_tags: tuple[str, ...] = ()
    effective_session: str = ""
    restored: dict[str, int] = field(default_factory=dict)


_TOOL_STEP_RENDERINGS = (
    re.compile(r"^\s*\[Tool result[:\]]"),
    re.compile(r"\[Called tools?: [^\]\n]{1,200}\]\s*$"),
)


def flattened_tool_history(items: list[dict[str, Any]]) -> bool:
    """Validity drop. An assistant message that is a text rendering of a tool step ("[Tool result: ...]" as the
    message, or a "[Called tools: ...]" footer) means the trace flattened the session's tool calls into prose. The
    model never saw its history in that form, and training on it teaches the policy to narrate tool calls instead of
    making them, so the row is dropped. Call syntax a model wrote into its own text (a parse failure that really
    happened) is not this rule and stays."""
    for it in items:
        if it.get("type") == "message" and it.get("role") == "assistant":
            text = message_text(it)
            if text and any(p.search(text) for p in _TOOL_STEP_RENDERINGS):
                return True
    return False


_DECLARED_UNREADABLE = re.compile(
    r'Cannot read "[^"]+" \(this model does not support image input\)|image could not be analyzed'
)
_DROPPED_INLINE_MEDIA = re.compile(
    r"(?<!!)\[Image #?\d+\](?!\()|\[Pasted (text|image) #?\d+[^\]]*\]|\[Image: original \d+x\d+"
)


def absent_input(rows: list[Row], reviewed: Mapping[str, str]) -> dict[str, str]:
    """Validity drop. The last user message refers to an input the row does not carry, so the policy can only ask
    for it or make it up. Dropped: rows on a --reviewed list (row id -> reason code), where the user refers to
    something they attached, sent or uploaded that nothing in the input carries; messages with an inline media
    placeholder whose media the trace left out ("[Image #1]", "[Pasted text #2 ...]"); and rows under
    the same system prompt whose typed message (>= 30 characters) equals a reviewed row's. A message in which the
    harness says it could not read the input (an unsupported image) is kept unless reviewed: that is the state the
    agent really was in."""
    reviewed_text = {(r.system_prompt, r.typed_last): r.row_id for r in rows if r.row_id in reviewed}
    reasons: dict[str, str] = {}
    for r in rows:
        if r.row_id in reviewed:
            reasons[r.row_id] = "reviewed:" + reviewed[r.row_id]
            continue
        if _DECLARED_UNREADABLE.search(r.last_user):
            continue
        if _DROPPED_INLINE_MEDIA.search(r.last_user):
            reasons[r.row_id] = "inline_media_placeholder_media_not_in_input"
        elif (r.system_prompt, r.typed_last) in reviewed_text and len(r.typed_last) >= 30:
            reasons[r.row_id] = "same_text_as_reviewed:" + reviewed_text[(r.system_prompt, r.typed_last)]
    return reasons


_TOOLING_HEADING = re.compile(r"(?m)^## Tooling[ \t]*$")
_TOOLING_ITEM = re.compile(r"^- ([A-Za-z0-9_.\-]+)(?::\s|$)")
MCP_INSTRUCTIONS = "# MCP Server Instructions"


def openclaw_tooling(system: str) -> list[str] | None:
    """The tool names an OpenClaw system prompt lists under its "## Tooling" heading (a line of exactly that), or
    None without one."""
    heading = _TOOLING_HEADING.search(system)
    if heading is None:
        return None
    names: list[str] = []
    for line in system[heading.end() :].split("\n")[1:]:
        if line.startswith("## "):
            break
        match = _TOOLING_ITEM.match(line)
        if match:
            names.append(match.group(1))
        elif names and line.strip() and not line.startswith(" "):
            break
    return names


def incomplete_tool_list(row: Mapping[str, Any]) -> str | None:
    """Validity drop. The harness offered the agent tools that the row's tool list lacks, so the env would gate a
    call the real agent could make. The missing schemas were never captured and guessing them is not a safe repair,
    so the row is dropped. Three recognised cases: a Hermes subagent row (the recorded list holds only the toolsets
    the child had already used), an OpenClaw row whose own "## Tooling" section names tools its list lacks (MCP or
    plugin tools), and a Claude Code row whose system prompt carries "# MCP Server Instructions" (connected MCP
    servers whose tools are not in the list). Returns the case name, or None."""
    metadata = row.get("verifier_metadata") or {}
    harness = select_harness(metadata)
    params = row.get("responses_create_params") or {}
    items = params.get("input") or []
    if isinstance(items, str):
        items = []
    if harness.name == "hermes" and metadata.get("record_kind") == "subagent":
        return "hermes_subagent"
    if harness.name == "openclaw":
        listed = openclaw_tooling(system_text(items))
        if listed is not None and set(listed) - set(harness.tool_names(params.get("tools") or [])):
            return "openclaw_tooling_gap"
    if harness.name == "claude_code" and MCP_INSTRUCTIONS in system_text(items):
        return "claude_code_mcp_instructions"
    return None


def placeholder_pattern(vocabulary: Mapping[str, Sequence[str]]) -> re.Pattern[str] | None:
    """The redaction placeholders of a vocabulary {"tags": [...], "numbered_tags": [...]}: `[TAG]` for each name in
    `tags`, `[TAG_<digits>]` for each name in `numbered_tags`. None for an empty vocabulary."""
    unknown = set(vocabulary) - {"tags", "numbered_tags"}
    if unknown:
        raise ValueError(f"unknown vocabulary keys {sorted(unknown)}")
    names = [re.escape(t) for t in vocabulary.get("tags", [])]
    numbered = [re.escape(t) + r"_\d+" for t in vocabulary.get("numbered_tags", [])]
    if not names and not numbered:
        return None
    return re.compile(r"\[(" + "|".join(sorted(names + numbered, key=len, reverse=True)) + r")\]")


def last_user_placeholder(row: Row, pattern: re.Pattern[str] | None) -> tuple[str, ...]:
    """Validity drop. The last user message holds a redaction placeholder from the --placeholders vocabulary
    (`[NAME]`, a numbered `[NAME_3]`, ...). A placeholder in the request the policy must answer makes that
    request unanswerable or ambiguous: the policy can only guess or ask what it stands for, and a judge often pays the
    ask, a habit RL training then amplifies. So the row is neither a clean act target nor a clean ask target, and it
    is dropped. The whole message counts, including blocks a harness injected into it; placeholders earlier in the
    history stay. It runs after the normalisations, so a path that restore_sandbox_home restored is no longer a
    placeholder. Returns the placeholders found, numbers as N."""
    if pattern is None:
        return ()
    return tuple(sorted({re.sub(r"\d+", "N", m) for m in pattern.findall(row.last_user)}))


def row_features(row: Mapping[str, Any], patterns: TextPatterns = HARNESS_PATTERNS) -> Row:
    """The fields the rules read from one parsed row. A row without `session_id` is its own session; one without
    `system_prompt` (a label grouping rows that share a system prompt) is grouped by its system-message text."""
    metadata = row["verifier_metadata"]
    items = row["responses_create_params"]["input"]
    users = [i for i, it in enumerate(items) if it.get("type") == "message" and it.get("role") == "user"]
    replies = [i for i, it in enumerate(items) if it.get("type") == "message" and it.get("role") == "assistant"]
    last = users[-1] if users else None
    last_user = message_text(items[last]) if last is not None else ""
    before = [i for i in replies if last is not None and i < last]
    prev_reply = message_text(items[before[-1]])[:3000] if before else ""
    typed_last, typed_prev = typed_text(last_user, patterns), typed_text(prev_reply, patterns)
    return Row(
        row_id=row_id(metadata),
        session_id=metadata.get("session_id") or row_id(metadata),
        system_prompt=metadata.get("system_prompt") or hashlib.sha256(system_text(items).encode()).hexdigest(),
        n_user=len(users),
        n_items=len(items),
        last_user=last_user,
        typed_last=typed_last,
        typed_prev=typed_prev,
        last_shingles=shingles(typed_last),
        prev_shingles=shingles(typed_prev),
    )


# ---------------------------------------------------------------------------------------------------- row pairs
def same_state(a: Row, b: Row) -> bool:
    """Two rows are the same conversation state: last user messages of similar length with 5-gram Jaccard >= 0.7,
    and either both are first turns (>= 30 characters) or the replies before them match the same way."""
    la, lb = len(a.typed_last), len(b.typed_last)
    if not (a.typed_last or a.typed_prev):
        return False
    if min(la, lb) < 0.6 * max(la, lb):
        return False
    if jaccard(a.last_shingles, b.last_shingles) < 0.7:
        return False
    if not a.typed_prev and not b.typed_prev:
        return min(la, lb) >= 30
    if min(len(a.typed_prev), len(b.typed_prev)) < 0.6 * max(len(a.typed_prev), len(b.typed_prev)):
        return False
    if max(la, len(a.typed_prev)) < 30:
        return False
    return jaccard(a.prev_shingles, b.prev_shingles) >= 0.7


def _by_system_prompt(rows: list[Row]) -> dict[str, list[Row]]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.system_prompt].append(r)
    return {k: sorted(v, key=lambda r: len(r.typed_last)) for k, v in groups.items()}


def effective_sessions(rows: list[Row]) -> None:
    """Sets each row's effective session: its session, merged with every session that holds a same-state copy of
    one of its rows under the same system prompt (the same conversation stored under several ids)."""
    uf = _UnionFind()
    for r in rows:
        uf.find(r.session_id)
    for group in _by_system_prompt(rows).values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if len(a.typed_last) < 0.6 * len(b.typed_last):
                    break
                if a.session_id == b.session_id or uf.find(a.session_id) == uf.find(b.session_id):
                    continue
                if same_state(a, b):
                    uf.union(a.session_id, b.session_id)
    for r in rows:
        r.effective_session = uf.find(r.session_id)


def duplicates(rows: list[Row], dropped: set[str]) -> tuple[set[str], set[str]]:
    """Validity drops duplicate_same_state and duplicate_repeat: a copy upweights one state and lets the policy learn
    a reply by heart. Groups rows under one system prompt whose typed last messages are identical, or 5-gram Jaccard
    >= 0.9 at 20+ characters. Within a group, rows whose preceding replies also match (Jaccard >= 0.7, or both first
    turns) are one state. One row is kept per group (the one from the smallest effective session, then fewest user
    turns, then lowest id). For a substantive message (median >= 30 characters) every other copy in the kept row's
    state is dropped (duplicate_same_state) and so is every copy from another effective session (duplicate_repeat:
    the same request re-sent elsewhere); a copy later in the kept row's own effective session is a new state and is
    left to the session cap (one row per such state: its other same-state copies are duplicate_same_state). Effective
    sessions come from every input row, including rows an earlier rule drops. For a short message ("continue", "yes")
    only same-state copies are dropped, since the same word in another context is a new turn. Rows in `dropped`
    (earlier rules) take part in grouping but are never kept. Returns (duplicate_same_state ids, duplicate_repeat
    ids)."""
    uf = _UnionFind()
    for group in _by_system_prompt(rows).values():
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                la, lb = len(a.typed_last), len(b.typed_last)
                if la < 0.8 * lb:
                    break
                if a.typed_last == b.typed_last or (la >= 20 and jaccard(a.last_shingles, b.last_shingles) >= 0.9):
                    uf.union(a.row_id, b.row_id)
    clusters: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        clusters[uf.find(r.row_id)].append(r)
    live_per_session = Counter(r.effective_session for r in rows if r.row_id not in dropped)

    def pick(candidates: list[Row]) -> Row:
        return min(candidates, key=lambda r: (live_per_session[r.effective_session], r.n_user, r.row_id))

    same, repeat = set(), set()
    for group in (g for g in clusters.values() if len(g) > 1):
        live = [r for r in group if r.row_id not in dropped]
        median = sorted(len(r.typed_last) for r in group)[len(group) // 2]
        states = _UnionFind()
        for r in group:
            states.find(r.row_id)
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                if (not a.typed_prev and not b.typed_prev) or (
                    a.typed_prev and b.typed_prev and jaccard(a.prev_shingles, b.prev_shingles) >= 0.7
                ):
                    states.union(a.row_id, b.row_id)
        if median >= 30:
            if len(live) > 1:
                keep = pick(live)
                later: dict[str, list[Row]] = defaultdict(list)
                for r in live:
                    if r is keep:
                        continue
                    if states.find(r.row_id) == states.find(keep.row_id):
                        same.add(r.row_id)
                    elif r.effective_session != keep.effective_session:
                        repeat.add(r.row_id)
                    else:
                        later[states.find(r.row_id)].append(r)
                for members in later.values():
                    if len(members) > 1:
                        first = pick(members)
                        same.update(r.row_id for r in members if r is not first)
        else:
            by_state: dict[str, list[Row]] = defaultdict(list)
            for r in live:
                by_state[states.find(r.row_id)].append(r)
            for members in by_state.values():
                if len(members) > 1:
                    keep = pick(members)
                    same.update(r.row_id for r in members if r is not keep)
    return same, repeat


def session_cap(rows: list[Row], dropped: set[str], cap: int) -> set[str]:
    """Run selection. Keeps at most `cap` (>= 1) rows per effective session, so one long session (or one
    conversation stored under several ids) cannot dominate a training step. The kept rows are spread over the
    session: `cap` targets evenly spaced from its first to its last user-turn position (the first position when cap
    is 1), and for each target the nearest remaining row (ties: fewer input items, then lowest id). Keeps only rows
    that earlier rules kept; the effective sessions come from every input row."""
    if cap < 1:
        raise ValueError("cap must be >= 1")
    sessions: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        if r.row_id not in dropped:
            sessions[r.effective_session].append(r)
    drop: set[str] = set()
    for members in sessions.values():
        if len(members) <= cap:
            continue
        lo, hi = min(r.n_user for r in members), max(r.n_user for r in members)
        chosen: list[Row] = []
        for k in range(cap):
            target = lo if cap == 1 else lo + k * (hi - lo) / (cap - 1)
            taken = {c.row_id for c in chosen}
            chosen.append(
                min(
                    (r for r in members if r.row_id not in taken),
                    key=lambda r: (abs(r.n_user - target), r.n_items, r.row_id),
                )
            )
        keep = {c.row_id for c in chosen}
        drop.update(r.row_id for r in members if r.row_id not in keep)
    return drop


def holdout_overlap(rows: list[Row], holdout: list[Row]) -> dict[str, str]:
    """Held-out rows that share a session with an input row, or are a same-state copy of one (any input row,
    dropped or not). Report only."""
    sessions = {r.session_id for r in rows}
    by_length = sorted(rows, key=lambda r: len(r.typed_last))
    found: dict[str, str] = {}
    for v in holdout:
        if v.session_id in sessions:
            found[v.row_id] = "same_session_id"
            continue
        for a in by_length:
            if len(a.typed_last) > len(v.typed_last) / 0.6 + 1:
                break
            if same_state(v, a):
                found[v.row_id] = "same_state_as:" + a.row_id
                break
    return found


# ---------------------------------------------------------------------------------------------------- driver
@dataclass(frozen=True)
class Config:
    """How to read rows: the levels to run, the text patterns and the placeholder vocabulary."""

    levels: frozenset[str] = frozenset(LEVELS)
    patterns: TextPatterns = HARNESS_PATTERNS
    placeholders: re.Pattern[str] | None = None


def prepare_row(line: bytes, config: Config, *, with_tool_rule: bool = True) -> tuple[bytes, Row]:
    """One line after the normalisations, and its features with the per-row rule results."""
    row = json.loads(line)
    restored: dict[str, int] = {}
    if "normalise" in config.levels:
        line, row, restored = restore_sandbox_home(line, row)
    r = row_features(row, config.patterns)
    r.restored = restored
    if "validity" in config.levels:
        r.flattened = flattened_tool_history(row["responses_create_params"]["input"])
        if with_tool_rule:
            r.tool_list_case = incomplete_tool_list(row)
        r.placeholder_tags = last_user_placeholder(r, config.placeholders)
    return line, r


def normalise_line(line: bytes, levels: Iterable[str]) -> bytes:
    """The line after the normalise rules (itself when the normalise level is off)."""
    if "normalise" not in levels:
        return line
    return restore_sandbox_home(line, json.loads(line))[0]


def iter_lines(path: Path) -> Iterator[bytes]:
    """Every line of a JSONL file as bytes; a blank line is an error."""
    with open(path, "rb", buffering=1 << 22) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                raise SystemExit(f"{path}:{n}: blank line")
            yield line


def read_rows(path: Path, config: Config = Config(), *, with_tool_rule: bool = True) -> list[Row]:
    """Every row's features and per-row rule results; fails on a repeated row id."""
    rows = []
    for n, line in enumerate(iter_lines(path), 1):
        try:
            rows.append(prepare_row(line, config, with_tool_rule=with_tool_rule)[1])
        except ValueError as e:
            raise SystemExit(f"{path}:{n}: {e}") from e
    ids = Counter(r.row_id for r in rows)
    if max(ids.values(), default=1) > 1:
        raise SystemExit(f"{path}: duplicate row ids, e.g. {next(k for k, v in ids.items() if v > 1)}")
    return rows


def plan_rules(
    rows: list[Row], reviewed: Mapping[str, str], *, cap: int, levels: Iterable[str] = LEVELS
) -> list[dict[str, Any]]:
    """Every rule in order. A normalise entry's `ids` are the rows it changed; a drop entry's `ids` are the rows
    that rule drops and no earlier rule did (`matched` counts all it matched). Rules of a level not in `levels` are
    listed with `enabled: false`."""
    levels = set(levels)
    validity, run = "validity" in levels, "run" in levels
    restored = {r.row_id: r.restored for r in rows if r.restored}
    occurrences: Counter[str] = Counter()
    for counts in restored.values():
        occurrences.update(counts)
    flattened = {r.row_id for r in rows if r.flattened} if validity else set()
    absent = absent_input(rows, reviewed) if validity else {}
    tools = {r.row_id: r.tool_list_case for r in rows if r.tool_list_case} if validity else {}
    tags = {r.row_id: r.placeholder_tags for r in rows if r.placeholder_tags} if validity else {}
    early = flattened | set(absent) | set(tools) | set(tags)
    effective_sessions(rows)
    same, repeat = duplicates(rows, early) if validity else (set(), set())
    capped = session_cap(rows, early | same | repeat, cap) if run else set()
    extras: dict[str, tuple[set[str], dict[str, Any]]] = {
        "flattened_tool_history": (flattened, {}),
        "absent_input": (set(absent), {"reasons": dict(sorted(absent.items()))}),
        "incomplete_tool_list": (set(tools), {"cases": dict(sorted(tools.items()))}),
        "last_user_placeholder": (
            set(tags),
            {"tag_rows": dict(sorted(Counter(t for ts in tags.values() for t in ts).items()))},
        ),
        "duplicate_same_state": (same, {}),
        "duplicate_repeat": (repeat, {}),
        "session_cap": (capped, {"cap": cap} if run else {}),
    }
    plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name, level in RULES:
        enabled = level in levels
        if level == "normalise":
            plan.append(
                {
                    "rule": name,
                    "level": level,
                    "enabled": enabled,
                    "count": len(restored),
                    "ids": sorted(restored),
                    "occurrences": dict(sorted(occurrences.items())),
                }
            )
            continue
        ids, extra = extras[name]
        new = sorted(ids - seen)
        plan.append(
            {"rule": name, "level": level, "enabled": enabled, "count": len(new), "ids": new, "matched": len(ids)}
            | extra
        )
        seen |= ids
    return plan


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_private(path: Path, lines: Iterable[bytes], *, overwrite: bool) -> None:
    """Writes lines to path (0600) through a temporary file; refuses an existing path unless overwrite."""
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists (pass --overwrite to replace it)")
    tmp = path.with_name(path.name + ".tmp")
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "wb") as f:
        for line in lines:
            f.write(line)
    os.replace(tmp, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="rows of one data source (JSONL)")
    parser.add_argument("output", type=Path, nargs="?", help="kept rows in input order (omit to write only the plan)")
    parser.add_argument(
        "--levels",
        default=",".join(LEVELS),
        help=f"comma-separated rule levels to run (default {','.join(LEVELS)}; a source file: normalise,validity)",
    )
    parser.add_argument(
        "--reviewed",
        type=Path,
        help='JSON {"absent_input": {row id: reason code}}: a review list of rows whose input is absent',
    )
    parser.add_argument(
        "--placeholders",
        type=Path,
        help='JSON {"tags": [...], "numbered_tags": [...]}: redaction placeholders for last_user_placeholder',
    )
    parser.add_argument(
        "--patterns", type=Path, help='JSON {"strip": [regex], "cut": [regex]}: dataset text that typed_text removes'
    )
    parser.add_argument("--cap", type=int, default=5, help="rows per effective session (default 5, at least 1)")
    parser.add_argument("--plan", type=Path, help="write the ids each rule changed or dropped (JSON, no text)")
    parser.add_argument("--holdout", type=Path, help="report rows of this file that overlap the input")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.levels = frozenset(x.strip() for x in args.levels.split(",") if x.strip())
    if not args.levels or args.levels - set(LEVELS):
        parser.error(f"--levels takes a comma-separated subset of {','.join(LEVELS)}")
    if args.cap < 1:
        parser.error("--cap must be >= 1")
    if args.output is not None and args.output.resolve() == args.input.resolve():
        parser.error("output must differ from input")
    if args.output is None and args.plan is None:
        parser.error("give OUTPUT, --plan, or both")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    reviewed = json.loads(args.reviewed.read_text())["absent_input"] if args.reviewed else {}
    placeholders = placeholder_pattern(json.loads(args.placeholders.read_text())) if args.placeholders else None
    config = Config(levels=args.levels, patterns=load_patterns(args.patterns), placeholders=placeholders)
    rows = read_rows(args.input, config)
    plan = plan_rules(rows, reviewed, cap=args.cap, levels=args.levels)
    drop = {i for rule in plan if rule["level"] != "normalise" for i in rule["ids"]}
    summary: dict[str, Any] = {
        "input": args.input.name,
        "output": args.output.name if args.output else None,
        "levels": [level for level in LEVELS if level in args.levels],
        "rows_in": len(rows),
        "rows_out": len(rows) - len(drop),
        "normalised": {rule["rule"]: rule["count"] for rule in plan if rule["level"] == "normalise"},
        "dropped": {rule["rule"]: rule["count"] for rule in plan if rule["level"] != "normalise"},
        "reviewed_ids_in_input": len({r.row_id for r in rows} & set(reviewed)),
    }
    if args.output is not None:
        ids = iter(r.row_id for r in rows)
        kept = (normalise_line(line, args.levels) for line in iter_lines(args.input) if next(ids) not in drop)
        write_private(args.output, kept, overwrite=args.overwrite)
        summary["output_sha256"] = sha256_file(args.output)
    holdout: dict[str, str] = {}
    if args.holdout:
        holdout = holdout_overlap(rows, read_rows(args.holdout, config, with_tool_rule=False))
        summary["holdout"] = {"file": args.holdout.name, "overlapping_rows": len(holdout)}
    print(json.dumps(summary))
    if args.plan:
        text = json.dumps({**summary, "cap": args.cap, "rules": plan, "holdout_rows": holdout}, indent=1) + "\n"
        write_private(args.plan, [text.encode()], overwrite=True)


if __name__ == "__main__":
    main()
