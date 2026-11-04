# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the Auto Ontology agent reads from BIRD-Interact's c-Interact protocol, and how it adapts what it sends.

The ``bird_interact`` Environment Server sends the BIRD-Interact ADK's fixed texts
(``environment_servers/bird_interact/adk_prompts.py``). The first activation's system message is the ADK's instruction,
with the database, its schema, the visible knowledge and the question budget; its user message opens phase 1. Each
later activation opens with the ADK's follow-up or debug message. These functions read those texts exactly and fail
with ``ProtocolError`` on a text they do not know, so a change on the server side cannot pass unnoticed.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Literal, Optional


LOGGER = logging.getLogger(__name__)

# The number of child entries a knowledge entry shows, when the session reads them (see ``format_knowledge``).
MAX_KNOWLEDGE_CHILDREN = 5

_DATABASE = re.compile(r'^You have a DB called "(?P<database>[^"]+)"\.$', re.MULTILINE)
_SCHEMA_AND_KNOWLEDGE = re.compile(
    r"^# DB Schema Info:\n(?P<schema>.*?)\n\n# External Knowledge:\n(?P<knowledge>.*?)\n\n# Instructions:$",
    re.MULTILINE | re.DOTALL,
)
_QUESTION_BUDGET = re.compile(r"^You have at most (?P<budget>\d+) clarification turns\.", re.MULTILINE)
_PHASE_1 = re.compile(r"\AUser Query:\n(?P<question>.*)\n\nYou have \d+ clarification turns\. ", re.DOTALL)
_FOLLOW_UP = re.compile(
    r"\APhase 1 is complete\. Here is a follow-up question:\n\n(?P<question>.*)\n\n"
    r"Generate the \w+ query and call submit_sql\.\Z",
    re.DOTALL,
)
_EXECUTION_ERROR = re.compile(
    r"\AYour SQL is not executable: (?P<error>.*)\nPlease fix and call submit_sql\.\Z", re.DOTALL
)
_WRONG_RESULT = "Your SQL is not correct. You have one more chance. Please fix and call submit_sql."
# The server appends this note to each answer of the user; the ADK's simulator answer has none.
_CLARIFICATIONS_LEFT_NOTE = re.compile(r"\n\n\[SYSTEM NOTE: Clarification turns remaining: \d+/\d+\]\Z")
_PHASE_PASSED = re.compile(r"\APhase (?P<phase>[12]) correct! ")


class ProtocolError(ValueError):
    """A message that is not one of the c-Interact texts this agent reads."""


@dataclass(frozen=True)
class TaskContext:
    """The task as the ADK's instruction gives it to the agent."""

    database: str
    schema: str
    knowledge: list[dict]
    question_budget: int


@dataclass(frozen=True)
class NextTurn:
    """What a later activation asks for: phase 2's question, or a retry of the failed phase.

    ``text`` is the follow-up question, or the execution error of a debug turn (``None`` when the SQL ran but its
    result was wrong).
    """

    kind: Literal["follow_up", "debug"]
    text: Optional[str]


def read_instructions(instructions: str) -> TaskContext:
    """Read the database, schema, visible knowledge and question budget from the ADK's c-Interact instruction."""
    database = _DATABASE.search(instructions)
    sections = _SCHEMA_AND_KNOWLEDGE.search(instructions)
    budget = _QUESTION_BUDGET.search(instructions)
    if database is None or sections is None or budget is None:
        raise ProtocolError("The system message is not the BIRD-Interact ADK's c-Interact instruction")
    try:
        knowledge = json.loads(sections["knowledge"])
    except json.JSONDecodeError as error:
        raise ProtocolError(f"The instruction's external knowledge is not JSON: {error}") from error
    if not isinstance(knowledge, list) or not all(isinstance(entry, dict) for entry in knowledge):
        raise ProtocolError("The instruction's external knowledge is not a list of entries")
    return TaskContext(
        database=database["database"],
        schema=sections["schema"],
        knowledge=knowledge,
        question_budget=int(budget["budget"]),
    )


def read_question(message: str) -> str:
    """The user's question from the ADK's phase 1 message."""
    match = _PHASE_1.search(message)
    if match is None:
        raise ProtocolError("The first user message is not the BIRD-Interact ADK's phase 1 message")
    return match["question"].strip()


def read_next_turn(message: str) -> NextTurn:
    """Classify the message that opens a later activation: the follow-up question, or a debug turn."""
    if follow_up := _FOLLOW_UP.search(message):
        return NextTurn("follow_up", follow_up["question"].strip())
    if execution_error := _EXECUTION_ERROR.search(message):
        return NextTurn("debug", execution_error["error"].strip())
    if message.strip() == _WRONG_RESULT:
        return NextTurn("debug", None)
    raise ProtocolError(f"Unknown c-Interact message: {message[:200]!r}")


def user_answer(reply: str) -> str:
    """The user's answer in an ``ask_user`` reply, without the server's note on the clarifications left."""
    return _CLARIFICATIONS_LEFT_NOTE.sub("", reply)


def submit_result(reply: str) -> dict:
    """The ``submit_sql`` reply as the ADK's ``/submit`` result that Auto Ontology reads: its message and phase.

    The reply's first line is the grader's message. ``phase_completed`` names the phase it passed, as the ADK's
    ``/submit`` does, and is ``None`` for a failed submission.
    """
    message = reply.split("\n", 1)[0]
    passed = _PHASE_PASSED.search(message)
    return {"message": message, "phase_completed": int(passed["phase"]) if passed else None}


def format_knowledge(
    entries: list[dict], children: Optional[dict[int, list[int]]] = None
) -> tuple[str, dict[str, list[str]]]:
    """The knowledge as Auto Ontology's session takes it: one text block, and each entry's children by its name.

    Each entry becomes ``- <name>`` with its description and definition on indented lines. ``children`` maps an entry's
    id to its ``children_knowledge`` ids. The ADK hides that field from agents, so it is empty unless the agent is
    configured to read it; each entry then lists up to ``MAX_KNOWLEDGE_CHILDREN`` of its children's texts.
    """
    texts: dict[int, str] = {}
    names: dict[int, str] = {}
    for entry in entries:
        entry_id, name = entry.get("id"), entry.get("knowledge") or ""
        lines = [f"- {name}"] if name else []
        if entry.get("description"):
            lines.append(f"  Description: {entry['description']}")
        if entry.get("definition"):
            lines.append(f"  Definition: {entry['definition']}")
        if lines and entry_id is not None:
            texts[entry_id], names[entry_id] = "\n".join(lines), name
    children_texts: dict[str, list[str]] = {}
    for entry_id, child_ids in (children or {}).items():
        if not names.get(entry_id):
            continue
        shown = [texts[child] for child in child_ids[:MAX_KNOWLEDGE_CHILDREN] if child in texts]
        if shown:
            children_texts[names[entry_id]] = shown
    return "\n".join(texts.values()), children_texts


def strip_catalog_qualifiers(sql: str) -> str:
    """Drop the database name that Auto Ontology puts before each table (``<database>.<schema>.<table>``).

    The grader runs the SQL in another database, ``<database>_template``, where PostgreSQL refuses a reference to
    ``<database>``. SQL that does not parse is returned as it is, so the grader reports its error.
    """
    import sqlglot
    from sqlglot import exp

    try:
        statements = sqlglot.parse(sql, dialect="postgres")
    except sqlglot.errors.ParseError:
        LOGGER.warning("Submitting SQL that does not parse unchanged: %r", sql)
        return sql
    changed = False
    for statement in statements:
        for table in statement.find_all(exp.Table) if statement is not None else ():
            if table.catalog:
                table.set("catalog", None)
                changed = True
    if not changed:
        return sql
    return ";\n".join(statement.sql(dialect="postgres") for statement in statements if statement is not None)
