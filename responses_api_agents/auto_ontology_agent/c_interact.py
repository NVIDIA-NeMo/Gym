# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BIRD-Interact's c-Interact for Auto Ontology's ReAct agent.

The ``bird_interact`` Environment Server runs c-Interact as the BIRD-Interact ADK does
(``environment_servers/bird_interact``): it seeds no agent context, and each turn's request holds the ADK's texts.
The first request's instructions name the database, the visible knowledge base and how many questions the agent may
ask, and its input is the phase-1 message with the user's ambiguous request. A later turn's input is the debug
message after a failed submission or the follow-up question that opens phase 2. The agent asks the user with
``ask_user`` and submits with ``submit_sql``, two tools the server grants.

This module reads those texts and writes what one ReAct episode gets for a turn: the question, the evidence (the
visible knowledge, the user's answers so far and the earlier submissions) and a prompt section on asking. Auto
Ontology's own prompts, tools and loop stay as they are; the episode adds ``ask_user``, and the query Auto Ontology
accepts goes to ``submit_sql``.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal


ASK_TOOL = "ask_user"
SUBMIT_TOOL = "submit_sql"
# A task has at most four turns: each phase, and a debug turn after a phase that failed.
MAX_TURNS = 4

# The ADK's texts (environment_servers/bird_interact/adk_prompts.py). With SQLite the server writes SQLite where the
# ADK writes PostgreSQL.
_DATABASE = re.compile(r'^You have a DB called "([A-Za-z0-9_-]+)"\.$', re.MULTILINE)
_MAX_QUESTIONS = re.compile(r"^You have at most (\d+) clarification turns\.", re.MULTILINE)
_KNOWLEDGE_HEADING = "\n\n# External Knowledge:\n"
_INSTRUCTIONS_HEADING = "\n\n# Instructions:\n"
_PHASE_1 = re.compile(r"\AUser Query:\n(.*)\n\nYou have \d+ clarification turns\. ", re.DOTALL)
_FOLLOW_UP = re.compile(
    r"\APhase 1 is complete\. Here is a follow-up question:\n\n(.*)\n\nGenerate the (?:PostgreSQL|SQLite) query and "
    r"call submit_sql\.\Z",
    re.DOTALL,
)
_DEBUG_PREFIXES = ("Your SQL is not executable: ", "Your SQL is not correct.")
# The note the server adds to each answer of the user.
_QUESTIONS_LEFT_NOTE = "\n\n[SYSTEM NOTE: Clarification turns remaining: "


class CInteractTextError(ValueError):
    """A request that does not hold the ADK's c-Interact texts."""


@dataclass(frozen=True)
class CInteractTask:
    """What the first request's instructions say about the task."""

    database: str
    dialect: str
    # The knowledge-base entries the user shares: id, knowledge, description and definition.
    knowledge: list[dict[str, Any]]
    max_questions: int


@dataclass(frozen=True)
class Turn:
    """What one turn's input asks for."""

    kind: Literal["phase_1", "debug", "follow_up"]
    # The user's request for phase 1 and the follow-up; the server's message for a debug turn.
    text: str


@dataclass
class Exchange:
    question: str
    answer: str

    @classmethod
    def from_reply(cls, question: str, reply: str) -> "Exchange":
        """A question and the user's answer, without the note on how many questions are left."""
        return cls(question=question, answer=reply.split(_QUESTIONS_LEFT_NOTE, 1)[0].strip())


@dataclass
class Submission:
    phase: int
    sql: str
    reply: str


@dataclass
class Dialogue:
    """What the agent knows across the task's turns."""

    task: CInteractTask
    phase: int = 1
    questions: dict[int, str] = field(default_factory=dict)
    exchanges: list[Exchange] = field(default_factory=list)
    submissions: list[Submission] = field(default_factory=list)


def read_task(instructions: str) -> CInteractTask:
    """The task, from the ADK instruction that the first request carries."""
    database = _DATABASE.search(instructions)
    max_questions = _MAX_QUESTIONS.search(instructions)
    # The schema comes first, and its text is the database's: the knowledge is after the last heading of it.
    _, knowledge_heading, rest = instructions.rpartition(_KNOWLEDGE_HEADING)
    knowledge_text, instructions_heading, _ = rest.partition(_INSTRUCTIONS_HEADING)
    if database is None or max_questions is None or not knowledge_heading or not instructions_heading:
        raise CInteractTextError("The first request's instructions are not BIRD-Interact's c-Interact instruction")
    try:
        knowledge = json.loads(knowledge_text)
    except json.JSONDecodeError as error:
        raise CInteractTextError(f"The instruction's knowledge base is not JSON: {error}") from error
    if not isinstance(knowledge, list):
        raise CInteractTextError("The instruction's knowledge base is not a list")
    dialect = "sqlite" if "SQLite writing ability" in instructions else "postgres"
    return CInteractTask(
        database=database.group(1), dialect=dialect, knowledge=knowledge, max_questions=int(max_questions.group(1))
    )


def read_turn(message: str) -> Turn:
    """What a turn's input message asks for."""
    if phase_1 := _PHASE_1.match(message):
        return Turn(kind="phase_1", text=phase_1.group(1).strip())
    if follow_up := _FOLLOW_UP.match(message):
        return Turn(kind="follow_up", text=follow_up.group(1).strip())
    if message.startswith(_DEBUG_PREFIXES):
        return Turn(kind="debug", text=message.strip())
    raise CInteractTextError(f"Not a c-Interact turn message: {message[:200]!r}")


def advance(dialogue: Dialogue, turn: Turn) -> str:
    """Record ``turn`` in ``dialogue`` and return the question of the turn's episode."""
    if turn.kind == "phase_1":
        dialogue.questions[1] = turn.text
    elif turn.kind == "follow_up":
        dialogue.phase = 2
        dialogue.questions[2] = turn.text
    if dialogue.phase not in dialogue.questions:
        raise CInteractTextError(f"A {turn.kind} turn came before the phase's question")
    return dialogue.questions[dialogue.phase]


def evidence(dialogue: Dialogue, turn: Turn) -> str:
    """What the episode knows besides the question: the shared knowledge, the user's answers and the submissions."""
    parts = []
    if dialogue.task.knowledge:
        entries = []
        for entry in dialogue.task.knowledge:
            line = f"- {entry.get('knowledge', '')}: {entry.get('description', '')}".rstrip(": ")
            if definition := entry.get("definition"):
                line += f"\n  Definition: {definition}"
            entries.append(line)
        parts.append("Knowledge the user shared:\n" + "\n".join(entries))
    if dialogue.exchanges:
        parts.append(
            "The user's answers to earlier questions:\n"
            + "\n".join(f"Q: {exchange.question}\nA: {exchange.answer}" for exchange in dialogue.exchanges)
        )
    if dialogue.phase == 2:
        accepted = [submission.sql for submission in dialogue.submissions if submission.phase == 1]
        if accepted:
            parts.append(
                f"This follows up on the user's earlier request: {dialogue.questions[1]}\n"
                f"The query that answered it:\n{accepted[-1]}"
            )
    if turn.kind == "debug":
        last = [submission for submission in dialogue.submissions if submission.phase == dialogue.phase]
        if last:
            parts.append(f"Your last query for this request:\n{last[-1].sql}\nThe user says: {turn.text}")
        else:
            parts.append(f"You submitted no query for this request. The user says: {turn.text}")
    return "\n\n".join(parts)


def asking_prompt(dialogue: Dialogue) -> str:
    """The system prompt's section on asking the user, for this turn."""
    remaining = max(dialogue.task.max_questions - len(dialogue.exchanges), 0)
    return (
        "## Asking the user\n"
        "The user's request may be ambiguous, and some of the knowledge it relies on is not written down. When the "
        "catalog and the evidence leave a term, a formula or a value unclear, ask the user with ask_user, one question "
        "at a time. The user knows what they want, but not the database: ask about meaning, never about tables or "
        f"columns. You may ask {remaining} more question{'s' if remaining != 1 else ''} in this task; after that, "
        "submit your best query."
    )
