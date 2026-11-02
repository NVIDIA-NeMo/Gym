# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The two-stage user simulator: classify the agent's question, then answer it.

BIRD-Interact's simulator first maps a clarifying question to one action: an annotated ambiguity (``labeled``), a
detail the ground truth settles that is not annotated (``unlabeled``), or a question to refuse (``unanswerable``).
It then writes the reply from that action alone, so the reply cannot reveal more than the action allows.
The prompts here are written for Gym. Parity runs against the published scores need the reference prompts.
"""

import json
import re
from dataclasses import dataclass
from typing import Literal

from environment_servers.bird_interact.task_data import AmbiguityPoint, BirdInteractTask


UNANSWERABLE_REPLY = "Sorry, I can't help with that question. Please work with what you have."
_ACTION = re.compile(r'\b(labeled|unlabeled)\(\s*"(.+?)"\s*\)|\bunanswerable\(\s*\)', re.DOTALL)


@dataclass(frozen=True)
class SimulatorAction:
    kind: Literal["labeled", "unlabeled", "unanswerable"]
    argument: str | None = None


def _sql(task: BirdInteractTask, phase: int) -> str:
    sql = task.sol_sql if phase == 1 else task.follow_up.sol_sql
    return "\n".join(sql)


def classifier_prompt(task: BirdInteractTask, question: str, *, phase: int) -> str:
    ambiguities = [{"term": point.term, "sql_snippet": point.sql_snippet} for point in task.ambiguities()]
    request = task.amb_user_query if phase == 1 else task.follow_up.query
    return f"""You play the user in a text-to-SQL conversation. An AI assistant writes SQL for your request.
It asked you a clarifying question. Decide what kind of question it is.
The assistant sees only its own question, never this prompt.

Your request:
{request}

Annotated ambiguities in your request. Each has a term and the SQL fragment that resolves it:
{json.dumps(ambiguities, indent=1)}

Ground-truth SQL:
```sql
{_sql(task, phase)}
```

The assistant's question:
{question}

Choose exactly one action:
- labeled("<term>"): the question is about an annotated ambiguity. Write its exact term.
- unlabeled("<SQL fragment>"): the question is about a detail that is not annotated, that the ground-truth SQL
  settles, and that a careful analyst needs. Quote the SQL fragment.
- unanswerable(): any other question, such as a request for the SQL, the result, or a detail the SQL does not settle.

Write only the action, on one line."""


def parse_action(text: str, task: BirdInteractTask) -> SimulatorAction:
    """Read the classifier's action. Anything unreadable, or a term that is not annotated, is ``unanswerable``."""
    match = _ACTION.search(text)
    if match is None or match.group(1) is None:
        return SimulatorAction("unanswerable")
    kind, argument = match.group(1), match.group(2).strip()
    if kind == "labeled" and _ambiguity(task, argument) is None:
        return SimulatorAction("unanswerable")
    return SimulatorAction(kind, argument)


def _ambiguity(task: BirdInteractTask, term: str) -> AmbiguityPoint | None:
    return next((point for point in task.ambiguities() if point.term.casefold() == term.casefold()), None)


def responder_prompt(
    task: BirdInteractTask,
    question: str,
    action: SimulatorAction,
    *,
    phase: int,
    schema_text: str,
    knowledge: dict[int, dict],
) -> str:
    """The prompt that writes the reply for a ``labeled`` or ``unlabeled`` action."""
    if action.kind == "labeled":
        point = _ambiguity(task, action.argument)
        fact = f"The term {point.term!r} means this SQL: {point.sql_snippet}"
        # A term in the question links to a hidden knowledge entry through the same SQL fragment.
        for linked in task.knowledge_ambiguity:
            if (linked is point or linked.sql_snippet == point.sql_snippet) and linked.deleted_knowledge in knowledge:
                fact += f"\nIt comes from this knowledge entry: {json.dumps(knowledge[linked.deleted_knowledge])}"
    else:
        fact = f"This part of the ground-truth SQL answers the question: {action.argument}"
    request = task.amb_user_query if phase == 1 else task.follow_up.query
    clear = f"\nThe same request without ambiguity (hidden from the assistant):\n{task.query}\n" if task.query else ""
    return f"""You play the user in a text-to-SQL conversation. Answer the assistant's clarifying question as the
person who wrote the request: briefly, in plain language, in one or two sentences.

Rules:
- Answer only what the question asks, with the fact below.
- State every detail the way the ground-truth SQL computes it, such as a fraction or a percentage, a unit or a
  sort order. Never add a requirement the SQL does not have; if it does not settle a point, say it does not matter.
- Never write SQL, never quote the ground truth, and never give the full solution.
- Do not ask a question back.

Database schema:
{schema_text}

Your request:
{request}
{clear}
The fact that answers the question (hidden from the assistant):
{fact}

Ground-truth SQL (hidden from the assistant):
```sql
{_sql(task, phase)}
```

The assistant's question:
{question}

Write only your reply."""
