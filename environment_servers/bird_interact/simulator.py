# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The user simulator: the BIRD-Interact ADK's two-stage pipeline, with its default (``v2``) prompts.

The action parser reads the agent's question, the annotated ambiguities and the solution SQL split into clauses, and
picks one action: ``labeled("<term>")``, ``unlabeled("<segment>")`` or ``unanswerable()``. The response generator
then writes the user's reply from that action, the clear question, the solution SQL and the schema. The prompts are
the ADK's (``adk_prompts.py``); this module fills them, and reads the model's output, as the ADK's user simulator
does. In phase 2 the prompts show no ambiguities and the follow-up's solution, and keep phase 1's clear question.

One difference: the ADK fills the placeholders one after another, so a question that contains ``[[GT_SQL]]`` would
get the solution SQL written into it. Here the placeholders are filled in one pass over the template.
"""

import json
import re

from sqlglot import tokenize
from sqlglot.errors import TokenError
from sqlglot.tokens import TokenType

from environment_servers.bird_interact.adk_prompts import (
    ACTION_PARSER_PROMPT,
    RESPONSE_GENERATOR_PROMPT,
    UNCLEAR_QUESTION_REPLY,
)
from environment_servers.bird_interact.task_data import BirdInteractTask


# The ADK's output limits for the two stages, which it samples at temperature 0.
ACTION_PARSER_MAX_TOKENS = 500
RESPONSE_GENERATOR_MAX_TOKENS = 1024
_PLACEHOLDER = re.compile(r"\[\[(\w+)\]\]")
# The tokens that start a clause of the solution SQL, and the clause's name.
_CLAUSES = {
    TokenType.SELECT: "SELECT",
    TokenType.FROM: "FROM",
    TokenType.WHERE: "WHERE",
    TokenType.HAVING: "HAVING",
    TokenType.GROUP_BY: "GROUP BY",
    TokenType.ORDER_BY: "ORDER BY",
    TokenType.LIMIT: "LIMIT",
    TokenType.OFFSET: "OFFSET",
    TokenType.JOIN: "JOIN",
}


def segment_sql(sql: str) -> list[tuple[str, str]]:
    """Split ``sql`` at each clause keyword into (clause, text up to the next clause), as PostgreSQL tokenizes it.

    SQL with no clause is one ``STATEMENT``; SQL that does not tokenize is one ``STATEMENT`` for each semicolon.
    """
    try:
        starts = [
            (token.start, _CLAUSES[token.token_type])
            for token in tokenize(sql, read="postgres")
            if token.token_type in _CLAUSES
        ]
    except TokenError:
        return [("STATEMENT", f"{part.strip()};") for part in sql.split(";") if part.strip()]
    if not starts:
        return [("STATEMENT", sql.strip())]
    ends = [start for start, _ in starts[1:]] + [len(sql)]
    return [(clause, sql[start:end].strip()) for (start, clause), end in zip(starts, ends)]


def _solution(task: BirdInteractTask, phase: int) -> list[str]:
    return task.sol_sql if phase == 1 else task.follow_up.sol_sql


def _segments(task: BirdInteractTask, phase: int) -> str:
    return "\n===\n".join(
        "\n\n".join(f"{clause}:\n{text}" for clause, text in segment_sql(sql)) for sql in _solution(task, phase)
    )


def _ambiguities(task: BirdInteractTask, phase: int) -> str:
    """The annotated ambiguities as the row writes them; phase 2's question has none."""
    if phase != 1:
        return json.dumps({}, indent=4)
    query_ambiguity = task.user_query_ambiguity.model_dump(mode="json", exclude_unset=True)
    knowledge_ambiguity = [point.model_dump(mode="json", exclude_unset=True) for point in task.knowledge_ambiguity]
    return (
        f"user_query_ambiguity: \n{json.dumps(query_ambiguity, indent=4)}\n\n"
        f"knowledge_ambiguity: \n{json.dumps(knowledge_ambiguity, indent=4)}"
    )


def _fill(template: str, values: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda match: values.get(match.group(1), match.group(0)), template)


def action_parser_prompt(task: BirdInteractTask, question: str, *, phase: int) -> str:
    return _fill(
        ACTION_PARSER_PROMPT,
        {"clarification_Q": question, "amb_json": _ambiguities(task, phase), "SQL_Glot": _segments(task, phase)},
    )


def response_generator_prompt(
    task: BirdInteractTask, question: str, action: str, *, phase: int, schema_text: str
) -> str:
    return _fill(
        RESPONSE_GENERATOR_PROMPT,
        {
            "clarification_Q": question,
            "Action": action,
            "clear_query": task.query if task.query is not None else task.amb_user_query,
            "amb_json": _ambiguities(task, phase),
            "GT_SQL": "\n".join(_solution(task, phase)),
            "SQL_Glot": _segments(task, phase),
            "DB_schema": schema_text,
        },
    )


def read_action(output: str) -> str:
    """The action in the action parser's output: what precedes ``</s>``, else its first line, after any ``<s>``."""
    output = output.strip()
    action = output.split("</s>")[0].strip() if "</s>" in output else output.split("\n")[0].strip()
    if "<s>" in action:
        action = action[action.find("<s>") :].replace("<s>", "").strip()
    return action


def read_reply(output: str) -> str:
    """The reply in the response generator's output, between ``<s>`` and ``</s>``, either of which may be missing."""
    output = output.strip()
    if "</s>" in output:
        reply = output.split("</s>")[0].strip()
        return reply[reply.find("<s>") :].replace("<s>", "").strip() if "<s>" in reply else reply
    if "<s>" in output:
        return output.split("<s>")[1].strip()
    return UNCLEAR_QUESTION_REPLY
