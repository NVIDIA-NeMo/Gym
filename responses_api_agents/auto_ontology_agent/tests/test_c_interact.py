# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading c-Interact's texts, as the bird_interact Environment Server writes them."""

import json

import pytest

from responses_api_agents.auto_ontology_agent import c_interact
from responses_api_agents.auto_ontology_agent.tests.adk_texts import (
    C_INTERACT_INSTRUCTION,
    CLARIFICATIONS_LEFT_NOTE,
    DEBUG_EXECUTION_ERROR_MESSAGE,
    DEBUG_WRONG_RESULT_MESSAGE,
    FOLLOW_UP_MESSAGE,
    PHASE_1_MESSAGE,
)


KNOWLEDGE = [
    {
        "id": 3,
        "knowledge": "Active Member",
        "description": "A member with a visit this year.",
        "definition": "visits > 0",
    },
    {"id": 4, "knowledge": "Churn", "description": "Members who left."},
]
# A schema with blank lines and brackets, as BIRD-Interact's schema files have.
SCHEMA = 'CREATE TABLE members (\n  id integer,\n  tags text[]\n);\n\nFirst 3 rows:\nid  tags\n1   ["a"]'


def _in_dialect(text: str, engine: str) -> str:
    # The server's in_dialect: the ADK's texts name PostgreSQL.
    return text if engine == "postgresql" else text.replace("PostgreSQL", "SQLite")


def _instruction(engine: str = "postgresql", knowledge: list | None = None) -> str:
    return _in_dialect(C_INTERACT_INSTRUCTION, engine).format(
        db_name="gym_members",
        db_schema=SCHEMA,
        external_kg=json.dumps(KNOWLEDGE if knowledge is None else knowledge, indent=2),
        max_turn=5,
    )


@pytest.mark.parametrize("engine", ["postgresql", "sqlite"])
def test_the_instruction_names_the_database_the_knowledge_and_the_questions(engine: str) -> None:
    task = c_interact.read_task(_instruction(engine))

    assert task.database == "gym_members"
    assert task.knowledge == KNOWLEDGE
    assert task.max_questions == 5
    assert task.dialect == ("postgres" if engine == "postgresql" else "sqlite")


def test_an_instruction_without_knowledge_has_none() -> None:
    assert c_interact.read_task(_instruction(knowledge=[])).knowledge == []


def test_another_instruction_is_refused() -> None:
    with pytest.raises(c_interact.CInteractTextError):
        c_interact.read_task("You are a helpful assistant.")


@pytest.mark.parametrize("engine", ["postgresql", "sqlite"])
def test_each_turn_message_is_read(engine: str) -> None:
    phase_1 = _in_dialect(PHASE_1_MESSAGE, engine).format(
        amb_user_query="Show the active ones.\n\nBy year.", max_turn=5
    )
    follow_up = _in_dialect(FOLLOW_UP_MESSAGE, engine).format(follow_up_query="Now only for 2024.")

    assert c_interact.read_turn(phase_1) == c_interact.Turn(kind="phase_1", text="Show the active ones.\n\nBy year.")
    assert c_interact.read_turn(follow_up) == c_interact.Turn(kind="follow_up", text="Now only for 2024.")
    executable = DEBUG_EXECUTION_ERROR_MESSAGE.format(error="column x does not exist")
    assert c_interact.read_turn(executable) == c_interact.Turn(kind="debug", text=executable)
    assert c_interact.read_turn(DEBUG_WRONG_RESULT_MESSAGE).kind == "debug"


def test_another_message_is_refused() -> None:
    with pytest.raises(c_interact.CInteractTextError, match="Not a c-Interact turn"):
        c_interact.read_turn("Hello")


def _dialogue() -> c_interact.Dialogue:
    return c_interact.Dialogue(task=c_interact.read_task(_instruction()))


def test_the_first_turn_asks_the_request_with_the_shared_knowledge() -> None:
    dialogue = _dialogue()
    turn = c_interact.Turn(kind="phase_1", text="Show the active ones.")

    assert c_interact.advance(dialogue, turn) == "Show the active ones."
    evidence = c_interact.evidence(dialogue, turn)

    assert evidence == (
        "Knowledge the user shared:\n"
        "- Active Member: A member with a visit this year.\n  Definition: visits > 0\n"
        "- Churn: Members who left."
    )


def test_a_debug_turn_keeps_the_request_and_shows_the_last_query_and_the_answers() -> None:
    dialogue = _dialogue()
    c_interact.advance(dialogue, c_interact.Turn(kind="phase_1", text="Show the active ones."))
    reply = "Members with a visit." + CLARIFICATIONS_LEFT_NOTE.format(remaining=4, max_turn=5)
    dialogue.exchanges.append(c_interact.Exchange.from_reply("What is active?", reply))
    dialogue.submissions.append(c_interact.Submission(phase=1, sql="SELECT 1", reply="SQL failed Phase 1."))
    turn = c_interact.Turn(kind="debug", text=DEBUG_WRONG_RESULT_MESSAGE)

    assert c_interact.advance(dialogue, turn) == "Show the active ones."
    evidence = c_interact.evidence(dialogue, turn)

    assert "Q: What is active?\nA: Members with a visit.\n" in evidence + "\n"
    assert "[SYSTEM NOTE" not in evidence
    assert evidence.endswith(
        f"Your last query for this request:\nSELECT 1\nThe user says: {DEBUG_WRONG_RESULT_MESSAGE}"
    )
    assert "You may ask 4 more questions" in c_interact.asking_prompt(dialogue)


def test_a_debug_turn_after_no_query_says_so() -> None:
    dialogue = _dialogue()
    c_interact.advance(dialogue, c_interact.Turn(kind="phase_1", text="q"))
    turn = c_interact.Turn(kind="debug", text=DEBUG_WRONG_RESULT_MESSAGE)

    assert "You submitted no query for this request." in c_interact.evidence(dialogue, turn)


def test_the_follow_up_carries_the_first_request_and_its_query() -> None:
    dialogue = _dialogue()
    c_interact.advance(dialogue, c_interact.Turn(kind="phase_1", text="Show the active ones."))
    dialogue.submissions.append(c_interact.Submission(phase=1, sql="SELECT wrong", reply="SQL failed Phase 1."))
    dialogue.submissions.append(c_interact.Submission(phase=1, sql="SELECT right", reply="Phase 1 correct!"))
    turn = c_interact.Turn(kind="follow_up", text="Now only for 2024.")

    assert c_interact.advance(dialogue, turn) == "Now only for 2024."
    evidence = c_interact.evidence(dialogue, turn)

    assert dialogue.phase == 2
    assert "This follows up on the user's earlier request: Show the active ones.\n" in evidence
    assert evidence.endswith("The query that answered it:\nSELECT right")


def test_a_debug_turn_before_any_request_is_refused() -> None:
    with pytest.raises(c_interact.CInteractTextError, match="before the phase's question"):
        c_interact.advance(_dialogue(), c_interact.Turn(kind="debug", text=DEBUG_WRONG_RESULT_MESSAGE))


def test_no_questions_are_left_once_the_budget_is_spent() -> None:
    dialogue = _dialogue()
    dialogue.exchanges = [c_interact.Exchange(question="q", answer="a")] * 6

    assert "You may ask 0 more questions" in c_interact.asking_prompt(dialogue)
