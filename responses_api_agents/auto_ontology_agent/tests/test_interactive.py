# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The agent's interactive pipeline mode: what a c-Interact turn gives the sandbox, and the runner's turn."""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from responses_api_agents.auto_ontology_agent import c_interact, episode_runner
from responses_api_agents.auto_ontology_agent.tests.adk_texts import C_INTERACT_INSTRUCTION
from responses_api_agents.auto_ontology_agent.tests.test_app import (
    C_INTERACT_GRANT,
    MODEL_URL,
    _agent,
    _c_interact_state,
    _first_turn,
    _turn,
    _TurnSession,
    staged,  # noqa: F401 (a fixture)
)


KNOWLEDGE = [
    {"id": 1, "knowledge": "Active", "description": "Visited this year.", "definition": "visits > 0"},
    {"id": 2, "knowledge": "Churn", "description": "Members who left."},
]


def test_the_instruction_gives_the_schema() -> None:
    instruction = C_INTERACT_INSTRUCTION.format(
        db_name="gym",
        db_schema="CREATE TABLE members (id integer);\n\nFirst 3 rows:\nid\n1",
        external_kg="[]",
        max_turn=3,
    )

    assert c_interact.read_task(instruction).schema == "CREATE TABLE members (id integer);\n\nFirst 3 rows:\nid\n1"


def test_a_debug_turn_gives_the_execution_error_only_when_the_sql_did_not_run() -> None:
    assert (
        c_interact.execution_error(
            c_interact.read_turn(
                "Your SQL is not executable: column x does not exist\nPlease fix and call submit_sql."
            )
        )
        == "column x does not exist"
    )
    wrong = "Your SQL is not correct. You have one more chance. Please fix and call submit_sql."
    assert c_interact.execution_error(c_interact.read_turn(wrong)) is None


def test_the_knowledge_is_written_as_auto_ontology_eval_writes_it_with_the_visible_children() -> None:
    # Entry 9 is hidden from the task, so Active's link to it is not given.
    text, children = c_interact.interactive_knowledge(KNOWLEDGE, {1: [2, 9], 2: [-1]})

    assert (
        text
        == "- Active\n  Description: Visited this year.\n  Definition: visits > 0\n- Churn\n  Description: Members who left."
    )
    assert children == {"Active": ["- Churn\n  Description: Members who left."]}


async def test_an_interactive_turn_gives_the_pipeline_the_task_and_keeps_its_state_in_the_session(
    monkeypatch: pytest.MonkeyPatch, staged: dict[str, str], tmp_path: Path
) -> None:
    (tmp_path / "gym_members").mkdir()
    (tmp_path / "gym_members" / "gym_members_kb.jsonl").write_text(
        "\n".join(json.dumps({"id": 1, "knowledge": "Active", "children_knowledge": [7]}) for _ in range(1))
    )
    monkeypatch.setattr(type(_agent()), "resolve_model_base_url", lambda *args, **kwargs: MODEL_URL)
    agent = _agent(pipeline="interactive", knowledge_dir=str(tmp_path))
    session = _TurnSession(
        [
            {"sql": "SELECT 1", "reason": "submitted", "tool_calls": 2, "environment_tool_calls": [],
             "submit_reply": "SQL failed Phase 1. Error executing submitted SQL: boom"},
            {"sql": "SELECT 2", "reason": "submitted", "tool_calls": 1, "environment_tool_calls": [],
             "submit_reply": "Phase 1 correct! (Reward: 0.5). Task finished."},
        ],
        staged,
    )  # fmt: skip
    state = _c_interact_state(agent, session)

    await agent._activate_agent_session_state(state, _first_turn(), activation_id=1)
    first = json.loads(staged[f"{session.root}/turn-1/task.json"])
    await agent._activate_agent_session_state(
        state, _turn("Your SQL is not executable: boom\nPlease fix and call submit_sql."), activation_id=2
    )
    second = json.loads(staged[f"{session.root}/turn-2/task.json"])

    assert first["pipeline"] == "interactive"
    assert (first["database"], first["schema"]) == ("gym_members", "CREATE TABLE members (id integer);")
    assert first["knowledge"] == "- Active\n  Description: Visited this year."
    assert first["knowledge_children"] == {}  # Entry 7 is not among the task's knowledge.
    assert (first["max_questions"], first["max_steps"]) == (5, 10)
    assert first["turn"] == {"kind": "phase_1", "text": "Show the active members.", "execution_error": None}
    assert first["submit"]["url"] == f"{C_INTERACT_GRANT['base_url']}/submit_sql"
    assert [tool["definition"]["function"]["name"] for tool in first["tools"]] == ["ask_user"]
    # Both turns keep the pipeline's session in the same file, outside the turns' own folders.
    assert first["state_path"] == second["state_path"] == f"{session.root}/interactive_state.pickle"
    assert second["turn"] == {"kind": "debug", "text": "Show the active members.", "execution_error": "boom"}


# ----- the runner, with a stand-in Auto Ontology ----- #


@dataclass
class _Session:
    db_name: str
    max_clarify_turns: int
    external_kb_children_map: dict
    answers: list[str] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    steps: list[tuple[str, dict]] = field(default_factory=list)
    data_retriever: Any = None
    semantic_retriever: Any = None
    connectors: list = field(default_factory=list)
    clarify_history: list = field(default_factory=list)
    path_state: dict = field(default_factory=dict)


class _Connector:
    database_name = "gym_members"


@dataclass
class AskUserAction:
    question: str


@dataclass
class SubmitSQLAction:
    sql: str


def _fake_auto_ontology(monkeypatch: pytest.MonkeyPatch, actions: list[Any]) -> list[_Session]:
    """Install a stand-in for the parts of Auto Ontology the runner imports; returns the sessions it creates.

    Its ``step`` returns ``actions`` in order.
    """
    created: list[_Session] = []

    def create_session(**kwargs: Any) -> _Session:
        session = _Session(kwargs["db_name"], kwargs["max_clarify_turns"], kwargs["external_kb_children_map"])
        for name in ("data_retriever", "semantic_retriever", "connectors"):
            setattr(session, name, kwargs[name])
        created.append(session)
        return session

    def step(session: _Session, *, turn_type: str, **options: Any) -> Any:
        assert session.connectors and session.data_retriever == "data", "each turn reattaches the connections"
        session.steps.append((turn_type, options))
        return actions.pop(0)

    interactive = types.SimpleNamespace(
        AskUserAction=AskUserAction,
        SubmitSQLAction=SubmitSQLAction,
        TurnType=types.SimpleNamespace(INITIAL="initial", DEBUG="debug", FOLLOW_UP="follow_up"),
        create_session=create_session,
        step=step,
        apply_user_answer=lambda session, answer: session.answers.append(answer),
        apply_submit_result=lambda session, result: session.results.append(result),
    )
    modules = {
        "auto_ontology": types.ModuleType("auto_ontology"),
        "auto_ontology.connectors": types.SimpleNamespace(get_connectors=lambda: [_Connector()]),
        "auto_ontology.retrieval": types.ModuleType("auto_ontology.retrieval"),
        "auto_ontology.retrieval.interactive": interactive,
        "auto_ontology.utils": types.SimpleNamespace(
            get_data_objects_retriever=lambda: "data", get_semantic_objects_retriever=lambda: None
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(episode_runner, "episode_environment", lambda task: {})
    return created


def _task(tmp_path: Path, kind: str, text: str, **turn: Any) -> dict[str, Any]:
    ask = {"definition": {"type": "function", "function": {"name": "ask_user"}}, "url": "http://env/ask_user"}
    return {
        "pipeline": "interactive",
        "database": "gym_members",
        "schema": "CREATE TABLE members (id integer);",
        "knowledge": "- Active",
        "knowledge_children": {"Active": ["- Churn"]},
        "max_questions": 3,
        "max_steps": 4,
        "turn": {"kind": kind, "text": text, "execution_error": None, **turn},
        "state_path": str(tmp_path / "interactive_state.pickle"),
        "tools": [ask],
        "submit": {"url": "http://env/submit_sql"},
    }


def test_an_interactive_turn_asks_until_it_submits_and_the_next_turn_resumes_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    replies = iter(
        [
            "Visits this year.\n\n[SYSTEM NOTE: Clarification turns remaining: 2/3]",
            "SQL failed Phase 1. Your SQL is not correct.",
            "Phase 1 correct! (Reward: 0.5). Task finished.",
        ]
    )
    sent: list[tuple[str, dict]] = []

    def call_tool(endpoint: dict, arguments: dict) -> str:
        sent.append((endpoint["url"], arguments))
        return next(replies)

    monkeypatch.setattr(episode_runner, "call_tool", call_tool)
    actions = [AskUserAction("What is active?"), SubmitSQLAction("SELECT 1")]
    created = _fake_auto_ontology(monkeypatch, actions)

    first = episode_runner.run(_task(tmp_path, "phase_1", "Show the active members."))

    assert (first["reason"], first["sql"], first["tool_calls"]) == ("submitted", "SELECT 1", 2)
    assert first["environment_tool_calls"][0]["arguments"] == {"question": "What is active?"}
    assert sent == [
        ("http://env/ask_user", {"question": "What is active?"}),
        ("http://env/submit_sql", {"sql": "SELECT 1"}),
    ]
    (session,) = created
    assert session.steps[0] == ("initial", {"initial_question": "Show the active members."})
    assert session.answers == ["Visits this year."]
    assert session.results == [{"reward": 0.0, "phase_completed": None, "message": first["submit_reply"]}]
    assert (session.max_clarify_turns, session.external_kb_children_map) == (3, {"Active": ["- Churn"]})

    actions.append(SubmitSQLAction("SELECT 2"))
    second = episode_runner.run(_task(tmp_path, "debug", "Show the active members.", execution_error="boom"))

    # The debug turn resumed the saved session: no new one was created.
    assert len(created) == 1
    assert (second["sql"], second["submit_reply"]) == ("SELECT 2", "Phase 1 correct! (Reward: 0.5). Task finished.")


def test_a_turn_that_never_submits_stops_at_its_step_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(episode_runner, "call_tool", lambda endpoint, arguments: "I don't know.")
    _fake_auto_ontology(monkeypatch, [AskUserAction("q")] * 10)

    result = episode_runner.run(_task(tmp_path, "phase_1", "Show the active members."))

    assert (result["reason"], result["sql"], result["tool_calls"]) == ("step_limit", None, 4)
    assert result["submit_reply"] is None


@pytest.mark.parametrize(
    ("reply", "result"),
    [
        ("Phase 1 correct! (Reward: 0.7). Moving to Phase 2.", {"reward": 0.7, "phase_completed": 1}),
        ("Phase 2 correct! (Reward: 0.2). Task finished.", {"reward": 0.2, "phase_completed": 2}),
        ("SQL failed Phase 1. Your SQL is not correct.", {"reward": 0.0, "phase_completed": None}),
    ],
)
def test_a_submission_reply_reads_as_the_pipeline_reads_a_result(reply: str, result: dict) -> None:
    assert episode_runner.submission_result(reply) == {**result, "message": reply}


def test_the_user_answer_drops_the_server_note() -> None:
    assert episode_runner.user_answer("Yes.\n\n[SYSTEM NOTE: Clarification turns remaining: 2/5]") == "Yes."


def test_the_database_qualifier_is_dropped_and_sql_that_does_not_parse_is_kept() -> None:
    pytest.importorskip("sqlglot")
    assert episode_runner.without_database_qualifier("SELECT a FROM gym.public.t") == "SELECT a FROM public.t"
    assert episode_runner.without_database_qualifier("SELECT a FROM public.t") == "SELECT a FROM public.t"
    assert episode_runner.without_database_qualifier("SELEC (") == "SELEC ("
