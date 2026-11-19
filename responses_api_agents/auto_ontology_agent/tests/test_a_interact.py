# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BIRD-Interact's a-Interact with Auto Ontology's interactive pipeline, as auto-ontology-eval runs it."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.agents.testing.agent_session_conformance import check_agent_seed_rejected
from nemo_gym.core import EpisodeId, NeMoGymResponseCreateParamsNonStreaming, TaskId
from responses_api_agents.auto_ontology_agent import a_interact, episode_runner
from responses_api_agents.auto_ontology_agent import app as agent_app
from responses_api_agents.auto_ontology_agent.app import AutoOntologyAgent, AutoOntologySessionState
from responses_api_agents.auto_ontology_agent.tests.test_app import (
    C_INTERACT_GRANT,
    MODEL_URL,
    _agent,
    _FakeSession,
    _seed,
    staged,  # noqa: F401 (a fixture)
)
from responses_api_agents.auto_ontology_agent.tests.test_interactive import (
    AskUserAction,
    SubmitSQLAction,
    _fake_auto_ontology,
)


# The bird_interact server's a-Interact request (environment_servers/bird_interact/app.py, A_INTERACT_INSTRUCTIONS).
INSTRUCTIONS = (
    "You are a data analyst who writes PostgreSQL queries for a user. ... Each tool call costs BIRD-COIN; your budget is "
    "18.0. Each tool result shows the coins you have left. When they run out, you can only submit."
)
MESSAGE = "Database: gym_members\n\nMy request:\nShow the active members."
FOLLOW_UP_REPLY = (
    "Phase 1 correct! (Reward: 0.7). Moving to Phase 2.\nReward: 0.7\nFollow-up question: Now count them by year."
)


def _request() -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": MESSAGE}]
    )


def test_the_request_names_the_database_the_request_and_the_budget() -> None:
    task = a_interact.read_request(INSTRUCTIONS, MESSAGE)

    assert task == a_interact.AInteractTask(database="gym_members", question="Show the active members.", budget=18.0)
    with pytest.raises(a_interact.AInteractTextError):
        a_interact.read_request(INSTRUCTIONS, "User Query:\nShow the active members.")


# ----- the budget, as the ADK's callbacks charge it ----- #


@pytest.mark.parametrize(
    ("remaining", "tool", "reserve", "charged", "after"),
    [
        (16.0, "execute", 4.0, True, 15.0),
        (5.0, "execute", 4.0, True, 4.0),
        # Phase 1's queries and questions leave its floor.
        (4.0, "execute", 4.0, False, 4.0),
        (5.0, "ask", 4.0, False, 5.0),
        (2.0, "ask", 0.0, True, 0.0),
        (1.0, "ask", 0.0, False, 1.0),
        # A submission is never refused, ignores the floor, and one that ends at zero or below leaves the budget spent.
        (10.0, "submit", 4.0, True, 7.0),
        (3.0, "submit", 0.0, True, -1.0),
        (2.0, "submit", 0.0, True, -1.0),
    ],
)
def test_the_budget_charges_as_the_adk_does(
    remaining: float, tool: str, reserve: float, charged: bool, after: float
) -> None:
    budget = episode_runner.Budget(initial=18.0, remaining=remaining)

    assert budget.charge(tool, reserve=reserve) is charged
    assert budget.remaining == after
    assert budget.exhausted is (after < 0)


@pytest.mark.parametrize(("remaining", "questions"), [(15.0, 5), (13.0, 4), (6.0, 1), (5.0, 0), (3.0, 0)])
def test_phase_1_asks_while_a_submission_and_its_floor_stay_paid(remaining: float, questions: int) -> None:
    assert episode_runner.Budget(18.0, remaining).affordable_questions(episode_runner.PHASE_1_FLOOR) == questions


# ----- Auto Ontology's queries ----- #


class _Database:
    database_name = "gym_members"
    dialect = "postgres"

    def __init__(self) -> None:
        self.ran: list[str] = []

    def execute(self, sql: str) -> list[tuple]:
        self.ran.append(sql)
        return [(1,)]


def _charged(
    monkeypatch: pytest.MonkeyPatch, remaining: float, reply: str = "1 rows\n(1,)", phase: int = 1
) -> tuple[episode_runner.ChargedConnector, _Database, list[dict], list[dict]]:
    sent: list[dict] = []

    def call(endpoint: dict, arguments: dict) -> dict:
        sent.append({"url": endpoint["url"], **arguments})
        return {"observation": reply, "coins_left": remaining - 1}

    monkeypatch.setattr(episode_runner, "call_charged_tool", call)
    database, calls = _Database(), []
    budget = episode_runner.Budget(18.0, remaining)
    connector = episode_runner.ChargedConnector(database, budget, {"url": "http://env/execute"}, calls, lambda: phase)
    return connector, database, sent, calls


def test_a_query_is_charged_by_the_server_then_read_from_the_sandbox_s_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector, database, sent, calls = _charged(monkeypatch, 10.0)

    # The whole result, as auto-ontology-eval's connector returns it; the server's reply only shows a preview.
    assert connector.execute("SELECT 1") == [(1,)]
    assert sent == [{"url": "http://env/execute", "sql": "SELECT 1"}]
    assert database.ran == ["SELECT 1"]
    assert calls == [{"name": "execute", "arguments": {"sql": "SELECT 1"}, "charged": True, "coins_left": 9.0}]
    # Other attributes are the connector's own, and free.
    assert (connector.database_name, connector.dialect) == ("gym_members", "postgres")


def test_a_query_the_budget_cannot_pay_for_runs_nowhere(monkeypatch: pytest.MonkeyPatch) -> None:
    connector, database, sent, calls = _charged(monkeypatch, 4.0)

    assert not connector.can_run_queries()
    with pytest.raises(episode_runner.BudgetExhausted):
        connector.execute("SELECT 1")
    assert (sent, database.ran) == ([], [])
    assert calls[0]["charged"] is False


def test_a_query_the_server_refuses_does_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    connector, database, _, _ = _charged(monkeypatch, 10.0, reply=episode_runner.BUDGET_SPENT_REPLY)

    with pytest.raises(episode_runner.BudgetExhausted):
        connector.execute("SELECT 1")
    assert database.ran == []


def test_the_react_agent_is_told_what_its_queries_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    phase_1, *_ = _charged(monkeypatch, 12.0)
    phase_2, *_ = _charged(monkeypatch, 5.0, phase=2)
    phase_2.reserve = 0.0
    spent, *_ = _charged(monkeypatch, 4.0)

    assert phase_1.cost_status().startswith("BUDGET: 12 coins left. run_sql costs 1 coin.")
    assert "submit with at least 6 coins unspent" in phase_1.cost_status()
    assert phase_2.cost_status().startswith("BUDGET: 5 coins left for this follow-up.")
    assert spent.cost_status() == "BUDGET: no coins left for run_sql. Submit your best query now."


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ("SQL failed Phase 1. Error executing submitted SQL: boom", "Error executing submitted SQL: boom"),
        ("SQL failed Phase 2. Submitted SQL execution timed out", "Submitted SQL execution timed out"),
        (
            "SQL failed Phase 1. Error executing submitted SQL: boom\nYour budget is spent, so the task ends here.",
            "Error executing submitted SQL: boom",
        ),
        ("SQL failed Phase 1. Your SQL is not correct.", None),
        (FOLLOW_UP_REPLY, None),
    ],
)
def test_a_failed_submission_s_error_is_the_grader_s_when_the_query_did_not_run(reply: str, error: str | None) -> None:
    assert episode_runner.a_interact_execution_error(reply) == error


def test_the_follow_up_question_gets_auto_ontology_eval_s_instructions() -> None:
    question = episode_runner.follow_up_question(FOLLOW_UP_REPLY)

    assert question == f"Now count them by year.\n\n{episode_runner.FOLLOW_UP_GUIDANCE}"
    assert episode_runner.follow_up_question("Phase 1 correct! (Reward: 0.7). Task finished.\nReward: 0.7") is None


# ----- the task ----- #


def _task() -> dict[str, Any]:
    return {
        "pipeline": "interactive",
        "mode": "a_interact",
        "database": "gym_members",
        "question": "Show the active members.",
        "schema": "CREATE TABLE members (id integer);",
        "knowledge": "- Active",
        "knowledge_children": {},
        # 18 at the start, 15 after the schema, the column meanings and the knowledge base.
        "budget": {"initial": 18.0, "remaining": 15.0},
        "tools": {name: {"url": f"http://env/{name}"} for name in ("ask", "submit", "execute")},
    }


def _server(monkeypatch: pytest.MonkeyPatch, replies: list[str]) -> list[tuple[str, dict]]:
    sent: list[tuple[str, dict]] = []

    def call(endpoint: dict, arguments: dict) -> dict:
        sent.append((endpoint["url"].rsplit("/", 1)[1], arguments))
        return {"observation": replies.pop(0), "coins_left": None}

    monkeypatch.setattr(episode_runner, "call_charged_tool", call)
    return sent


def test_a_task_runs_phase_1_its_retry_and_the_follow_up_in_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _server(
        monkeypatch,
        [
            "Visits this year.",
            "SQL failed Phase 1. Error executing submitted SQL: boom",
            FOLLOW_UP_REPLY,
            "Phase 2 correct! (Reward: 0.3). Task finished.\nReward: 0.3",
        ],
    )
    created = _fake_auto_ontology(
        monkeypatch,
        [
            AskUserAction("What is active?"),
            SubmitSQLAction("SELECT x"),
            SubmitSQLAction("SELECT 1"),
            SubmitSQLAction("SELECT 2"),
        ],
    )

    result = episode_runner.run(_task())

    assert [name for name, _ in sent] == ["ask", "submit", "submit", "submit"]
    assert (result["reason"], result["sql"]) == ("done", "SELECT 2")
    # 15 - 2 for the question - 3 for each submission.
    assert result["budget"]["remaining"] == 4.0
    (session,) = created
    turns = [
        (turn_type, options["debug_error"], options["follow_up_question"]) for turn_type, options in session.steps
    ]
    assert turns == [
        ("initial", None, None),
        ("initial", None, None),
        ("debug", "Error executing submitted SQL: boom", None),
        ("follow_up", None, f"Now count them by year.\n\n{episode_runner.FOLLOW_UP_GUIDANCE}"),
    ]
    assert all(options["initial_question"] == "Show the active members." for _, options in session.steps)
    # Each phase-1 step lets the pipeline ask what the budget can still pay, keeping a submission and the floor: at
    # the retry, after the question and a submission, (10 - 4) // 2.
    assert session.max_clarify_turns == 3
    assert session.answers == ["Visits this year."]
    # The follow-up's queries keep no floor.
    assert session.connectors[0].reserve == 0.0


def test_a_retry_that_repeats_a_failed_query_ends_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _server(monkeypatch, ["SQL failed Phase 1. Your SQL is not correct."])
    _fake_auto_ontology(monkeypatch, [SubmitSQLAction("SELECT 1"), SubmitSQLAction("SELECT 1")])

    result = episode_runner.run(_task())

    assert result["reason"] == "repeated_sql"
    assert [name for name, _ in sent] == ["submit"]


def test_an_empty_submission_falls_back_to_the_last_generated_query(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _server(monkeypatch, ["Phase 1 correct! (Reward: 0.7). Task finished.\nReward: 0.7"])
    _fake_auto_ontology(monkeypatch, [SubmitSQLAction("  ")])

    def with_draft(original: Any) -> Any:
        def create_session(**kwargs: Any) -> Any:
            session = original(**kwargs)
            session.path_state = {"sql_code": "SELECT draft"}
            return session

        return create_session

    interactive = sys.modules["auto_ontology.retrieval.interactive"]
    monkeypatch.setattr(interactive, "create_session", with_draft(interactive.create_session))

    result = episode_runner.run(_task())

    assert sent == [("submit", {"sql": "SELECT draft"})]
    assert result["reason"] == "done"


def test_with_no_query_at_all_nothing_is_paid_for(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _server(monkeypatch, [])
    _fake_auto_ontology(monkeypatch, [SubmitSQLAction("")])

    result = episode_runner.run(_task())

    assert (result["reason"], sent) == ("empty_sql", [])


def test_a_submission_that_spends_the_budget_ends_the_task(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _server(monkeypatch, ["SQL failed Phase 1. Your SQL is not correct."])
    _fake_auto_ontology(monkeypatch, [SubmitSQLAction("SELECT 1"), SubmitSQLAction("SELECT 2")])
    task = _task()
    task["budget"]["remaining"] = 3.0

    result = episode_runner.run(task)

    assert result["reason"] == "budget_exhausted"
    assert [name for name, _ in sent] == ["submit"]


def test_after_the_user_fails_to_answer_the_pipeline_asks_no_more(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _server(
        monkeypatch,
        [episode_runner.USER_FAILED_REPLY, "Phase 1 correct! (Reward: 0.7). Task finished.\nReward: 0.7"],
    )
    created = _fake_auto_ontology(monkeypatch, [AskUserAction("q1"), AskUserAction("q2"), SubmitSQLAction("SELECT 1")])

    result = episode_runner.run(_task())

    # The second question is not sent: the pipeline is stepped again with no questions left, and submits.
    assert [name for name, _ in sent] == ["ask", "submit"]
    assert result["reason"] == "done"
    assert created[0].max_clarify_turns == len(created[0].clarify_history)


def test_the_non_reasoning_role_runs_on_its_own_model_server() -> None:
    task = {"model_base_url": MODEL_URL, "model": "openai/policy"}
    other = "http://10.0.0.1:9001/ng-rollout/rollout/v1"

    settings = episode_runner.episode_environment(
        {**task, "non_reasoning_model_base_url": other, "non_reasoning_model": "openai/non-reasoning"}
    )

    assert (settings["REASONING_ENDPOINT"], settings["REASONING_MODEL"]) == (MODEL_URL, "openai/policy")
    assert (settings["NON_REASONING_ENDPOINT"], settings["NON_REASONING_MODEL"]) == (other, "openai/non-reasoning")
    alone = episode_runner.episode_environment(task)
    assert (alone["NON_REASONING_ENDPOINT"], alone["NON_REASONING_MODEL"]) == (MODEL_URL, "openai/policy")


# ----- the Agent Server ----- #


def _a_interact_seed(**kwargs: Any) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="session",
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="bird_interact:a_interact", task_id="gym_members_1"),
        **{"tool_accesses": [C_INTERACT_GRANT], **kwargs},
    )


def test_the_interactive_pipeline_refuses_a_seed_with_a_task() -> None:
    reason = check_agent_seed_rejected(_agent(pipeline="interactive").setup_webserver(), _seed())

    assert "reads its task from its requests" in reason


def test_the_interactive_pipeline_refuses_an_a_interact_seed_without_the_server_s_grant() -> None:
    reason = check_agent_seed_rejected(
        _agent(pipeline="interactive").setup_webserver(), _a_interact_seed(tool_accesses=[])
    )

    assert "direct HTTP grant" in reason


class _Reply:
    def __init__(self, body: dict) -> None:
        self.status, self.ok, self._body = 200, True, json.dumps(body).encode()
        self.content = self

    async def read(self) -> bytes:
        return self._body


async def test_an_a_interact_activation_buys_the_task_s_context_then_runs_the_task_in_the_sandbox(
    monkeypatch: pytest.MonkeyPatch, staged: dict[str, str], tmp_path: Path
) -> None:
    (tmp_path / "gym_members").mkdir()
    (tmp_path / "gym_members" / "gym_members_kb.jsonl").write_text(
        json.dumps({"id": 1, "knowledge": "Active", "children_knowledge": [2]})
    )
    knowledge = [
        {"id": 1, "knowledge": "Active", "description": "Visited this year."},
        {"id": 2, "knowledge": "Churn", "description": "Members who left."},
    ]
    observations = {
        "get_schema": "CREATE TABLE members (id integer);",
        "get_all_column_meanings": "{}",
        "get_all_knowledge_definitions": json.dumps(knowledge, indent=2),
    }
    bought: list[tuple[str, dict]] = []

    async def request(method: str, url: str, **kwargs: Any) -> _Reply:
        name = url.rsplit("/", 1)[1]
        bought.append((name, kwargs))
        return _Reply({"observation": observations[name], "coins_left": 18.0 - len(bought)})

    async def raise_for_status(response: _Reply) -> None:
        pass

    monkeypatch.setattr(agent_app, "request", request)
    monkeypatch.setattr(agent_app, "raise_for_status", raise_for_status)
    monkeypatch.setattr(AutoOntologyAgent, "resolve_model_base_url", lambda *args, **kwargs: MODEL_URL)
    agent = _agent(pipeline="interactive", knowledge_dir=str(tmp_path))
    session = _FakeSession({})
    staged[f"{session.session_dir}/result.json"] = json.dumps(
        {"sql": "SELECT 1", "reason": "done", "tool_calls": 3, "submit_reply": "Phase 1 correct! (Reward: 0.7)."}
    )
    seed = _a_interact_seed()
    state = AutoOntologySessionState(
        request=seed,
        task=None,
        session=session,  # type: ignore[arg-type]
        catalog_port=34567,
        tool_access=agent.effective_tool_accesses(seed)[0],
    )

    response = await agent._run_episode(state, _request())

    # The schema, the column meanings and the knowledge base, in auto-ontology-eval's order, each called once.
    assert [name for name, _ in bought] == ["get_schema", "get_all_column_meanings", "get_all_knowledge_definitions"]
    assert all(options["_max_num_tries"] == 1 for _, options in bought)
    assert bought[0][1]["headers"] == C_INTERACT_GRANT["headers"]
    task = json.loads(staged[f"{session.session_dir}/task.json"])
    assert (task["mode"], task["database"], task["question"]) == (
        "a_interact",
        "gym_members",
        "Show the active members.",
    )
    assert task["schema"] == "CREATE TABLE members (id integer);"
    assert (
        task["knowledge"] == "- Active\n  Description: Visited this year.\n- Churn\n  Description: Members who left."
    )
    assert task["knowledge_children"] == {"Active": ["- Churn\n  Description: Members who left."]}
    assert task["budget"] == {"initial": 18.0, "remaining": 15.0}
    assert task["tools"]["execute"]["url"] == f"{C_INTERACT_GRANT['base_url']}/execute"
    assert "```sql\nSELECT 1\n```" in response.output[0].content[0].text
    assert state.invocations[0].stop_reason == "done"
