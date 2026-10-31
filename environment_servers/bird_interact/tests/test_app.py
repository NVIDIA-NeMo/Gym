# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD-Interact episodes end to end: a scripted agent and user model, and real database sandboxes on the host."""

import asyncio
import importlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx
import orjson
import pytest
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, ValidationError

from environment_servers.bird_interact.app import (
    SANDBOX_FAILED,
    USER_SIMULATOR_FAILED,
    BirdInteractEnvironmentServer,
    BirdInteractEnvironmentServerConfig,
    BirdInteractRequest,
    BirdInteractResponse,
    extract_sql,
)
from environment_servers.bird_interact.grading import grade, main, split_statements, strip_round_calls
from environment_servers.bird_interact.simulator import UNANSWERABLE_REPLY, parse_action
from environment_servers.bird_interact.task_data import BirdInteractTask
from nemo_gym.config_types import AgentServerRef, ModelServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient


DATA = Path(__file__).parents[1] / "data"
TASKS = {row["instance_id"]: row for row in map(json.loads, (DATA / "example.jsonl").read_text().splitlines())}
CORRECT_1 = TASKS["bookshop_1"]["sol_sql"][0]
WRONG = "SELECT name FROM authors"
# Episode sandboxes run on the host; each test gets its own workspace (see `sandbox_workspace`).
LOCAL_SANDBOX: dict[str, dict[str, str]] = {"local": {}}


@pytest.fixture(autouse=True)
def sandbox_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "sandboxes"
    monkeypatch.setitem(LOCAL_SANDBOX["local"], "workspace_root", str(workspace))
    return workspace


def _sql(sql: str) -> str:
    return f"Here is my query.\n```sqlite\n{sql}\n```"


def _sandboxes(workspace: Path) -> list[Path]:
    """The episode directories in the local sandboxes, which hold the runner and the database."""
    return sorted(path.parent for path in workspace.rglob("bird_interact/grading.py"))


def _message(text: str) -> dict:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=False,
        tools=[],
    ).model_dump(mode="json")


class _Content:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def read(self) -> bytes:
        return self.body


class _HTTPResponse:
    def __init__(self, body: dict) -> None:
        self.status = 200
        self.ok = True
        self.body = orjson.dumps(body)
        self.content = _Content(self.body)
        self.cookies: dict[str, Any] = {}
        self.headers: dict[str, str] = {}

    async def read(self) -> bytes:
        return self.body

    def raise_for_status(self) -> None:
        pass


class _Tools:
    """The agent's side of its seed's tool grants: it calls the episode's tools on the server's app."""

    def __init__(self, app: FastAPI, grants: list[dict]) -> None:
        self.app = app
        self.grants = {grant["kind"]: grant for grant in grants}

    async def call(self, name: str, **arguments: str) -> httpx.Response:
        grant = self.grants["direct_http"]
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url=grant["base_url"]) as http:
            return await http.post(f"/{name}", json=arguments, headers=grant["headers"])

    async def observe(self, name: str, **arguments: str) -> dict:
        reply = await self.call(name, **arguments)
        assert reply.status_code == 200, reply.text
        return reply.json()

    @asynccontextmanager
    async def mcp(self) -> AsyncIterator[ClientSession]:
        # Each SDK's real HTTP transport against the app, without a network listener: MCP 2 uses httpx2, MCP 1 httpx.
        http = importlib.import_module("httpx2" if version("mcp").startswith("2.") else "httpx")
        connection = self.grants["mcp"]["connection"]
        async with http.AsyncClient(
            transport=http.ASGITransport(app=self.app), headers=connection["headers"]
        ) as http_client:
            async with streamable_http_client(connection["url"], http_client=http_client) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    yield session


Turn = str | Callable[[_Tools], Awaitable[str]]


class _Client(ServerClient):
    """The agent and the user model: each agent turn and each user-model reply comes from a script."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_turns: list[Turn]
    user_replies: list[str | BaseException]
    app: Any = None
    seeds: list[dict] = []
    agent_inputs: list[dict] = []
    user_prompts: list[str] = []

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _HTTPResponse:
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            body = orjson.loads(orjson.dumps(body.model_dump(mode="json", exclude_unset=True)))
        if server_name == "user_model":
            self.user_prompts.append(body["input"][0]["content"])
            reply = self.user_replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return _HTTPResponse(_message(reply))
        if url_path == "/v1/agent_sessions":
            self.seeds.append(body)
            seeded = {"agent_session_id": body["agent_session_id"]}
            if body.get("continuation") is not None:
                seeded["capabilities"] = {"mode": "resume_harness_session", "observations": ["trajectory"]}
            return _HTTPResponse(seeded)
        if url_path == "/v1/agent_sessions/close":
            return _HTTPResponse({"agent_session_id": body["agent_session_id"]})
        self.agent_inputs.append(body)
        turn = self.agent_turns.pop(0)
        text = turn if isinstance(turn, str) else await turn(_Tools(self.app, self.seeds[-1]["tool_accesses"]))
        return _HTTPResponse(_message(text))

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:1"


def _server(
    mode: str = "c_interact", turns: list[Turn] = (), user: list[str | BaseException] = (), **config: Any
) -> tuple[BirdInteractEnvironmentServer, _Client]:
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(
            {"agent": {"responses_api_agents": {"agent": {"host": "agent", "port": 1, "entrypoint": "app.py"}}}}
        ),
        agent_turns=list(turns),
        user_replies=list(user),
        seeds=[],
        agent_inputs=[],
        user_prompts=[],
    )
    server = BirdInteractEnvironmentServer(
        config=BirdInteractEnvironmentServerConfig(
            name="bird_interact",
            host="bird",
            port=1,
            entrypoint="app.py",
            mode=mode,
            agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
            user_model=ModelServerRef(type="responses_api_models", name="user_model"),
            database_dir=str(DATA / "example_databases"),
            turn_timeout_seconds=60,
            cleanup_timeout_seconds=30,
            **({"sandbox_provider": LOCAL_SANDBOX, "max_turns": 20 if mode == "c_interact" else 1} | config),
        ),
        server_client=client,
    )
    client.app = server.setup_webserver()
    return server, client


async def _run(server: BirdInteractEnvironmentServer, task: str = "bookshop_1", **row: Any) -> BirdInteractResponse:
    request = BirdInteractRequest(
        episode_id=EpisodeId(rollout_id=f"{task}-0"),
        task=MaterializedTask(
            task_id=TaskId(taskset="bird_interact:example", task_id=task),
            task_input=BirdInteractTask.model_validate(TASKS[task] | row),
        ),
    )
    return await server.run_request(request)


def _phases(response: BirdInteractResponse) -> list[dict]:
    return [phase.model_dump() for phase in response.result.phases]


class TestGrading:
    def test_round_calls_are_removed_with_their_nested_arguments(self) -> None:
        sql = "SELECT ROUND(AVG(ROUND(x, 1)), 2), round( y ) FROM t"
        assert strip_round_calls(sql) == "SELECT AVG(x), y FROM t"

    def test_a_semicolon_inside_a_string_does_not_split_the_statement(self) -> None:
        assert split_statements("SELECT 'a;b'; SELECT 2") == ["SELECT 'a;b';", "SELECT 2;"]

    def test_unordered_results_compare_as_sets_and_ordered_results_by_position(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        reversed_sql = CORRECT_1.replace("DESC", "ASC")
        assert grade(database, reversed_sql, CORRECT_1, ordered=False, timeout_seconds=5).passed
        assert not grade(database, reversed_sql, CORRECT_1, ordered=True, timeout_seconds=5).passed

    def test_float_results_match_to_two_decimal_places(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        solution = TASKS["bookshop_2"]["sol_sql"][0]
        rounded = solution.replace(
            "AVG(1 - s.unit_price / b.list_price)", "ROUND(AVG(1 - s.unit_price / b.list_price), 2)"
        )
        assert grade(database, rounded, solution, ordered=False, timeout_seconds=5).passed

    def test_an_error_is_reported_and_an_empty_result_fails(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        broken = grade(database, "SELECT nope FROM books", CORRECT_1, ordered=False, timeout_seconds=5)
        empty = grade(database, "SELECT name FROM authors WHERE 0", CORRECT_1, ordered=False, timeout_seconds=5)
        assert broken.execution_error == "no such column: nope"
        assert not empty.passed and empty.execution_error is None

    def test_a_submission_that_writes_leaves_the_database_unchanged(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        grade(database, "DELETE FROM authors; SELECT 1", CORRECT_1, ordered=False, timeout_seconds=5)
        assert grade(database, CORRECT_1, CORRECT_1, ordered=True, timeout_seconds=5).passed

    def test_the_sandbox_runner_sets_up_the_database_and_reports_a_broken_solution(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def run(command: str, request: dict) -> dict:
            request_path = tmp_path / f"{command}.json"
            request_path.write_text(json.dumps(request | {"database": str(tmp_path / "episode.sqlite")}))
            main([command, str(request_path)])
            assert not request_path.exists()  # A grade request holds the solution SQL.
            return json.loads(capsys.readouterr().out)

        source = str(DATA / "example_databases/bookshop/bookshop.sql")
        assert run("setup", {"source": source, "preprocess_sql": ["DELETE FROM authors WHERE author_id > 1"]}) == {}
        executed = run("execute", {"sql": "SELECT COUNT(*) FROM authors", "timeout_seconds": 5})
        graded = run(
            "grade",
            {"predicted_sql": CORRECT_1, "solution_sql": "SELECT nope", "ordered": False, "timeout_seconds": 5},
        )

        assert executed == {"error": None, "row_count": 1, "preview": ["(1,)"]}
        assert graded == {"solution_error": "The ground-truth SQL failed: no such column: nope"}


def _bookshop(tmp_path: Path) -> Path:
    import sqlite3

    database = tmp_path / "bookshop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript((DATA / "example_databases/bookshop/bookshop.sql").read_text())
    return database


class TestParsing:
    def test_a_term_that_is_not_annotated_is_unanswerable(self) -> None:
        task = BirdInteractTask.model_validate(TASKS["bookshop_1"])
        assert parse_action('labeled("bestsellers")', task).kind == "labeled"
        assert parse_action('labeled("price")', task).kind == "unanswerable"
        assert parse_action("I think it is unlabeled", task).kind == "unanswerable"
        assert parse_action('unlabeled("ORDER BY bestsellers DESC")', task).argument == "ORDER BY bestsellers DESC"

    def test_the_last_sql_block_of_a_reply_is_the_submission(self) -> None:
        assert extract_sql("```sql\nSELECT 1\n```\nor rather\n```SQLite\nSELECT 2\n```") == "SELECT 2"
        assert extract_sql("Which year do you mean?") is None
        assert extract_sql("```sql\n\n```") is None


class TestConfig:
    def test_a_interact_runs_in_one_turn(self) -> None:
        with pytest.raises(ValidationError, match="max_turns to 1"):
            _server("a_interact", max_turns=2)


class TestConversationalMode:
    async def test_the_first_request_shows_the_schema_and_hides_the_knowledge_the_task_masks(self) -> None:
        server, client = _server(turns=[_sql(CORRECT_1)])

        response = await _run(server)

        assert response.failure is None
        assert (response.result.reward, response.result.stop_reason) == (0.7, "solved")
        system, user = (message["content"] for message in client.agent_inputs[0]["input"])
        # One critical and one knowledge ambiguity, plus the default patience of 3.
        assert "at most 5 in total" in system
        assert "CREATE TABLE" in user and "Revenue" in user and "Bestseller" not in user
        assert client.agent_inputs[0]["tools"] == []
        # The agent gets no tools in c-Interact, and continues its session across turns.
        assert client.seeds[0]["tool_accesses"] == [] and client.seeds[0]["continuation"] is not None

    async def test_a_question_goes_through_both_simulator_stages(self) -> None:
        server, client = _server(
            turns=["What counts as a bestseller?", _sql(CORRECT_1)],
            user=['labeled("bestsellers")', "At least 10 copies."],
        )

        response = await _run(server)

        assert client.agent_inputs[1]["input"] == [{"role": "user", "content": "At least 10 copies."}]
        assert "HAVING SUM(s.quantity) >= 10" in client.user_prompts[1]
        assert "Bestseller" in client.user_prompts[1]  # The masked knowledge entry backs the answer.
        assert (response.result.questions_asked, response.result.reward) == (1, 0.7)

    async def test_an_unanswerable_question_gets_a_refusal_without_a_second_model_call(self) -> None:
        server, client = _server(turns=["What is the price?", _sql(CORRECT_1)], user=['labeled("price")'])

        await _run(server)

        assert client.agent_inputs[1]["input"][0]["content"] == UNANSWERABLE_REPLY
        assert len(client.user_prompts) == 1

    async def test_a_question_after_the_budget_is_spent_ends_the_episode(self) -> None:
        server, client = _server(
            turns=["First?", "Second?", "Third?"], user=["unanswerable()", "unanswerable()"], question_patience=0
        )

        response = await _run(server)

        assert "That was your last question" in client.agent_inputs[2]["input"][0]["content"]
        assert (response.result.stop_reason, response.result.questions_asked) == ("no_sql_submitted", 2)
        assert response.result.reward == 0.0

    async def test_the_debug_turn_after_a_failed_submission_takes_no_question(self) -> None:
        server, client = _server(turns=[_sql(WRONG), "Did you mean all authors?"])

        response = await _run(server)

        assert "Send your corrected SQL" in client.agent_inputs[1]["input"][0]["content"]
        assert response.result.stop_reason == "no_sql_submitted"
        assert _phases(response) == [{"phase": 1, "passed": False, "attempts": 1, "reward": 0.0}]

    async def test_a_debugged_submission_earns_the_debug_reward(self) -> None:
        server, _ = _server(turns=[_sql(WRONG), _sql(CORRECT_1)])

        response = await _run(server)

        assert response.result.reward == 0.5
        assert _phases(response) == [{"phase": 1, "passed": True, "attempts": 2, "reward": 0.5}]

    async def test_two_failed_submissions_end_the_task(self) -> None:
        server, _ = _server(turns=[_sql(WRONG), _sql(WRONG)])

        response = await _run(server)

        assert (response.result.stop_reason, response.result.reward, response.result.finished) == ("failed", 0.0, True)

    async def test_the_follow_up_question_is_a_second_phase(self) -> None:
        task = TASKS["bookshop_3"]
        server, client = _server(turns=[_sql(task["sol_sql"][0]), _sql(task["follow_up"]["sol_sql"][0])])

        response = await _run(server, "bookshop_3")

        assert task["follow_up"]["query"] in client.agent_inputs[1]["input"][0]["content"]
        assert response.result.reward == pytest.approx(1.0)
        assert [phase["phase"] for phase in _phases(response)] == [1, 2]

    async def test_a_single_phase_reward_schedule_skips_the_follow_up(self) -> None:
        server, client = _server(
            turns=[_sql(TASKS["bookshop_3"]["sol_sql"][0])], phase_rewards=[{"first_attempt": 1.0, "after_debug": 0.5}]
        )

        response = await _run(server, "bookshop_3")

        assert (response.result.reward, response.result.stop_reason, len(client.agent_inputs)) == (1.0, "solved", 1)


class TestAgenticMode:
    async def test_the_agent_calls_the_tools_over_its_grant_and_each_call_is_charged(self) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            seen["names"] = await tools.observe("get_all_external_knowledge_names")
            seen["meaning"] = await tools.observe("get_column_meaning", table_name="SALES", column_name="quantity")
            seen["submitted"] = await tools.observe("submit", sql=CORRECT_1)
            return "Done."

        server, client = _server("a_interact", turns=[agent])

        response = await _run(server)

        # 6 + 2 for each of the two ambiguities + 6 patience = 16; the two lookups cost 0.5 each and submit 3.
        assert seen["meaning"] == {"observation": "Number of copies sold in the sale.", "coins_left": 15.0}
        assert "Bestseller" not in seen["names"]["observation"]
        assert "That's right" in seen["submitted"]["observation"]
        assert (response.result.reward, response.result.coins_spent) == (0.7, 4.0)
        assert response.result.stop_reason == "agent_finished"
        assert [call.name for call in response.result.tool_calls] == [
            "get_all_external_knowledge_names",
            "get_column_meaning",
            "submit",
        ]
        (grant,) = client.seeds[0]["tool_accesses"]
        assert grant["kind"] == "direct_http" and "continuation" not in client.seeds[0]

    async def test_the_first_request_declares_the_nine_tools_and_leaves_out_what_they_charge_for(self) -> None:
        server, client = _server("a_interact", turns=["Done."])

        await _run(server)

        request = client.agent_inputs[0]
        assert "your budget is 16.0" in request["input"][0]["content"]
        assert "CREATE TABLE" not in request["input"][1]["content"]
        assert {tool["name"] for tool in request["tools"]} == {
            "ask",
            "submit",
            "execute",
            "get_schema",
            "get_all_column_meanings",
            "get_column_meaning",
            "get_all_external_knowledge_names",
            "get_knowledge_definition",
            "get_all_knowledge_definitions",
        }
        column_meaning = next(tool for tool in request["tools"] if tool["name"] == "get_column_meaning")
        assert column_meaning["strict"] is True
        assert column_meaning["parameters"] == {
            "type": "object",
            "properties": {
                "table_name": {"description": "The table.", "type": "string"},
                "column_name": {"description": "The column.", "type": "string"},
            },
            "required": ["table_name", "column_name"],
            "additionalProperties": False,
        }

    async def test_over_mcp_alone_the_request_declares_no_tools_and_mcp_serves_the_nine(self) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            async with tools.mcp() as session:
                seen["tools"] = {tool.name for tool in (await session.list_tools()).tools}
                result = await session.call_tool("submit", {"sql": CORRECT_1})
                seen["submitted"] = json.loads(result.content[0].text)
            return "Done."

        server, client = _server("a_interact", turns=[agent], tool_transports=["mcp"])

        async with client.app.router.lifespan_context(client.app):
            response = await _run(server)

        assert client.agent_inputs[0]["tools"] == []
        assert len(seen["tools"]) == 9 and "submit" in seen["tools"]
        assert "That's right" in seen["submitted"]["observation"]
        assert response.result.reward == 0.7

    async def test_once_the_budget_is_spent_only_submit_runs(self) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            for _ in range(10):
                await tools.observe("execute", sql="SELECT 1")
            seen["refused"] = await tools.observe("get_schema")
            seen["submitted"] = await tools.observe("submit", sql=CORRECT_1)
            return "Done."

        server, _ = _server("a_interact", turns=[agent], coin_patience=0)

        response = await _run(server)

        assert seen["refused"]["observation"] == "Your budget is spent. Only submit is allowed now."
        assert "That's right" in seen["submitted"]["observation"]
        assert (response.result.reward, response.result.finished) == (0.7, True)

    async def test_a_failed_submit_with_no_budget_left_ends_the_task_and_records_the_phase(self) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            for _ in range(10):
                await tools.observe("execute", sql="SELECT 1")
            seen["submitted"] = await tools.observe("submit", sql=WRONG)
            return "Done."

        server, _ = _server("a_interact", turns=[agent], coin_patience=0)

        response = await _run(server)

        assert "the task ends here" in seen["submitted"]["observation"]
        assert response.result.finished is True
        assert _phases(response) == [{"phase": 1, "passed": False, "attempts": 1, "reward": 0.0}]

    async def test_an_exploratory_query_returns_a_bounded_preview(self) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            seen["result"] = await tools.observe("execute", sql="SELECT s.sale_id FROM sales s, authors a")
            return "Done."

        server, _ = _server("a_interact", turns=[agent])

        await _run(server)

        assert seen["result"]["observation"].startswith("95 rows\n(1,)")
        assert seen["result"]["observation"].endswith("... 75 more rows")

    async def test_a_simulator_failure_masks_the_sample(self) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            seen["status"] = (await tools.call("ask", question="?")).status_code
            return "Done."

        server, _ = _server("a_interact", turns=[agent], user=[RuntimeError("user model down")])

        response = await _run(server)

        assert seen["status"] == 500
        assert response.failure is None
        assert (response.result.mask_sample, response.result.failure_kind) == (True, USER_SIMULATOR_FAILED)


class TestDatabaseSandbox:
    async def test_each_episode_has_its_own_sandbox_and_stops_it_when_the_episode_ends(
        self, sandbox_workspace: Path
    ) -> None:
        during: list[list[Path]] = []

        def agent(sql: str) -> Callable[[_Tools], Awaitable[str]]:
            async def turn(tools: _Tools) -> str:
                await asyncio.sleep(0.2)  # Both episodes are running.
                during.append([path for path in _sandboxes(sandbox_workspace) if (path / "episode.sqlite").exists()])
                return _sql(sql)

            return turn

        first, _ = _server(turns=[agent(CORRECT_1)])
        second, _ = _server(turns=[agent(TASKS["bookshop_2"]["sol_sql"][0])])

        await asyncio.gather(_run(first), _run(second, "bookshop_2"))

        assert [len(sandboxes) for sandboxes in during] == [2, 2]
        assert _sandboxes(sandbox_workspace) == []

    async def test_no_request_stays_in_the_sandbox(self, sandbox_workspace: Path) -> None:
        left: list[Path] = []

        async def after_a_submission(tools: _Tools) -> str:
            (sandbox,) = _sandboxes(sandbox_workspace)
            left.extend(sandbox.glob("request-*.json"))
            return _sql(CORRECT_1)

        server, _ = _server(turns=[_sql(WRONG), after_a_submission])

        await _run(server)

        # The grade request held the solution SQL.
        assert left == []

    async def test_preprocess_sql_runs_in_the_sandbox_before_the_episode(self) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            seen["result"] = await tools.observe("execute", sql="SELECT COUNT(*) FROM sales")
            return "Done."

        server, _ = _server("a_interact", turns=[agent])

        await _run(server, preprocess_sql=["DELETE FROM sales"])

        assert seen["result"]["observation"] == "1 rows\n(0,)"

    async def test_a_command_that_fails_runs_again_on_a_fresh_sandbox(self, sandbox_workspace: Path) -> None:
        broken: list[Path] = []

        async def break_the_sandbox(tools: _Tools) -> str:
            (sandbox,) = _sandboxes(sandbox_workspace)
            (sandbox / "grading.py").unlink()
            broken.append(sandbox)
            return _sql(CORRECT_1)

        server, _ = _server(turns=[break_the_sandbox])

        response = await _run(server)

        assert response.result.reward == 0.7
        assert not broken[0].parent.exists()

    async def test_a_sandbox_that_keeps_failing_fails_a_c_interact_episode_for_a_retry(
        self, sandbox_workspace: Path
    ) -> None:
        async def break_the_sandbox(tools: _Tools) -> str:
            (sandbox,) = _sandboxes(sandbox_workspace)
            (sandbox / "grading.py").unlink()
            return _sql(CORRECT_1)

        server, _ = _server(turns=[break_the_sandbox], inconclusive_verification_retries=0)

        response = await _run(server)

        assert (response.failure.stage, response.failure.terminal) == ("verification", False)
        assert "grading.py grade failed in the sandbox" in response.failure.failure_reason
        assert len(response.failure.turns) == 1
        assert _sandboxes(sandbox_workspace) == []

    async def test_a_sandbox_failure_masks_an_a_interact_sample(self, sandbox_workspace: Path) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            (sandbox,) = _sandboxes(sandbox_workspace)
            (sandbox / "grading.py").unlink()
            seen["status"] = (await tools.call("execute", sql="SELECT 1")).status_code
            return "Done."

        server, _ = _server("a_interact", turns=[agent], inconclusive_verification_retries=0)

        response = await _run(server)

        assert seen["status"] == 500
        assert (response.result.mask_sample, response.result.failure_kind) == (True, SANDBOX_FAILED)

    async def test_a_sandbox_that_cannot_start_fails_the_episode_at_seed_for_a_retry(self) -> None:
        server, client = _server(sandbox_provider={"local": {"shell": "/missing/shell"}})

        response = await _run(server)

        assert (response.failure.stage, response.failure.terminal) == ("seed", False)
        assert client.seeds == []

    @pytest.mark.parametrize(
        ("row", "message"),
        [
            ({"sol_sql": []}, "has no sol_sql"),
            ({"test_cases": ["def test_case(): pass"]}, "custom test cases"),
            ({"selected_database": "missing"}, "Missing database file"),
        ],
    )
    async def test_a_task_the_server_cannot_run_fails_for_good(self, row: dict, message: str) -> None:
        server, client = _server()

        response = await _run(server, **row)

        assert (response.failure.stage, response.failure.terminal) == ("seed", True)
        assert message in response.failure.failure_reason
        assert client.seeds == []

    async def test_shutdown_stops_the_sandbox_of_a_running_episode(self, sandbox_workspace: Path) -> None:
        started = asyncio.Event()

        async def stuck(tools: _Tools) -> str:
            started.set()
            await asyncio.Event().wait()
            return "never"

        server, _ = _server(turns=[stuck])
        episode = asyncio.create_task(_run(server))
        await started.wait()
        assert len(_sandboxes(sandbox_workspace)) == 1

        await server.finish_running_cleanups()

        assert _sandboxes(sandbox_workspace) == []
        episode.cancel()
        await asyncio.gather(episode, return_exceptions=True)
