# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BIRD-Interact episodes end to end: a scripted agent and user model, and real database sandboxes on the host."""

import asyncio
import datetime
import importlib
import json
import shutil
import subprocess
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx
import orjson
import pytest
from aiohttp import ClientResponseError, RequestInfo
from fastapi import FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, ValidationError
from yarl import URL

from environment_servers.bird_interact.adk_prompts import (
    ASK_USER_DESCRIPTION,
    DEBUG_WRONG_RESULT_MESSAGE,
    SUBMITTED_REPLY,
    TASK_COMPLETED_REPLY,
    UNCLEAR_QUESTION_REPLY,
)
from environment_servers.bird_interact.app import (
    SANDBOX_FAILED,
    SUBMISSION_NOT_GRADED_REPLY,
    USER_SIMULATOR_FAILED,
    BirdInteractEnvironmentServer,
    BirdInteractEnvironmentServerConfig,
    BirdInteractEpisode,
    BirdInteractRequest,
    BirdInteractResponse,
    SandboxCommandError,
)
from environment_servers.bird_interact.grading import (
    GRADE_DATABASE,
    WORK_DATABASE,
    Grade,
    PostgresDatabase,
    QueryResult,
    SQLiteDatabase,
    TestCaseHelpers,
    grade,
    grade_by_test_cases,
    main,
    passes_test_cases,
    remove_comments,
    remove_distinct,
    remove_round,
    split_statements,
)
from environment_servers.bird_interact.simulator import (
    action_parser_prompt,
    read_action,
    read_reply,
    response_generator_prompt,
    segment_sql,
)
from environment_servers.bird_interact.task_input import BirdInteractTask
from nemo_gym.client import ServerClient
from nemo_gym.config import AgentServerRef, BaseServerConfig, ModelServerRef
from nemo_gym.core import EpisodeId, MaterializedTask, NeMoGymResponse, TaskId
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxSpec
from nemo_gym.server_runtime import END_ACTIVATION_HEADER
from nemo_gym.trajectory import AgentObservationBundle


DATA = Path(__file__).parents[1] / "data"
TASKS = {row["instance_id"]: row for row in map(json.loads, (DATA / "example.jsonl").read_text().splitlines())}
CORRECT_1 = TASKS["bookshop_1"]["sol_sql"][0]
PHASE_1_OF_3 = TASKS["bookshop_3"]["sol_sql"][0]
PHASE_2_OF_3 = TASKS["bookshop_3"]["follow_up"]["sol_sql"][0]
WRONG = "SELECT name FROM authors"
# The simulator's two stages, as a model writes them.
LABELED_BESTSELLERS = (
    '<think>The question is about the bestsellers term.</think>\n\n- Action:\n<s>labeled("bestsellers")</s>'
)
TEN_COPIES = "<s>At least 10 copies sold in total.</s>"
# Episode sandboxes run on the host; each test gets its own workspace (see `sandbox_workspace`).
LOCAL_SANDBOX: dict[str, dict[str, str]] = {"local": {}}
_AGENT_URL = URL("http://agent:1")
_REQUEST_INFO = RequestInfo(_AGENT_URL, "POST", CIMultiDictProxy(CIMultiDict()), _AGENT_URL)


@pytest.fixture(autouse=True)
def sandbox_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "sandboxes"
    monkeypatch.setitem(LOCAL_SANDBOX["local"], "workspace_root", str(workspace))
    return workspace


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


class _Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class _HTTPResponse:
    def __init__(self, body: dict, *, cookies: dict[str, str] | None = None) -> None:
        self.status = 200
        self.ok = True
        self.body = orjson.dumps(body)
        self.content = _Content(self.body)
        self.cookies = {name: _Cookie(value) for name, value in (cookies or {}).items()}
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

    async def observe(self, name: str, **arguments: str) -> Any:
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


def _submit(sql: str, replies: list | None = None) -> Turn:
    """A c-Interact turn that submits ``sql``; ``replies`` collects the tool's reply."""

    async def turn(tools: _Tools) -> str:
        reply = await tools.observe("submit_sql", sql=sql)
        if replies is not None:
            replies.append(reply)
        return "Submitted."

    return turn


def _ask_then_submit(question: str, sql: str, replies: list | None = None) -> Turn:
    async def turn(tools: _Tools) -> str:
        answer = await tools.observe("ask_user", question=question)
        if replies is not None:
            replies.append(answer)
        await tools.observe("submit_sql", sql=sql)
        return "Submitted."

    return turn


class _Client(ServerClient):
    """The agent and the user model: each agent turn and each user-model reply comes from a script."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_turns: list[Turn]
    user_replies: list[str | BaseException]
    app: Any = None
    seeds: list[dict] = []
    agent_inputs: list[dict] = []
    # Each agent turn's path, headers and cookies.
    agent_calls: list[dict] = []
    user_requests: list[dict] = []
    continuation_supported: bool = True
    close_error: BaseException | None = None
    closes: int = 0

    @property
    def user_prompts(self) -> list[str]:
        return [request["input"][0]["content"] for request in self.user_requests]

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _HTTPResponse:
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            body = orjson.loads(orjson.dumps(body.model_dump(mode="json", exclude_unset=True)))
        if server_name == "user_model":
            self.user_requests.append(body)
            reply = self.user_replies.pop(0)
            if isinstance(reply, BaseException):
                raise reply
            return _HTTPResponse(_message(reply))
        if url_path == "/v1/agent_sessions":
            self.seeds.append(body)
            seeded = {"agent_session_id": body["agent_session_id"]}
            if body.get("continuation") is not None and self.continuation_supported:
                seeded["capabilities"] = {"mode": "resume_harness_session", "observations": ["trajectory"]}
            return _HTTPResponse(seeded, cookies={"agent": "agent-cookie"})
        if url_path == "/v1/agent_sessions/close":
            self.closes += 1
            if self.close_error is not None:
                raise self.close_error
            observations = AgentObservationBundle(source="test", records=[]).model_dump(mode="json")
            return _HTTPResponse({"agent_session_id": body["agent_session_id"], "agent_observations": observations})
        self.agent_inputs.append(body)
        self.agent_calls.append({"path": url_path, "headers": kwargs.get("headers"), "cookies": kwargs.get("cookies")})
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
        agent_calls=[],
        user_requests=[],
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
            cleanup_timeout_seconds=30,
            **({"sandbox_provider": LOCAL_SANDBOX, "activation_timeout_seconds": 60} | config),
        ),
        server_client=client,
    )
    client.app = server.setup_webserver()
    return server, client


async def _run(
    server: BirdInteractEnvironmentServer, task: str = "bookshop_1", *, skip_token_capture: bool = False, **row: Any
) -> BirdInteractResponse:
    request = BirdInteractRequest(
        episode_id=EpisodeId(rollout_id=f"{task}-0"),
        skip_token_capture=skip_token_capture,
        task=MaterializedTask(
            task_id=TaskId(taskset="bird_interact:example", task_id=task),
            task_input=BirdInteractTask.model_validate(TASKS[task] | row),
        ),
    )
    return await server.run_request(request)


def _phases(response: BirdInteractResponse) -> list[dict]:
    return [phase.model_dump() for phase in response.result.phases]


def _opening(client: _Client, turn: int) -> str:
    """The message that opened an agent turn after the first."""
    (message,) = client.agent_inputs[turn - 1]["input"]
    return message["content"]


def _bookshop(tmp_path: Path) -> SQLiteDatabase:
    import sqlite3

    database = tmp_path / "bookshop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript((DATA / "example_databases/bookshop/bookshop.sql").read_text())
    return SQLiteDatabase(database)


class TestGrading:
    """The ADK's default test case: its SQL rewrites, the order the queries run in, and the comparison."""

    def test_round_calls_are_removed_with_their_nested_arguments(self) -> None:
        sql = "SELECT ROUND(AVG(ROUND(x, 1)), 2), round( y ) FROM t"
        assert remove_round(sql) == "SELECT AVG(x), y FROM t"

    def test_distinct_is_removed_only_as_a_word_between_spaces(self) -> None:
        assert remove_distinct("SELECT DISTINCT a, COUNT(DISTINCT b) FROM t") == "SELECT a, COUNT(DISTINCT b) FROM t"

    def test_only_the_line_comments_a_line_break_ends_are_removed(self) -> None:
        assert remove_comments("SELECT 1 /* a */ -- b\n\n\nFROM t -- c") == "SELECT 1  \nFROM t -- c"

    def test_a_semicolon_inside_a_string_does_not_split_a_sqlite_statement(self) -> None:
        assert split_statements("SELECT 'a;b'; SELECT 2") == ["SELECT 'a;b';", "SELECT 2;"]

    def test_agent_sql_cannot_attach_the_episode_file(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        script = f"ATTACH DATABASE '{database.path}' AS original; DELETE FROM original.authors; SELECT 1"

        (attached,) = database.run([script], timeout_seconds=5)

        assert "not authorized" in attached.error
        assert grade(database, CORRECT_1, [CORRECT_1], ordered=False, timeout_seconds=5).passed

    def test_a_submission_that_breaks_the_solution_s_state_is_wrong_not_bad_ground_truth(self, tmp_path: Path) -> None:
        # As in the ADK, the solution runs on the state the submission left; here it finds no authors table.
        database = _bookshop(tmp_path)

        outcome = grade(
            database, "DROP TABLE IF EXISTS authors; SELECT 1", [CORRECT_1], ordered=False, timeout_seconds=5
        )

        assert (outcome.passed, outcome.execution_error) == (False, None)
        assert grade(database, CORRECT_1, [CORRECT_1], ordered=False, timeout_seconds=5).passed

    def test_unordered_results_compare_as_sets_and_ordered_results_by_position(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        reversed_sql = CORRECT_1.replace("DESC", "ASC")
        assert grade(database, reversed_sql, [CORRECT_1], ordered=False, timeout_seconds=5).passed
        assert not grade(database, reversed_sql, [CORRECT_1], ordered=True, timeout_seconds=5).passed

    def test_float_results_match_to_two_decimal_places(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        solution = TASKS["bookshop_2"]["sol_sql"][0]
        rounded = solution.replace(
            "AVG(1 - s.unit_price / b.list_price)", "ROUND(AVG(1 - s.unit_price / b.list_price), 2)"
        )
        assert grade(database, rounded, [solution], ordered=False, timeout_seconds=5).passed

    def test_an_error_is_reported_and_an_empty_result_fails(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        broken = grade(database, "SELECT nope FROM books", [CORRECT_1], ordered=False, timeout_seconds=5)
        empty = grade(database, "SELECT name FROM authors WHERE 0", [CORRECT_1], ordered=False, timeout_seconds=5)
        assert broken.execution_error == "no such column: nope"
        assert not empty.passed and empty.execution_error is None

    def test_the_submission_runs_as_written_then_normalized_on_one_state(self, tmp_path: Path) -> None:
        # As in the ADK, a write that cannot run a second time fails the comparison, not the execution.
        database = _bookshop(tmp_path)
        once = grade(database, f"CREATE TABLE note (x); {CORRECT_1}", [CORRECT_1], ordered=True, timeout_seconds=5)
        twice = grade(
            database, f"CREATE TABLE IF NOT EXISTS note (x); {CORRECT_1}", [CORRECT_1], ordered=True, timeout_seconds=5
        )
        assert (once.passed, once.execution_error) == (False, None)
        assert twice.passed

    def test_nothing_stays_in_the_database_unless_a_passing_submission_is_applied(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        write = f"CREATE TABLE IF NOT EXISTS note (x); {CORRECT_1}"

        def note_exists() -> bool:
            (result,) = database.run(["SELECT 1 FROM note"], timeout_seconds=5)
            return result.ok

        grade(database, f"DELETE FROM authors; {write}", [CORRECT_1], ordered=True, timeout_seconds=5)
        grade(database, write, [CORRECT_1], ordered=True, timeout_seconds=5)
        assert not note_exists()
        assert grade(database, CORRECT_1, [CORRECT_1], ordered=True, timeout_seconds=5).passed
        grade(database, write, [CORRECT_1], ordered=True, timeout_seconds=5, apply_if_passed=True)
        assert note_exists()

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
        setup = {
            "source": source,
            "preprocess_sql": ["DELETE FROM authors WHERE author_id > 2"],
            "applied_sql": ["DELETE FROM authors WHERE author_id > 1"],
            "timeout_seconds": 5,
        }
        assert run("setup", setup) == {}
        executed = run("execute", {"sql": "SELECT COUNT(*) FROM authors", "timeout_seconds": 5})
        graded = run(
            "grade",
            {"predicted_sql": CORRECT_1, "solution_sql": ["SELECT nope"], "ordered": False, "timeout_seconds": 5},
        )

        assert executed == {"error": None, "row_count": 1, "truncated": False, "preview": ["(1,)"]}
        assert graded == {"solution_error": "The ground-truth SQL failed: no such column: nope"}

    def test_a_task_setup_that_fails_fails_the_runner_command(self, tmp_path: Path) -> None:
        request_path = tmp_path / "setup.json"
        request = {
            "database": str(tmp_path / "episode.sqlite"),
            "source": str(DATA / "example_databases/bookshop/bookshop.sql"),
            "preprocess_sql": ["SELECT nope"],
            "timeout_seconds": 5,
        }
        request_path.write_text(json.dumps(request))

        # A non-zero exit, so the server runs the setup again on a fresh sandbox, and then fails the episode.
        with pytest.raises(RuntimeError, match="Setup SQL failed: no such column: nope"):
            main(["setup", str(request_path)])

    @pytest.mark.parametrize("sql", ["SELECT 'a\x00b'", "SELECT '\ud800'"], ids=["nul", "lone-surrogate"])
    def test_sql_the_driver_cannot_send_is_the_query_s_error(self, sql: str, tmp_path: Path) -> None:
        # Not a crash of grading.py, which the server would retry as a sandbox failure and mask.
        (result,) = _bookshop(tmp_path).run([sql], timeout_seconds=5)
        assert result.error

    def test_a_failing_submission_is_not_applied(self, tmp_path: Path) -> None:
        database = _bookshop(tmp_path)
        grade(
            database,
            "CREATE TABLE IF NOT EXISTS note (x); SELECT name FROM authors",
            ["SELECT title FROM books"],
            ordered=False,
            timeout_seconds=5,
            apply_if_passed=True,
        )
        (result,) = database.run(["SELECT 1 FROM note"], timeout_seconds=5)
        assert not result.ok

    def test_comments_are_removed_before_the_round_rewrite(self, tmp_path: Path) -> None:
        # A comment holding an unbalanced "ROUND(" would stop the rewrite if it stayed.
        outcome = grade(
            _bookshop(tmp_path),
            "/* ROUND( */ SELECT ROUND(list_price / 7, 0) FROM books",
            ["SELECT list_price / 7 FROM books"],
            ordered=False,
            timeout_seconds=5,
        )
        assert outcome.passed


class TestTestCases:
    """A management phase's test cases, run as the ADK runs them: in order, until one raises."""

    def test_each_test_case_gets_the_submission_the_solution_the_database_and_the_connection(self) -> None:
        seen: list[tuple] = []
        code = (
            "def test_case(pred_sqls, sol_sqls, db_name, conn):\n    seen.append((pred_sqls, sol_sqls, db_name, conn))"
        )

        assert passes_test_cases([code, code], {"seen": seen}, ["SELECT 1"], ["SELECT 2"], "shop", "connection")
        assert seen == [(["SELECT 1"], ["SELECT 2"], "shop", "connection")] * 2

    @pytest.mark.parametrize("failing", ["assert False, 'wrong'", "raise ValueError('broken')"])
    def test_the_first_test_case_that_raises_fails_the_phase_and_the_rest_do_not_run(self, failing: str) -> None:
        ran: list[int] = []
        record = "def test_case(*args):\n    ran.append(1)"

        assert not passes_test_cases([f"def test_case(*args):\n    {failing}", record], {"ran": ran}, [], [], "", None)
        assert ran == []

    def test_a_top_level_import_is_not_visible_to_the_test_case(self) -> None:
        # The ADK runs the code with separate globals and locals, so its functions see only the helpers.
        top_level = "import json\n\ndef test_case(*args):\n    json.dumps(1)"
        inside = "def test_case(*args):\n    import json\n    json.dumps(1)"

        assert not passes_test_cases([top_level], {}, [], [], "", None)
        assert passes_test_cases([inside], {}, [], [], "", None)

    def test_code_without_a_test_case_passes_and_its_output_goes_to_standard_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert passes_test_cases(["print('checked')"], {}, [], [], "", None)
        captured = capsys.readouterr()
        assert (captured.out, captured.err) == ("", "checked\n")

    def test_the_sql_rewrites_take_and_return_lists(self) -> None:
        namespace = TestCaseHelpers(timeout_seconds=5).namespace(pred_query_result=[(1,)])
        sql = ["SELECT DISTINCT ROUND(amount, 1) -- note\n"]

        rewritten = namespace["remove_round"](namespace["remove_distinct"](namespace["remove_comments"](sql)))

        assert rewritten == ["SELECT amount"]
        assert namespace["pred_query_result"] == [(1,)]


class _TypedRows:
    """A database whose submission and solution return fixed rows, typed as PostgreSQL's driver types them."""

    def __init__(self, predicted: list[tuple], solution: list[tuple]) -> None:
        self.predicted, self.solution = predicted, solution

    def run(self, scripts: list[str], *, timeout_seconds: float, commit: bool = False) -> list[QueryResult]:
        return [QueryResult(rows=self.predicted)] * 2 + [QueryResult(rows=self.solution)]


class TestPostgresGrading:
    @pytest.mark.parametrize(
        ("predicted", "solution", "passed"),
        [
            # Decimals round half-up to two places, where a float would round 3.145 down.
            ((Decimal("3.145"),), (Decimal("3.15"),), True),
            ((Decimal("1.01"),), (Decimal("1.02"),), False),
            # A datetime compares as its date.
            ((datetime.datetime(2024, 1, 2, 15, 30),), (datetime.date(2024, 1, 2),), True),
            # JSON values compare with sorted keys and their numbers rounded. psycopg2 decodes JSON with
            # `json.loads`, so its numbers are floats.
            (({"b": 1, "a": [2.499]},), ({"a": [2.5], "b": 1},), True),
            # An array of decimals cannot be written as JSON, so it fails even when equal, as in the ADK.
            (([Decimal("1.5")],), ([Decimal("1.5")],), False),
        ],
    )
    def test_values_are_normalized_by_the_benchmark_rules(
        self, predicted: tuple, solution: tuple, passed: bool
    ) -> None:
        outcome = grade(
            _TypedRows([predicted], [solution]), "SELECT x", ["SELECT x"], ordered=False, timeout_seconds=5
        )
        assert outcome.passed is passed

    def test_a_numeric_too_large_to_round_is_a_failed_comparison(self) -> None:
        huge = (Decimal("1E+30"),)
        outcome = grade(_TypedRows([huge], [huge]), "SELECT x", ["SELECT x"], ordered=False, timeout_seconds=5)
        assert outcome.passed is False

    def test_a_passing_phase_1_that_cannot_be_applied_fails_the_command(self) -> None:
        class ApplicationFails(_TypedRows):
            def run(self, scripts: list[str], *, timeout_seconds: float, commit: bool = False) -> list[QueryResult]:
                if commit:
                    return [QueryResult(rows=None, error="database is locked")]
                return super().run(scripts, timeout_seconds=timeout_seconds)

        database = ApplicationFails([(1,)], [(1,)])
        with pytest.raises(RuntimeError, match="database is locked"):
            grade(database, "SELECT x", ["SELECT x"], ordered=False, timeout_seconds=5, apply_if_passed=True)

    def test_the_runner_routes_a_postgresql_request_to_the_image_server(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        started: list[bool] = []
        ran: list[tuple[str, list[str], bool]] = []
        copies: list[tuple[str, str, bool]] = []

        def run(
            self: PostgresDatabase, scripts: list[str], *, timeout_seconds: float, commit: bool = False
        ) -> list[QueryResult]:
            ran.append((self.name, list(scripts), commit))
            # Typed as psycopg2 types them: numeric, date, text and bytea.
            row = (Decimal("2.50"), datetime.date(2024, 1, 2), "x", memoryview(b"\x01"))
            return [QueryResult(rows=[row]) for _ in scripts]

        def administer(self: PostgresDatabase, name: str, *, create: bool) -> None:
            copies.append((self.name, name, create))

        monkeypatch.setattr("environment_servers.bird_interact.grading.start_postgres", lambda: started.append(True))
        monkeypatch.setattr(PostgresDatabase, "run", run)
        monkeypatch.setattr(PostgresDatabase, "_administer", administer)

        def command(name: str, request: dict) -> dict:
            request_path = tmp_path / f"{name}.json"
            request_path.write_text(json.dumps(request | {"engine": "postgresql", "database": "bookshop_template"}))
            main([name, str(request_path)])
            return json.loads(capsys.readouterr().out)

        setup = {"preprocess_sql": ["UPDATE books SET price = 1"], "applied_sql": [], "timeout_seconds": 5}
        assert command("setup", setup) == {}
        executed = command("execute", {"sql": "SELECT 1", "timeout_seconds": 5})
        graded = command(
            "grade",
            {
                "predicted_sql": "SELECT 1",
                "solution_sql": ["SELECT 1"],
                "ordered": False,
                "test_cases": [],
                "timeout_seconds": 5,
                "apply_if_passed": True,
            },
        )

        assert started == [True]
        # The task's setup is committed to the episode's database. Agent SQL runs only on copies of it: an exploratory
        # query on the working copy, a submission on a fresh copy; only a passing phase 1 is applied to the database.
        assert ran == [
            ("bookshop_template", ["UPDATE books SET price = 1"], True),
            ("bird_interact_work", ["SELECT 1"], False),
            ("bird_interact_grade", ["SELECT 1", "SELECT 1", "SELECT 1"], False),
            ("bookshop_template", ["SELECT 1"], True),
        ]
        # Setup and each grade leave a fresh working copy; the grade's copy is dropped.
        assert copies == [
            ("bookshop_template", "bird_interact_work", True),
            ("bookshop_template", "bird_interact_grade", True),
            ("bookshop_template", "bird_interact_grade", False),
            ("bookshop_template", "bird_interact_work", True),
        ]
        assert executed["preview"] == ["(2.5, '2024-01-02', 'x', b'\\x01')"]
        assert graded["passed"] is True

    def test_the_runner_grades_by_test_cases_when_the_request_has_them(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        graded: list[tuple] = []

        def by_test_cases(
            database: PostgresDatabase,
            predicted_sql: str,
            solution_sql: list[str],
            test_cases: list[str],
            *,
            timeout_seconds: float,
            apply_if_passed: bool,
        ) -> Grade:
            graded.append((database.name, predicted_sql, solution_sql, test_cases, apply_if_passed))
            return Grade(passed=False, execution_error="relation missing")

        monkeypatch.setattr("environment_servers.bird_interact.grading.grade_by_test_cases", by_test_cases)
        refreshed: list[str] = []
        monkeypatch.setattr(
            PostgresDatabase, "_administer", lambda self, name, *, create: refreshed.append(name) if create else None
        )
        request = {
            "predicted_sql": "DELETE FROM sales",
            "solution_sql": ["DELETE FROM sales WHERE amount > 1"],
            "ordered": False,
            "test_cases": ["def test_case(*args): pass"],
            "timeout_seconds": 5,
            "apply_if_passed": True,
        }

        def command(engine: str, database: str) -> None:
            request_path = tmp_path / "grade.json"
            request_path.write_text(json.dumps(request | {"engine": engine, "database": database}))
            main(["grade", str(request_path)])

        command("postgresql", "shop_template")
        assert json.loads(capsys.readouterr().out) == {
            "passed": False,
            "execution_error": "relation missing",
            "timed_out": False,
        }
        assert graded == [
            ("shop_template", "DELETE FROM sales", request["solution_sql"], request["test_cases"], True),
        ]
        # Exploration starts again from the episode's database.
        assert refreshed == ["bird_interact_work"]
        with pytest.raises(SystemExit, match="PostgreSQL only"):
            command("sqlite", str(tmp_path / "episode.sqlite"))


_POSTGRES = shutil.which("initdb") and shutil.which("pg_ctl")


@pytest.mark.skipif(not _POSTGRES, reason="needs PostgreSQL's initdb and pg_ctl")
class TestPostgresDatabase:
    """The runner's PostgreSQL engine against a real server, started for the test as the current user.

    The server has the agent role as the image has it: not a superuser, owning the episode's database and its tables.
    """

    @pytest.fixture
    def database(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> PostgresDatabase:
        psycopg2 = pytest.importorskip("psycopg2")
        data, socket = tmp_path / "data", tmp_path / "socket"
        socket.mkdir()
        subprocess.run(
            ["initdb", "--auth=trust", "--username=bird", "--pgdata", data], check=True, capture_output=True
        )
        options = f"-c listen_addresses= -c unix_socket_directories={socket}"
        subprocess.run(
            ["pg_ctl", "start", "--wait", "--pgdata", data, "--log", tmp_path / "log", "--options", options],
            check=True,
            capture_output=True,
        )
        try:
            monkeypatch.setenv("PGHOST", str(socket))
            monkeypatch.setenv("PGUSER", "bird")
            monkeypatch.setattr("environment_servers.bird_interact.grading.AGENT_ROLE", "bird_agent")
            # Since psycopg2 2.9, `with connection` opens a transaction even in autocommit, and CREATE DATABASE
            # cannot run in one.
            connection = psycopg2.connect(dbname="postgres")
            connection.autocommit = True
            try:
                connection.cursor().execute("CREATE ROLE bird_agent LOGIN NOSUPERUSER")
                connection.cursor().execute("CREATE DATABASE shop_template OWNER bird_agent")
            finally:
                connection.close()
            connection = psycopg2.connect(dbname="shop_template")
            with connection, connection.cursor() as cursor:
                cursor.execute("CREATE TABLE sales (amount numeric, sold date)")
                cursor.execute("INSERT INTO sales VALUES (1.005, '2024-01-02'), (2.5, '2024-01-03')")
                cursor.execute("ALTER TABLE sales OWNER TO bird_agent")
            # A database with a session open cannot be copied, and test-case grading copies it.
            connection.close()
            yield PostgresDatabase("shop_template")
        finally:
            subprocess.run(["pg_ctl", "stop", "--pgdata", data, "--mode", "immediate"], capture_output=True)

    def test_agent_sql_cannot_read_the_server_s_files_run_programs_or_change_its_settings(
        self, database: PostgresDatabase
    ) -> None:
        # Through any of them, agent SQL could log the solution SQL that grading runs, and read the log.
        results = [
            database.run([sql], timeout_seconds=5)[0]
            for sql in (
                "SELECT pg_read_file('postgresql.conf')",
                "COPY (SELECT 1) TO PROGRAM 'true'",
                "SET log_statement = 'all'",
                "SELECT pg_reload_conf()",
            )
        ]
        refused = ("permission denied", "must be superuser")
        assert all(result.error and any(word in result.error for word in refused) for result in results), results
        # The role owns the tables, so a management task can still change them.
        (altered,) = database.run(["ALTER TABLE sales ADD COLUMN note text; SELECT 1"], timeout_seconds=5)
        assert altered.ok

    def test_sql_that_breaks_the_driver_or_its_connection_is_the_query_s_error(
        self, database: PostgresDatabase
    ) -> None:
        late, killed = (
            database.run([sql], timeout_seconds=5)[0]
            for sql in ("SELECT DATE '9999-12-31' + 1", "SELECT pg_terminate_backend(pg_backend_pid())")
        )
        assert "year" in late.error and killed.error, (late, killed)

    def test_a_query_that_writes_leaves_the_database_unchanged(self, database: PostgresDatabase) -> None:
        (deleted,) = database.run(["DELETE FROM sales; SELECT COUNT(*) FROM sales"], timeout_seconds=5)
        (counted,) = database.run(["SELECT COUNT(*) FROM sales"], timeout_seconds=5)
        assert (deleted.rows, counted.rows) == ([(0,)], [(2,)])

    def test_a_slow_query_times_out_and_an_error_is_reported(self, database: PostgresDatabase) -> None:
        (slow,) = database.run(["SELECT pg_sleep(5)"], timeout_seconds=0.2)
        (broken,) = database.run(["SELECT nope FROM sales"], timeout_seconds=5)
        assert slow.timed_out and "nope" in broken.error

    def test_a_management_submission_has_one_deadline_like_a_default_one(self, database: PostgresDatabase) -> None:
        never = "def test_case(*args):\n    raise AssertionError('ran')"
        # Each statement is under the timeout; together they are over it.
        sleeps = "SELECT pg_sleep(0.15); " * 8 + "SELECT 1"

        by_rows = grade(database, sleeps, ["SELECT 1"], ordered=False, timeout_seconds=0.5)
        by_cases = grade_by_test_cases(database, sleeps, ["SELECT 1"], [never], timeout_seconds=0.5)

        assert (by_rows.passed, by_rows.timed_out) == (False, True)
        assert (by_cases.passed, by_cases.timed_out) == (False, True)

    def test_a_passing_management_submission_that_cannot_be_applied_fails_the_command(
        self, database: PostgresDatabase
    ) -> None:
        passes = "def test_case(*args):\n    pass"
        # Runs on the test cases' copy, and fails on the episode's database.
        copy_only = "DO $$ BEGIN IF current_database() = 'shop_template' THEN RAISE 'not here'; END IF; END $$"

        with pytest.raises(RuntimeError, match="not here"):
            grade_by_test_cases(database, copy_only, ["SELECT 1"], [passes], timeout_seconds=5, apply_if_passed=True)

    def test_a_solution_run_again_alone_cannot_commit_to_the_episode_s_database(
        self, database: PostgresDatabase
    ) -> None:
        # The submission breaks the solution, which then runs again alone, with a COMMIT of its own.
        solution = ["UPDATE sales SET amount = 1000; COMMIT; SELECT amount FROM sales"]
        with database.copy(GRADE_DATABASE) as scratch:
            outcome = grade(scratch, "DROP TABLE sales; SELECT 1", solution, ordered=False, timeout_seconds=5,
                            baseline=database)  # fmt: skip

        (amounts,) = database.run(["SELECT max(amount) FROM sales"], timeout_seconds=5)
        assert not outcome.passed and amounts.rows == [(Decimal("2.5"),)]

    def test_a_script_has_one_deadline_whatever_its_statements(self, database: PostgresDatabase) -> None:
        (sleeps,) = database.run(["SELECT pg_sleep(0.15); " * 8 + "SELECT 1"], timeout_seconds=0.5)
        (committed,) = database.run(["SELECT 1; COMMIT; SELECT pg_sleep(5)"], timeout_seconds=0.5)
        assert sleeps.timed_out and committed.timed_out

    def test_a_commit_in_agent_sql_changes_only_a_copy(
        self, database: PostgresDatabase, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def command(name: str, request: dict) -> dict:
            request_path = tmp_path / f"{name}.json"
            request_path.write_text(json.dumps(request | {"engine": "postgresql", "database": database.name}))
            main([name, str(request_path)])
            return json.loads(capsys.readouterr().out)

        # The working copy that setup makes.
        database.refresh(WORK_DATABASE)
        write = "UPDATE sales SET amount = 0; COMMIT; SELECT amount, sold FROM sales"
        command("execute", {"sql": write, "timeout_seconds": 5})
        submission = {"predicted_sql": write, "solution_sql": ["SELECT 1"], "ordered": False, "test_cases": []}
        command("grade", submission | {"timeout_seconds": 5, "apply_if_passed": False})

        (amounts,) = database.run(["SELECT amount FROM sales ORDER BY sold"], timeout_seconds=5)
        assert amounts.rows == [(Decimal("1.005"),), (Decimal("2.5"),)]

    def test_a_dollar_quoted_body_runs_as_one_query(self, database: PostgresDatabase) -> None:
        (result,) = database.run(["DO $$ BEGIN PERFORM 1; PERFORM 2; END $$; SELECT 3"], timeout_seconds=5)
        assert result.rows == [(3,)]

    def test_grading_compares_numeric_and_date_results_by_the_benchmark_rules(
        self, database: PostgresDatabase
    ) -> None:
        solution = ["SELECT amount, sold FROM sales"]
        predicted = "SELECT ROUND(amount, 2), sold::timestamp FROM sales ORDER BY amount DESC"
        assert grade(database, predicted, solution, ordered=False, timeout_seconds=5).passed
        assert not grade(database, predicted, solution, ordered=True, timeout_seconds=5).passed

    def test_test_cases_see_the_submission_s_writes_on_a_copy_that_is_dropped(
        self, database: PostgresDatabase
    ) -> None:
        three_sales = (
            "def test_case(pred_sqls, sol_sqls, db_name, conn):\n"
            "    rows, error, timed_out = execute_queries(['SELECT COUNT(*) FROM sales'], db_name, conn)\n"
            "    assert rows == [(3,)], rows\n"
        )
        insert = "INSERT INTO sales VALUES (4, '2024-01-04')"

        def count() -> list[tuple] | None:
            (result,) = database.run(["SELECT COUNT(*) FROM sales"], timeout_seconds=5)
            return result.rows

        passed = grade_by_test_cases(database, insert, [insert], [three_sales], timeout_seconds=5)
        failed = grade_by_test_cases(database, "DELETE FROM sales", [insert], [three_sales], timeout_seconds=5)
        assert (passed.passed, failed.passed, count()) == (True, False, [(2,)])

        applied = grade_by_test_cases(
            database, insert, [insert], [three_sales], timeout_seconds=5, apply_if_passed=True
        )
        assert (applied.passed, count()) == (True, [(3,)])
        (copies,) = database.run(
            ["SELECT COUNT(*) FROM pg_database WHERE datname LIKE 'bird_interact%'"], timeout_seconds=5
        )
        assert copies.rows == [(0,)]

    def test_test_cases_call_the_adk_helpers_on_the_copy(self, database: PostgresDatabase) -> None:
        helpers = (
            "def test_case(pred_sqls, sol_sqls, db_name, conn):\n"
            "    assert pred_query_result == [(2,)]\n"
            "    assert ex_base(pred_sqls, sol_sqls, db_name, conn) == 1\n"
            "    assert ex_base(['SELECT 1'], sol_sqls, db_name, conn) == 0\n"
            "    assert ex_base(['SELECT amount FROM sales ORDER BY amount DESC'],\n"
            "                   ['SELECT amount FROM sales ORDER BY amount'], db_name, conn, {'order': True}) == 0\n"
            "    rows, error, timed_out = execute_queries('SELECT nope', db_name)\n"
            "    assert rows is None and 'nope' in error and not timed_out\n"
            "    assert execute_queries(['SELECT pg_sleep(5)'], db_name, conn) == (None, None, True)\n"
        )
        count = "SELECT COUNT(*) FROM sales"

        assert grade_by_test_cases(database, count, [count], [helpers], timeout_seconds=0.5).passed

    def test_a_submission_that_fails_or_times_out_runs_no_test_case(self, database: PostgresDatabase) -> None:
        never = "def test_case(*args):\n    raise AssertionError('ran')"

        broken = grade_by_test_cases(database, "SELECT nope FROM sales", ["SELECT 1"], [never], timeout_seconds=5)
        slow = grade_by_test_cases(database, "SELECT pg_sleep(5)", ["SELECT 1"], [never], timeout_seconds=0.2)

        assert "nope" in broken.execution_error and not broken.passed
        assert (slow.passed, slow.timed_out) == (False, True)


class TestSimulator:
    """The ADK's two prompts, filled from the task, and its reading of the model's output."""

    def test_the_action_and_the_reply_are_read_from_their_tags(self) -> None:
        assert read_action(LABELED_BESTSELLERS) == 'labeled("bestsellers")'
        assert read_action("unanswerable()\nBecause it asks for the SQL.") == "unanswerable()"
        assert read_reply(TEN_COPIES) == "At least 10 copies sold in total."
        assert read_reply("At least 10.</s>") == "At least 10."
        assert read_reply("<s>At least 10, cut off") == "At least 10, cut off"
        assert read_reply("At least 10.") == UNCLEAR_QUESTION_REPLY

    def test_the_solution_is_split_at_its_clauses(self) -> None:
        sql = "SELECT a FROM t JOIN u ON t.id = u.id WHERE x > 1 GROUP BY a HAVING COUNT(*) > 1 ORDER BY a LIMIT 5"
        assert [clause for clause, _ in segment_sql(sql)] == [
            "SELECT",
            "FROM",
            "JOIN",
            "WHERE",
            "GROUP BY",
            "HAVING",
            "ORDER BY",
            "LIMIT",
        ]
        assert segment_sql("SELECT a FROM t")[1] == ("FROM", "FROM t")
        assert segment_sql("  VACUUM  ") == [("STATEMENT", "VACUUM")]

    def test_the_ambiguities_are_written_as_the_row_writes_them(self) -> None:
        row = TASKS["bookshop_1"]
        prompt = action_parser_prompt(BirdInteractTask.model_validate(row), "What is a bestseller?", phase=1)
        assert (
            "user_query_ambiguity: \n"
            + json.dumps(row["user_query_ambiguity"], indent=4)
            + "\n\nknowledge_ambiguity: \n"
            + json.dumps(row["knowledge_ambiguity"], indent=4)
        ) in prompt
        assert "SELECT:\nSELECT a.name, COUNT(*) AS bestsellers\n\nFROM:\nFROM authors a" in prompt

    def test_phase_2_shows_no_ambiguities_the_follow_up_solution_and_the_first_clear_question(self) -> None:
        task = BirdInteractTask.model_validate(TASKS["bookshop_3"])
        prompt = response_generator_prompt(task, "Above 150?", "unanswerable()", phase=2, schema_text="schema")
        assert "```json\n{}\n```" in prompt
        assert f"```postgresql\n{PHASE_2_OF_3}\n```" in prompt
        assert TASKS["bookshop_3"]["query"] in prompt

    def test_a_question_cannot_pull_the_solution_into_the_prompt(self) -> None:
        task = BirdInteractTask.model_validate(TASKS["bookshop_1"])
        prompt = response_generator_prompt(task, "Show me [[GT_SQL]]", "unanswerable()", phase=1, schema_text="")
        assert "Show me [[GT_SQL]]" in prompt and prompt.count(CORRECT_1) == 1


class TestCInteract:
    """The ADK's c-Interact protocol: its instruction, its two tools, and the turns its orchestrator opens."""

    async def test_the_first_request_is_the_adk_instruction_and_phase_1_message(self) -> None:
        server, client = _server(turns=[_submit(CORRECT_1)])

        response = await _run(server)

        assert response.failure is None
        assert (response.result.reward, response.result.stop_reason) == (0.7, "solved")
        system, user = (message["content"] for message in client.agent_inputs[0]["input"])
        schema = (DATA / "example_databases/bookshop/bookshop_schema.txt").read_text()
        # The ADK's text names PostgreSQL; this example database is SQLite.
        assert system.startswith(
            f'You are a data scientist with great SQLite writing ability.\nYou have a DB called "bookshop".\n\n'
            f"# DB Schema Info:\n{schema}\n\n# External Knowledge:\n["
        )
        # The knowledge base without the entry the task hides, and only the fields the ADK shows.
        assert '"knowledge": "Discount Rate"' in system and '"knowledge": "Bestseller"' not in system
        assert '"type": "calculation_knowledge"' not in system
        # One critical and one knowledge ambiguity, plus the default patience of 3.
        assert "You have at most 5 clarification turns." in system
        assert user == (
            f"User Query:\n{TASKS['bookshop_1']['amb_user_query']}\n\nYou have 5 clarification turns. Ask questions "
            "with ask_user to resolve ambiguities, then call submit_sql with your final SQLite query."
        )
        tools = {tool["name"]: tool for tool in client.agent_inputs[0]["tools"]}
        assert tools["ask_user"]["description"] == ASK_USER_DESCRIPTION
        assert "sql: The final SQLite SQL query to submit." in tools["submit_sql"]["description"]
        (grant,) = client.seeds[0]["tool_accesses"]
        assert grant["kind"] == "direct_http" and client.seeds[0]["continuation"] is not None

    async def test_a_question_goes_through_both_simulator_stages(self) -> None:
        answers: list[str] = []
        server, client = _server(
            turns=[_ask_then_submit("What counts as a bestseller?", CORRECT_1, answers)],
            user=[LABELED_BESTSELLERS, TEN_COPIES],
        )

        response = await _run(server)

        assert answers == ["At least 10 copies sold in total.\n\n[SYSTEM NOTE: Clarification turns remaining: 4/5]"]
        parser, generator = client.user_prompts
        assert "What counts as a bestseller?" in parser and '"term": "bestsellers"' in parser
        assert 'labeled("bestsellers")' in generator and "CREATE TABLE" in generator
        assert TASKS["bookshop_1"]["query"] in generator
        # The ADK samples both stages at temperature 0, with its own output limits.
        assert [(request["temperature"], request["max_output_tokens"]) for request in client.user_requests] == [
            (0.0, 500),
            (0.0, 1024),
        ]
        assert (response.result.questions_asked, response.result.reward) == (1, 0.7)

    async def test_questions_are_not_capped_and_the_note_counts_down_to_zero(self) -> None:
        answers: list[str] = []

        async def turn(tools: _Tools) -> str:
            for question in ("First?", "Second?", "Third?"):
                reply = await tools.call("ask_user", question=question)
                # A question does not end the turn.
                assert END_ACTIVATION_HEADER not in reply.headers
                answers.append(reply.json())
            await tools.observe("submit_sql", sql=CORRECT_1)
            return "Submitted."

        server, _ = _server(turns=[turn], user=["<s>unanswerable()</s>", "<s>No.</s>"] * 3, question_patience=0)

        response = await _run(server)

        assert [answer.rsplit("remaining: ", 1)[1] for answer in answers] == ["1/2]", "0/2]", "0/2]"]
        assert (response.result.questions_asked, response.result.reward) == (3, 0.7)

    async def test_a_turn_ends_at_its_submission(self) -> None:
        replies: list[Any] = []
        ends: list[bool] = []

        async def turn(tools: _Tools) -> str:
            for name, arguments in (
                ("submit_sql", {"sql": WRONG}),
                ("submit_sql", {"sql": CORRECT_1}),
                ("ask_user", {"question": "Was that right?"}),
            ):
                reply = await tools.call(name, **arguments)
                replies.append(reply.json())
                ends.append(reply.headers.get(END_ACTIVATION_HEADER))
            return "Submitted."

        server, client = _server(turns=[turn, _submit(CORRECT_1)])

        response = await _run(server)

        assert replies == [
            "SQL failed Phase 1. Your SQL is not correct.\nBudget remaining: 0 bird-coins",
            SUBMITTED_REPLY,
            SUBMITTED_REPLY,
        ]
        # The submission's reply ends the agent's activation, as the ADK's callback ends its run; a question does not.
        assert ends == ["true", "true", "true"]
        assert client.user_requests == []
        assert _opening(client, 2) == DEBUG_WRONG_RESULT_MESSAGE
        assert _phases(response) == [{"phase": 1, "passed": True, "attempts": 2, "reward": 0.5}]

    async def test_an_sql_error_opens_the_debug_turn_and_a_second_failure_ends_the_task(self) -> None:
        server, client = _server(turns=[_submit("SELECT nope FROM books"), _submit(WRONG)])

        response = await _run(server)

        assert _opening(client, 2) == (
            "Your SQL is not executable: Error executing submitted SQL: no such column: nope\n"
            "Please fix and call submit_sql."
        )
        assert (response.result.stop_reason, response.result.reward, response.result.finished) == ("failed", 0.0, True)
        assert _phases(response) == [{"phase": 1, "passed": False, "attempts": 2, "reward": 0.0}]

    async def test_a_turn_without_a_submission_gets_the_debug_turn(self) -> None:
        # As in the ADK, the reward follows the phase's submissions, so the debug turn's first one earns 0.7.
        server, client = _server(turns=["I am not sure yet.", _submit(CORRECT_1)])

        response = await _run(server)

        assert _opening(client, 2) == DEBUG_WRONG_RESULT_MESSAGE
        assert _phases(response) == [{"phase": 1, "passed": True, "attempts": 1, "reward": 0.7}]

    async def test_two_turns_without_a_submission_end_the_task(self) -> None:
        server, client = _server(turns=["Hmm.", "Still thinking."])

        response = await _run(server)

        assert (response.result.stop_reason, response.result.reward, len(client.agent_inputs)) == (
            "no_sql_submitted",
            0.0,
            2,
        )

    async def test_the_follow_up_question_is_phase_2_and_each_phase_has_its_debug_turn(self) -> None:
        replies: list[str] = []
        server, client = _server(
            turns=[_submit(PHASE_1_OF_3, replies), _submit(WRONG), _submit(PHASE_2_OF_3, replies)]
        )

        response = await _run(server, "bookshop_3")

        follow_up = TASKS["bookshop_3"]["follow_up"]["query"]
        assert replies == [
            "Phase 1 correct! (Reward: 0.7). Moving to Phase 2.\nReward: 0.7\n"
            f"Follow-up question: {follow_up}\nBudget remaining: 0 bird-coins",
            "Phase 2 correct! (Reward: 0.2). Task finished.\nReward: 0.2\nBudget remaining: 0 bird-coins",
        ]
        assert _opening(client, 2) == (
            f"Phase 1 is complete. Here is a follow-up question:\n\n{follow_up}\n\n"
            "Generate the SQLite query and call submit_sql."
        )
        assert _opening(client, 3) == DEBUG_WRONG_RESULT_MESSAGE
        assert (response.result.reward, response.result.stop_reason) == (pytest.approx(0.9), "solved")
        assert [(phase["phase"], phase["attempts"], phase["reward"]) for phase in _phases(response)] == [
            (1, 1, 0.7),
            (2, 2, 0.2),
        ]

    async def test_a_call_after_the_task_is_done_gets_the_adk_reply(self) -> None:
        replies: list[Any] = []

        async def turn(tools: _Tools) -> str:
            replies.append(await tools.observe("submit_sql", sql=CORRECT_1))
            replies.append(await tools.observe("ask_user", question="Anything else?"))
            return "Done."

        server, client = _server(turns=[turn])

        response = await _run(server)

        assert replies == [
            "Phase 1 correct! (Reward: 0.7). Task finished.\nReward: 0.7\nBudget remaining: 0 bird-coins",
            TASK_COMPLETED_REPLY,
        ]
        assert client.user_requests == [] and response.result.stop_reason == "solved"

    async def test_a_single_phase_reward_schedule_skips_the_follow_up(self) -> None:
        replies: list[Any] = []
        server, client = _server(
            turns=[_submit(PHASE_1_OF_3, replies)], phase_rewards=[{"first_attempt": 1.0, "after_debug": 0.5}]
        )

        response = await _run(server, "bookshop_3")

        assert replies == [
            "Phase 1 correct! (Reward: 1.0). Task finished.\nReward: 1.0\nBudget remaining: 0 bird-coins"
        ]
        assert (response.result.reward, response.result.stop_reason, len(client.agent_inputs)) == (1.0, "solved", 1)

    async def test_the_dialect_changes_the_adk_texts_but_not_the_task(self) -> None:
        server, client = _server(turns=[_submit(CORRECT_1)])
        question = "Which authors are tagged 'PostgreSQL'?"

        await _run(server, amb_user_query=question)

        system, user = (item["content"] for item in client.agent_inputs[0]["input"])
        assert "great SQLite writing ability" in system and "PostgreSQL" not in system
        assert question in user and "final SQLite query" in user

    @pytest.mark.parametrize(
        ("tool", "stall", "kind"),
        [("ask_user", "_user_model", USER_SIMULATOR_FAILED), ("submit_sql", "_run_command", SANDBOX_FAILED)],
    )
    async def test_a_tool_call_cut_off_by_its_deadline_masks_the_sample(
        self, tool: str, stall: str, kind: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        statuses: list[int] = []

        async def turn(tools: _Tools) -> str:
            arguments = {"question": "?"} if tool == "ask_user" else {"sql": CORRECT_1}
            statuses.append((await tools.call(tool, **arguments)).status_code)
            return "Waiting."

        server, client = _server(turns=[turn], tool_call_timeout_seconds=0.2)
        original = getattr(server, stall)

        async def stalled(*args: Any, **kwargs: Any) -> Any:
            # The sandbox's setup runs before any tool call, with no tool deadline; only the tool's own work stalls.
            if args[1:2] == ("setup",):
                return await original(*args, **kwargs)
            await asyncio.sleep(30)

        monkeypatch.setattr(server, stall, stalled)

        response = await _run(server)

        assert statuses == [504] and response.failure is None
        assert (response.result.mask_sample, response.result.stop_reason) == (True, kind)
        assert len(client.agent_inputs) == 1

    async def test_a_simulator_failure_masks_the_sample_and_ends_the_turns(self) -> None:
        statuses: list[int] = []

        async def turn(tools: _Tools) -> str:
            statuses.append((await tools.call("ask_user", question="?")).status_code)
            return "Waiting."

        server, client = _server(turns=[turn], user=[RuntimeError("user model down")])

        response = await _run(server)

        assert statuses == [500] and response.failure is None
        assert (response.result.mask_sample, response.result.stop_reason) == (True, USER_SIMULATOR_FAILED)
        assert len(client.agent_inputs) == 1

    async def test_phase_2_sees_what_phase_1_changed_on_the_same_sandbox(self) -> None:
        server, _ = _server(
            turns=[
                _submit(f"CREATE TABLE IF NOT EXISTS note (x); {PHASE_1_OF_3}"),
                _submit(f"SELECT * FROM ({PHASE_2_OF_3}) WHERE NOT EXISTS (SELECT 1 FROM note)"),
            ]
        )

        response = await _run(server, "bookshop_3")

        assert response.result.reward == pytest.approx(1.0)

    async def test_a_submission_over_the_sql_timeout_gets_the_timeout_message(self) -> None:
        replies: list = []
        endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT MAX(x) FROM c"
        server, client = _server(turns=[_submit(endless, replies), _submit("SELECT 1")], sql_timeout_seconds=0.5)

        await _run(server)

        assert replies == ["SQL failed Phase 1. Submitted SQL execution timed out\nBudget remaining: 0 bird-coins"]
        assert client.agent_inputs[1]["input"][0]["content"].startswith(
            "Your SQL is not executable: Submitted SQL execution timed out"
        )


class TestAInteract:
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
        assert seen["submitted"]["observation"] == "Phase 1 correct! (Reward: 0.7). Task finished.\nReward: 0.7"
        assert (response.result.reward, response.result.coins_spent) == (0.7, 4.0)
        assert response.result.stop_reason == "agent_finished"
        assert [call.name for call in response.result.tool_calls] == [
            "get_all_external_knowledge_names",
            "get_column_meaning",
            "submit",
        ]
        (grant,) = client.seeds[0]["tool_accesses"]
        assert grant["kind"] == "direct_http" and "continuation" not in client.seeds[0]

    async def test_the_knowledge_tools_show_the_fields_the_adk_shows_and_not_the_hidden_entry(self) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            seen["all"] = await tools.observe("get_all_knowledge_definitions")
            seen["one"] = await tools.observe("get_knowledge_definition", knowledge_name="discount rate")
            return "Done."

        server, _ = _server("a_interact", turns=[agent])

        await _run(server)

        # The ADK's db_environment /knowledge: a JSON array of the visible entries, each with these four fields only.
        entries = json.loads(seen["all"]["observation"])
        assert {entry["knowledge"] for entry in entries} >= {"Discount Rate"}
        assert "Bestseller" not in {entry["knowledge"] for entry in entries}
        assert all(set(entry) <= {"id", "knowledge", "description", "definition"} for entry in entries)
        assert seen["all"]["observation"] == json.dumps(entries, indent=2)
        one = json.loads(seen["one"]["observation"])
        assert one["knowledge"] == "Discount Rate" and "type" not in one

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
        assert seen["submitted"]["observation"].startswith("Phase 1 correct!")
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
        assert seen["submitted"]["observation"].startswith("Phase 1 correct!")
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

        assert seen["submitted"]["observation"] == (
            "SQL failed Phase 1. Your SQL is not correct.\nYour budget is spent, so the task ends here."
        )
        assert response.result.finished is True
        assert _phases(response) == [{"phase": 1, "passed": False, "attempts": 1, "reward": 0.0}]

    async def test_a_passing_submit_that_spends_the_budget_still_opens_the_follow_up(self) -> None:
        seen: dict[str, Any] = {}
        task = TASKS["bookshop_3"]

        async def agent(tools: _Tools) -> str:
            while (await tools.observe("execute", sql="SELECT 1"))["coins_left"] > 0:
                pass
            seen["phase_1"] = await tools.observe("submit", sql=task["sol_sql"][0])
            seen["phase_2"] = await tools.observe("submit", sql=task["follow_up"]["sol_sql"][0])
            return "Done."

        server, _ = _server("a_interact", turns=[agent], coin_patience=0)

        response = await _run(server, "bookshop_3")

        assert "the task ends here" not in seen["phase_1"]["observation"]
        assert seen["phase_2"]["observation"] != "The task is finished."
        assert (response.result.reward, response.result.finished) == (pytest.approx(1.0), True)

    async def test_an_exploratory_query_returns_a_bounded_preview(self) -> None:
        seen: dict[str, Any] = {}

        async def agent(tools: _Tools) -> str:
            seen["result"] = await tools.observe("execute", sql="SELECT s.sale_id FROM sales s, authors a")
            return "Done."

        server, _ = _server("a_interact", turns=[agent])

        await _run(server)

        assert seen["result"]["observation"].startswith("95 rows\n(1,)")
        assert seen["result"]["observation"].endswith("... 75 more rows")

    @pytest.mark.parametrize(
        ("rows", "count", "more"), [(10_000, "10000 rows", "9980 more"), (10_001, "10000+ rows", "9980+ more")]
    )
    async def test_a_result_over_the_row_cap_reports_its_count_as_a_lower_bound(
        self, rows: int, count: str, more: str
    ) -> None:
        seen: dict[str, Any] = {}
        sql = f"WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < {rows}) SELECT x FROM c"

        async def agent(tools: _Tools) -> str:
            seen["result"] = await tools.observe("execute", sql=sql)
            return "Done."

        server, _ = _server("a_interact", turns=[agent])

        await _run(server)

        assert seen["result"]["observation"].startswith(f"{count}\n(1,)")
        assert seen["result"]["observation"].endswith(f"... {more} rows")

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


async def _hang(_: _Tools) -> str:
    await asyncio.sleep(10)
    raise AssertionError("unreachable")


class TestAgentSession:
    async def test_each_turn_runs_on_the_same_agent_session_through_the_capture_route(self) -> None:
        server, client = _server(turns=[_submit(WRONG), _submit(CORRECT_1)])

        response = await _run(server)

        assert response.result.stop_reason == "solved"
        assert [call["path"] for call in client.agent_calls] == ["/ng-rollout/bookshop_1-0/v1/responses"] * 2
        assert [call["headers"] for call in client.agent_calls] == [
            {"x-ng-agent-activation": "1"},
            {"x-ng-agent-activation": "2"},
        ]
        assert client.agent_calls[1]["cookies"] == {"agent": "agent-cookie"}
        assert float(client.agent_inputs[1]["metadata"]["agent_timeout_sec"]) <= 60
        assert [turn.turn for turn in response.result.turns] == [1, 2]
        assert client.closes == 1 and response.result.ng_agent_observations.source == "test"

    async def test_the_tool_grant_ends_with_the_episode(self) -> None:
        granted: list[_Tools] = []

        async def agent(tools: _Tools) -> str:
            granted.append(tools)
            return "Done."

        server, _ = _server("a_interact", turns=[agent])

        response = await _run(server)

        assert response.result.stop_reason == "agent_finished"
        assert (await granted[0].call("get_schema")).status_code == 410

    async def test_an_agent_that_cannot_resume_its_session_fails_the_seed_for_good(self) -> None:
        server, client = _server(turns=[])
        client.continuation_supported = False

        response = await _run(server)

        assert (response.failure.stage, response.failure.terminal) == ("seed", True)
        # The close was registered before the seed, so the session is still closed.
        assert client.agent_calls == [] and client.closes == 1

    async def test_a_turn_over_its_timeout_fails_the_episode_for_a_retry(self) -> None:
        server, _ = _server(turns=[_hang], activation_timeout_seconds=0.05)

        response = await _run(server)

        assert (response.failure.stage, response.failure.terminal) == ("agent", False)
        assert "timed out" in response.failure.failure_reason

    async def test_the_interaction_deadline_stops_the_turns_and_the_episode_is_graded(self) -> None:
        server, client = _server(turns=["Hmm.", _hang], interaction_timeout_seconds=0.5)

        response = await _run(server)

        assert response.failure is None
        assert (response.result.stop_reason, response.result.reward) == ("interaction_timeout", 0.0)
        assert [turn.turn for turn in response.result.turns] == [1]
        assert client.closes == 1

    async def test_a_failed_turn_keeps_the_completed_turns_and_the_agent_evidence(self) -> None:
        async def unavailable(_: _Tools) -> str:
            raise ClientResponseError(_REQUEST_INFO, history=(), status=503, message="model unavailable")

        server, client = _server(turns=[_submit(WRONG), unavailable])

        response = await _run(server)

        assert (response.failure.stage, response.failure.terminal) == ("agent", False)
        assert [turn.turn for turn in response.failure.turns] == [1]
        assert [call.name for call in response.failure.tool_calls] == ["submit_sql"]
        assert response.failure.agent_observations.source == "test" and client.closes == 1

    @pytest.mark.parametrize(("mode", "tool"), [("c_interact", "submit_sql"), ("a_interact", "submit")])
    async def test_a_broken_solution_fails_the_submission_as_the_adk_grades_it(self, mode: str, tool: str) -> None:
        async def submit(tools: _Tools) -> str:
            assert (await tools.call(tool, sql=CORRECT_1)).status_code == 200
            return "Submitted."

        server, _ = _server(mode, turns=[submit, submit])

        response = await _run(server, sol_sql=["SELECT missing_column FROM sales"])

        # The ADK's ex_base returns 0 when the solution does not run: the phase fails, and the episode goes on.
        assert response.failure is None
        assert (response.result.reward, response.result.mask_sample) == (0.0, False)
        assert response.result.solution_errors
        assert all("bookshop_1, phase 1" in e and "missing_column" in e for e in response.result.solution_errors)

    async def test_a_broken_follow_up_solution_keeps_phase_1_s_reward(self) -> None:
        replies: list[str] = []
        server, client = _server(
            turns=[_submit(PHASE_1_OF_3, replies), _submit(PHASE_2_OF_3, replies), _submit(PHASE_2_OF_3, replies)]
        )
        follow_up = TASKS["bookshop_3"]["follow_up"] | {"sol_sql": ["SELECT missing_column FROM sales"]}

        response = await _run(server, "bookshop_3", follow_up=follow_up)

        assert replies[1:] == ["SQL failed Phase 2. Your SQL is not correct.\nBudget remaining: 0 bird-coins"] * 2
        assert _opening(client, 3) == DEBUG_WRONG_RESULT_MESSAGE
        assert (response.result.reward, response.result.stop_reason) == (pytest.approx(0.7), "failed")
        assert [(phase["phase"], phase["passed"]) for phase in _phases(response)] == [(1, True), (2, False)]
        assert [e.split(":")[0] for e in response.result.solution_errors] == ["Task bookshop_3, phase 2"] * 2

    async def test_a_failed_close_fails_the_episode_at_cleanup(self) -> None:
        server, client = _server(turns=[_submit(CORRECT_1)])
        client.close_error = ClientResponseError(
            _REQUEST_INFO, history=(), status=500, message="harness still running"
        )

        response = await _run(server)

        assert response.failure.stage == "cleanup"
        assert [turn.turn for turn in response.failure.turns] == [1]

    async def test_token_capture_moves_the_turns_onto_the_training_capture_path(self) -> None:
        server, client = _server(turns=[_submit(CORRECT_1)])
        agent = {"host": "agent", "port": 1, "entrypoint": "app.py", "token_id_capture": True}
        client.global_config_dict = OmegaConf.create(
            {"agent": {"responses_api_agents": {"agent": agent}}, "token_id_capture": {"enabled": True}}
        )

        await _run(server)

        assert client.agent_calls[0]["path"] == "/ng-rollout/bookshop_1-0/training-token-capture/v1/responses"

    @pytest.mark.parametrize("all_agents", [False, True])
    async def test_a_request_that_skips_token_capture_stays_off_the_training_path(self, all_agents: bool) -> None:
        server, client = _server(turns=["Done."])
        agent = {"host": "agent", "port": 1, "entrypoint": "app.py", "token_id_capture": not all_agents}
        client.global_config_dict = OmegaConf.create(
            {
                "agent": {"responses_api_agents": {"agent": agent}},
                "token_id_capture": {"enabled": True, "all_agents": all_agents},
            }
        )

        await _run(server, skip_token_capture=True)

        assert client.agent_calls[0]["path"] == "/ng-rollout/bookshop_1-0/v1/responses"


def _answer_for_the_image_server(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict, float]]:
    """Answer each runner command as passing, for the image's PostgreSQL server that the tests do not have. Return
    each command with its request and its time limit, as the server sends them."""
    commands: list[tuple[str, dict, float]] = []
    exec_in_sandbox = AsyncSandbox.exec

    async def exec(self: AsyncSandbox, command: str, **kwargs: Any) -> SandboxExecResult:
        argv = command.split()
        if len(argv) != 4 or not argv[1].endswith("grading.py"):
            return await exec_in_sandbox(self, command, **kwargs)
        request_path = Path(argv[3])
        commands.append((argv[2], json.loads(request_path.read_text()), kwargs["timeout_s"]))
        request_path.unlink()
        reply = {"setup": {}, "grade": {"passed": True, "execution_error": None, "timed_out": False}}[argv[2]]
        return SandboxExecResult(stdout=json.dumps(reply), stderr="", return_code=0)

    monkeypatch.setattr(AsyncSandbox, "exec", exec)
    return commands


class TestPostgreSQL:
    """Lite and Full: each task's database is in the sandbox image, and the runner uses the image's server."""

    def test_postgresql_needs_its_image(self) -> None:
        with pytest.raises(ValidationError, match="benchmarks/bird_interact/Dockerfile"):
            _server(database_engine="postgresql")

    @pytest.mark.parametrize(
        ("engine", "sandbox_config", "image"),
        [
            ("sqlite", {}, "python:3.12-slim"),
            ("postgresql", {"image": "bird-interact-postgres:lite"}, "bird-interact-postgres:lite"),
        ],
    )
    async def test_the_sandbox_starts_the_configured_image_or_the_sqlite_one(
        self, engine: str, sandbox_config: dict, image: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _answer_for_the_image_server(monkeypatch)
        server, _ = _server(turns=[_submit(CORRECT_1)], database_engine=engine, sandbox_config=sandbox_config)
        images: list[str] = []
        start = type(server).start_episode_sandbox

        async def start_and_record(self: Any, request: Any, cleanup: Any, spec: SandboxSpec, *args, **kwargs) -> Any:
            images.append(spec.image)
            return await start(self, request, cleanup, spec, *args, **kwargs)

        monkeypatch.setattr(type(server), "start_episode_sandbox", start_and_record)

        response = await _run(server)

        assert response.failure is None, response.failure
        assert images == [image]

    async def test_the_a_interact_instructions_name_postgresql(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _answer_for_the_image_server(monkeypatch)
        server, client = _server(
            "a_interact",
            turns=["Done."],
            database_engine="postgresql",
            sandbox_config={"image": "bird-interact-postgres:lite"},
        )

        await _run(server)

        system = client.agent_inputs[0]["input"][0]["content"]
        assert "writes PostgreSQL queries" in system and "SQLite" not in system

    async def test_the_episode_runs_on_the_task_database_in_the_image(
        self, sandbox_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = _answer_for_the_image_server(monkeypatch)
        server, client = _server(
            turns=[_ask_then_submit("What counts as a bestseller?", "SELECT 1")],
            user=[LABELED_BESTSELLERS, TEN_COPIES],
            database_engine="postgresql",
            sandbox_config={"image": "bird-interact-postgres:lite"},
        )

        response = await _run(server)

        assert response.failure is None, response.failure
        assert (response.result.stop_reason, response.result.reward) == ("solved", 0.7)
        assert [(command, request["engine"], request["database"]) for command, request, _ in requests] == [
            ("setup", "postgresql", "bookshop_template"),
            ("grade", "postgresql", "bookshop_template"),
        ]
        # Nothing to upload: the database is in the image.
        assert "source" not in requests[0][1] and not list(sandbox_workspace.rglob("bookshop.sql"))
        assert requests[1][1]["solution_sql"] == [CORRECT_1] and requests[1][1]["apply_if_passed"] is False
        system = client.agent_inputs[0]["input"][0]["content"]
        assert system.startswith("You are a data scientist with great PostgreSQL writing ability.")

    async def test_a_management_phase_is_graded_by_its_test_cases_with_time_for_them(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requests = _answer_for_the_image_server(monkeypatch)
        server, _ = _server(
            turns=[_submit("CREATE TABLE note (x)"), _submit(PHASE_2_OF_3)],
            database_engine="postgresql",
            sandbox_config={"image": "bird-interact-postgres:lite"},
        )
        test_case = "def test_case(pred_sqls, sol_sqls, db_name, conn): pass"

        response = await _run(server, "bookshop_3", category="Management", test_cases=[test_case])

        assert response.failure is None, response.failure
        assert response.result.reward == pytest.approx(1.0)
        (phase_1, phase_1_limit), (phase_2, phase_2_limit) = [
            (request, limit) for command, request, limit in requests if command == "grade"
        ]
        # Phase 1 is a management phase, and its follow-up a query.
        assert (phase_1["test_cases"], phase_1["apply_if_passed"]) == ([test_case], True)
        assert phase_2["test_cases"] == []
        # Each script the grader may run has its SQL timeout. Phase 1: the submission, up to ten queries for the test
        # case, and the application. Phase 2: the submission twice, and the solution, which may run again alone.
        timeout, follow_up_scripts = (
            server.config.sql_timeout_seconds,
            len(TASKS["bookshop_3"]["follow_up"]["sol_sql"]),
        )
        assert phase_1_limit == pytest.approx(12 * timeout + 60)
        assert phase_2_limit == pytest.approx((2 + 2 * follow_up_scripts) * timeout + 60)


class TestDatabaseSandbox:
    async def test_each_episode_has_its_own_sandbox_and_stops_it_when_the_episode_ends(
        self, sandbox_workspace: Path
    ) -> None:
        during: list[list[Path]] = []
        both_running = asyncio.Barrier(2)

        def agent(sql: str) -> Callable[[_Tools], Awaitable[str]]:
            async def turn(tools: _Tools) -> str:
                # Both episodes are in a turn, so both sandboxes exist.
                await asyncio.wait_for(both_running.wait(), 30)
                during.append([path for path in _sandboxes(sandbox_workspace) if (path / "episode.sqlite").exists()])
                return await _submit(sql)(tools)

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
            return await _submit(CORRECT_1)(tools)

        server, _ = _server(turns=[_submit(WRONG), after_a_submission])

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
            return await _submit(CORRECT_1)(tools)

        server, _ = _server(turns=[break_the_sandbox])

        response = await _run(server)

        assert response.result.reward == 0.7
        assert not broken[0].parent.exists()
        assert _sandboxes(sandbox_workspace) == []

    async def test_an_ending_episode_starts_no_replacement_sandbox(
        self, sandbox_workspace: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The episode's cleanup would never stop a sandbox started while it runs, for a tool call still in flight.
        episodes: list[BirdInteractEpisode] = []
        start = BirdInteractEnvironmentServer._start_episode

        async def start_and_keep(self: BirdInteractEnvironmentServer, *args: Any) -> BirdInteractEpisode:
            episodes.append(await start(self, *args))
            return episodes[-1]

        monkeypatch.setattr(BirdInteractEnvironmentServer, "_start_episode", start_and_keep)
        server, _ = _server(turns=[_submit(CORRECT_1)])
        await _run(server)

        with pytest.raises(SandboxCommandError, match="The episode is ending"):
            await server._replace_sandbox(episodes[0])
        assert _sandboxes(sandbox_workspace) == []

    async def test_phase_2_sees_what_phase_1_changed_even_on_a_fresh_sandbox(self, sandbox_workspace: Path) -> None:
        async def break_the_sandbox_then_submit(tools: _Tools) -> str:
            (sandbox,) = _sandboxes(sandbox_workspace)
            (sandbox / "grading.py").unlink()
            # Runs only where phase 1's (empty) table exists.
            return await _submit(f"SELECT * FROM ({PHASE_2_OF_3}) WHERE NOT EXISTS (SELECT 1 FROM note)")(tools)

        server, _ = _server(
            turns=[_submit(f"CREATE TABLE IF NOT EXISTS note (x); {PHASE_1_OF_3}"), break_the_sandbox_then_submit]
        )

        response = await _run(server, "bookshop_3")

        assert response.failure is None, response.failure.failure_reason
        assert response.result.reward == pytest.approx(1.0)

    async def test_a_sandbox_that_keeps_failing_masks_the_sample_and_ends_the_turns(
        self, sandbox_workspace: Path
    ) -> None:
        async def break_the_sandbox(tools: _Tools) -> str:
            (sandbox,) = _sandboxes(sandbox_workspace)
            (sandbox / "grading.py").unlink()
            reply = await tools.call("submit_sql", sql=CORRECT_1)
            # The turn ends at the submission, as a graded one does, so the policy is not called again.
            seen.append((reply.status_code, reply.json(), reply.headers.get(END_ACTIVATION_HEADER)))
            return "Submitted."

        seen: list[tuple] = []
        server, _ = _server(turns=[break_the_sandbox], inconclusive_verification_retries=0)

        response = await _run(server)

        assert seen == [(200, SUBMISSION_NOT_GRADED_REPLY, "true")]

        assert response.failure is None
        assert (response.result.mask_sample, response.result.stop_reason) == (True, SANDBOX_FAILED)
        assert "grading.py grade failed in the sandbox" in response.result.failure_reason
        assert len(response.result.turns) == 1
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
            ({"category": "Management", "test_cases": ["def test_case(): pass"]}, "custom test cases"),
            (
                {
                    "follow_up": {
                        "query": "And now?",
                        "sol_sql": ["SELECT 1"],
                        "category": "Management",
                        "test_cases": ["def test_case(): pass"],
                    }
                },
                "custom test cases",
            ),
            ({"selected_database": "missing"}, "Missing database file"),
        ],
    )
    async def test_a_task_the_server_cannot_run_fails_for_good(self, row: dict, message: str) -> None:
        server, client = _server()

        response = await _run(server, **row)

        assert (response.failure.stage, response.failure.terminal) == ("seed", True)
        assert message in response.failure.failure_reason
        assert client.seeds == []

    async def test_a_query_with_test_cases_is_graded_by_its_result(self) -> None:
        # As in BIRD-Interact, a query's test cases do not count.
        server, _ = _server(turns=[_submit(CORRECT_1)])

        response = await _run(server, test_cases=["def test_case(): assert False"])

        assert (response.failure, response.result.reward) == (None, 0.7)

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
