# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The bird_sql catalog server: its tools, the sandbox each episode gets, and what verify grades."""

import json
import shlex
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.sandbox import SandboxExecResult, SandboxSpec
from nemo_gym.server_runtime import TOOL_ERROR_HEADER, ResourcesSessionLostError
from resources_servers.bird_sql import catalog_app
from resources_servers.bird_sql.app import BirdSqlVerifyRequest, FailureCode
from resources_servers.bird_sql.catalog_app import (
    SANDBOX_EPISODE,
    SANDBOX_TOOL_SERVER,
    BirdSqlCatalogResourcesServer,
    BirdSqlCatalogResourcesServerConfig,
    BirdSqlCatalogSeedRequest,
    _Episode,
)
from resources_servers.bird_sql.catalog_tool_server import Episode, request, serve, wait
from resources_servers.bird_sql.catalog_tools import ACCEPTED, tool_definitions


def _server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs) -> BirdSqlCatalogResourcesServer:
    monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: tmp_path)
    config = BirdSqlCatalogResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        bird_sql_dir=str(tmp_path),
        sandbox_provider="bird_catalog_sandbox",
        sandbox_config={"image": "catalog:bird"},
        catalog_environment={"DEFAULT_MODELS_API_KEY": "embed-key"},
        **kwargs,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"bird_catalog_sandbox": {"docker": {}}}
    return BirdSqlCatalogResourcesServer(config=config, server_client=client)


def _call(call_id: str, name: str, arguments: dict) -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        id=f"fc_{call_id}", call_id=call_id, name=name, arguments=json.dumps(arguments), type="function_call"
    )


def _output(call_id: str, text: str) -> NeMoGymFunctionCallOutput:
    return NeMoGymFunctionCallOutput(call_id=call_id, output=text, type="function_call_output")


def _verify_request(*items, gt_sql: str = "SELECT x FROM t WHERE x > 1") -> BirdSqlVerifyRequest:
    response = NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=list(items),
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return BirdSqlVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
        question="Which x are above 1?",
        gt_sql=gt_sql,
        db_id="TestDB",
    )


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    (tmp_path / "TestDB").mkdir()
    connection = sqlite3.connect(tmp_path / "TestDB" / "TestDB.sqlite")
    connection.execute("CREATE TABLE t (x INTEGER)")
    connection.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    connection.commit()
    connection.close()
    return tmp_path


# ----- tool definitions ----- #


def test_the_rows_offer_the_three_tools_with_the_routes_arguments() -> None:
    tools = {tool["name"]: tool for tool in tool_definitions()}

    assert set(tools) == {"retrieve", "run_sql", "submit"}
    assert tools["retrieve"]["parameters"]["required"] == ["query"]
    assert tools["run_sql"]["parameters"]["required"] == ["sql"]
    assert tools["submit"]["parameters"]["required"] == ["sql"]
    assert set(tools["submit"]["parameters"]["properties"]) == {"sql", "response", "assumptions"}
    kinds = tools["retrieve"]["parameters"]["properties"]["kinds"]["anyOf"][0]["items"]["enum"]
    assert kinds == ["term", "attribute", "definition", "analysis"]
    assert all(tool["type"] == "function" and tool["description"] for tool in tools.values())


# ----- what verify grades ----- #


async def test_verify_grades_the_last_accepted_submission(tmp_path, tiny_db, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch)
    body = _verify_request(
        _call("1", "submit", {"sql": "SELECT x FROM missing"}),
        _output("1", "Not accepted. Fix these first:\n- no table missing"),
        _call("2", "run_sql", {"sql": "SELECT x FROM t"}),
        _output("2", "x\n1\n2\n3"),
        _call("3", "submit", {"sql": "SELECT x FROM t WHERE x > 1"}),
        _output("3", ACCEPTED),
    )

    graded = await server.verify(SimpleNamespace(session={}), body)

    assert graded.reward == 1.0
    assert graded.extracted_sql == "SELECT x FROM t WHERE x > 1"


async def test_a_rejected_submission_or_a_query_in_text_is_not_an_answer(tmp_path, tiny_db, monkeypatch) -> None:
    server = _server(tmp_path, monkeypatch)
    body = _verify_request(
        _call("1", "submit", {"sql": "SELECT x FROM t WHERE x > 1"}),
        _output("1", "Not accepted. The database returned an error: locked"),
    )

    graded = await server.verify(SimpleNamespace(session={}), body)

    assert graded.reward == 0.0
    assert graded.failure_reason == FailureCode.NO_SQL_EXTRACTED


# ----- the episode's sandbox ----- #


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="", return_code=0)
    monkeypatch.setattr(catalog_app, "create_provider", lambda config: AsyncMock())
    monkeypatch.setattr(catalog_app, "AsyncSandbox", MagicMock(return_value=sandbox))
    return sandbox


def _seed() -> BirdSqlCatalogSeedRequest:
    return BirdSqlCatalogSeedRequest(question="Which x are above 1?", db_id="TestDB", gt_sql="SELECT 1")


async def test_the_seed_starts_the_catalog_and_the_tools_and_release_stops_the_sandbox(
    tmp_path, monkeypatch, sandbox
) -> None:
    server = _server(tmp_path, monkeypatch)

    await server.initialize_session_state("s1", _seed())

    [spec] = sandbox.start.await_args.args
    assert isinstance(spec, SandboxSpec)
    assert spec.image == "catalog:bird"
    assert spec.metadata["bird-db-id"] == "TestDB"
    assert json.loads(spec.files[SANDBOX_EPISODE]) == {
        "database": "TestDB",
        "question": "Which x are above 1?",
        "max_tool_calls": 24,
    }
    assert "def serve(" in spec.files[SANDBOX_TOOL_SERVER]
    start, serve_tools, wait_tools = sandbox.exec.await_args_list
    assert start.args == ("/opt/auto-ontology-eval/start-catalog",)
    assert " serve " in serve_tools.args[0] and serve_tools.args[0].endswith("&")
    assert serve_tools.kwargs["env"] == {"DEFAULT_MODELS_API_KEY": "embed-key"}
    assert " wait " in wait_tools.args[0]
    assert server.has_session_state("s1")

    await server.release_session_state("s1")

    sandbox.stop.assert_awaited_once()
    assert not server.has_session_state("s1")


async def test_a_catalog_that_does_not_start_fails_the_seed(tmp_path, monkeypatch, sandbox) -> None:
    sandbox.exec.return_value = SandboxExecResult(stdout="", stderr="no PGDATA", return_code=1)

    with pytest.raises(RuntimeError, match="no PGDATA"):
        await _server(tmp_path, monkeypatch).initialize_session_state("s1", _seed())


# ----- tool routes ----- #


def _seated(tmp_path, monkeypatch, reply: SandboxExecResult) -> tuple[BirdSqlCatalogResourcesServer, AsyncMock]:
    server = _server(tmp_path, monkeypatch)
    sandbox = AsyncMock()
    sandbox.exec.return_value = reply
    server._episodes["s1"] = _Episode(sandbox=sandbox)
    return server, sandbox


async def test_a_tool_call_runs_in_the_episodes_tool_process(tmp_path, monkeypatch) -> None:
    reply = SandboxExecResult(stdout=json.dumps({"output": "x\n1", "accepted": False}), stderr="", return_code=0)
    server, sandbox = _seated(tmp_path, monkeypatch, reply)

    response = await server.run_sql(
        SimpleNamespace(session={SESSION_ID_KEY: "s1"}), catalog_app.RunSqlArguments(sql="SELECT x FROM t")
    )

    assert response.body.decode() == "x\n1"
    [call] = sandbox.exec.await_args_list
    assert json.loads(shlex.split(call.args[0])[-1]) == {
        "tool": "run_sql",
        "args": {"sql": "SELECT x FROM t"},
    }


async def test_an_accepted_submission_answers_with_the_marker_verify_reads(tmp_path, monkeypatch) -> None:
    reply = SandboxExecResult(stdout=json.dumps({"output": "Accepted.", "accepted": True}), stderr="", return_code=0)
    server, _ = _seated(tmp_path, monkeypatch, reply)

    response = await server.submit(
        SimpleNamespace(session={SESSION_ID_KEY: "s1"}), catalog_app.SubmitArguments(sql="SELECT 1")
    )

    assert response.body.decode() == ACCEPTED


async def test_a_tool_process_that_fails_is_a_tool_error(tmp_path, monkeypatch) -> None:
    server, _ = _seated(
        tmp_path, monkeypatch, SandboxExecResult(stdout="", stderr="", return_code=None, error_type="timeout")
    )

    response = await server.retrieve(
        SimpleNamespace(session={SESSION_ID_KEY: "s1"}), catalog_app.RetrieveArguments(query="schools")
    )

    assert response.headers[TOOL_ERROR_HEADER] == "true"
    assert "timeout" in response.body.decode()


async def test_a_tool_call_without_a_seeded_session_is_refused(tmp_path, monkeypatch) -> None:
    with pytest.raises(ResourcesSessionLostError):
        await _server(tmp_path, monkeypatch).run_sql(
            SimpleNamespace(session={SESSION_ID_KEY: "unknown"}), catalog_app.RunSqlArguments(sql="SELECT 1")
        )


# ----- the tool process in the sandbox ----- #


def _episode(max_tool_calls: int = 3) -> tuple[Episode, list]:
    calls: list = []

    def retrieve(context, query, kinds=None):
        calls.append(("retrieve", query, kinds))
        return f"found {query}"

    def run_sql(context, sql):
        if "boom" in sql:
            raise RuntimeError("database is locked")
        return "rows"

    def submit(context, sql="", response="", assumptions=""):
        return ("Accepted.", object()) if "good" in sql else ("Not accepted. Fix it.", None)

    functions = {"retrieve": retrieve, "run_sql": run_sql, "submit": submit}
    return Episode(functions, context=object(), max_tool_calls=max_tool_calls), calls


def test_the_tools_call_auto_ontologys_functions_and_report_failures_as_observations() -> None:
    episode, calls = _episode()

    assert episode.call("retrieve", {"query": "schools", "kinds": ["term"]})["output"] == "found schools"
    assert calls == [("retrieve", "schools", ["term"])]
    assert episode.call("run_sql", {"sql": "boom"})["output"] == "run_sql failed: database is locked"
    assert episode.call("run_sql", {"query": "SELECT 1"})["output"].startswith("Wrong arguments for run_sql")
    assert episode.call("drop_table", {})["output"].startswith("Unknown tool 'drop_table'")


def test_past_the_budget_only_submit_is_served() -> None:
    episode, _ = _episode(max_tool_calls=1)

    episode.call("run_sql", {"sql": "SELECT 1"})

    assert "budget of 1 calls is spent" in episode.call("retrieve", {"query": "schools"})["output"]
    assert episode.call("submit", {"sql": "good"}) == {"output": "Accepted.", "accepted": True}


def test_an_accepted_submission_ends_the_episodes_tools() -> None:
    episode, _ = _episode()

    assert episode.call("submit", {"sql": "bad"}) == {"output": "Not accepted. Fix it.", "accepted": False}
    assert episode.call("submit", {"sql": "good"})["accepted"] is True
    assert "already submitted" in episode.call("run_sql", {"sql": "SELECT 1"})["output"]


def test_requests_reach_the_episode_over_its_socket(tmp_path) -> None:
    socket_path = str(tmp_path / "tools.sock")
    episode, _ = _episode()
    threading.Thread(target=serve, args=(socket_path, episode), daemon=True).start()

    assert wait(socket_path, 5)
    assert request(socket_path, {"tool": "retrieve", "args": {"query": "schools"}}) == {
        "output": "found schools",
        "accepted": False,
    }
    assert request(socket_path, {"tool": "submit", "args": {"sql": "good"}})["accepted"] is True
