# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.server_utils import ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.bird_interact.app import (
    SANDBOX_FAILED,
    USER_SIMULATOR_FAILED,
    BirdInteractResourcesServer,
    BirdInteractResourcesServerConfig,
)
from resources_servers.bird_interact.grading import grade, main, split_statements, strip_round_calls
from resources_servers.bird_interact.schemas import BirdInteractSeedSessionResponse
from resources_servers.bird_interact.simulator import UNANSWERABLE_REPLY, parse_action
from resources_servers.bird_interact.task_data import BirdInteractTask


DATA = Path(__file__).parents[1] / "data"
TASKS = {row["instance_id"]: row for row in map(json.loads, (DATA / "example.jsonl").read_text().splitlines())}
EPISODE = {"rollout_id": "rollout", "attempt": 0}
CORRECT_1 = TASKS["bookshop_1"]["sol_sql"][0]
# Session sandboxes run on the host; each test gets its own workspace (see `sandbox_workspace`).
LOCAL_SANDBOX: dict[str, dict[str, str]] = {"local": {}}


@pytest.fixture(autouse=True)
def sandbox_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    workspace = tmp_path / "sandboxes"
    monkeypatch.setitem(LOCAL_SANDBOX["local"], "workspace_root", str(workspace))
    return workspace


def _model_reply(text: str) -> MagicMock:
    body = {
        "id": "user-model",
        "created_at": 0,
        "model": "user",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "tools": [],
    }
    response = MagicMock(ok=True, status=200, cookies={})
    response.read = AsyncMock(return_value=json.dumps(body).encode())
    return response


def _server(**config) -> tuple[BirdInteractResourcesServer, MagicMock]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server_client.post = AsyncMock()
    server = BirdInteractResourcesServer(
        config=BirdInteractResourcesServerConfig(
            name="bird_interact",
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            database_dir=str(DATA / "example_databases"),
            user_model=ModelServerRef(type="responses_api_models", name="user_model"),
            **({"sandbox_provider": LOCAL_SANDBOX} | config),
        ),
        server_client=server_client,
    )
    return server, server_client


def _sandbox_root(server: BirdInteractResourcesServer, task: str) -> Path:
    """The host directory of a session's sandbox files, which the local provider exposes."""
    return Path(server._sessions[f"session-{task}"].sandbox.root)


def _seed(client: TestClient, task: str, mode: str = "c_interact", **overrides):
    task_data = TASKS[task] | {"interaction_mode": mode} | overrides
    return client.post(
        "/seed_session",
        json={
            "resources_session_id": f"session-{task}",
            "episode_id": EPISODE,
            "task_id": {"taskset": "bird_interact:example", "task_id": task},
            "task_data": task_data,
        },
    )


def _verify(client: TestClient) -> dict:
    response = client.post("/verify", json={"responses_create_params": {"input": "q"}, "response": _response()})
    assert response.status_code == 200
    return response.json()


def _response() -> dict:
    return json.loads(_model_reply("").read.return_value)


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


class TestSimulatorActions:
    def test_a_term_that_is_not_annotated_is_unanswerable(self) -> None:
        task = BirdInteractTask.model_validate(TASKS["bookshop_1"])
        assert parse_action('labeled("bestsellers")', task).kind == "labeled"
        assert parse_action('labeled("price")', task).kind == "unanswerable"
        assert parse_action("I think it is unlabeled", task).kind == "unanswerable"
        assert parse_action('unlabeled("ORDER BY bestsellers DESC")', task).argument == "ORDER BY bestsellers DESC"


class TestSeed:
    def test_the_session_contract_holds(self) -> None:
        server, _ = _server()
        check_resources_session_contract(
            server.setup_webserver(),
            ResourcesSeedSessionRequest(
                resources_session_id="contract",
                episode_id=EpisodeId(**EPISODE),
                task_id=TaskId(taskset="bird_interact:example", task_id="bookshop_1"),
                task_data=TASKS["bookshop_1"] | {"interaction_mode": "c_interact"},
            ),
            keeps_state=True,
            holds_state=lambda session_id: session_id in server._sessions,
            verify_body={"responses_create_params": {"input": "q"}, "response": _response()},
            seed_response_model=BirdInteractSeedSessionResponse,
        )

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"sol_sql": []}, "has no sol_sql"),
            ({"test_cases": ["def test_case(): pass"]}, "custom test cases"),
            ({"interaction_mode": "chat"}, "interaction_mode"),
            ({"selected_database": "missing"}, "Missing database file"),
        ],
    )
    def test_a_task_the_server_cannot_run_is_refused_for_good(self, overrides: dict, message: str) -> None:
        client = TestClient(_server()[0].setup_webserver())
        refused = _seed(client, "bookshop_1", **overrides)
        assert refused.status_code == 422
        assert message in refused.json()["detail"]

    def test_the_seed_reply_s_context_hides_the_knowledge_the_task_masks(self) -> None:
        client = TestClient(_server()[0].setup_webserver())

        context = _seed(client, "bookshop_1").json()["context"]

        assert "Bestseller" not in context["knowledge"]
        assert "Revenue" in context["knowledge"]
        # One critical and one knowledge ambiguity, plus the default patience of 3.
        assert context["question_budget"] == 5

    def test_the_a_interact_context_leaves_out_what_the_tools_charge_for(self) -> None:
        client = TestClient(_server()[0].setup_webserver())

        context = _seed(client, "bookshop_1", mode="a_interact").json()["context"]

        assert (context["schema_text"], context["column_meanings"], context["knowledge"]) == (None, None, None)
        # 6, plus one ask (2) for each of the two ambiguities, plus the default patience of 6.
        assert context["coin_budget"] == 16.0
        # No other route returns the context, so the agent cannot reach it through its tool grant.
        assert client.post("/episode_context").status_code == 404


class TestConversationalMode:
    def test_a_question_goes_through_both_simulator_stages(self) -> None:
        server, server_client = _server()
        server_client.post.side_effect = [_model_reply('labeled("bestsellers")'), _model_reply("At least 10 copies.")]
        client = TestClient(server.setup_webserver())
        _seed(client, "bookshop_1")

        answer = client.post("/c_interact/ask", json={"question": "What counts as a bestseller?"}).json()

        assert answer == {"reply": "At least 10 copies.", "answered": True, "questions_left": 4}
        responder_prompt = server_client.post.await_args_list[1].kwargs["json"].input[0].content
        assert "HAVING SUM(s.quantity) >= 10" in responder_prompt
        assert "Bestseller" in responder_prompt  # The masked knowledge entry backs the answer.

    def test_an_unanswerable_question_gets_a_refusal_without_a_second_model_call(self) -> None:
        server, server_client = _server()
        server_client.post.side_effect = [_model_reply("unanswerable()")]
        client = TestClient(server.setup_webserver())
        _seed(client, "bookshop_1")

        answer = client.post("/c_interact/ask", json={"question": "Just give me the SQL?"}).json()

        assert answer["reply"] == UNANSWERABLE_REPLY
        assert server_client.post.await_count == 1

    def test_questions_stop_when_the_budget_is_spent(self) -> None:
        server, server_client = _server(question_patience=0)
        server_client.post.side_effect = [_model_reply("unanswerable()")] * 2
        client = TestClient(server.setup_webserver())
        _seed(client, "bookshop_1")

        answers = [client.post("/c_interact/ask", json={"question": "?"}).json() for _ in range(3)]

        assert [answer["answered"] for answer in answers] == [True, True, False]
        assert "last question" in answers[1]["reply"]

    def test_the_debug_turn_after_a_failed_submission_takes_no_question(self) -> None:
        server, server_client = _server()
        client = TestClient(server.setup_webserver())
        _seed(client, "bookshop_1")
        client.post("/c_interact/submit", json={"sql": "SELECT name FROM authors"})

        answer = client.post("/c_interact/ask", json={"question": "What did I get wrong?"}).json()

        assert answer["answered"] is False
        server_client.post.assert_not_awaited()

    def test_a_debugged_submission_earns_the_debug_reward(self) -> None:
        client = TestClient(_server()[0].setup_webserver())
        _seed(client, "bookshop_1")

        first = client.post("/c_interact/submit", json={"sql": "SELECT name FROM authors"}).json()
        second = client.post("/c_interact/submit", json={"sql": CORRECT_1}).json()

        assert (first["passed"], first["finished"], "one more attempt" in first["feedback"]) == (False, False, True)
        assert (second["passed"], second["finished"]) == (True, True)
        verification = _verify(client)
        assert verification["reward"] == 0.5
        assert verification["phases"] == [{"phase": 1, "passed": True, "attempts": 2, "reward": 0.5}]

    def test_two_failed_submissions_end_the_task(self) -> None:
        client = TestClient(_server()[0].setup_webserver())
        _seed(client, "bookshop_1")

        client.post("/c_interact/submit", json={"sql": "SELECT nope"})
        last = client.post("/c_interact/submit", json={"sql": "SELECT name FROM authors"}).json()

        assert (last["passed"], last["finished"]) == (False, True)
        assert _verify(client)["reward"] == 0.0

    def test_the_follow_up_question_is_a_second_phase(self) -> None:
        client = TestClient(_server()[0].setup_webserver())
        _seed(client, "bookshop_3")

        first = client.post("/c_interact/submit", json={"sql": TASKS["bookshop_3"]["sol_sql"][0]}).json()
        second = client.post("/c_interact/submit", json={"sql": TASKS["bookshop_3"]["follow_up"]["sol_sql"][0]}).json()

        assert TASKS["bookshop_3"]["follow_up"]["query"] in first["feedback"]
        assert (first["finished"], second["finished"], second["phase"]) == (False, True, 2)
        assert _verify(client)["reward"] == pytest.approx(1.0)

    def test_a_single_phase_reward_schedule_skips_the_follow_up(self) -> None:
        client = TestClient(_server(phase_rewards=[{"first_attempt": 1.0, "after_debug": 0.5}])[0].setup_webserver())
        _seed(client, "bookshop_3")

        result = client.post("/c_interact/submit", json={"sql": TASKS["bookshop_3"]["sol_sql"][0]}).json()

        assert result["finished"] is True
        assert _verify(client)["reward"] == 1.0

    def test_an_a_interact_session_refuses_the_c_interact_routes(self) -> None:
        server, server_client = _server()
        client = TestClient(server.setup_webserver())
        _seed(client, "bookshop_1", mode="a_interact")

        # Without the check, this submission would be graded without its BIRD-COIN cost.
        assert client.post("/c_interact/submit", json={"sql": CORRECT_1}).status_code == 409
        assert client.post("/c_interact/ask", json={"question": "?"}).status_code == 409

        server_client.post.assert_not_awaited()
        verification = _verify(client)
        assert (verification["reward"], verification["phases"], verification["questions_asked"]) == (0.0, [], 0)


class TestAgenticMode:
    def test_each_tool_charges_its_cost_and_hides_masked_knowledge(self) -> None:
        client = TestClient(_server()[0].setup_webserver())
        _seed(client, "bookshop_1", mode="a_interact")

        names = client.post("/a_interact/get_all_external_knowledge_names").json()
        meaning = client.post(
            "/a_interact/get_column_meaning", json={"table_name": "SALES", "column_name": "quantity"}
        ).json()

        # 6 + 2 for each of the two ambiguities + 6 patience = 16; the two lookups cost 0.5 each.
        assert meaning["coins_left"] == 15.0
        assert "Bestseller" not in names["observation"]
        assert meaning["observation"] == "Number of copies sold in the sale."

    def test_once_the_budget_is_spent_only_submit_runs(self) -> None:
        client = TestClient(_server(coin_patience=0)[0].setup_webserver())
        _seed(client, "bookshop_1", mode="a_interact")
        for _ in range(10):
            client.post("/a_interact/execute", json={"sql": "SELECT 1"})

        refused = client.post("/a_interact/get_schema").json()
        submitted = client.post("/a_interact/submit", json={"sql": CORRECT_1}).json()

        assert refused["observation"] == "Your budget is spent. Only submit is allowed now."
        assert "That's right" in submitted["observation"]
        verification = _verify(client)
        assert (verification["reward"], verification["finished"]) == (0.7, True)

    def test_a_failed_submit_with_no_budget_left_ends_the_task_and_records_the_phase(self) -> None:
        client = TestClient(_server(coin_patience=0)[0].setup_webserver())
        _seed(client, "bookshop_1", mode="a_interact")
        for _ in range(10):
            client.post("/a_interact/execute", json={"sql": "SELECT 1"})

        submitted = client.post("/a_interact/submit", json={"sql": "SELECT name FROM authors"}).json()

        assert "the task ends here" in submitted["observation"]
        verification = _verify(client)
        assert verification["finished"] is True
        assert verification["phases"] == [{"phase": 1, "passed": False, "attempts": 1, "reward": 0.0}]

    def test_an_exploratory_query_returns_a_bounded_preview(self) -> None:
        client = TestClient(_server()[0].setup_webserver())
        _seed(client, "bookshop_1", mode="a_interact")

        result = client.post("/a_interact/execute", json={"sql": "SELECT s.sale_id FROM sales s, authors a"}).json()

        assert result["observation"].startswith("95 rows\n(1,)")
        assert result["observation"].endswith("... 75 more rows")

    def test_a_simulator_failure_masks_the_sample(self) -> None:
        server, server_client = _server()
        server_client.post.side_effect = RuntimeError("user model down")
        client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
        _seed(client, "bookshop_1", mode="a_interact")

        assert client.post("/a_interact/ask", json={"question": "?"}).status_code == 500

        verification = _verify(client)
        assert (verification["mask_sample"], verification["failure_kind"]) == (True, USER_SIMULATOR_FAILED)

    def test_a_c_interact_session_refuses_the_tools(self) -> None:
        client = TestClient(_server()[0].setup_webserver())
        _seed(client, "bookshop_1")

        assert client.post("/a_interact/get_schema").status_code == 409


class TestSessionSandbox:
    def test_each_session_has_its_own_sandbox_and_close_stops_it(self) -> None:
        server, _ = _server()
        client = TestClient(server.setup_webserver())
        _seed(client, "bookshop_1")
        _seed(client, "bookshop_2")
        first, second = _sandbox_root(server, "bookshop_1"), _sandbox_root(server, "bookshop_2")
        assert first != second
        assert (first / "episode.sqlite").exists() and (second / "episode.sqlite").exists()

        closed = client.post(
            "/close_session", json={"resources_session_id": "session-bookshop_1", "episode_id": EPISODE}
        )

        assert closed.status_code == 200
        assert not first.parent.exists() and second.exists()

    def test_preprocess_sql_runs_in_the_sandbox_before_the_episode(self) -> None:
        client = TestClient(_server()[0].setup_webserver())
        _seed(client, "bookshop_1", mode="a_interact", preprocess_sql=["DELETE FROM sales"])

        result = client.post("/a_interact/execute", json={"sql": "SELECT COUNT(*) FROM sales"}).json()

        assert result["observation"] == "1 rows\n(0,)"

    def test_a_command_that_fails_runs_again_on_a_fresh_sandbox(self) -> None:
        server, _ = _server()
        client = TestClient(server.setup_webserver())
        _seed(client, "bookshop_1")
        broken = _sandbox_root(server, "bookshop_1")
        (broken / "grading.py").unlink()

        submitted = client.post("/c_interact/submit", json={"sql": CORRECT_1})

        assert submitted.json()["passed"] is True
        assert _sandbox_root(server, "bookshop_1") != broken
        assert not broken.parent.exists()

    def test_a_sandbox_that_keeps_failing_is_retryable_in_c_interact(self) -> None:
        server, _ = _server(inconclusive_verification_retries=0)
        client = TestClient(server.setup_webserver())
        _seed(client, "bookshop_1")
        (_sandbox_root(server, "bookshop_1") / "grading.py").unlink()

        submitted = client.post("/c_interact/submit", json={"sql": CORRECT_1})

        assert submitted.status_code == 503
        assert "grading.py grade failed in the sandbox" in submitted.json()["detail"]

    def test_a_sandbox_failure_masks_an_a_interact_sample(self) -> None:
        server, _ = _server(inconclusive_verification_retries=0)
        client = TestClient(server.setup_webserver())
        _seed(client, "bookshop_1", mode="a_interact")
        (_sandbox_root(server, "bookshop_1") / "grading.py").unlink()

        assert client.post("/a_interact/execute", json={"sql": "SELECT 1"}).status_code == 503

        verification = _verify(client)
        assert (verification["mask_sample"], verification["failure_kind"]) == (True, SANDBOX_FAILED)

    def test_a_sandbox_that_cannot_start_fails_the_seed_as_retryable(self) -> None:
        client = TestClient(_server(sandbox_provider={"local": {"shell": "/missing/shell"}})[0].setup_webserver())

        seeded = _seed(client, "bookshop_1")

        assert seeded.status_code == 503
        assert "Could not start the session sandbox" in seeded.json()["detail"]

    def test_a_failed_close_keeps_the_session_so_the_next_close_stops_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server, _ = _server()
        client = TestClient(server.setup_webserver(), raise_server_exceptions=False)
        _seed(client, "bookshop_1")
        stop = AsyncSandbox.stop
        failures = iter([RuntimeError("provider unavailable")])

        async def flaky_stop(sandbox: AsyncSandbox) -> None:
            if (error := next(failures, None)) is not None:
                raise error
            await stop(sandbox)

        monkeypatch.setattr(AsyncSandbox, "stop", flaky_stop)
        close = {"resources_session_id": "session-bookshop_1", "episode_id": EPISODE}

        assert client.post("/close_session", json=close).status_code == 500
        assert "session-bookshop_1" in server._sessions
        assert client.post("/close_session", json=close).status_code == 200
        assert "session-bookshop_1" not in server._sessions

    def test_shutdown_stops_the_sandboxes_left_open(self, sandbox_workspace: Path) -> None:
        server, _ = _server()
        with TestClient(server.setup_webserver()) as client:
            _seed(client, "bookshop_1")
            assert any(sandbox_workspace.iterdir())

        assert not any(sandbox_workspace.iterdir())
