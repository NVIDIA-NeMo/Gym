# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The resources server's session lifecycle, tools and scoring, on a real AutomationBench example task."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
)
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.automationbench import app as app_module
from resources_servers.automationbench.app import (
    AutomationBenchResourcesServer,
    AutomationBenchResourcesServerConfig,
)
from resources_servers.automationbench.task_data import TaskData


EXAMPLE_FPATH = Path(__file__).parents[3] / "benchmarks" / "automationbench" / "data" / "example.jsonl"
# The first example task: mark the Meridian Corp deal won, then send the win notice to two of five team mailboxes.
# Its first assertion is the deal's stage; the three "not sent to" assertions already pass, so they are guardrails.
TASK = json.loads(EXAMPLE_FPATH.read_text().splitlines()[0])
OPPORTUNITY_URL = "https://yourinstance.salesforce.com/services/data/v61.0/sobjects/Opportunity/006xx000004MER1"


def _server(**overrides) -> AutomationBenchResourcesServer:
    config = AutomationBenchResourcesServerConfig(
        **{"host": "0.0.0.0", "port": 8080, "entrypoint": "", "name": "automationbench_resources_server"} | overrides
    )
    return AutomationBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def client():
    with TestClient(_server().setup_webserver()) as client:
        yield client


def _seed(session_id: str = "session", task: dict[str, Any] = TASK) -> ResourcesSeedSessionRequest:
    # As single_agent_turn seeds: task_data without the params, and the params beside it.
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=f"{session_id}-0"),
        task_id=TaskId(taskset="automationbench", task_id=task["task_id"]),
        task_data={key: value for key, value in task.items() if key != "responses_create_params"},
        responses_create_params=task["responses_create_params"],
    )


def _post_seed(client: TestClient, seed: ResourcesSeedSessionRequest):
    return client.post("/seed_session", json=seed.model_dump(mode="json"))


def _verify_body(tools: list[dict[str, Any]] = TASK["responses_create_params"]["tools"]) -> dict[str, Any]:
    return {
        "task_id": TASK["task_id"],
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=[], tools=tools).model_dump(
            mode="json"
        ),
        "response": NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        ).model_dump(mode="json"),
    }


def _verify(client: TestClient) -> dict[str, Any]:
    return client.post("/verify", json=_verify_body()).json()


def _win_the_deal(client: TestClient) -> str:
    response = client.post(
        "/api_fetch",
        json={"method": "PATCH", "url": OPPORTUNITY_URL, "params": {}, "body": {"StageName": "Closed Won"}},
    )
    assert response.status_code == 200
    return response.text


def test_example_rows_are_valid_task_data_and_carry_the_tools() -> None:
    for line in EXAMPLE_FPATH.read_text().splitlines():
        row = json.loads(line)
        TaskData.model_validate(row)
        assert [tool["name"] for tool in row["responses_create_params"]["tools"]] == [
            "api_search",
            "api_fetch",
            "base64_encode",
        ]


async def test_exported_fixture_checks_actual_tool_mutations_and_guardrail_scoring() -> None:
    results = await exercise_verifier_fixture(
        app_module.VERIFIER_FIXTURE, reward_range=(0.0, 1.0), determinism="unknown"
    )
    assert [(result.name, result.observed_rewards) for result in results] == [
        ("objective_completed", (1.0,)),
        ("objective_unfinished", (0.0,)),
        ("broken_guardrail_zeroes_completed_objective", (0.0,)),
        ("missing_task_info", ()),
    ]


def test_the_row_tools_are_the_environment_tools() -> None:
    tools = {tool["name"]: tool for tool in TASK["responses_create_params"]["tools"]}

    for tool in app_module.tool_env().tool_defs:
        assert (tools[tool.name]["parameters"], tools[tool.name]["strict"]) == (tool.parameters, tool.strict)


def test_an_untouched_world_scores_zero_with_its_guardrails_kept(client):
    assert _post_seed(client, _seed()).status_code == 200

    result = _verify(client)

    assert (result["reward"], result["aa_headline"], result["partial_credit"]) == (0.0, 0.0, 0.0)
    assert (result["guardrails_total"], result["guardrails_violated"]) == (3.0, 0.0)
    assert (result["objectives_total"], result["objectives_passed"]) == (3.0, 0.0)
    assert (result["task_name"], result["domain"], result["mask_sample"]) == ("sales.multi_hop_lookup", "sales", False)
    assert "assertion_results" not in result


def test_tool_calls_change_the_session_world_that_verify_scores(client):
    _post_seed(client, _seed())

    found = client.post("/api_search", json={"query": "salesforce update opportunity", "top_k": 1})
    assert json.loads(found.text)["results"][0]["url"].endswith("/sobjects/Opportunity/{id}")
    _win_the_deal(client)

    result = _verify(client)

    assert result["objectives_passed"] == 1.0
    assert result["reward"] == result["aa_headline"] == pytest.approx(1 / 3)
    assert result["task_completed_correctly"] == 0.0


def test_sessions_hold_separate_worlds(client):
    _post_seed(client, _seed("first"))
    _win_the_deal(client)
    first = dict(client.cookies)
    _post_seed(client, _seed("second"))

    assert _verify(client)["objectives_passed"] == 0.0
    # A seed that replaced one shared world would have lost the first session's won deal.
    client.cookies.clear()
    client.cookies.update(first)
    assert _verify(client)["objectives_passed"] == 1.0


def test_tool_errors_reach_the_model_as_verifiers_words_them(client):
    _post_seed(client, _seed())

    assert client.post("/no_such_tool", json={}).text == "'no_such_tool'"
    assert client.post("/api_fetch", json=["GET"]).text == "Expected tool arguments to be a dict, got list: ['GET']"
    assert client.post("/base64_encode", json={"text": "hi"}).text == "aGk="
    missing = client.post("/base64_encode", json={})
    assert (missing.status_code, "text" in missing.text) == (200, True)


def test_a_tool_call_or_verify_without_a_seeded_session_is_a_lost_session(client):
    response = client.post("/api_search", json={"query": "gmail"})

    assert (response.status_code, response.json()["detail"]["failure_kind"]) == (404, "session_lost")
    result = _verify(client)
    assert (result["mask_sample"], result["failure_kind"], result["reward"]) == (True, "session_lost", 0.0)


def test_a_repeated_verify_returns_the_first_scores_and_the_world_is_gone(client):
    _post_seed(client, _seed())
    _win_the_deal(client)

    first = _verify(client)

    assert _verify(client) == first
    assert client.post("/api_search", json={"query": "gmail"}).status_code == 404


def test_partial_credit_can_carry_the_reward_and_assertions_can_be_exported():
    with TestClient(_server(reward_fn="partial_credit", export_assertion_results=True).setup_webserver()) as client:
        _post_seed(client, _seed())
        _win_the_deal(client)
        result = _verify(client)

    assert result["reward"] == result["partial_credit"] == pytest.approx(1 / 3)
    records = result["assertion_results"]
    assert [(r["role"], r["passed"]) for r in records] == [
        ("objective", True),
        ("objective", False),
        ("objective", False),
        ("guardrail", True),
        ("guardrail", True),
        ("guardrail", True),
    ]


def test_a_rubric_that_raises_masks_the_sample(client, monkeypatch):
    def broken(state, **kwargs):
        raise KeyError("assertion")

    monkeypatch.setattr(app_module, "score", lambda state, **kwargs: broken(state))
    _post_seed(client, _seed())

    result = _verify(client)

    assert (result["mask_sample"], result["failure_kind"], result["reward"]) == (True, "verifier_error", 0.0)
    assert result["failure_reason"] == "AutomationBench rubric raised: KeyError: 'assertion'"


def test_a_row_prepared_without_tools_is_refused_at_seed(client):
    # Rows prepared before this server carry no tools; the model could not act, so the episode fails before any
    # model call instead of scoring a zero that is no grade of it.
    task = TASK | {
        "responses_create_params": {
            key: value for key, value in TASK["responses_create_params"].items() if key != "tools"
        }
    }

    response = _post_seed(client, _seed(task=task))

    assert response.status_code == 422
    assert "gym eval prepare --benchmark automationbench" in response.json()["detail"]
    assert client.post("/api_search", json={"query": "gmail"}).status_code == 404


def test_a_seed_without_the_params_is_served_and_a_toolless_row_is_masked_at_verify(client):
    # A caller that sends only task_data cannot be checked at seed; /verify still masks a row without tools.
    seed = _seed().model_copy(update={"responses_create_params": None})

    assert _post_seed(client, seed).status_code == 200
    result = client.post("/verify", json=_verify_body(tools=[])).json()

    assert (result["mask_sample"], result["failure_kind"], result["reward"]) == (True, "verifier_error", 0.0)
    assert "gym eval prepare --benchmark automationbench" in result["failure_reason"]
    assert result["task_name"] == "sales.multi_hop_lookup"


def test_a_task_counted_as_zero_before_verify_stays_in_its_domain():
    # With count_missing_rollouts_as_zero or count_failure_classes_as_zero, a task whose every repeat failed before
    # /verify reaches the metrics as reward-0 rows carrying task_id but no domain.
    scores = {"aa_headline": 1.0, "partial_credit": 1.0, "task_completed_correctly": 1.0}
    scored = {"task_id": "sales.win/1", "domain": "sales", "reward": 1.0, **scores}
    counted = {"task_id": "sales.lose/2", "reward": 0.0}
    metrics = _server().compute_metrics([[scored], [counted]])
    for name in scores:
        assert metrics[f"pass@1[avg-of-1]/{name}"] == metrics[f"sales/pass@1[avg-of-1]/{name}"] == 50

    # A domain a repeat reports wins over the task_id's prefix.
    metrics = _server().compute_metrics([[counted, {**scored, "task_id": "sales.lose/2", "domain": "marketing"}]])
    assert metrics["marketing/pass@1[avg-of-2]/aa_headline"] == 50
    assert not any(key.startswith("sales/") for key in metrics)


def test_a_task_without_assertions_is_refused(client):
    task = TASK | {"info": TASK["info"] | {"assertions": []}}

    assert _post_seed(client, _seed(task=task)).status_code == 422


def test_an_initial_state_the_world_rejects_is_refused_at_seed(client):
    task = TASK | {"info": TASK["info"] | {"initial_state": {"salesforce": {"opportunities": "not-a-list"}}}}

    response = _post_seed(client, _seed(task=task))

    assert response.status_code == 422
    assert "not a valid AutomationBench world" in response.json()["detail"]


def test_close_releases_the_world_and_the_session_follows_the_contract():
    server = _server()
    app = server.setup_webserver()
    with TestClient(app) as client:
        seed = _seed()
        _post_seed(client, seed)
        close = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=seed.episode_id)

        assert client.post("/close_session", json=close.model_dump(mode="json")).status_code == 200
        assert client.post("/api_search", json={"query": "gmail"}).status_code == 404
        assert _post_seed(client, seed).status_code == 409

    check_resources_session_contract(
        app,
        _seed("contract"),
        keeps_state=True,
        holds_state=lambda session_id: session_id in server._episodes,
        verify_body=_verify_body(),
    )
