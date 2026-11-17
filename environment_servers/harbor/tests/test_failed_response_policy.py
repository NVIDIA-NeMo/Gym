# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failed agent responses retain provenance and obey Harbor outcome policy."""

import pytest

import environment_servers.harbor.tests.test_app as harbor_tests
from environment_servers.harbor.tests.conftest import make_server, run_episode
from nemo_gym.client import MODEL_RESPONSE_TIMEOUT_REASON


failed_turn = harbor_tests.TestAgentStage.failed_turn
reported_turn = harbor_tests.TestAgentStage.reported_turn


async def test_failed_turn_failure_is_caused_by_the_selected_event(tmp_path, monkeypatch):
    """The episode failure's report chains to the agent's selected event (``raise ... from cause``)."""
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    server.server_client.turn_response = reported_turn(
        '{"failure_kind":"model_transport_error"}', "model_transport_error"
    )

    response = await run_episode(server, task)

    report = response.failure_report
    episode_event = report.events[report.failure_id]
    assert episode_event.failure_kind == "model_transport_error"
    assert episode_event.cause_id == "agent"
    assert report.events["agent"].failure_kind == "model_transport_error"


async def test_failed_turn_whose_report_selects_nothing_follows_the_outcome_policy(tmp_path, monkeypatch):
    """A provider 400 after a reply with an empty report is graded, as on single_agent_turn."""
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    turn = failed_turn('{"failure_kind":"model_request_error","status_code":400}')
    turn["failure_report"] = {"failure_id": None, "events": {}}
    server.server_client.turn_response = turn

    response = await run_episode(server, task)

    assert response.failure is None, response.failure
    assert any("test.sh" in call["command"] for call in sandbox.execs)


@pytest.mark.parametrize(("connected", "terminal"), [(False, False), (True, True)], ids=["never-sent", "sent"])
async def test_enclosing_500_is_retried_only_when_the_request_was_never_sent(
    tmp_path, monkeypatch, connected, terminal
):
    """The model-request-not-sent rule now applies on Harbor too."""
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    turn = failed_turn('{"failure_kind":"model_request_error","status_code":500}')
    turn["failure_report"] = {
        "failure_id": "selected",
        "events": {
            "selected": {
                "event_id": "selected",
                "boundary": "model",
                "failure_kind": "model_request_error",
                "operation": "model_request",
                "operation_id": "operation",
                "attempt_id": "attempt-1",
                "details": {"request_upload_complete": connected},
                "foreign_error": {"exception_type": "aiohttp.client_exceptions.ClientConnectorError", "message": ""},
            }
        },
    }
    server.server_client.turn_response = turn

    response = await run_episode(server, task)

    assert response.result is None
    assert response.failure.terminal is terminal
    assert response.failure.failure_kind == "model_request_error"
    assert not any("test.sh" in call["command"] for call in sandbox.execs)


async def test_confirmed_policy_timeout_follows_the_outcome_policy_without_an_agent_timeout_rule(
    tmp_path, monkeypatch
):
    """Harbor passes no ``timeout_action``: a confirmed policy-response timeout takes the policy default (FAIL),
    where single_agent_turn falls back to the agent's rule (RETRY by default)."""
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    turn = failed_turn('{"failure_kind":"model_policy_response_timeout"}')
    turn["failure_report"] = {
        "failure_id": "agent",
        "events": {
            "agent": {
                "event_id": "agent",
                "boundary": "agent",
                "failure_kind": "model_policy_response_timeout",
                "operation": "agent.policy_request",
                "operation_id": "operation",
                "cause_id": "model",
            },
            "model": {
                "event_id": "model",
                "boundary": "model",
                "failure_kind": "model_request_error",
                "failure_reason": MODEL_RESPONSE_TIMEOUT_REASON,
                "operation": "model_request",
                "operation_id": "operation",
                "attempt_id": "attempt",
                "details": {"request_upload_complete": True},
            },
        },
    }
    server.server_client.turn_response = turn

    response = await run_episode(server, task)

    assert response.result is None
    assert response.failure.terminal is True
    assert response.failure.failure_kind == "model_policy_response_timeout"
    assert not any("test.sh" in call["command"] for call in sandbox.execs)


@pytest.mark.parametrize("report", [None, {"failure_id": None, "events": {}, "omitted": True}])
async def test_null_or_omitted_report_fails_the_episode(tmp_path, monkeypatch, report):
    server, task, sandbox, _ = make_server(tmp_path, monkeypatch)
    turn = failed_turn('{"failure_kind":"model_transport_error"}')
    turn["failure_report"] = report
    server.server_client.turn_response = turn
    response = await run_episode(server, task)
    assert response.failure.terminal is True
    assert not any("test.sh" in call["command"] for call in sandbox.execs)
