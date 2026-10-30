# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared mock-tool server with the example backend, loaded by import path as a plugin's backend is."""

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.core import NeMoGymResponse
from nemo_gym.mcp_auto_exposure import harvest_tools
from nemo_gym.server_utils import ServerClient, SimpleServer, entrypoint_ray_enabled
from nemo_gym.testing.session_conformance import check_resources_session_contract
from nemo_gym.verifier_fixture import exercise_verifier_fixture, load_verifier_fixture
from resources_servers.partner_if import mock_tools_app as api
from resources_servers.partner_if.evidence import MAX_EVIDENCE_CHARS
from resources_servers.partner_if.example_tools import NotesBackend
from resources_servers.partner_if.mock_tools_evidence import RecordedToolCall, to_evidence
from resources_servers.partner_if.tests.test_app import response
from resources_servers.partner_if.tests.test_configs import server_config


TOOLS = [
    {
        "type": "function",
        "name": name,
        "description": f"{name} tool",
        "parameters": {"type": "object"},
        "strict": False,
    }
    for name in ("lookup", "append")
]


class StepCappedNotes(NotesBackend):
    """A backend whose episodes may end on a tool result, like an episode cut off at the step cap."""

    requires_final_answer = False


class FailingNotes(NotesBackend):
    def call(self, state, name, arguments):
        raise RuntimeError("emulator bug")


class LostNotes(NotesBackend):
    """A backend whose episode is gone by the time the episode is verified."""

    def artifacts(self, state):
        raise RuntimeError("state lost")


def row():
    rubrics = [
        ("logged", "artifact:log", "Logs HELLO."),
        ("answered", "answer", "Says HELLO."),
    ]
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": "Look up the greeting, log it and say it."}],
            "tools": TOOLS,
        },
        "verifier_metadata": {
            "partner_if": {
                "task_id": "smoke",
                "origin": "synthetic",
                "source": "authored",
                "rubric_version": "tools-smoke-v1",
                "audit_signoff": True,
                "data_kind": "synthetic_smoke",
                "evidence_mode": "live_agent",
                "rubrics": [
                    {
                        "id": rid,
                        "constraint": constraint,
                        "verifier": {"id": "text_contains_v1", "params": {"source": source, "literal": "HELLO"}},
                    }
                    for rid, source, constraint in rubrics
                ],
            }
        },
    }


@pytest.fixture
def plugin(monkeypatch):
    module = ModuleType("synthetic_tools")
    module.NotesBackend, module.StepCappedNotes, module.FailingNotes = NotesBackend, StepCappedNotes, FailingNotes
    module.LostNotes = LostNotes
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def make_server(tmp_path, backend="synthetic_tools:NotesBackend", task=None, **config):
    (tmp_path / "tasks.jsonl").write_text(json.dumps(task or row()) + "\n")
    (tmp_path / "notes.json").write_text(json.dumps({"greeting": "HELLO"}))
    return api.MockToolsResourcesServer(
        config=api.MockToolsServerConfig(
            host="127.0.0.1",
            port=8001,
            name="notes",
            entrypoint="mock_tools_app.py",
            backend=backend,
            fixture_path=tmp_path / "notes.json",
            tasks_path=tmp_path / "tasks.jsonl",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
            **config,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def seed_for(task, session="session-one"):
    return {
        "resources_session_id": session,
        "episode_id": {"rollout_id": "r", "attempt": 0},
        "task_id": {"taskset": "synthetic", "task_id": "smoke"},
        "task_data": {"verifier_metadata": task["verifier_metadata"]},
    }


@pytest.fixture
def server(tmp_path, plugin):
    task = row()
    return make_server(tmp_path), task, seed_for(task)


def test_backend_is_a_class_named_by_import_path(tmp_path, plugin):
    with pytest.raises(ValidationError, match="backend"):
        make_server(tmp_path, backend="notes")
    for backend in ("synthetic_tools:Missing", "missing_tools:NotesBackend"):
        with pytest.raises(ValueError, match="Cannot load tool backend"):
            make_server(tmp_path, backend=backend)


def test_episodes_are_not_exported_so_checkpoints_restart_them(server):
    assert (api.MockToolsResourcesServer.checkpoint_mode, api.MockToolsResourcesServer.checkpoint_verify) == (
        "restart_only",
        "wait",
    )
    # The episodes are gone after the run, so reverify cannot grade the rollouts again.
    resource, _, _ = server
    assert TestClient(resource.setup_webserver()).get("/reverify_mode").json() == "unsupported"
    # The server grades its own episodes, so it has no setting for captures an agent sends.
    assert "accept_verification_input" not in api.MockToolsServerConfig.model_fields


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda task: task["responses_create_params"].update(tools=[*TOOLS, TOOLS[0] | {"name": "erase"}]), "erase"),
        (lambda task: task["verifier_metadata"]["partner_if"]["rubrics"][0]["verifier"].update(id="x.v1"), "x.v1"),
        (
            lambda task: task["verifier_metadata"]["partner_if"]["rubrics"][0]["verifier"]["params"].pop("literal"),
            "literal",
        ),
    ],
)
def test_tasks_the_server_cannot_serve_or_grade_fail_at_startup(tmp_path, plugin, change, error):
    task = row()
    change(task)
    with pytest.raises(ValueError, match=error):
        make_server(tmp_path, task=task)


def test_a_task_can_call_only_the_tools_it_advertises(tmp_path, plugin):
    task = row()
    task["responses_create_params"]["tools"] = TOOLS[:1]
    resource = make_server(tmp_path, task=task)
    app = resource.setup_webserver()
    assert set(harvest_tools(app, resource)) == {"lookup"}
    client = TestClient(app)
    seed = seed_for(task)
    assert client.post("/seed_session", json=seed).status_code == 200
    assert resource.mcp_allowed_tools_for_session(seed["task_data"]) == ["lookup"]
    assert client.post("/lookup", json={"key": "greeting"}).json() == {"note": "HELLO"}
    assert client.post("/append", json={"text": "HELLO"}).status_code == 404


def test_sessions_isolate_backend_state_and_only_tool_routes_are_added(server):
    resource, _, seed = server
    app = resource.setup_webserver()
    assert set(harvest_tools(app, resource)) == {"lookup", "append"}
    assert resource.mcp_allowed_tools_for_session(seed["task_data"]) == ["append", "lookup"]
    first, second = TestClient(app), TestClient(app)
    assert first.post("/lookup", json={"key": "greeting"}).status_code == 409
    assert first.post("/seed_session", json=seed).status_code == 200
    assert first.post("/seed_session", json=seed).status_code == 200
    assert second.post("/seed_session", json=seed | {"resources_session_id": "session-two"}).status_code == 200
    assert first.post("/lookup", json={"key": "greeting"}).json() == {"note": "HELLO"}
    appended = first.post("/append", json={"text": "HELLO"})
    assert appended.headers["content-type"].startswith("text/plain") and appended.text == "Appended"
    assert resource._episodes["session-one"].backend.state == ["HELLO"]
    assert resource._episodes["session-two"].backend.state == []
    assert first.post("/invented", json={}).status_code == 404
    close = {k: seed[k] for k in ("resources_session_id", "episode_id")}
    assert first.post("/close_session", json=close).status_code == 200
    assert first.post("/close_session", json=close).status_code == 200
    assert first.post("/seed_session", json=seed).status_code == 409
    assert first.post("/lookup", json={"key": "greeting"}).status_code == 409


def test_seed_identity_conflicts_and_close_before_seed_are_rejected(server):
    resource, _, seed = server
    client = TestClient(resource.setup_webserver())
    client.post("/seed_session", json=seed)
    changed = seed | {"episode_id": {"rollout_id": "other", "attempt": 0}}
    assert client.post("/seed_session", json=changed).status_code == 409
    future = {"resources_session_id": "future", "episode_id": seed["episode_id"]}
    assert client.post("/close_session", json=future).status_code == 200
    assert client.post("/seed_session", json=seed | {"resources_session_id": "future"}).status_code == 409


def call(name="lookup", args=None, identifier="call1"):
    arguments = json.dumps({"key": "greeting"} if args is None else args)
    return {"type": "function_call", "id": identifier, "call_id": identifier, "name": name, "arguments": arguments}


def output(value, identifier="call1"):
    text = value if isinstance(value, str) else json.dumps(value)
    return {"type": "function_call_output", "call_id": identifier, "output": text}


def answer(text="HELLO"):
    return response(text).output[0].model_dump(mode="json")


def episode(events, **overrides):
    return NeMoGymResponse.model_validate(response("HELLO").model_dump() | {"output": events} | overrides)


def evidence(raw, records, *, artifacts=None):
    params = {"input": [{"role": "user", "content": "Look up"}], "tools": TOOLS}
    return to_evidence(params, raw, records, artifacts=artifacts)


def test_evidence_correlates_parallel_calls_and_uses_only_the_final_answer():
    found = {"note": "HELLO"}
    records = [
        RecordedToolCall(name="append", arguments={"text": "HELLO"}, output="Appended"),
        RecordedToolCall(name="lookup", arguments={"key": "greeting"}, output=found),
    ]
    events = [
        answer("Looking up"),
        call(),
        call("append", {"text": "HELLO"}, "call2"),
        output("Appended", "call2"),
        output(found),
        answer(),
    ]
    bundle = evidence(episode(events), records)
    assert bundle.trajectory_error is None and bundle.answer == "HELLO"
    assert len(bundle.trajectory) == 6 and bundle.trajectory[0].text == "Looking up"
    assert bundle.trajectory[4].text == json.dumps(found)


@pytest.mark.parametrize(
    "defect", ["forged", "missing_output", "missing_receipt", "duplicate_id", "unfinished", "extra_receipt"]
)
def test_capture_defects_are_unscorable_not_if_failures(defect):
    found = {"note": "HELLO"}
    records = [RecordedToolCall(name="lookup", arguments={"key": "greeting"}, output=found)]
    events = [call(), output(found), answer()]
    if defect == "forged":
        events[1] = output({"note": "invented"})
    if defect == "missing_output":
        events.pop(1)
    if defect == "missing_receipt":
        records.clear()
    if defect == "duplicate_id":
        events.insert(1, call())
    if defect == "extra_receipt":
        records.append(records[0])
    raw = episode(events)
    if defect == "unfinished":
        raw.status = "incomplete"
    assert evidence(raw, records).trajectory_error


def test_unknown_tools_and_malformed_arguments_are_visible_actor_errors():
    events = [call("invented"), output({"error": "'invented' is not an advertised tool"}), answer()]
    bundle = evidence(episode(events), [])
    assert not bundle.trajectory_error and bundle.trajectory[0].tool_name == "invented"
    events[0] = call()
    events[0]["arguments"] = "{bad"
    events[1] = output({"error": "Invalid tool call arguments: bad JSON"})
    assert not evidence(episode(events), []).trajectory_error


def test_a_turn_ending_on_a_tool_result_is_complete_evidence():
    records = [RecordedToolCall(name="append", arguments={"text": "HELLO"}, output="Appended")]
    events = [call("append", {"text": "HELLO"}), output("Appended")]
    bundle = evidence(episode(events), records, artifacts={"log": "HELLO"})
    assert bundle.trajectory_error is None and bundle.answer == ""
    assert bundle.artifact_text("log") == "HELLO"


CUTOFF = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}


def test_output_cap_cutoff_ends_a_turn_without_a_final_answer_like_the_step_cap():
    records = [RecordedToolCall(name="append", arguments={"text": "HELLO"}, output="Appended")]
    events = [
        call("append", {"text": "HELLO"}),
        output("Appended"),
        # The response the cap cut off: its calls never ran, so they have no results or receipts.
        call("append", {"text": "BYE"}, "call2"),
        call("lookup", {"key": "greeting"}, "call3") | {"status": "in_progress"},
    ]
    bundle = evidence(episode(events, **CUTOFF), records)
    assert bundle.trajectory_error is None
    assert [event.call_id for event in bundle.trajectory if event.kind == "tool_call"] == ["call1"]
    cut_before_any_result = episode([call("append", {"text": "HELLO"}) | {"status": "in_progress"}], **CUTOFF)
    bundle = evidence(cut_before_any_result, [])
    assert bundle.trajectory_error is None and bundle.trajectory == []


@pytest.mark.parametrize("case", ["other_reason", "executed_call_hidden"])
def test_other_incomplete_turns_stay_unscorable(case):
    records = [RecordedToolCall(name="lookup", arguments={"key": "greeting"}, output={"note": "HELLO"})]
    events = [call(), output({"note": "HELLO"})]
    details = {"reason": "content_filter" if case == "other_reason" else "max_output_tokens"}
    if case == "executed_call_hidden":
        # An executed call keeps its receipt, so a missing result cannot be dropped as cut off.
        records.append(RecordedToolCall(name="append", arguments={"text": "HELLO"}, output="Appended"))
        events.append(call("append", {"text": "HELLO"}, "call2"))
    raw = episode(events, status="incomplete", incomplete_details=details)
    assert evidence(raw, records).trajectory_error


def test_http_verify_grades_backend_artifacts_and_returns_its_receipts(server):
    resource, task, seed = server
    client = TestClient(resource.setup_webserver())
    client.post("/seed_session", json=seed)
    found = client.post("/lookup", json={"key": "greeting"}).json()
    appended = client.post("/append", json={"text": "HELLO"}).text
    events = [call(), output(found), call("append", {"text": "HELLO"}, "call2"), output(appended, "call2"), answer()]
    raw = episode(events).model_dump(mode="json")
    checked = client.post("/verify", json=task | {"response": raw})
    assert checked.status_code == 200, checked.text
    body = checked.json()
    assert body["reward"] == 1 and not body["mask_sample"] and body["judge_request"] is None
    assert body["environment_receipts"] == {"appended": ["HELLO"]}
    assert [recorded["name"] for recorded in body["tool_calls"]] == ["lookup", "append"]

    # Without this session's receipts the same trajectory cannot be scored.
    unseeded = TestClient(resource.setup_webserver()).post("/verify", json=task | {"response": raw}).json()
    assert unseeded["mask_sample"] and unseeded["reward"] == 0
    changed = task | {
        "responses_create_params": {"input": [{"role": "user", "content": "Other task"}], "tools": TOOLS}
    }
    assert client.post("/verify", json=changed | {"response": raw}).status_code == 409


def test_uuid_rule_checks_assistant_messages_before_the_last_tool_result(tmp_path, plugin):
    task = row()
    task["verifier_metadata"]["partner_if"]["rubrics"] = [
        {"id": "no_uuid", "constraint": "Shows no UUIDs.", "verifier": {"id": "text_no_uuid_v1"}}
    ]
    client = TestClient(make_server(tmp_path, task=task).setup_webserver())
    seed = {
        "resources_session_id": "session-one",
        "episode_id": {"rollout_id": "r", "attempt": 0},
        "task_id": {"taskset": "synthetic", "task_id": "smoke"},
        "task_data": {"verifier_metadata": task["verifier_metadata"]},
    }
    assert client.post("/seed_session", json=seed).status_code == 200
    found = client.post("/lookup", json={"key": "greeting"}).json()
    # The final answer is clean; the UUID is only in assistant text before the tool result.
    events = [answer("Opening 0f1e2d3c-4b5a-4968-8776-a5b4c3d2e1f0."), call(), output(found), answer()]
    body = client.post("/verify", json=task | {"response": episode(events).model_dump(mode="json")}).json()
    assert body["grading_result"]["items"][0]["raw_passed"] is False
    assert body["reward"] == 0 and body["mask_sample"] is False


REASONING_ONLY = {"status": "incomplete", "metadata": {"ng_termination_reason": "incomplete_reasoning"}}


@pytest.mark.parametrize(
    "ending,overrides,kind",
    [
        ([], {}, "partner_if:no_answer"),
        ([answer(" ")], {}, "partner_if:no_answer"),
        ([{"type": "reasoning", "id": "r", "summary": []}], REASONING_ONLY, "partner_if:no_answer"),
        ([call("append", {"text": "BYE"}, "call3") | {"status": "in_progress"}], CUTOFF, "partner_if:truncated"),
    ],
)
def test_an_episode_without_its_required_final_answer_fails_every_item(tmp_path, plugin, ending, overrides, kind):
    task = row()
    task["verifier_metadata"]["partner_if"]["rubrics"].append(
        {"id": "kept", "constraint": "Never erases the log.", "verifier": {"id": "llm_judge_v1"}}
    )
    client = TestClient(make_server(tmp_path, task=task).setup_webserver())
    client.post("/seed_session", json=seed_for(task))
    appended = client.post("/append", json={"text": "HELLO"}).text
    events = [call("append", {"text": "HELLO"}), output(appended), *ending]
    raw = episode(events, **overrides).model_dump(mode="json")
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock) as judge:
        body = client.post("/verify", json=task | {"response": raw}).json()
    judge.assert_not_called()  # So the prohibition cannot pass by omission.
    assert body["reward"] == 0 and body["mask_sample"] is False and body["failure_kind"] == kind
    assert [item["raw_passed"] for item in body["grading_result"]["items"]] == [False, False, False]


def test_a_backend_that_needs_no_final_answer_grades_the_completed_exchanges(tmp_path, plugin):
    task = row()
    client = TestClient(make_server(tmp_path, backend="synthetic_tools:StepCappedNotes").setup_webserver())
    client.post("/seed_session", json=seed_for(task))
    appended = client.post("/append", json={"text": "HELLO"}).text
    raw = episode([call("append", {"text": "HELLO"}), output(appended)]).model_dump(mode="json")
    body = client.post("/verify", json=task | {"response": raw}).json()
    assert body["mask_sample"] is False and body["failure_kind"] is None
    assert {item["id"]: item["passed"] for item in body["grading_result"]["items"]} == {
        "logged": True,
        "answered": False,
    }


def test_a_tool_call_error_is_the_models_call_and_keeps_its_receipt(server):
    resource, task, seed = server
    client = TestClient(resource.setup_webserver())
    client.post("/seed_session", json=seed)
    missing = client.post("/lookup", json={"key": "missing"})
    assert missing.status_code == 200 and missing.json() == {"error": "No note has the key 'missing'"}
    events = [call(args={"key": "missing"}), output(missing.json()), answer()]
    body = client.post("/verify", json=task | {"response": episode(events).model_dump(mode="json")}).json()
    assert body["mask_sample"] is False and body["tool_calls"][0]["output"] == missing.json()


def test_a_tool_call_error_is_caught_when_gym_runs_the_server_as_a_script(monkeypatch):
    # Gym starts `python mock_tools_app.py`, so the running server's classes are __main__'s.
    monkeypatch.setattr(SimpleServer, "run_webserver", MagicMock())
    script = runpy.run_path(api.__file__, run_name="__main__")
    config = script["MockToolsServerConfig"].model_validate(server_config("mock_tools.yaml", "partner_if_mock_tools"))
    resource = script["MockToolsResourcesServer"](config=config, server_client=MagicMock(spec=ServerClient))
    client = TestClient(resource.setup_webserver())
    task = json.loads(config.tasks_path.read_text().splitlines()[0])
    assert client.post("/seed_session", json=seed_for(task)).status_code == 200
    missing = client.post("/lookup", json={"key": "missing"})
    assert missing.status_code == 200 and missing.json() == {"error": "No note has the key 'missing'"}
    assert [call.model_dump() for call in resource._episodes["session-one"].calls] == [
        {"name": "lookup", "arguments": {"key": "missing"}, "output": missing.json()}
    ]


def test_other_backend_failures_are_logged_infrastructure_errors_that_mask_the_rollout(tmp_path, plugin, caplog):
    resource = make_server(tmp_path, backend="synthetic_tools:FailingNotes")
    client = TestClient(resource.setup_webserver(), raise_server_exceptions=False)
    client.post("/seed_session", json=seed_for(row()))
    failed = client.post("/lookup", json={"key": "greeting"})
    # The agent shows the body to the model as the call's output, so it names no exception.
    assert failed.status_code == 500 and failed.text == "Internal Server Error"
    assert resource._episodes["session-one"].calls == []
    assert "synthetic_tools:FailingNotes failed in lookup" in caplog.text and "emulator bug" in caplog.text
    # The call has no receipt, and the environment, not the model, failed: the rollout is masked and says so.
    events = [call(), output(failed.text), answer()]
    body = client.post("/verify", json=row() | {"response": episode(events).model_dump(mode="json")}).json()
    assert body["mask_sample"] and body["reward"] == 0 and body["failure_kind"] == "partner_if:tool_backend_failed"
    assert body["failure_reason"] == "The tool backend failed in lookup" and body["tool_sandbox_id"] is None


def test_a_backend_lost_before_verify_masks_the_rollout(tmp_path, plugin, caplog):
    resource = make_server(tmp_path, backend="synthetic_tools:LostNotes")
    client = TestClient(resource.setup_webserver(), raise_server_exceptions=False)
    client.post("/seed_session", json=seed_for(row()))
    found = client.post("/lookup", json={"key": "greeting"}).json()
    checked = client.post(
        "/verify", json=row() | {"response": episode([call(), output(found), answer()]).model_dump(mode="json")}
    )
    assert checked.status_code == 200, checked.text
    body = checked.json()
    # The environment failed, not the model: masked with the backend's failure kind, not a failed /verify.
    assert body["mask_sample"] and body["reward"] == 0 and body["failure_kind"] == "partner_if:tool_backend_failed"
    assert body["failure_reason"] == "The tool backend failed before verify" and body["environment_receipts"] == {}
    assert body["evaluation_completed"] is False
    assert "failed before verify" in caplog.text and "state lost" in caplog.text


async def test_a_retired_session_frees_its_episode_and_stays_closed(server):
    resource, _, seed = server
    client = TestClient(resource.setup_webserver())
    client.post("/seed_session", json=seed)
    await resource.retire_session_state("session-one")
    assert "session-one" not in resource._episodes
    assert client.post("/seed_session", json=seed).status_code == 409


@pytest.mark.parametrize("seed_first", [False, True])
def test_close_retains_episode_identity_for_retries(server, seed_first):
    resource, _, seed = server
    client = TestClient(resource.setup_webserver())
    if seed_first:
        assert client.post("/seed_session", json=seed).status_code == 200
    close = {key: seed[key] for key in ("resources_session_id", "episode_id")}
    assert client.post("/close_session", json=close).status_code == 200
    assert client.post("/close_session", json=close).status_code == 200
    other = close | {"episode_id": {**close["episode_id"], "rollout_id": "other"}}
    assert client.post("/close_session", json=other).status_code == 409
    assert client.post("/seed_session", json=seed).status_code == 409


async def test_retire_retains_a_known_episode_identity(server):
    resource, _, seed = server
    client = TestClient(resource.setup_webserver())
    assert client.post("/seed_session", json=seed).status_code == 200
    await resource.retire_session_state(seed["resources_session_id"])
    close = {key: seed[key] for key in ("resources_session_id", "episode_id")}
    other = close | {"episode_id": {**close["episode_id"], "rollout_id": "other"}}
    assert client.post("/close_session", json=other).status_code == 409
    assert client.post("/close_session", json=close).status_code == 200
    assert client.post("/seed_session", json=seed).status_code == 409


async def test_closed_ids_are_remembered_for_the_close_retry_window(tmp_path, plugin, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("nemo_gym.base_resources_server.monotonic", lambda: clock[0])
    resource = make_server(tmp_path, session_close_retry_window_seconds=10)
    client = TestClient(resource.setup_webserver())
    closed, retired = seed_for(row()), seed_for(row(), session="session-two")
    client.post("/seed_session", json=closed)
    close = {k: closed[k] for k in ("resources_session_id", "episode_id")}
    assert client.post("/close_session", json=close).status_code == 200
    await resource.retire_session_state("session-two")
    clock[0] = 9.5
    assert [client.post("/seed_session", json=seed).status_code for seed in (closed, retired)] == [409, 409]
    # No record outlives the routing claim on its ID, so with several workers every worker then treats it as new.
    clock[0] = 10.0
    assert len(resource._closed) == 0
    assert [client.post("/seed_session", json=seed).status_code for seed in (closed, retired)] == [200, 200]


def test_an_unsigned_contract_is_masked_rather_than_rejected(tmp_path, plugin):
    task = row()
    task["verifier_metadata"]["partner_if"]["audit_signoff"] = False
    client = TestClient(make_server(tmp_path, task=task).setup_webserver())
    client.post("/seed_session", json=seed_for(task))
    raw = episode([answer()]).model_dump(mode="json")
    body = client.post("/verify", json=task | {"response": raw}).json()
    assert body["mask_sample"] and body["failure_kind"] == "partner_if:unsigned_contract"
    assert body["if_result"]["status"] == "unscorable" and body["grading_result"] is None


def test_an_oversized_artifact_is_a_capture_error_for_the_items_that_need_it(server):
    resource, task, seed = server
    client = TestClient(resource.setup_webserver())
    client.post("/seed_session", json=seed)
    text = "x" * (MAX_EVIDENCE_CHARS + 1)
    appended = client.post("/append", json={"text": text}).text
    events = [call("append", {"text": text}), output(appended), answer()]
    body = client.post("/verify", json=task | {"response": episode(events).model_dump(mode="json")}).json()
    assert body["mask_sample"] and body["failure_kind"] == "partner_if:evidence_too_large"
    assert body["failure_reason"] == f"Artifact log exceeds {MAX_EVIDENCE_CHARS:,} characters"
    assert {item["id"]: item["raw_passed"] for item in body["grading_result"]["items"]} == {
        "logged": None,
        "answered": True,
    }


def test_episodes_live_in_one_worker_and_the_server_never_joins_ray(tmp_path, plugin):
    for workers in (None, 1):
        make_server(tmp_path, num_workers=workers)
    with pytest.raises(ValueError, match="num_workers=1"):
        make_server(tmp_path, num_workers=2)
    # Declared on the class the entrypoint runs, where Gym reads it before it decides whether to start Ray.
    assert api.MockToolsResourcesServer.ray_enabled is False
    assert entrypoint_ray_enabled(Path(api.__file__)) is False


def test_in_process_sessions_keep_the_resources_session_contract(tmp_path, plugin):
    seed = ResourcesSeedSessionRequest.model_validate(seed_for(row()))
    check_resources_session_contract(make_server(tmp_path).setup_webserver(), seed, keeps_state=True)


FIXTURE_CASES = [
    ("logs_and_says_the_note", "full_reward"),
    ("says_goodbye_without_logging", "zero_reward"),
    ("forged_tool_output_is_masked", "zero_reward"),
    ("missing_contract", "malformed"),
]


async def test_the_verifier_fixture_replays_each_case_through_the_example_backend_without_a_judge():
    with patch("resources_servers.partner_if.grading_service.call_judge", new_callable=AsyncMock) as judge:
        results = await exercise_verifier_fixture(
            api.VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
        )
    judge.assert_not_called()
    assert [(result.name, result.kind) for result in results] == FIXTURE_CASES


async def test_gym_env_test_finds_the_fixture_in_the_mock_tool_entrypoint():
    # The loader `gym env test` and `gym env publish` use: the entrypoint is imported apart from this test's modules.
    from nemo_gym.environment.onboarding import _import_resources_server

    server_dir = PARENT_DIR / "resources_servers/partner_if"
    module = _import_resources_server(server_dir / "mock_tools_app.py", PARENT_DIR, server_dir)
    results = await exercise_verifier_fixture(
        module.VERIFIER_FIXTURE, reward_range=(0.0, 1.0), higher_is_better=True, determinism="stochastic"
    )
    assert [(result.name, result.kind) for result in results] == FIXTURE_CASES


async def test_a_replayed_output_the_backend_does_not_reproduce_is_masked_not_graded():
    [case] = [
        case for case in load_verifier_fixture(api.VERIFIER_FIXTURE.cases_path) if case.name == FIXTURE_CASES[2][0]
    ]
    request = api.VerifyRequest.model_validate(case.request)
    body = await api.replay_and_verify(api.VERIFIER_FIXTURE.server_factory(), request)
    assert body.mask_sample and body.reward == 0 and body.failure_kind == "partner_if:unscorable"
    assert body.failure_reason == "Tool output lacks a matching authoritative receipt"
