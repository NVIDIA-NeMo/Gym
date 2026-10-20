# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared mock-tool server with a synthetic backend, loaded by import path as a plugin's backend is."""

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.mcp_auto_exposure import harvest_tools
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.partner_if import mock_tools_app as api
from resources_servers.partner_if.tests.test_app import response


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


class NotesBackend:
    """Reads packaged notes and appends to a per-session log, which is the episode's artifact."""

    requires_final_answer = True

    def __init__(self, fixture_path: Path):
        self.notes = json.loads(fixture_path.read_text())

    def tools(self, row):
        return {tool["name"]: tool for tool in row["responses_create_params"]["tools"]}

    def new_state(self, row):
        return []

    def call(self, state, name, arguments):
        if name == "lookup":
            return {"note": self.notes.get(arguments.get("key"))}
        state.append(str(arguments.get("text", "")))
        return "Appended"

    def artifacts(self, state):
        return {"log": "\n".join(state)}

    def receipts(self, state):
        return {"appended": list(state)}


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
    module.NotesBackend = NotesBackend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def make_server(tmp_path, backend="synthetic_tools:NotesBackend"):
    (tmp_path / "tasks.jsonl").write_text(json.dumps(row()) + "\n")
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
        ),
        server_client=MagicMock(spec=ServerClient),
    )


@pytest.fixture
def server(tmp_path, plugin):
    task = row()
    seed = {
        "resources_session_id": "session-one",
        "episode_id": {"rollout_id": "r", "attempt": 0},
        "task_id": {"taskset": "synthetic", "task_id": "smoke"},
        "task_data": {"verifier_metadata": task["verifier_metadata"]},
    }
    return make_server(tmp_path), task, seed


def test_backend_is_a_class_named_by_import_path(tmp_path, plugin):
    with pytest.raises(ValidationError, match="backend"):
        make_server(tmp_path, backend="notes")
    for backend in ("synthetic_tools:Missing", "missing_tools:NotesBackend"):
        with pytest.raises(ValueError, match="Cannot load tool backend"):
            make_server(tmp_path, backend=backend)


class ServiceBackend(NotesBackend):
    """Serves its whole tool family whatever a task advertises, as a search service would."""

    def tools(self, row):
        return {tool["name"]: tool for tool in TOOLS}


def test_a_task_can_call_only_the_tools_it_advertises(tmp_path, plugin):
    plugin.ServiceBackend = ServiceBackend
    task = row()
    task["responses_create_params"]["tools"] = TOOLS[:1]
    (tmp_path / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    (tmp_path / "notes.json").write_text(json.dumps({"greeting": "HELLO"}))
    resource = api.MockToolsResourcesServer(
        config=api.MockToolsServerConfig(
            host="127.0.0.1",
            port=8001,
            name="notes",
            entrypoint="mock_tools_app.py",
            backend="synthetic_tools:ServiceBackend",
            fixture_path=tmp_path / "notes.json",
            tasks_path=tmp_path / "tasks.jsonl",
            judge_model_server={"type": "responses_api_models", "name": "judge"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    app = resource.setup_webserver()
    assert set(harvest_tools(app, resource)) == {"lookup"}
    client = TestClient(app)
    seed = {
        "resources_session_id": "session-one",
        "episode_id": {"rollout_id": "r", "attempt": 0},
        "task_id": {"taskset": "synthetic", "task_id": "smoke"},
        "task_data": {"verifier_metadata": task["verifier_metadata"]},
    }
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
    assert resource._episodes["session-one"].state == ["HELLO"]
    assert resource._episodes["session-two"].state == []
    assert first.post("/grade", json={}).status_code == 404
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


def evidence(raw, records, *, requires_final_answer=True, artifacts=None):
    params = {"input": [{"role": "user", "content": "Look up"}], "tools": TOOLS}
    return api.to_evidence(params, raw, records, requires_final_answer=requires_final_answer, artifacts=artifacts)


def test_evidence_correlates_parallel_calls_and_uses_only_the_final_answer():
    found = {"note": "HELLO"}
    records = [
        api.RecordedToolCall(name="append", arguments={"text": "HELLO"}, output="Appended"),
        api.RecordedToolCall(name="lookup", arguments={"key": "greeting"}, output=found),
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
    records = [api.RecordedToolCall(name="lookup", arguments={"key": "greeting"}, output=found)]
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


def test_backends_decide_whether_a_turn_may_end_on_a_tool_result():
    records = [api.RecordedToolCall(name="append", arguments={"text": "HELLO"}, output="Appended")]
    events = [call("append", {"text": "HELLO"}), output("Appended")]
    assert evidence(episode(events), records).trajectory_error
    bundle = evidence(episode(events), records, requires_final_answer=False, artifacts={"log": "HELLO"})
    assert bundle.trajectory_error is None and bundle.answer == ""
    assert bundle.artifact_text("log") == "HELLO"


CUTOFF = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}


def test_output_cap_cutoff_ends_a_turn_without_a_final_answer_like_the_step_cap():
    records = [api.RecordedToolCall(name="append", arguments={"text": "HELLO"}, output="Appended")]
    events = [
        call("append", {"text": "HELLO"}),
        output("Appended"),
        # The response the cap cut off: its calls never ran, so they have no results or receipts.
        call("append", {"text": "BYE"}, "call2"),
        call("lookup", {"key": "greeting"}, "call3") | {"status": "in_progress"},
    ]
    bundle = evidence(episode(events, **CUTOFF), records, requires_final_answer=False)
    assert bundle.trajectory_error is None
    assert [event.call_id for event in bundle.trajectory if event.kind == "tool_call"] == ["call1"]
    cut_before_any_result = episode([call("append", {"text": "HELLO"}) | {"status": "in_progress"}], **CUTOFF)
    bundle = evidence(cut_before_any_result, [], requires_final_answer=False)
    assert bundle.trajectory_error is None and bundle.trajectory == []


@pytest.mark.parametrize("case", ["other_reason", "answer_required", "executed_call_hidden"])
def test_other_incomplete_turns_stay_unscorable(case):
    records = [api.RecordedToolCall(name="lookup", arguments={"key": "greeting"}, output={"note": "HELLO"})]
    events = [call(), output({"note": "HELLO"})]
    details = {"reason": "content_filter" if case == "other_reason" else "max_output_tokens"}
    if case == "executed_call_hidden":
        # An executed call keeps its receipt, so a missing result cannot be dropped as cut off.
        records.append(api.RecordedToolCall(name="append", arguments={"text": "HELLO"}, output="Appended"))
        events.append(call("append", {"text": "HELLO"}, "call2"))
    raw = episode(events, status="incomplete", incomplete_details=details)
    assert evidence(raw, records, requires_final_answer=case == "answer_required").trajectory_error


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
