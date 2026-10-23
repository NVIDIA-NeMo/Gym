# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden equivalence: ``if-boundary.mjs`` against recorded input -> output pairs (``fixtures/``), byte for byte.

``fixtures/index.json`` lists every fixture; each must match:

- the request body the plugin sends == the recorded outgoing request (``orjson.dumps`` of it);
- the reply OpenCode receives == the recorded reply (its SSE text, or compact JSON);
- refusals: the plugin refuses with the recorded exception type and message and sends nothing (four fixtures
  expect the plugin's wording, which names OpenCode's user prompts, permission configuration and history);
- the call records, read back by ``records.py``, == the recorded observations and issues.
"""

import json
from copy import deepcopy
from pathlib import Path

import orjson
import pytest

from resources_servers.agentic_if.opencode_bundle.boundary import compile_boundary
from resources_servers.agentic_if.task_variants.builder import content_text
from resources_servers.agentic_if.task_variants.runtime import RuntimeSpec

from .harness import BoundaryRun, call, needs_node, run_boundary, sse_wire, upstream


FIXTURES = Path(__file__).with_name("fixtures")
INDEX = json.loads((FIXTURES / "index.json").read_text())
OK = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}


def _spec(fixture: dict, entry: dict) -> RuntimeSpec:
    spec = deepcopy(fixture["spec"])
    if entry.get("normalize_aliases") and spec["tools"].get("tool_names"):
        # The fixture's spec names an alias; the call bypasses tool presentation, so identity names give the same output.
        spec["tools"]["tool_names"] = {name: name for name in spec["tools"]["tool_names"]}
    return RuntimeSpec.model_validate(spec)


def _system(params: dict) -> str | None:
    systems = [content_text(item["content"]) for item in params["input"] if item.get("role") == "system"]
    return systems[0] if systems else None


def _reply(recorded: dict) -> dict:
    if "sse_events" in recorded:
        wire = recorded.get("sse_wire")
        return upstream(
            wire.encode() if wire is not None else sse_wire(recorded["sse_events"]), content_type="text/event-stream"
        )
    if "status" in recorded:
        return upstream(recorded["error_body"].encode(), status=recorded["status"])
    return upstream(json.dumps(recorded["json"]).encode())


def _identity(headers: dict) -> dict:
    return {
        "session": headers["x-session-id"],
        "assistant": headers.get("x-opencode-assistant-message-id"),
        "agent": "compaction" if headers.get("x-opencode-request-kind") == "compaction" else "build",
    }


def _error_text(entry: dict) -> str:
    return f"{entry['error']['name']}: {entry['error']['message']}"


def _check_refusal(actual: dict, expected_error: str) -> None:
    assert actual["sent"] is None, "a refused call must not reach the model"
    assert actual["error"] is not None
    assert expected_error in _error_text(actual)


def _replay_calls(tmp_path: Path, fixture: dict, entry: dict) -> None:
    calls = [
        call(item["opencode_request"], _reply(item["upstream_reply"]), **_identity(item["headers"]))
        for item in fixture["calls"]
    ]
    run = run_boundary(
        tmp_path, boundary=compile_boundary(_spec(fixture, entry)), calls=calls, system=_system(fixture["params"])
    )
    for index, (expected, actual) in enumerate(zip(fixture["calls"], run.calls, strict=True)):
        where = f"{fixture['case']} call {index}"
        if expected["outgoing_request"] is None:
            assert actual["sent"] is None, where
        else:
            assert actual["sent"] == orjson.dumps(expected["outgoing_request"]).decode(), where
        if expected["error"] is not None and entry.get("returned") == "pass_through":
            # The upstream HTTP error reaches OpenCode unchanged (the fixture recorded a refusal here).
            assert actual["status"] == expected["upstream_reply"]["status"], where
            assert run.returned(index) == expected["upstream_reply"]["error_body"].encode(), where
        elif expected["error"] is not None:
            assert actual["error"] is not None, where
            assert expected["error"] in _error_text(actual), where
        elif "sse_text" in expected["returned"]:
            assert actual["error"] is None, where
            assert actual["content_type"] == "text/event-stream; charset=utf-8", where
            assert run.returned(index) == expected["returned"]["sse_text"].encode(), where
        else:
            assert actual["error"] is None, where
            assert actual["content_type"] == "application/json", where
            rendered = json.dumps(expected["returned"]["json"], ensure_ascii=False, separators=(",", ":")).encode()
            assert run.returned(index) == rendered, where
    assert run.records.observations == fixture["observations"]
    assert run.records.issues == fixture["issues"]
    _check_receipts(run)


def _replay_prepare(tmp_path: Path, fixture: dict, entry: dict) -> None:
    calls = []
    if fixture["child_session"]:
        calls.append(
            call(
                {"messages": [{"role": "user", "content": "Task"}], "tools": []},
                upstream(json.dumps(OK).encode()),
                assistant="prime",
            )
        )
    session = "child" if fixture["child_session"] else "root"
    calls.append(
        call(fixture["opencode_request"], upstream(json.dumps(OK).encode()), session=session, assistant="msg-prepare")
    )
    run = run_boundary(
        tmp_path, boundary=compile_boundary(_spec(fixture, entry)), calls=calls, system=_system(fixture["params"])
    )
    actual = run.calls[-1]
    if fixture["error"] is not None:
        _check_refusal(actual, fixture["error"])
        assert run.records.issues[-1] == fixture["error"].split(":")[0]
        return
    assert actual["error"] is None
    assert actual["sent"] == orjson.dumps(fixture["prepared"]["payload"]).decode()
    observation = run.records.observations[-1]
    assert observation["session_id"] == session and observation["assistant_message_id"] == "msg-prepare"
    assert observation["execution_tools"] == fixture["prepared"]["execution_tools"]
    _check_receipts(run)


def _check_receipts(run: BoundaryRun) -> None:
    assert [row["kind"] for row in run.records.loads] == ["loaded"]
    assert [row["kind"] for row in run.records.configurations] == ["configured"]
    assert run.provider_fetch_installed


@needs_node
@pytest.mark.parametrize("entry", INDEX["cases"], ids=lambda entry: entry["case"])
def test_fixture_matches_byte_for_byte(tmp_path: Path, entry: dict) -> None:
    fixture = json.loads((FIXTURES / f"{entry['case']}.json").read_text())
    if fixture["kind"] == "prepare":
        _replay_prepare(tmp_path, fixture, entry)
    else:
        _replay_calls(tmp_path, fixture, entry)


def test_index_lists_every_fixture() -> None:
    files = {path.stem for path in FIXTURES.glob("*.json")} - {"index", "profile-goldens"}
    assert files == {entry["case"] for entry in INDEX["cases"]}
