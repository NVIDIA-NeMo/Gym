# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.agents import AGENT_OUTCOME_METADATA_KEY
from nemo_gym.core import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.models import CaptureStore
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRecord,
    ModelCallRef,
    ToolCallObservation,
)
from responses_api_agents.claude_code_agent import observability
from responses_api_agents.claude_code_agent.observability import extract_claude_code_observations
from responses_api_agents.claude_code_agent.parsing import _invocation_outcome, parse_stream_json
from responses_api_agents.claude_code_agent.tests.test_app import _make_agent
from responses_api_agents.claude_code_agent.tests.test_observability import MODEL_REF, _agent, _write


ERROR_BODY = {"error": {"message": "controlled model failure", "type": "conformance_error", "code": "400"}}
ERROR_TEXT = "API Error: 400 " + json.dumps(ERROR_BODY, separators=(",", ":"))


def _error_event(**overrides) -> dict:
    # 2.1.108 Od8 emits error/model/parent_tool_use_id; its persisted r3/WLK
    # record additionally carries isApiErrorMessage. Neither UUID is served.
    return {
        "type": "assistant",
        "session_id": "root",
        "parent_tool_use_id": None,
        "uuid": "local-event",
        "error": "unknown",
        "message": {
            "id": "local-message",
            "model": "<synthetic>",
            "role": "assistant",
            "content": [{"type": "text", "text": ERROR_TEXT}],
        },
        **overrides,
    }


def _result(**overrides) -> dict:
    return {"type": "result", "subtype": "success", "is_error": True, "result": ERROR_TEXT, **overrides}


@pytest.mark.parametrize("capture", [False, True])
def test_synthetic_http_error_keeps_text_and_classifies_original_status(capture: bool) -> None:
    items, metadata = parse_stream_json(
        "\n".join(map(json.dumps, [_error_event(), _result()])), fold_thinking=not capture
    )
    assert items[0].content[0].text == ERROR_TEXT
    assert "terminal_response_id" not in metadata
    assert _invocation_outcome(metadata, 1) == ("failed", "model_request_error")
    response = _make_agent()._response_from_output(
        NeMoGymResponseCreateParamsNonStreaming(input=[]), items, "model", metadata, status="failed"
    )
    assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY]) == {
        "failure_kind": "model_request_error",
        "status_code": 400,
    }
    assert response.output[0].content[0].text == ERROR_TEXT


@pytest.mark.parametrize(
    "case", ["ordinary_text", "child", "different_result", "recovered", "tool_error", "transport"]
)
def test_http_text_without_terminal_root_api_error_is_not_classified(case: str) -> None:
    event, result = _error_event(), _result()
    events = [event, result]
    if case == "ordinary_text":
        event["message"]["model"] = "served-model"
        event.pop("error")
    elif case == "child":
        event["parent_tool_use_id"] = "spawn"
    elif case == "different_result":
        result["result"] = "A tool failed"
    elif case == "recovered":
        result["is_error"] = False
    elif case == "tool_error":
        events[0] = {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool", "is_error": True, "content": ERROR_TEXT}],
            },
        }
    elif case == "transport":
        event["message"]["content"][0]["text"] = "API Error: Connection error."
        result["result"] = "API Error: Connection error."
    _, metadata = parse_stream_json("\n".join(map(json.dumps, events)))
    assert _make_agent().classify_outcome(metadata).failure_kind == "unknown"


def test_synthetic_transcript_error_is_not_a_model_response_reference(tmp_path: Path) -> None:
    error = _error_event()
    error.update(sessionId="root", isApiErrorMessage=True, timestamp="2026-11-08T00:00:00Z")
    _write(tmp_path / "projects" / "test" / "session.jsonl", error)
    bundle = extract_claude_code_observations(tmp_path, model_ref=MODEL_REF, root_status="failed")
    [invocation] = [record for record in bundle.records if isinstance(record, AgentInvocation)]
    assert invocation.model_calls == []
    assert invocation.conversation[0].content[0].text == ERROR_TEXT
    assert "model_response_id_missing" not in {gap.code for gap in bundle.gaps}


def test_local_failure_status_is_preserved_in_response() -> None:
    items, metadata = parse_stream_json("\n".join(map(json.dumps, [_error_event(), _result()])))
    metadata["status"] = "failed"
    response = _make_agent()._response_from_output(
        NeMoGymResponseCreateParamsNonStreaming(input=[]), items, "model", metadata
    )
    assert response.status == "failed"
    assert json.loads(response.metadata[AGENT_OUTCOME_METADATA_KEY])["status_code"] == 400


def _call(index: int, *, failed: bool) -> ModelCallRecord:
    return ModelCallRecord(
        call_index=index,
        model_call_id=f"call-{index}",
        response_id=None if failed else f"response-{index}",
        dialect="messages",
        model_ref=MODEL_REF,
        status_code=400 if failed else 200,
        request={
            "model": "served-model",
            "metadata": {"user_id": json.dumps({"session_id": "root"})},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Check this task."}]}],
            "tools": [{"name": "Bash"}],
        },
        response=ERROR_BODY if failed else {"id": f"response-{index}", "content": [{"type": "text", "text": "done"}]},
    )


def _failure_bundle() -> AgentObservationBundle:
    items, _ = parse_stream_json("\n".join(map(json.dumps, [_error_event(), _result()])))
    return AgentObservationBundle(
        source="claude_code",
        records=[
            AgentInvocation(
                invocation_id="root",
                status="failed",
                error_type="model_request_error",
                conversation=[NeMoGymEasyInputMessage(role="user", content="Check this task."), *items],
            )
        ],
    )


def _join(bundle, calls):
    return observability.associate_claude_code_failed_calls(bundle, calls, model_ref=MODEL_REF)


def test_failed_attempt_joins_exact_successful_retry_and_retains_status() -> None:
    failed, success = _call(0, failed=True), _call(1, failed=False)
    bundle = AgentObservationBundle(
        source="claude_code",
        records=[
            AgentInvocation(
                invocation_id="root",
                status="completed",
                model_calls=[ModelCallRef(model_ref=MODEL_REF, response_id=success.response_id)],
            )
        ],
    )
    original = bundle.model_dump()
    joined = _join(bundle, [failed, success])
    assert [ref.model_call_id for ref in joined.records[0].model_calls] == ["call-0", None]
    assert _join(joined, [failed, success]) == joined
    assert bundle.model_dump() == original and failed.status_code == 400


@pytest.mark.parametrize(
    "case",
    [
        "changed_request",
        "later_attempt",
        "different_owner",
        "child_anchor",
        "conflicting_session",
        "anchor_session",
        "anchor_response",
        "anchor_model",
    ],
)
def test_retry_join_requires_one_exact_owned_anchor(case: str) -> None:
    failed, success = _call(0, failed=True), _call(1, failed=False)
    bundle = AgentObservationBundle(
        source="claude_code",
        records=[
            AgentInvocation(
                invocation_id="root", model_calls=[ModelCallRef(model_ref=MODEL_REF, response_id=success.response_id)]
            )
        ],
    )
    if case == "changed_request":
        failed.request["model"] = "different"
    elif case == "later_attempt":
        failed.call_index = 2
    elif case == "different_owner":
        bundle.records.append(
            AgentInvocation(
                invocation_id="child", parent_invocation_id="root", model_calls=list(bundle.records[0].model_calls)
            )
        )
    elif case == "child_anchor":
        bundle.records[0].parent_invocation_id = "parent"
    elif case == "conflicting_session":
        failed.client_session_id = "other"
    elif case == "anchor_session":
        success.client_session_id = "other"
    elif case in {"anchor_response", "anchor_model"}:
        bundle.records[0].model_calls = [
            ModelCallRef(
                model_call_id=success.model_call_id,
                response_id="other" if case == "anchor_response" else success.response_id,
                model_ref=MODEL_REF.model_copy(update={"name": "other"}) if case == "anchor_model" else MODEL_REF,
            )
        ]
    assert _join(bundle, [failed, success]) == bundle


def test_initial_terminal_error_joins_original_status_and_json_body_only() -> None:
    call, bundle = _call(0, failed=True), _failure_bundle()
    joined = _join(bundle, [call])
    assert [ref.model_call_id for ref in joined.records[0].model_calls] == ["call-0"]
    assert joined.records[0].model_calls[0].response_id is None
    assert joined.records[0].conversation == bundle.records[0].conversation
    assert _join(joined, [call]) == joined


@pytest.mark.parametrize(
    "case",
    [
        "wrong_body",
        "wrong_status",
        "wrong_prompt",
        "extra_message",
        "ambiguous",
        "child",
        "tool_history",
        "ordinary_error",
        "later_failure",
        "missing_identity",
        "other_model",
    ],
)
def test_terminal_join_leaves_ambiguous_and_unsupported_failures_unowned(case: str) -> None:
    call, bundle = _call(0, failed=True), _failure_bundle()
    calls = [call]
    if case == "wrong_body":
        call.response = {"error": {"message": "different"}}
    elif case == "wrong_status":
        call.status_code = 500
    elif case == "wrong_prompt":
        call.request["messages"][0]["content"][0]["text"] = "Child task"
    elif case == "extra_message":
        call.request["messages"].append({"role": "user", "content": "Check this task."})
    elif case == "ambiguous":
        calls.append(_call(1, failed=True))
    elif case == "child":
        bundle.records.append(AgentInvocation(invocation_id="child", parent_invocation_id="root"))
    elif case == "tool_history":
        bundle.records.append(ToolCallObservation(invocation_id="root", tool_call_id="spawn", tool_name="Agent"))
    elif case == "ordinary_error":
        bundle.records[0].error_type = "agent_error"
    elif case == "later_failure":
        bundle.records[0].model_calls = [ModelCallRef(model_call_id="prior")]
    elif case == "missing_identity":
        call.request.pop("metadata")
    elif case == "other_model":
        call.model_ref = None
    assert _join(bundle, calls) == bundle


@pytest.mark.asyncio
async def test_capture_join_distinguishes_failed_title_and_first_main_request(tmp_path: Path) -> None:
    main, title = _call(1, failed=True), _call(0, failed=True)
    title.model_call_id = "title-0"
    title.request.update(
        system=(
            "Generate a concise, sentence-case title (3-7 words) that captures the main topic "
            "or goal of this coding session."
        ),
        tools=[],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
            },
        },
    )
    main.request["messages"][0]["content"].insert(
        0, {"type": "text", "text": "<system-reminder>\nContext\n</system-reminder>\n\n"}
    )
    store = CaptureStore(tmp_path)
    for call in (title, main):
        store.record("0-0", call.model_dump(exclude={"call_index"}))
    source = _failure_bundle()
    joined = await _agent(tmp_path)._associate_helper_calls(source, rollout_id="0-0")
    [owner] = [record for record in joined.records if isinstance(record, AgentInvocation)]
    assert [ref.model_call_id for ref in owner.model_calls] == ["title-0", "call-1"]
    assert all(ref.response_id is None for ref in owner.model_calls)
    assert owner.conversation == source.records[0].conversation
    assert joined.gaps == []
