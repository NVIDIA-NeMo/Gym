# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nemo_gym.config import ModelServerRef
from nemo_gym.models import CaptureStore
from nemo_gym.rollouts import merge_model_call_capture_into_record
from nemo_gym.rollouts.trajectory_builder import _build_trajectory_record
from nemo_gym.trajectory import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ModelCallRecord,
    ModelCallRef,
    ObservationGap,
    SessionTitleObservation,
)
from responses_api_agents.claude_code_agent import observability
from responses_api_agents.claude_code_agent.tests.test_observability import MODEL_REF, _agent, _captured_call, _records


def _title_call(index: int = 0, *, failed: bool = False) -> ModelCallRecord:
    return ModelCallRecord(
        call_index=index,
        model_call_id=f"title-{index}",
        model_ref=MODEL_REF,
        dialect="messages",
        status_code=429 if failed else 200,
        response_id=None if failed else f"title-response-{index}",
        error_category="http_error" if failed else None,
        request={
            "system": [
                {"type": "text", "text": "You are Claude Code, Anthropic's official CLI for Claude."},
                {
                    "type": "text",
                    "text": (
                        "Generate a concise, sentence-case title (3-7 words) that captures the main topic "
                        "or goal of this coding session.\nExamples: Fix authentication flow"
                    ),
                },
            ],
            "messages": [{"role": "user", "content": "Run the shell checks."}],
            "tools": [],
            "metadata": {"user_id": json.dumps({"session_id": "root"})},
            "output_config": {
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
        },
        response=(
            {"error": {"message": "retry"}}
            if failed
            else {"id": f"title-response-{index}", "type": "message", "content": [{"type": "text", "text": "{}"}]}
        ),
    )


def _bundle() -> AgentObservationBundle:
    return AgentObservationBundle(
        source="claude_code",
        records=[AgentInvocation(invocation_id="root", status="completed")],
        gaps=[ObservationGap(code="existing_source_gap")],
    )


def _associate(bundle: AgentObservationBundle, calls: list[ModelCallRecord]) -> AgentObservationBundle:
    return observability.associate_claude_code_title_calls(bundle, calls, model_ref=MODEL_REF)


def test_title_attempts_preserve_failure_identity_and_capture_order_without_mutating_source() -> None:
    bundle = _bundle()
    failed, successful = _title_call(failed=True), _title_call(1)
    original = bundle.model_dump()

    associated = _associate(bundle, [successful, failed])

    [title] = _records(associated, SessionTitleObservation)
    [owner] = _records(associated, AgentInvocation)
    assert [ref.model_call_id for ref in title.model_calls] == ["title-0", "title-1"]
    assert [ref.response_id for ref in title.model_calls] == [None, "title-response-1"]
    assert owner.model_calls == title.model_calls
    assert title.included_in_response_usage is False
    assert owner.conversation == []
    assert associated.gaps == bundle.gaps
    assert bundle.model_dump() == original
    assert _associate(associated, [successful, failed]) == associated
    assert failed.status_code == 429 and failed.error_category == "http_error"
    assert successful.tokens_in is None and successful.tokens_out is None


@pytest.mark.parametrize("scope", [None, True, False])
def test_title_usage_scope_requires_revalidation_and_preserves_conflicts(scope: bool | None) -> None:
    call = _title_call()
    bundle = _associate(_bundle(), [call])
    [title] = _records(bundle, SessionTitleObservation)
    title.included_in_response_usage = scope
    joined = _associate(bundle, [call])
    assert _records(joined, SessionTitleObservation)[0].included_in_response_usage is (
        True if scope is True else False
    )
    assert title.included_in_response_usage is scope


def test_title_usage_scope_stays_unknown_when_a_reference_cannot_be_revalidated() -> None:
    bundle = _associate(_bundle(), [_title_call(), _title_call(1)])
    [title] = _records(bundle, SessionTitleObservation)
    title.included_in_response_usage = None
    joined = _associate(bundle, [_title_call()])
    assert _records(joined, SessionTitleObservation)[0].included_in_response_usage is None


@pytest.mark.parametrize(
    "case",
    [
        "other_source",
        "other_dialect",
        "other_model",
        "missing_call_id",
        "missing_identity",
        "malformed_json",
        "identity_list",
        "identity_dict",
        "empty_session",
        "unknown_session",
        "child_identity",
        "metadata_child",
        "request_child",
        "client_identity",
        "child_owner",
        "orphan_child",
        "ambiguous_owner",
        "instruction",
        "schema",
        "tools",
        "malformed_system",
        "malformed_format",
        "duplicate_call_id",
        "conflicting_owner",
        "conflicting_purpose",
        "conflicting_model_ref",
        "conflicting_response_id",
    ],
)
def test_title_join_rejects_unsupported_or_ambiguous_evidence(case: str) -> None:
    bundle, call = _bundle(), _title_call()
    calls = [call]
    if case == "other_source":
        bundle.source = "other"
    elif case == "other_dialect":
        call.dialect = "chat_completions"
    elif case == "other_model":
        call.model_ref = ModelServerRef(type="responses_api_models", name="other")
    elif case == "missing_call_id":
        call.model_call_id = None
    elif case == "missing_identity":
        call.request.pop("metadata")
    elif case in {"malformed_json", "identity_list", "identity_dict"}:
        call.request["metadata"]["user_id"] = {
            "malformed_json": "{",
            "identity_list": "[]",
            "identity_dict": {"session_id": "root"},
        }[case]
    elif case in {"empty_session", "unknown_session", "child_identity"}:
        identity = {"session_id": "" if case == "empty_session" else "other" if case == "unknown_session" else "root"}
        if case == "child_identity":
            identity["agentId"] = "child"
        call.request["metadata"]["user_id"] = json.dumps(identity)
    elif case == "metadata_child":
        call.request["metadata"]["agent_id"] = "child"
    elif case == "request_child":
        call.request["agentId"] = "child"
    elif case == "client_identity":
        call.client_session_id = "other"
    elif case == "child_owner":
        bundle.records[0].parent_invocation_id = "parent"
    elif case == "orphan_child":
        bundle.gaps.append(ObservationGap(code="subagent_parent_unavailable", invocation_id="root"))
    elif case == "ambiguous_owner":
        bundle.records.append(bundle.records[0].model_copy(deep=True))
    elif case == "instruction":
        call.request["system"] = "Return the title field for the task."
    elif case == "schema":
        call.request["output_config"]["format"]["schema"]["properties"]["extra"] = {"type": "string"}
    elif case == "tools":
        call.request["tools"] = [{"name": "Bash"}]
    elif case == "malformed_system":
        call.request["system"] = 5
    elif case == "malformed_format":
        call.request["output_config"] = ["not a mapping"]
    elif case == "duplicate_call_id":
        calls.append(call.model_copy(deep=True))
    elif case in {"conflicting_model_ref", "conflicting_response_id"}:
        bundle.records[0].model_calls = [
            ModelCallRef(
                model_call_id=call.model_call_id,
                model_ref=ModelServerRef(type="responses_api_models", name="other")
                if case == "conflicting_model_ref"
                else MODEL_REF,
                response_id="other-response" if case == "conflicting_response_id" else call.response_id,
            )
        ]
    else:
        reference = ModelCallRef(model_call_id=call.model_call_id, model_ref=MODEL_REF)
        bundle.records.append(
            AgentInvocation(invocation_id="other", model_calls=[reference])
            if case == "conflicting_owner"
            else ContextCompactionObservation(invocation_id="root", model_calls=[reference])
        )
    original = bundle.model_dump()

    result = _associate(bundle, calls)

    assert _records(result, SessionTitleObservation) == []
    assert _records(result, AgentInvocation)[0].model_calls == bundle.records[0].model_calls
    assert bundle.model_dump() == original


@pytest.mark.asyncio
async def test_adapter_joins_title_without_compaction_after_capture_quiescence(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path)
    store.mark_pending("0-0", "title-0")
    failed, successful = _title_call(failed=True), _title_call(1)
    bundle = _bundle()
    bundle.records[0].model_calls = [ModelCallRef(model_ref=MODEL_REF, response_id="main-response")]

    async def finish_capture():
        await asyncio.sleep(0.01)
        for call in (failed, successful):
            store.record("0-0", call.model_dump(exclude={"call_index"}))
        store.record("0-0", _captured_call("main", "main-response"))
        store.clear_pending("0-0", "title-0")

    associated, _ = await asyncio.gather(
        _agent(tmp_path)._associate_helper_calls(bundle, rollout_id="0-0"), finish_capture()
    )
    [title] = _records(associated, SessionTitleObservation)
    assert [ref.model_call_id for ref in title.model_calls] == ["title-0", "title-1"]
    record = {"_ng_task_index": 0, "_ng_rollout_index": 0, "ng_agent_observations": associated.model_dump(mode="json")}
    merge_model_call_capture_into_record(record, [tmp_path], include_payloads=True)
    trajectory = _build_trajectory_record(record, record)
    assert [ref.model_call_id for turn in trajectory.turns for ref in turn.model_calls] == ["main"]
    assert {call.model_call_id for call in trajectory.model_calls} == {"title-0", "title-1", "main"}
    assert "existing_source_gap" in {gap.code for gap in trajectory.gaps}


@pytest.mark.asyncio
async def test_title_capture_failure_preserves_source_gaps(tmp_path: Path) -> None:
    bundle = _bundle()
    with patch("responses_api_agents.claude_code_agent.app.read_available_model_call_records", side_effect=OSError):
        result = await _agent(tmp_path)._associate_helper_calls(bundle, rollout_id="0-0")
    assert _records(result, SessionTitleObservation) == []
    assert {gap.code for gap in result.gaps} >= {"existing_source_gap", "session_title_model_call_join_failed"}


@pytest.mark.asyncio
async def test_adapter_filters_other_model_title_calls(tmp_path: Path) -> None:
    call = _title_call()
    call.model_ref = ModelServerRef(type="responses_api_models", name="other")
    CaptureStore(tmp_path).record("0-0", call.model_dump(exclude={"call_index"}))
    bundle = _bundle()
    assert await _agent(tmp_path)._associate_helper_calls(bundle, rollout_id="0-0") == bundle
