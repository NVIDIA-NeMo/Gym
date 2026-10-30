# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy

import pytest

from environment_servers.swe_together.judge_evidence import aggregate_judge_evidence, collect_judge_evidence
from nemo_gym.base_responses_api_agent import AgentCloseSessionResponse
from nemo_gym.harness_capabilities.checker import inspect_record
from nemo_gym.rollout_collection import _build_trajectory_record
from nemo_gym.rollout_observability import AgentObservationBundle, ModelCallRef, TrajectoryRecord


MODEL = {"type": "responses_api_models", "name": "grading"}


def exchange(identity="call", response="response", model=MODEL):
    return {
        "model_call_id": identity,
        "model_ref": model,
        "dialect": "anthropic_messages",
        "status_code": 200,
        "request": {"messages": [{"role": "user", "content": "Grade this patch"}]},
        "response": {"id": response, "content": [{"type": "text", "text": "verdict"}]},
    }


def receipt(*responses):
    return AgentCloseSessionResponse(
        agent_session_id="judge-session",
        cleanup_confirmed=True,
        agent_observations=AgentObservationBundle.model_validate(
            {
                "source": "claude_code",
                "records": [
                    {
                        "kind": "agent_invocation",
                        "invocation_id": "judge-invocation",
                        "status": "completed",
                        "model_calls": [{"model_ref": MODEL, "response_id": response} for response in responses],
                    }
                ],
            }
        ),
    )


def parent():
    return TrajectoryRecord.model_validate(
        {
            "task_id": "task",
            "rollout_id": "parent",
            "invocations": [{"invocation_id": "candidate", "status": "incomplete"}],
            "gaps": [{"code": "turns_unavailable"}],
        }
    )


def title():
    call = exchange("title", "title-response")
    call["request"] = {
        "tools": [],
        "system": [{"type": "text", "text": "Generate a concise, sentence-case title (3-7 words) for this session"}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                    "additionalProperties": False,
                },
            }
        },
    }
    return call


def test_preserves_exchanges_and_distinct_receipt_ownership_without_inventing_steps():
    calls = [exchange(str(i), str(i)) for i in range(11)] + [title()]
    original = deepcopy(calls)
    result, provenance = aggregate_judge_evidence(
        parent(), receipt(*(str(i) for i in range(11))), calls, source_episode="child"
    )
    assert calls == original
    assert len(result.model_calls) == 11
    assert len(result.verifier_invocations) == 1
    assert result.verifier_invocations[0].invocation_id == "judge-invocation"
    assert len(result.verifier_invocations[0].model_calls) == 11
    assert result.invocations == parent().invocations
    assert result.turns == []
    assert result.gaps == parent().gaps
    assert provenance["excluded"] == [{"model_call_id": "title", "reason": "claude_session_title"}]
    assert provenance["kind"] == "judge_receipt_reconstruction"
    assert result.model_calls[0].request == calls[0]["request"]
    assert result.model_calls[0].response == calls[0]["response"]
    again, _ = aggregate_judge_evidence(result, receipt(*(str(i) for i in range(11))), calls, source_episode="child")
    assert again == result


@pytest.mark.parametrize("change", ["model", "response", "ambiguous"])
def test_identity_mismatch_or_ambiguity_keeps_call_without_owner(change):
    close = receipt("response")
    call = exchange()
    if change == "model":
        call["model_ref"] = {**MODEL, "name": "different"}
    elif change == "response":
        call["response"]["id"] = "different"
    else:
        close.agent_observations.records.append(
            close.agent_observations.records[0].model_copy(update={"invocation_id": "other-judge"})
        )
    result, _ = aggregate_judge_evidence(parent(), close, [call], source_episode="child")
    assert len(result.model_calls) == 1
    assert not result.verifier_invocations
    assert "judge_call_owner_unresolved" in {gap.code for gap in result.gaps}


def test_unreferenced_reasoning_is_not_housekeeping_and_receipt_has_precedence():
    result, provenance = aggregate_judge_evidence(parent(), receipt(), [exchange()], source_episode="child")
    assert len(result.model_calls) == 1
    assert provenance["excluded"] == []
    result, provenance = aggregate_judge_evidence(
        parent(), receipt("title-response"), [title()], source_episode="child"
    )
    assert len(result.verifier_invocations[0].model_calls) == 1
    assert provenance["excluded"] == []


def test_missing_receipt_exchange_is_explicit():
    result, _ = aggregate_judge_evidence(parent(), receipt("missing"), [], source_episode="child")
    assert "judge_receipt_calls_missing" in {gap.code for gap in result.gaps}


def test_existing_collector_accepts_producer_verifier_mapping_and_preserves_candidate_gap():
    result, _ = aggregate_judge_evidence(parent(), receipt("response"), [exchange()], source_episode="child")
    record = {"ng_trajectory": result.model_dump(mode="json")}
    row = {"trajectory_identity": {"task_id": "task", "rollout_id": "parent"}}
    canonical = _build_trajectory_record(row, record)
    assert canonical.model_calls == result.model_calls
    assert canonical.verifier_invocations == result.verifier_invocations
    assert canonical.turns == []
    assert "turns_unavailable" in {gap.code for gap in canonical.gaps}
    record["ng_trajectory"] = canonical.model_dump(mode="json")
    checks = {check["id"]: check["status"] for check in inspect_record(record)["checks"]}
    assert checks["ownership.call_owner"] == "pass"
    assert checks["steps.gap"] == "fail"


def test_judge_aggregation_keeps_unowned_simulator_capture_visible():
    simulator = exchange("simulator", "simulator-response", {**MODEL, "name": "simulator"})
    row = {"trajectory_identity": {"task_id": "task", "rollout_id": "parent"}}
    original = _build_trajectory_record(
        row, {"ng_trajectory": parent().model_dump(mode="json"), "ng_model_call_capture": {"calls": [simulator]}}
    )
    result, _ = aggregate_judge_evidence(original, receipt("response"), [exchange()], source_episode="child")
    canonical = _build_trajectory_record(row, {"ng_trajectory": result.model_dump(mode="json")})

    assert canonical.model_calls[0] == original.model_calls[0]
    assert [call.model_call_id for call in canonical.model_calls] == ["simulator", "call"]
    assert canonical.invocations == original.invocations and not canonical.turns
    assert [ref.model_call_id for owner in canonical.verifier_invocations for ref in owner.model_calls] == ["call"]
    checks = {
        check["id"]: check["status"]
        for check in inspect_record({"ng_trajectory": canonical.model_dump(mode="json")})["checks"]
    }
    assert checks["ownership.call_target"] == "pass"
    assert checks["ownership.call_owner"] == "fail"


async def test_collection_retains_raw_housekeeping_and_records_exact_source(tmp_path):
    captures = tmp_path / "captures"
    captures.mkdir()
    raw = [exchange(), title()]
    (captures / "explicit-child.capture.jsonl").write_text("\n".join(json.dumps(call) for call in raw))
    verification = {
        "judge_capture_episode": "explicit-child",
        "judge_close": receipt("response").model_dump(mode="json"),
    }
    result = await collect_judge_evidence(
        parent(), verification, {"observability_enabled": True, "model_call_capture_dir": str(captures)}, tmp_path
    )
    assert len(result.model_calls) == 1
    assert json.loads((tmp_path / "judge-model-calls.json").read_text()) == raw
    assert verification["judge_evidence_provenance"]["source_episode"] == "explicit-child"
    assert verification["judge_evidence_provenance"]["excluded"][0]["model_call_id"] == "title"


async def test_missing_capture_keeps_candidate_and_exposes_gap(tmp_path):
    verification = {"judge_capture_episode": "missing", "judge_close": receipt("response").model_dump(mode="json")}
    result = await collect_judge_evidence(
        parent(), verification, {"observability_enabled": True, "model_call_capture_dir": str(tmp_path)}, tmp_path
    )
    assert result.invocations == parent().invocations
    assert not result.model_calls
    assert "judge_capture_unavailable" in {gap.code for gap in result.gaps}
    assert "FileNotFoundError" in verification["judge_evidence_error"]


@pytest.mark.parametrize(
    "reference,identities,owned,gap",
    [
        (
            {"model_call_id": "expected", "model_ref": MODEL, "response_id": "response"},
            ["actual"],
            [],
            "judge_receipt_calls_missing",
        ),
        (
            {"model_ref": MODEL, "response_id": "response"},
            ["first", "second"],
            [],
            "judge_receipt_reference_ambiguous",
        ),
        (
            {"model_call_id": "second", "model_ref": MODEL, "response_id": "response"},
            ["first", "second"],
            ["second"],
            "judge_call_owner_unresolved",
        ),
        ({"model_call_id": "call"}, ["call"], ["call"], None),
    ],
)
def test_receipt_resolves_all_identifiers_to_one_capture(reference, identities, owned, gap):
    close = receipt()
    close.agent_observations.records[0].model_calls = [ModelCallRef.model_validate(reference)]
    result, _ = aggregate_judge_evidence(
        parent(), close, [exchange(identity) for identity in identities], source_episode="child"
    )
    assert [call.model_call_id for call in result.model_calls] == identities
    assert [ref.model_call_id for owner in result.verifier_invocations for ref in owner.model_calls] == owned
    if gap:
        assert gap in {item.code for item in result.gaps}
    assert result.invocations == parent().invocations and result.turns == parent().turns


def test_conflicting_reference_does_not_hide_a_title_shaped_call():
    close = receipt("title-response")
    close.agent_observations.records[0].model_calls[0].model_call_id = "wrong"
    result, provenance = aggregate_judge_evidence(parent(), close, [title()], source_episode="child")
    assert [call.model_call_id for call in result.model_calls] == ["title"]
    assert not result.verifier_invocations and not provenance["excluded"]
    assert "judge_call_owner_unresolved" in {gap.code for gap in result.gaps}


async def test_missing_close_retains_captured_calls_and_original_failure(tmp_path):
    raw = exchange()
    (tmp_path / "child.capture.jsonl").write_text(json.dumps(raw))
    verification = {
        "judge_capture_episode": "child",
        "judge_close_error": "connection lost",
        "mask_sample": True,
        "evaluation_completed": False,
        "reward": 0,
    }
    original = deepcopy(verification)
    result = await collect_judge_evidence(
        parent(), verification, {"observability_enabled": True, "model_call_capture_dir": str(tmp_path)}, tmp_path
    )
    assert result.model_calls[0].request == raw["request"]
    assert result.model_calls[0].response == raw["response"]
    assert not result.verifier_invocations
    assert {"judge_receipt_unavailable", "judge_call_owner_unresolved"} <= {gap.code for gap in result.gaps}
    assert result.invocations == parent().invocations and result.turns == parent().turns
    assert all(verification[key] == value for key, value in original.items())
    provenance = verification["judge_evidence_provenance"]
    assert provenance["judge_session_id"] is None and provenance["receipt_artifact"] is None


@pytest.mark.parametrize("placement", ["middle", "last", "only"])
async def test_damaged_capture_retains_intact_exchanges_and_raw_bytes(tmp_path, placement):
    first, second = exchange("first", "first"), exchange("second", "second")
    lines = {
        "middle": [json.dumps(first), '{"broken":', json.dumps(second)],
        "last": [json.dumps(first), json.dumps(second), '{"broken":'],
        "only": ['{"broken":'],
    }[placement]
    raw = "\n".join(lines).encode()
    (tmp_path / "child.capture.jsonl").write_bytes(raw)
    verification = {
        "judge_capture_episode": "child",
        "judge_close": receipt("first", "second").model_dump(mode="json"),
    }
    result = await collect_judge_evidence(
        parent(), verification, {"observability_enabled": True, "model_call_capture_dir": str(tmp_path)}, tmp_path
    )
    expected = [] if placement == "only" else [first, second]
    assert [call.model_call_id for call in result.model_calls] == [call["model_call_id"] for call in expected]
    assert json.loads((tmp_path / "judge-model-calls.json").read_text()) == expected
    assert (tmp_path / "judge-model-calls.capture.jsonl").read_bytes() == raw
    assert [(gap.code, gap.detail) for gap in result.gaps if gap.code == "judge_capture_records_unreadable"] == [
        ("judge_capture_records_unreadable", "count=1")
    ]
    if expected:
        assert [ref.model_call_id for ref in result.verifier_invocations[0].model_calls] == ["first", "second"]
        assert [call.response for call in result.model_calls] == [first["response"], second["response"]]
