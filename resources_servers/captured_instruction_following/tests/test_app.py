# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic-only tests: no judge, real backend, source data, or network."""

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetricsRequest
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.captured_instruction_following.app import (
    VERIFIER_FIXTURE,
    CapturedInstructionFollowingConfig,
    CapturedInstructionFollowingResourcesServer,
    VerifyRequest,
)
from resources_servers.captured_instruction_following.metrics import summarize
from resources_servers.captured_instruction_following.scoring import canonical_sha256, compare_value, score_native
from resources_servers.captured_instruction_following.task_data import CapturedMetadata, TaskData
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in (ROOT / "data/example.jsonl").read_text().splitlines()]


@pytest.fixture
def server() -> CapturedInstructionFollowingResourcesServer:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client.post = AsyncMock(side_effect=AssertionError("Verifier must never call a judge or external backend"))
    config = CapturedInstructionFollowingConfig(host="127.0.0.1", port=8080, entrypoint="app.py", name="captured")
    return CapturedInstructionFollowingResourcesServer(config=config, server_client=client)


def candidate(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": "synthetic-response",
        "created_at": 0,
        "model": "synthetic-checkpoint",
        "object": "response",
        "status": "completed",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def call(name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": "synthetic-call",
        "type": "function_call",
        "call_id": "synthetic-call",
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


def message(text: str) -> dict[str, Any]:
    return {
        "id": "synthetic-message",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


@pytest.mark.parametrize(
    "profile,expected,actual,passed",
    [
        ("case_insensitive_scalars", "AbC", "abc", True),
        ("case_insensitive_scalars", "abc", " abc ", False),
        ("case_insensitive_scalars", True, 1, True),
        ("case_insensitive_scalars", False, 0, True),
        ("case_insensitive_scalars", True, "yes", False),
        ("normalized_scalars", True, " YES ", True),
        ("normalized_scalars", False, "0", True),
        ("normalized_scalars", 7, "7.0", True),
        ("normalized_scalars", "AbC", " abc ", True),
        ("normalized_scalars", None, None, True),
        ("normalized_scalars", None, "none", False),
        ("normalized_scalars", 100, "1e2", False),
    ],
)
def test_scalar_profiles_preserve_the_documented_contract(profile, expected, actual, passed):
    assert compare_value(expected, actual, profile) is passed


def test_first_call_and_subset_semantics_are_explicit():
    output = [call("choose_record", '{"id":"abc","extra":"ignored"}'), call("irrelevant_later_call", "{}")]
    result = score_native(output, "choose_record", {"id": "ABC"}, "case_insensitive_scalars")
    assert result["reward"] == 1
    assert result["extra_argument_keys"] == ["extra"]
    assert result["ignored_call_count"] == 1
    assert result["parameter_scope"] == "expected_subset"
    assert (
        score_native(list(reversed(output)), "choose_record", {"id": "ABC"}, "case_insensitive_scalars")["reward"] == 0
    )
    assert (
        score_native([call("choose_record", "{}")], "choose_record", {"optional": None}, "normalized_scalars")[
            "reward"
        ]
        == 1
    )
    assert score_native([call("choose_record", "{}")], "choose_record", {}, "case_insensitive_scalars")["reward"] == 1


@pytest.mark.parametrize("arguments", ["{", "[]", "null", '{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}'])
def test_malformed_or_nonobject_arguments_fail_even_without_expected_params(arguments):
    result = score_native([call("choose_record", arguments)], "choose_record", {}, "normalized_scalars")
    assert result["reward"] == 0
    assert result["arguments_valid"] is False


def test_no_json_text_fallback_or_supplied_history_answer():
    output = [message('{"function":"choose_record","params":{"id":"ABC"}}')]
    assert score_native(output, "choose_record", {"id": "ABC"}, "case_insensitive_scalars")["reward"] == 0
    assert score_native([], "choose_record", {}, "case_insensitive_scalars")["reward"] == 0


def test_synthetic_data_schema_and_hash_roundtrip(rows):
    assert len(rows) == 5
    for row in rows:
        TaskData.model_validate(
            {**row["verifier_metadata"], **{key: value for key, value in row.items() if key != "verifier_metadata"}}
        )
        body = VerifyRequest.model_validate({**row, "response": candidate([message("Synthetic answer")])})
        params = body.responses_create_params.model_dump(mode="json", exclude_unset=True)
        assert canonical_sha256(params) == row["verifier_metadata"]["evaluation_request_sha256"]
        assert row["synthetic"] is True and row["do_not_use_for_scoring"] is True
        assert "Old synthetic" not in json.dumps(params)


@pytest.mark.asyncio
async def test_native_response_passes_and_unreviewed_text_is_masked(server, rows):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.setup_webserver()), base_url="http://resource"
    ) as client:
        row = rows[0]
        metadata = row["verifier_metadata"]
        result = await client.post(
            "/verify",
            json={
                **row,
                "response": candidate([call(metadata["expected_function"], json.dumps(metadata["expected_params"]))]),
            },
        )
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1 and result.json()["mask_sample"] is False
        assert result.json()["benchmark"] == row["benchmark"]
        assert result.json()["orig_trajectory_id"] == row["orig_trajectory_id"]
        text = await client.post(
            "/verify", json={**rows[-1], "response": candidate([message("A useful synthetic answer")])}
        )
        assert text.status_code == 200, text.text
        assert text.json()["reward"] == 0
        assert text.json()["mask_sample"] is True
        assert text.json()["scoring_status"] == "not_scored"
        assert text.json()["scoring_reason"] == "criteria_require_review"
        assert text.json()["evaluation_completed"] is False
    server.server_client.post.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ["prompt", "identity", "incomplete", "provider_error", "queued", "in_progress", "pending_call", "incomplete_call"],
)
async def test_binding_and_incomplete_candidate_are_masked(server, rows, change):
    row = copy.deepcopy(rows[0])
    response = candidate([call("get_open_invoice", '{"customer_id":"SYN-CUST-1"}')])
    if change == "prompt":
        row["responses_create_params"]["input"][-1]["content"] += "changed"
    elif change == "identity":
        row["orig_trajectory_id"] = "wrong-source"
    elif change == "incomplete":
        response["incomplete_details"] = {"reason": "max_output_tokens"}
        response["status"] = "incomplete"
    elif change == "provider_error":
        response["error"] = {"code": "server_error", "message": "synthetic provider failure"}
        response["status"] = "failed"
    elif change in {"queued", "in_progress"}:
        response["status"] = change
    elif change == "pending_call":
        response["output"][0]["status"] = "in_progress"
    else:
        response["output"][0]["status"] = "incomplete"
    result = await server.verify(VerifyRequest.model_validate({**row, "response": response}))
    assert result.reward == 0 and result.mask_sample is True and result.scoring_status == "not_scored"


def test_unreviewed_and_missing_items_do_not_become_failures():
    def row(identifier, reward, status, masked=False):
        return {
            "benchmark": "synthetic",
            "orig_trajectory_id": identifier,
            "reward": reward,
            "scoring_status": status,
            "mask_sample": masked,
            "verifier_metadata": {"cohort_size": 7},
        }

    values = [row("1", 1, "scored"), row("2", 0, "scored"), row("3", 0, "not_scored", True)]
    summary = summarize(values)
    assert summary["passed"] == 1 and summary["failed"] == 1
    assert summary["scored"] == 2 and summary["not_scored"] == 1
    assert summary["scored_subset_accuracy"] == 0.5
    assert summary["scoring_coverage"] == 2 / 7
    assert summary["missing_items"] == 4
    assert summary["complete_accuracy"] is None
    entirely_unreviewed = summarize([row("3", 0, "not_scored", True)])
    assert entirely_unreviewed["failed"] == 0 and entirely_unreviewed["scored_subset_accuracy"] is None
    values[2].update(scoring_status="scored", mask_sample=False)
    for value in values:
        value["verifier_metadata"]["cohort_size"] = 3
    assert summarize(values)["complete_accuracy"] == 1 / 3


@pytest.mark.asyncio
async def test_aggregate_api_keeps_masked_samples_out_of_accuracy(server, rows):
    native = await server.verify(
        VerifyRequest.model_validate(
            {**rows[0], "response": candidate([call("get_open_invoice", '{"customer_id":"SYN-CUST-1"}')])}
        )
    )
    text = await server.verify(
        VerifyRequest.model_validate({**rows[-1], "response": candidate([message("Synthetic report")])})
    )
    aggregate = await server.aggregate_metrics(
        AggregateMetricsRequest(
            verify_responses=[
                native.model_dump() | {"_ng_task_index": 0, "_ng_rollout_index": 0},
                text.model_dump() | {"_ng_task_index": 1, "_ng_rollout_index": 0},
            ]
        )
    )
    assert aggregate.agent_metrics["failed"] == 0
    assert aggregate.agent_metrics["scored"] == 1 and aggregate.agent_metrics["not_scored"] == 1
    assert aggregate.agent_metrics["scoring_coverage"] == 1 / 5
    assert aggregate.agent_metrics["scored_subset_accuracy"] == 1
    assert aggregate.agent_metrics["complete_accuracy"] is None


def test_native_metadata_requires_a_reviewed_reference(rows):
    metadata = copy.deepcopy(rows[0]["verifier_metadata"])
    metadata.pop("expected_params")
    with pytest.raises(ValidationError):
        CapturedMetadata.model_validate(metadata)
    with pytest.raises(ValidationError):
        CapturedMetadata.model_validate(rows[-1]["verifier_metadata"] | {"expected_function": "must_not_grade_text"})


@pytest.mark.asyncio
async def test_public_verifier_fixture_contract():
    results = await exercise_verifier_fixture(VERIFIER_FIXTURE, reward_range=[0, 1], determinism="seeded")
    assert {result.kind for result in results} == {"full_reward", "zero_reward", "malformed", "determinism"}


class MockHTTPResponse:
    def __init__(self, payload):
        self.status = 200
        self.ok = True
        self.headers = {}
        self.cookies = {}
        self._bytes = json.dumps(payload).encode()
        self.content = self

    async def read(self):
        return self._bytes

    async def json(self, **kwargs):
        return json.loads(self._bytes)

    def raise_for_status(self):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("mode_index", [0, 4])
async def test_real_simple_agent_session_returns_native_calls_unexecuted(server, rows, mode_index):
    row = rows[mode_index]
    metadata = row["verifier_metadata"]
    output = (
        [call(metadata["expected_function"], json.dumps(metadata["expected_params"]))]
        if mode_index == 0
        else [message("Synthetic text completion")]
    )
    recorded = []

    async def post(**kwargs):
        recorded.append(kwargs)
        assert kwargs["server_name"] == "policy_model", "No judge or resources tool may be called"
        return MockHTTPResponse(candidate(output))

    client_double = MagicMock(spec=ServerClient)
    client_double.global_config_dict = {}
    client_double.post = AsyncMock(side_effect=post)
    config = SimpleAgentConfig(
        host="127.0.0.1",
        port=8081,
        name="captured_agent",
        entrypoint="app.py",
        execute_tools=False,
        max_steps=1,
        model_server={"type": "responses_api_models", "name": "policy_model"},
        resources_server={"type": "resources_servers", "name": "captured"},
    )
    agent = SimpleAgent(config=config, server_client=client_double)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent.setup_webserver()), base_url="http://agent"
    ) as client:
        seed = AgentSeedSessionRequest(
            agent_session_id=f"synthetic-{mode_index}",
            episode_id=EpisodeId(rollout_id=f"synthetic-{mode_index}", attempt=0),
            task_id=TaskId(taskset="captured:synthetic", task_id=str(mode_index)),
        )
        result = await client.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))
        assert result.status_code == 200, result.text
        generated = await client.post("/v1/responses", json=row["responses_create_params"])
        assert generated.status_code == 200, generated.text
    assert len(recorded) == 1
    body = recorded[0]["json"]
    actual = body.model_dump(mode="json", exclude_unset=True) if hasattr(body, "model_dump") else body
    assert actual["input"] == row["responses_create_params"]["input"]
    assert "verifier_metadata" not in actual and "source_identity" not in actual
    result = await server.verify(VerifyRequest.model_validate({**row, "response": generated.json()}))
    assert result.orig_trajectory_id == row["orig_trajectory_id"]
    assert result.mask_sample is (mode_index != 0)
    if mode_index == 0:
        assert result.reward == 1
