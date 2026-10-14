# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.spatialclaw_agent.app import (
    SpatialClawAgent,
    SpatialClawAgentConfig,
    SpatialClawAgentRunRequest,
)
from responses_api_agents.spatialclaw_agent.observability import read_observations, response_usage


def agent_config(**overrides):
    return SpatialClawAgentConfig(
        host="127.0.0.1",
        port=1234,
        name="agent",
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        dataset_config="videomme_spatialclaw",
        video_root="/videos",
        **overrides,
    )


def test_capture_path_is_carried_into_native_model_url():
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {
        "model": {"responses_api_models": {"openai_model": {"host": "localhost", "port": 99}}}
    }
    client._build_server_base_url.return_value = "http://localhost:99"
    agent = SpatialClawAgent(config=agent_config(), server_client=client)
    for rollout in ("first", "second"):
        request = SimpleNamespace(
            path_params={"rollout_id": rollout}, url=SimpleNamespace(path=f"/ng-rollout/{rollout}/v1/responses")
        )
        assert agent._model_base_url(request) == f"http://localhost:99/ng-rollout/{rollout}/v1"
    assert agent._model_base_url() == "http://localhost:99/v1"


def write_journal(path: Path, usages):
    path.write_text(
        "".join(
            json.dumps({"response_id": f"response-{index}", "usage": usage}) + "\n"
            for index, usage in enumerate(usages)
        )
    )


def test_usage_is_unknown_when_any_response_omits_counts(tmp_path):
    journal = tmp_path / "calls.jsonl"
    assert response_usage(journal) is None
    write_journal(journal, [{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}, None])
    assert response_usage(journal) is None
    write_journal(journal, [{"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}])
    usage = response_usage(journal)
    assert usage.total_tokens == 5
    assert usage.input_tokens_details.cached_tokens is None
    assert usage.output_tokens_details.reasoning_tokens is None


def test_usage_includes_kernel_and_workflow_calls(tmp_path):
    journal = tmp_path / "calls.jsonl"
    write_journal(
        journal,
        [
            {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "total_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 2},
                "completion_tokens_details": {"reasoning_tokens": 1},
            },
            {
                "prompt_tokens": 11,
                "completion_tokens": 5,
                "total_tokens": 16,
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 3},
            },
        ],
    )
    usage = response_usage(journal)
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (18, 8, 26)
    assert usage.input_tokens_details.cached_tokens == 6
    assert usage.output_tokens_details.reasoning_tokens == 4


def test_native_execution_failure_and_partial_evidence_survive(tmp_path):
    session = "native-session"
    trace = tmp_path / f"session-{session}" / "trace.jsonl"
    trace.parent.mkdir()
    trace.write_text(
        json.dumps(
            {
                "event_type": "execution",
                "step_index": 4,
                "timestamp": "2026-10-14T00:00:00+00:00",
                "code": "raise ValueError('bad')",
                "stdout": "",
                "stderr": "",
                "error": "ValueError: bad",
                "execution_time_sec": 0.25,
            }
        )
        + "\n"
        + "{malformed}\n"
    )
    write_journal(tmp_path / "gym-model-responses.jsonl", [None])
    bundle = read_observations(
        tmp_path,
        session,
        model_ref=agent_config().model_server,
        result={"termination_reason": "timeout"},
        failure={"failure_kind": "agent_timeout"},
        duration_ms=400,
    )
    invocation, tool = bundle.records
    assert invocation.status == "failed"
    assert invocation.model_calls[0].response_id == "response-0"
    assert tool.status == "failed" and tool.duration_ms == 250
    assert tool.tool_call_id == "native-session:execute:4"
    assert invocation.conversation[0].call_id == invocation.conversation[1].call_id == tool.tool_call_id
    assert "ValueError: bad" in invocation.conversation[1].output
    assert bundle.gaps[0].code == "agent_artifact_record_unparseable"


@pytest.mark.parametrize("failure_kind", ["agent_timeout", "agent_run_error", "verifier_error"])
async def test_infrastructure_failure_is_masked_and_preserves_observations(monkeypatch, failure_kind):
    client = MagicMock(spec=ServerClient)
    client.post = AsyncMock(return_value=MagicMock(cookies={}))
    response = {
        "id": "r",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    observations = {
        "source": "spatialclaw",
        "records": [{"kind": "agent_invocation", "invocation_id": "session"}],
        "gaps": [],
    }
    response["_ng_agent_observations"] = observations
    if failure_kind != "verifier_error":
        response["_ng_agent_failure"] = {"failure_kind": failure_kind, "failure_reason": "injected outage"}
    monkeypatch.setattr("responses_api_agents.spatialclaw_agent.app.raise_for_status", AsyncMock())
    monkeypatch.setattr(
        "responses_api_agents.spatialclaw_agent.app.get_response_json",
        AsyncMock(side_effect=[response, RuntimeError("scorer unavailable")]),
    )
    result = await SpatialClawAgent(config=agent_config(), server_client=client).run(
        SimpleNamespace(cookies={}), SpatialClawAgentRunRequest(responses_create_params={"input": "question"})
    )
    assert result.mask_sample and result.reward == 0
    assert result.failure_kind == failure_kind
    assert result.ng_agent_observations.records[0].invocation_id == "session"
    assert client.post.await_count == (3 if failure_kind == "verifier_error" else 2)
