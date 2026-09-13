# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from nemo_gym.config_types import AggregateMetricsRequest
from nemo_gym.openai_utils import NeMoGymResponse
from resources_servers.string_match.app import StringMatchResourcesServer, StringMatchResourcesServerConfig
from responses_api_agents.simple_agent.app import (
    SimpleAgentRunRequest,
    SimpleAgentVerifyRequest,
    SimpleAgentVerifyResponse,
)
from responses_api_agents.simple_agent.tests.test_app import _make_agent, _mock_response


def _response(token_count=0):
    message = {
        "id": "message-1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "Paris", "annotations": []}],
    }
    if token_count:
        message.update(
            prompt_token_ids=[1, 2, 3],
            generation_token_ids=list(range(token_count)),
            generation_log_probs=[-0.25] * token_count,
        )
    # /v1/responses returns a serialized NeMoGymResponse, including its defaults.
    return NeMoGymResponse.model_validate(
        {
            "id": "terminal-response",
            "created_at": 1.0,
            "model": "model",
            "object": "response",
            "output": [message],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "metadata": {"terminal_call_id": "call-1"},
        }
    ).model_dump(mode="json")


@pytest.mark.parametrize("case", ["base", "extras", "training_64k", "large_metadata"])
async def test_verify_wire_matches_previous_serialization_without_revalidation(case, monkeypatch):
    agent, client = _make_agent(False)
    data = {"responses_create_params": {"input": [{"role": "user", "content": "Capital of France?"}]}}
    if case != "base":
        data.update(
            task_id="task-1",
            _ng_rollout_id="capture-1",
            _ng_task_index=4,
            _ng_rollout_index=2,
            verifier_metadata={"expected": "Paris", "nested": [{"labels": ["a", None, 3]}]},
            terminal_call_id="call-1",
            response={"stale": "this must be replaced"},
        )
    if case == "large_metadata":
        data["verifier_metadata"]["records"] = [{"text": "metadata" * 128, "index": i} for i in range(1024)]
    body = SimpleAgentRunRequest.model_validate(data)
    response = _response(65536 if case == "training_64k" else 0)
    original_body, original_response = deepcopy(body), deepcopy(response)
    expected = SimpleAgentVerifyRequest.model_validate(body.model_dump() | {"response": response}).model_dump()
    expected_wire = orjson.loads(orjson.dumps(expected))
    seed = _mock_response()
    seed.cookies = {"session": "seeded"}
    generated = _mock_response(response)
    generated.cookies = {"session": "after-tools"}

    async def post(*, url_path, json, cookies, **kwargs):
        if url_path == "/seed_session":
            assert cookies == {"caller": "cookie"}
            return seed
        if url_path == "/v1/responses":
            assert cookies == seed.cookies
            return generated
        assert url_path == "/verify"
        assert cookies == generated.cookies
        assert orjson.loads(orjson.dumps(json)) == expected_wire
        assert "capture_rollout_id" not in json and "_ng_rollout_id" not in json
        return _mock_response(json | {"reward": 1.0})

    client.post = AsyncMock(side_effect=post)
    validate = MagicMock(side_effect=AssertionError("verify payload must not be revalidated locally"))
    dump = MagicMock(side_effect=AssertionError("verify payload must not be dumped twice"))
    monkeypatch.setattr(SimpleAgentVerifyRequest, "model_validate", validate)
    monkeypatch.setattr(SimpleAgentVerifyRequest, "model_dump", dump)
    result = await agent.run(MagicMock(cookies={"caller": "cookie"}), body)
    assert result.reward == 1.0
    assert result.response.id == "terminal-response"
    assert body == original_body and response == original_response
    assert [item.kwargs["url_path"] for item in client.post.await_args_list] == [
        "/seed_session",
        "/v1/responses",
        "/verify",
    ]
    validate.assert_not_called()
    dump.assert_not_called()


@pytest.mark.parametrize("skip", [False, True])
async def test_trajectory_extraction_does_not_mutate_response(skip, monkeypatch):
    agent, client = _make_agent(True)
    agent.config.skip_verification = skip
    agent.config.skip_verification_reward = 0.25
    body = SimpleAgentRunRequest.model_validate(
        {"responses_create_params": {"input": "question"}, "task_id": "task-1", "_ng_rollout_id": "rollout-1"}
    )
    response = _response()
    response["_ng_trajectory"] = {
        "task_id": "unscoped",
        "rollout_id": "rollout-1",
        "invocations": [],
        "turns": [],
        "tool_calls": [],
    }
    before = deepcopy(response)
    # Return the caller-owned dictionary directly so mutations are observable.
    monkeypatch.setattr(
        "responses_api_agents.simple_agent.app.get_response_json", AsyncMock(side_effect=lambda value: value.payload)
    )

    async def post(*, url_path, **kwargs):
        if url_path == "/seed_session":
            return _mock_response()
        if url_path.endswith("/v1/responses"):
            return MagicMock(ok=True, cookies={}, payload=response)
        assert url_path == "/verify" and not skip
        assert "_ng_trajectory" not in kwargs["json"]["response"]
        return MagicMock(ok=True, payload=kwargs["json"] | {"reward": 1.0})

    client.post = AsyncMock(side_effect=post)
    result = await agent.run(MagicMock(cookies={}), body)
    assert result.reward == (0.25 if skip else 1.0)
    assert result.model_dump()["ng_trajectory"]["task_id"] == "task-1"
    assert response == before
    if skip:
        assert result.model_dump()["verification_skipped"] is True
        assert client.post.await_count == 2


@pytest.mark.parametrize("stage", ["seed", "model", "verify"])
async def test_run_propagates_http_failures(stage):
    agent, client = _make_agent(False)
    error = RuntimeError(f"{stage} failed")
    failure = _mock_response(status=500)
    failure.raise_for_status.side_effect = error
    replies = [_mock_response(), _mock_response(_response()), _mock_response()]
    index = ["seed", "model", "verify"].index(stage)
    replies[index] = failure
    client.post = AsyncMock(side_effect=replies)
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        await agent.run(MagicMock(cookies={}), SimpleAgentRunRequest(responses_create_params={"input": "question"}))
    assert client.post.await_count == index + 1


async def test_final_verify_response_still_validated():
    agent, client = _make_agent(False)
    client.post = AsyncMock(side_effect=[_mock_response(), _mock_response(_response()), _mock_response({"reward": 1})])
    with pytest.raises(ValidationError):
        await agent.run(MagicMock(cookies={}), SimpleAgentRunRequest(responses_create_params={"input": "question"}))


async def test_verified_rewards_use_resources_aggregate_metrics():
    agent, client = _make_agent(False)
    metrics = {"group_level_metrics": [], "agent_metrics": {"mean/reward": 0.5}, "key_metrics": {"accuracy": 0.5}}
    client.post = AsyncMock(return_value=_mock_response(metrics))
    body = AggregateMetricsRequest(verify_responses=[{"reward": 0.0}, {"reward": 1.0}])
    result = await agent.aggregate_metrics(body)
    assert result.agent_metrics == {"mean/reward": 0.5}
    assert result.key_metrics == {"accuracy": 0.5}
    assert result.group_level_metrics == []
    client.post.assert_awaited_once_with(server_name="resources", url_path="/aggregate_metrics", json=body)


async def test_skipped_verification_does_not_report_artificial_reward_metrics():
    agent, client = _make_agent(False)
    agent.config.skip_verification = True
    body = AggregateMetricsRequest(verify_responses=[{"reward": 0.25, "verification_skipped": True}])
    with pytest.warns(RuntimeWarning, match="skip_verification=True"):
        result = await agent.aggregate_metrics(body)
    assert result.agent_metrics == {} and result.key_metrics == {} and result.group_level_metrics == []
    client.post.assert_not_called()


@pytest.mark.parametrize("malformation", ["missing_metadata", "invalid_output", "partial_tokens"])
def test_resources_ingress_rejects_malformed_verify_payload(malformation):
    agent, client = _make_agent(False)
    resources = StringMatchResourcesServer(
        config=StringMatchResourcesServerConfig(name="resources", entrypoint="app.py", host="127.0.0.1", port=0),
        server_client=client,
    )
    payload = {
        "responses_create_params": {"input": "Capital of France?"},
        "response": _response(),
        "expected_answer": "Paris",
        "extraction_mode": "full_response",
    }
    with TestClient(resources.setup_webserver()) as http:
        valid = http.post("/verify", json=payload)
        assert valid.status_code == 200 and SimpleAgentVerifyResponse.model_validate(valid.json()).reward == 1.0
        if malformation == "missing_metadata":
            del payload["expected_answer"]
        elif malformation == "invalid_output":
            payload["response"]["output"] = "not an output list"
        else:
            payload["response"]["output"][0]["generation_token_ids"] = [1, 2]
        rejected = http.post("/verify", json=payload)
        assert rejected.status_code == 422
        assert rejected.json()["detail"]
