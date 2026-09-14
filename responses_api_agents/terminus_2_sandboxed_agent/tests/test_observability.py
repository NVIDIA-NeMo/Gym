# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.base_responses_api_model import CaptureStore, ModelCallRecord, merge_model_call_capture_into_record
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.rollout_collection import _attach_trajectory_record
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.rollout_observability import AgentObservationBundle, join_model_call_observations
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_sandboxed_agent import app as app_module
from responses_api_agents.terminus_2_sandboxed_agent.app import (
    Terminus2Agent,
    Terminus2AgentConfig,
    Terminus2AgentRunRequest,
)


@pytest.fixture
def execution(monkeypatch):
    config = Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_1_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="swebench_resources_server"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_turns=100,
        enable_summarize=True,
        proactive_summarization_threshold=8000,
        tmux_pane_width=160,
        tmux_pane_height=40,
        dump_trajectory=False,
        debug=False,
        model_context_limit=32_000,
        model_output_limit=4_000,
        llm_request_timeout=60,
        sandbox_provider="opensandbox",
        sandbox_timeout=10,
        remote_tmux_binary_path=None,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"observability_enabled": True}
    server = Terminus2Agent(config=config, server_client=client)
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, stdout="", stderr="")), stop=AsyncMock()
    )
    monkeypatch.setattr(Terminus2Agent, "_connect_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(app_module, "get_server_url", lambda _: "http://model")
    calls = []
    mode = SimpleNamespace(value="success")

    async def transport(self, **kwargs):
        # Intercept after the real client merges per-instance headers.
        await asyncio.sleep(0)
        index = len(calls)
        failed = mode.value == "all_fail" or (mode.value == "retry" and index == 0)
        response = {
            "id": f"resp_{index}",
            "created_at": 0,
            "model": "policy_model",
            "object": "response",
            "output": [
                {
                    "id": f"msg_{index}",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}],
                }
            ],
            "tool_choice": "auto",
            "tools": [],
            "parallel_tool_calls": True,
        }
        calls.append(
            ModelCallRecord(
                call_index=index,
                model_call_id=f"call_{index}",
                client_session_id=kwargs["headers"].get("x-session-id"),
                response_id=None if failed else response["id"],
                status_code=None if failed else 200,
                response_status=None if failed else "completed",
                error_category="timeout" if failed else None,
                request=kwargs["json"],
                response=None if failed else response,
                tokens_in=3,
                tokens_out=2,
                tokens_total=5,
            )
        )
        assert kwargs["url"].endswith("/v1/responses")
        if failed:
            raise TimeoutError
        return SimpleNamespace(status=200, ok=True, read=AsyncMock(return_value=json.dumps(response).encode()))

    monkeypatch.setattr(app_module.NeMoGymAsyncOpenAI, "_request_with_retry", transport)

    class Agent:
        def __init__(self, **kwargs):
            self.llm = kwargs["llm"]
            self._times_spent = []
            self._num_proactive_compactions = 0

        async def setup(self, environment):
            pass

        async def run(self, instruction, environment, context):
            await self.llm.call(instruction)
            if mode.value == "compaction":
                self.llm._is_compacting = True
                await self.llm.call("summarize", message_history=[{"role": "user", "content": instruction}])
                self.llm._is_compacting = False
                await self.llm.call("continue", message_history=[{"role": "user", "content": "summary"}])
            if mode.value == "failure":
                raise ValueError("agent failed after a response")

    monkeypatch.setattr(app_module, "NeMoGymTerminus2", Agent)

    async def run(rollout_id="test-rollout"):
        body = Terminus2AgentRunRequest(responses_create_params={"input": "solve this"}, _ng_rollout_id=rollout_id)
        request = SimpleNamespace(
            json=AsyncMock(return_value=body.model_dump(by_alias=True) | {"_ng_rollout_id": rollout_id}),
            session={app_module.SESSION_ID_KEY: rollout_id},
            cookies={},
        )

        async def post(**kwargs):
            if kwargs["url_path"] == "/seed_session":
                return SimpleNamespace(
                    status=200, ok=True, cookies={}, json=AsyncMock(return_value={"sandbox_handle": "sandbox"})
                )
            # The resources verifier knows nothing about agent observations.
            result = kwargs["json"] | {"reward": 1.0}
            return SimpleNamespace(status=200, ok=True, read=AsyncMock(return_value=json.dumps(result).encode()))

        client.post = post
        result = await server.run(request, body)
        return json.loads(result.model_dump_json(by_alias=True))

    return SimpleNamespace(run=run, calls=calls, mode=mode, client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,count,status",
    [
        ("success", 1, "completed"),
        ("retry", 2, "completed"),
        ("all_fail", 10, "incomplete"),
        ("compaction", 3, "completed"),
        ("failure", 1, "failed"),
    ],
)
async def test_saved_invocation_owns_calls_and_enables_health(execution, tmp_path, mode, count, status):
    execution.mode.value = mode
    result = await execution.run()
    bundle = AgentObservationBundle.model_validate(result["ng_agent_observations"])
    [invocation] = bundle.records
    assert invocation.status == status
    assert invocation.duration_ms >= 0
    assert invocation.error_type == ({"all_fail": "TimeoutError", "failure": "ValueError"}.get(mode))
    assert len(execution.calls) == count
    assert {call.client_session_id for call in execution.calls} == {invocation.invocation_id}
    if mode in {"retry", "all_fail"}:
        assert execution.calls[0].response_id is None
    joined = join_model_call_observations(bundle, execution.calls)
    assert [ref.model_call_id for ref in joined.records[0].model_calls] == [
        call.model_call_id for call in execution.calls
    ]
    store = CaptureStore(tmp_path / "capture")
    for call in execution.calls:
        exchange = call.model_dump(mode="json")
        if exchange["response"] is not None:
            exchange["response"]["usage"] = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        store.record("test-rollout", exchange)
    result["_ng_rollout_id"] = "test-rollout"
    merge_model_call_capture_into_record(result, [store.root], include_payloads=True)
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    result.update(row)
    _attach_trajectory_record(row, result)
    assert result["ng_trajectory"]["invocations"][0]["model_calls"] == joined.records[0].model_dump()["model_calls"]
    assert not result["ng_trajectory"]["turns"]
    assert "turns_unavailable" in {gap["code"] for gap in result["ng_trajectory"]["gaps"]}
    path = tmp_path / "rollouts.jsonl"
    path.write_text(json.dumps(result) + "\n")
    [health] = run_health_checks(path, workers=1).rollouts
    assert {
        "model_call_failed",
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "model_call_runaway_generation",
        "trajectory_capture_mismatch",
    }.isdisjoint(health.unobserved)
    assert {"agent_turn_hollow", "rollout_missing_agent_turns"} <= set(health.unobserved)
    assert ("model_call_failed" in {finding.check for finding in health.findings}) == (mode in {"retry", "all_fail"})


@pytest.mark.asyncio
async def test_concurrent_executions_have_distinct_ownership(execution):
    results = await asyncio.gather(execution.run("first"), execution.run("second"))
    ids = [result["ng_agent_observations"]["records"][0]["invocation_id"] for result in results]
    assert len(set(ids)) == 2
    assert {call.client_session_id for call in execution.calls} == set(ids)
    for result in results:
        bundle = AgentObservationBundle.model_validate(result["ng_agent_observations"])
        joined = join_model_call_observations(bundle, execution.calls)
        [owned] = joined.records[0].model_calls
        assert (
            next(call for call in execution.calls if call.model_call_id == owned.model_call_id).client_session_id
            == bundle.records[0].invocation_id
        )


@pytest.mark.asyncio
async def test_disabled_observability_preserves_output_and_headers(execution):
    execution.client.global_config_dict = {"observability_enabled": False}
    result = await execution.run()
    assert "ng_agent_observations" not in result
    assert result["terminus2_completed"] is True
    assert result["response"]["output"][-1]["content"][0]["text"] == "done"
    assert execution.calls[0].client_session_id is None


@pytest.mark.asyncio
async def test_missing_rollout_identity_does_not_emit_observations(execution):
    result = await execution.run(None)
    assert "ng_agent_observations" not in result
    assert execution.calls[0].client_session_id is None
