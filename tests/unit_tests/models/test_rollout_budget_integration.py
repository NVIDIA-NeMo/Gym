# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from NVIDIA NeMo Gym (Apache-2.0), original path: tests/unit_tests/models/test_rollout_budget_integration.py
# Source: https://gitlab-master.nvidia.com/tongc/gym/-/tree/d8c548915431e59169890d34cd8cea56ef2a905e
"""HTTP budget/capture integration with synthetic worker acknowledgements.

This exercises production request dispatch, capture and stop handlers. It does not
run a policy, fetch staged tensors, or perform a trainer optimizer update.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from aiohttp import ClientResponseError, ClientSession, web
from aiohttp.test_utils import TestServer
from fastapi import FastAPI, Request
from omegaconf import OmegaConf
from pydantic import PrivateAttr

from nemo_gym.agents.base_responses_api_agent import RolloutStopRequest
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.episode_context import rollout_context
from nemo_gym.core.generation_budget import EpisodeGenerationBudget, generation_budget_context, with_generation_budget
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.models.rollout_budget import BUDGET_STOP_REASON, RolloutGenerationBudget
from nemo_gym.models.token_capture.external_capture import VLLMWorkerCaptureHandler
from nemo_gym.models.token_capture.sink import current_capture_context
from nemo_gym.models.token_capture.staging.records import RolloutReceipt
from tests.unit_tests.agents.test_agent_sessions import _seed_request, _stoppable_agent
from tests.unit_tests.models.test_external_capture_handlers import _staged_coords


@pytest.mark.asyncio
@pytest.mark.parametrize("first_status", [404, 501, 502])
async def test_stop_checks_actual_http_status_before_identity(tmp_path, monkeypatch, first_status):
    """Real ServerClient/aiohttp replies cross ASGI routes; no client exception mock."""
    from fastapi import HTTPException

    import nemo_gym.client.http_client as transport

    owner = _stoppable_agent([True])
    owner._installed.set()
    await owner.seed_agent_session(MagicMock(session={}), _seed_request())
    missing = _stoppable_agent([])
    app = FastAPI()
    seen = []

    async def first(rollout_id: str, body: RolloutStopRequest):
        seen.append("first")
        if first_status == 404:
            return await missing.stop_rollout(rollout_id, body)
        raise HTTPException(first_status, "provider cannot deliver expiry")

    async def second(rollout_id: str, body: RolloutStopRequest):
        seen.append("owner")
        return await owner.stop_rollout(rollout_id, body)

    app.post("/first/rollouts/{rollout_id}/stop")(first)
    app.post("/owner/rollouts/{rollout_id}/stop")(second)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://asgi") as asgi:

        async def bridge(request):
            response = await asgi.post(
                request.path, content=await request.read(), headers={"content-type": "application/json"}
            )
            return web.Response(status=response.status_code, body=response.content, content_type="application/json")

        http_app = web.Application()
        http_app.router.add_post("/{path:.*}", bridge)
        async with TestServer(http_app) as http_server, ClientSession() as session:
            monkeypatch.setattr(transport, "get_global_aiohttp_client", lambda: session)
            client = ServerClient(
                head_server_config={"host": "localhost", "port": 0}, global_config_dict=OmegaConf.create({})
            )
            client._server_base_urls.update(
                {name: str(http_server.make_url(f"/{name}")) for name in ("first", "owner")}
            )
            budget = RolloutGenerationBudget(tmp_path)
            for _ in range(10):
                assert budget.reserve("rollout", 10)
            with (
                rollout_context("rollout-a2"),
                generation_budget_context(
                    EpisodeGenerationBudget(max_calls=10, rollout_id="rollout", agent_server="first")
                ),
            ):
                with pytest.raises(ClientResponseError) as error:
                    await budget.admit(SimpleNamespace(), client)
                assert error.value.status == first_status
                assert seen == ["first"]
                owner._sandbox_session.expire.assert_not_awaited()
            assert (tmp_path / "rollout.generation-count").read_text() == "10"


class CapturingModel(SimpleResponsesAPIModel):
    _calls: list[str] = PrivateAttr(default_factory=list)

    async def chat_completions(self, body):
        raise NotImplementedError

    async def responses(self, body):
        context = current_capture_context()
        call_id = context.model_call_id
        self._calls.append(call_id)
        handler = VLLMWorkerCaptureHandler()
        handler.prepare_request({"model": "fixture", "messages": [{"role": "user", "content": "go"}]})
        payload = {
            "ng_commit_coords": _staged_coords(
                capture_key=context.capture_key,
                model_call_id=call_id,
                staging_key=f"{context.capture_key}/{call_id}",
            )
        }
        handler.prepare_response(payload)
        response = NeMoGymResponse(
            id=f"response-{call_id}",
            created_at=0,
            model="fixture",
            object="response",
            status="completed",
            output=[],
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        )
        await handler.finalize_response(response.model_dump(mode="json"))
        return response


@pytest.mark.asyncio
async def test_http_eleventh_call_stops_without_poisoning_ten_call_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_GYM_TOKEN_CAPTURE_CONTROL_TOKEN", "test-only")
    # The sandbox supervisor is a fixture; HTTP request disconnect after expiry
    # is modeled separately from the production stop handler itself.
    agent = _stoppable_agent([])
    agent._installed.set()
    agent._sandbox_session.expired = False

    async def expire(*, timeout, reason):
        agent._sandbox_session.expired = True
        agent._sandbox_session.expire_reason = reason
        return True

    agent._sandbox_session.expire = AsyncMock(side_effect=expire)
    await agent.seed_agent_session(MagicMock(session={}), _seed_request())

    async def stop(*, server_name, url_path, json):
        assert server_name == "agent"
        result = await agent.stop_rollout(url_path.split("/")[2], RolloutStopRequest(**json))
        return SimpleNamespace(ok=True, json=AsyncMock(return_value=result.model_dump(mode="json")))

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {
        "token_id_capture": {
            "enabled": True,
            "external_staging": True,
            "parent_resolution": "source_history",
            "dir": str(tmp_path / "capture"),
            "rebuild_response": False,
        }
    }
    client.post = AsyncMock(side_effect=stop)
    model = CapturingModel(
        config=BaseResponsesAPIModelConfig(
            name="model",
            host="",
            port=0,
            entrypoint="",
            openai_api_key="test",
            generation_budget_state_dir=str(tmp_path / "budget"),
        ),
        server_client=client,
    )
    app = model.setup_webserver()
    monkeypatch.setattr(Request, "is_disconnected", AsyncMock(return_value=True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as http:
        endpoint = with_generation_budget(
            "/ng-rollout/rollout-a2/training-token-capture/v1/responses",
            EpisodeGenerationBudget(max_calls=10, rollout_id="rollout", agent_server="agent"),
        )
        for _ in range(10):
            response = await http.post(endpoint, json={"model": "fixture", "input": "go"})
            assert response.status_code == 200, response.text
        blocked = await http.post(endpoint, json={"model": "fixture", "input": "go"})
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == BUDGET_STOP_REASON
    assert len(model._calls) == 10
    agent._sandbox_session.expire.assert_awaited_once()
    assert agent._sandbox_session.expire_reason == BUDGET_STOP_REASON
    manifest = await model._capture_lineage.close_admissions("rollout-a2")
    assert len(manifest["records"]) == len(manifest["attempted_call_ids"]) == 10
    assert manifest["pending_call_ids"] == manifest["failures"] == []
    receipt = RolloutReceipt(
        capture_key="rollout-a2",
        reward=0.0,
        terminal_model_call_id=model._calls[-1],
        terminal_selection="declared",
        manifest=manifest["records"],
        attempted_call_ids=manifest["attempted_call_ids"],
        pending_call_ids=manifest["pending_call_ids"],
    )
    assert not receipt.capture_poisoned
    assert RolloutReceipt.model_validate_json(receipt.model_dump_json()).reward == 0.0
    stopped = agent.report_rollout_stop(
        NeMoGymResponse.model_validate(response.json()).model_copy(update={"status": "incomplete"}),
        agent._sandbox_session,
    )
    assert stopped.status == "incomplete"
    assert stopped.error is None
    assert stopped.metadata["ng_rollout_stop"] == '{"reason":"model_call_budget_exhausted"}'


@pytest.mark.asyncio
async def test_episode_budget_survives_agent_url_and_retry_but_not_uncapped_episode(tmp_path, monkeypatch):
    """Exercise the environment URL, agent middleware/URL builder and real model dispatch together."""
    from environment_servers.single_agent_turn.tests.test_app import _environment_server
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
    from nemo_gym.client.server_client import apply_rollout_prefix
    from nemo_gym.core.correlation import parse_path
    from nemo_gym.core.generation_budget import current_generation_budget
    from nemo_gym.server_runtime.request_handling.generation_budget import GenerationBudgetMiddleware
    from nemo_gym.server_runtime.request_handling.model_sampling import ModelSamplingContextMiddleware
    from nemo_gym.server_runtime.request_handling.rollout_context import RolloutContextMiddleware

    class Model(SimpleResponsesAPIModel):
        _calls: list[str] = PrivateAttr(default_factory=list)

        async def chat_completions(self, body):
            raise NotImplementedError

        async def responses(self, body):
            self._calls.append(body.input)
            if body.input == "go":
                assert (body.temperature, body.top_p, body.max_output_tokens) == (0.6, 0.95, 64)
            return NeMoGymResponse(
                id=f"response-{len(self._calls)}",
                created_at=0,
                model="fixture",
                object="response",
                status="completed",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
            )

    env, env_client = _environment_server()
    env.config.max_model_calls = 10
    adapter = SingleAgentTurnLegacyEnvironmentServer(config=env.config, server_client=env_client)
    # The same task can have many independent samples: only EpisodeId.rollout_id is the counter key.
    row = {"task_id": "same-task", "_ng_rollout_id": "sample-one", "responses_create_params": {"input": "go"}}
    row["_ng_model_sampling"] = {"temperature": 0.6, "top_p": 0.95, "max_output_tokens": 64}
    first = adapter._episode_request_from_row(row)
    retry = adapter._episode_request_from_row(row | {"_ng_attempt_index": 1})
    other_sample = adapter._episode_request_from_row(row | {"_ng_rollout_id": "sample-two"})
    assert first.task == retry.task == other_sample.task
    assert first.episode_id.rollout_id == retry.episode_id.rollout_id == "sample-one"
    assert retry.episode_id.attempt == 1
    expected_owner = env.config.agent_server.name
    stops = []

    async def stop(*, server_name, url_path, json):
        stops.append((server_name, url_path))
        return SimpleNamespace(ok=True, json=AsyncMock(return_value={"rollout_id": url_path.split("/")[2]}))

    model_client = MagicMock(spec=ServerClient)
    model_client.global_config_dict = {}
    model_client.post = AsyncMock(side_effect=stop)
    model = Model(
        config=BaseResponsesAPIModelConfig(
            name="model",
            host="",
            port=0,
            entrypoint="",
            openai_api_key="test",
            generation_budget_state_dir=str(tmp_path),
        ),
        server_client=model_client,
    )
    model_app = model.setup_webserver()
    monkeypatch.setattr(Request, "is_disconnected", AsyncMock(return_value=True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=model_app), base_url="http://model") as model_http:
        agent_app = FastAPI()
        agent_app.add_middleware(RolloutContextMiddleware)
        agent_app.add_middleware(ModelSamplingContextMiddleware)
        agent_app.add_middleware(GenerationBudgetMiddleware)

        @agent_app.post("/v1/responses")
        async def relay(request: Request):
            from fastapi.responses import JSONResponse

            from nemo_gym.core.episode_context import current_rollout_id

            budget = current_generation_budget()
            assert budget is None or budget.agent_server == expected_owner
            url = apply_rollout_prefix("http://model", current_rollout_id(), trace_context=True) + "/v1/responses"
            response = await model_http.post(url, json=await request.json())
            return JSONResponse(response.json(), status_code=response.status_code)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=agent_app), base_url="http://agent"
        ) as agent_http:
            for episode, count in [(first, 6), (retry, 4)]:
                for _ in range(count):
                    result = await agent_http.post(env._agent_responses_path(episode), json={"input": "go"})
                    assert result.status_code == 200, result.text
            denied = await agent_http.post(env._agent_responses_path(retry), json={"input": "go"})
            assert denied.status_code == 409
            assert stops == [(expected_owner, "/rollouts/sample-one-a1/stop")]
            for _ in range(11):
                result = await agent_http.post("/ng-rollout/uncapped/v1/responses", json={"input": "uncapped"})
                assert result.status_code == 200
            result = await agent_http.post(env._agent_responses_path(other_sample), json={"input": "other"})
            assert result.status_code == 200
        assert len(model._calls) == 22
        assert (tmp_path / "sample-one.generation-count").read_text() == "10"
        assert not (tmp_path / "sample-one-a1.generation-count").exists()
        assert not (tmp_path / "uncapped.generation-count").exists()
        # Seeding receives the same budget; retries never create a fresh allowance.
        seed_path = env._budget_path(retry, "/v1/agent_sessions")
        encoded = parse_path(seed_path)[0].generation_budget
        assert EpisodeGenerationBudget.from_segment(encoded).rollout_id == "sample-one"


@pytest.mark.asyncio
@pytest.mark.parametrize("segment", ["!", "e30", "x" * 1025])
async def test_malformed_budget_never_reaches_backend(tmp_path, segment):
    from nemo_gym.server_runtime.request_handling.generation_budget import GenerationBudgetMiddleware

    app = FastAPI()
    seen = []

    @app.post("/v1/responses")
    async def backend():
        seen.append(True)
        return {}

    app.add_middleware(GenerationBudgetMiddleware)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as client:
        response = await client.post(f"/ng-rollout/sample/ng-generation-budget/{segment}/v1/responses", json={})
    assert response.status_code == 422
    assert seen == []
