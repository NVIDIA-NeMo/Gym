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
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.models.base_responses_api_model import BaseResponsesAPIModelConfig, SimpleResponsesAPIModel
from nemo_gym.models.rollout_budget import BUDGET_STOP_REASON, RolloutGenerationBudget, RolloutGenerationBudgetConfig
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
            budget = RolloutGenerationBudget(
                RolloutGenerationBudgetConfig(max_calls=10, state_dir=tmp_path, agent_servers=["first", "owner"])
            )
            for _ in range(10):
                assert budget.reserve("rollout-a2")
            with rollout_context("rollout-a2"):
                if first_status == 404:
                    with pytest.raises(HTTPException) as error:
                        await budget.admit(SimpleNamespace(is_disconnected=AsyncMock(return_value=True)), client)
                    assert error.value.status_code == 409
                    assert seen == ["first", "owner"]
                    owner._sandbox_session.expire.assert_awaited_once()
                else:
                    with pytest.raises(ClientResponseError) as error:
                        await budget.admit(SimpleNamespace(), client)
                    assert error.value.status == first_status
                    assert b"provider cannot deliver expiry" in error.value.response_content
                    assert seen == ["first"]
                    owner._sandbox_session.expire.assert_not_awaited()
            assert (tmp_path / "rollout-a2.generation-count").read_text() == "10"


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
            rollout_generation_budget={"max_calls": 10, "state_dir": tmp_path / "budget", "agent_servers": ["agent"]},
        ),
        server_client=client,
    )
    app = model.setup_webserver()
    monkeypatch.setattr(Request, "is_disconnected", AsyncMock(return_value=True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fixture") as http:
        endpoint = "/ng-rollout/rollout-a2/training-token-capture/v1/responses"
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
