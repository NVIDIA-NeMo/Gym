# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from omegaconf import DictConfig

from nemo_gym.base_resources_server import BaseResourcesServerConfig
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseOutputMessage, NeMoGymResponseOutputText
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.gymnasium import EnvResetRequest, EnvStepRequest, GymnasiumServer, extract_text


def _make_response(*parts: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id=f"msg_{i}",
                content=[NeMoGymResponseOutputText(annotations=[], text=p, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
            for i, p in enumerate(parts)
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


class _FakeRequest:
    def __init__(self, session_id="sid-1"):
        self.session = {SESSION_ID_KEY: session_id}


class _TerminatingEnv(GymnasiumServer):
    async def step(self, action, metadata, session_id=None):
        return None, 1.0, True, False, {}


class _OngoingEnv(GymnasiumServer):
    async def step(self, action, metadata, session_id=None):
        return "keep going", 0.0, False, False, {}


class _TruncatingEnv(GymnasiumServer):
    async def step(self, action, metadata, session_id=None):
        return None, 0.0, False, True, {}


_close_log: list = []


class _CustomCloseEnv(GymnasiumServer):
    async def step(self, action, metadata, session_id=None):
        return None, 0.0, True, False, {}

    async def close_session(self, session_id):
        _close_log.append(session_id)
        await super().close_session(session_id)


def _make_env(cls):
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="")
    return cls(config=config, server_client=MagicMock(spec=ServerClient))


class TestGymnasiumServer:
    def test_routes_registered(self):
        env = _make_env(_TerminatingEnv)
        routes = {r.path for r in env.setup_webserver().routes}
        assert {"/reset", "/step", "/aggregate_metrics"}.issubset(routes)

    def test_rollout_prefixed_reset_and_step(self):
        client = TestClient(_make_env(_TerminatingEnv).setup_webserver())
        reset = client.post(
            "/ng-rollout/4-2/reset",
            json={"responses_create_params": {"input": []}},
        )
        assert reset.status_code == 200

        step = client.post(
            "/ng-rollout/4-2/step",
            json={
                "responses_create_params": {"input": []},
                "response": _make_response("x").model_dump(mode="json"),
            },
        )
        assert step.status_code == 200
        assert step.json()["terminated"] is True

    def test_verify_raises(self):
        env = _make_env(_TerminatingEnv)
        with pytest.raises(NotImplementedError):
            import asyncio

            asyncio.run(env.verify(SimpleNamespace()))

    @pytest.mark.asyncio
    async def test_reset_default_returns_empty(self):
        env = _make_env(_TerminatingEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvResetRequest(responses_create_params={"input": []})
        resp = await env._reset_endpoint(body, _FakeRequest())
        assert resp.observation is None
        assert resp.info == {}

    @pytest.mark.asyncio
    async def test_step_pops_on_terminated(self):
        env = _make_env(_TerminatingEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvStepRequest(responses_create_params={"input": []}, response=_make_response("x"))
        resp = await env._step_endpoint(body, _FakeRequest("sid-1"))
        assert resp.terminated is True
        assert "sid-1" not in env.session_state

    @pytest.mark.asyncio
    async def test_step_pops_on_truncated(self):
        env = _make_env(_TruncatingEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvStepRequest(responses_create_params={"input": []}, response=_make_response("x"))
        resp = await env._step_endpoint(body, _FakeRequest("sid-1"))
        assert resp.truncated is True
        assert "sid-1" not in env.session_state

    @pytest.mark.asyncio
    async def test_step_keeps_state_when_ongoing(self):
        env = _make_env(_OngoingEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvStepRequest(responses_create_params={"input": []}, response=_make_response("x"))
        resp = await env._step_endpoint(body, _FakeRequest("sid-1"))
        assert resp.terminated is False
        assert resp.truncated is False
        assert "sid-1" in env.session_state

    @pytest.mark.asyncio
    async def test_close_session_override_invoked(self):
        _close_log.clear()
        env = _make_env(_CustomCloseEnv)
        env.session_state["sid-1"] = {"x": 1}
        body = EnvStepRequest(responses_create_params={"input": []}, response=_make_response("x"))
        await env._step_endpoint(body, _FakeRequest("sid-1"))
        assert _close_log == ["sid-1"]
        assert "sid-1" not in env.session_state


class TestExtractText:
    def test_concats_output_text(self):
        r = _make_response("hello ", "world")
        assert extract_text(r) == "hello world"

    def test_empty_output(self):
        r = NeMoGymResponse(
            id="r",
            created_at=0.0,
            model="m",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        assert extract_text(r) == ""


class TestSharedSetup:
    def test_route_table_is_the_gymnasium_protocol_plus_the_shared_routes(self):
        routes = {r.path for r in _make_env(_TerminatingEnv).setup_webserver().routes}
        assert routes - {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"} == {
            "/reset",
            "/step",
            "/aggregate_metrics",
            "/reverify_mode",
        }

    def test_reverify_mode_is_served(self):
        client = TestClient(_make_env(_TerminatingEnv).setup_webserver())
        response = client.get("/reverify_mode")
        assert response.status_code == 200
        assert response.json() == "unknown"


AUTH = {"authorization": "Bearer t"}
RESET = {"responses_create_params": {"input": []}}


def _step_body(text: str = "x") -> dict:
    return {"responses_create_params": {"input": []}, "response": _make_response(text).model_dump(mode="json")}


def _control(checkpoint_id: str = "c1", *, timeout: float = 5, **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


class _CountingEnv(GymnasiumServer):
    """Counts steps per session and terminates on "stop"; its state is plain JSON."""

    ray_enabled = False

    async def reset(self, metadata, session_id=None):
        self.session_state[session_id] = {"steps": 0}
        return "start", {}

    async def step(self, action, metadata, session_id=None):
        if session_id not in self.session_state:
            raise HTTPException(status_code=404, detail="no episode for this session; call /reset first")
        state = self.session_state[session_id]
        state["steps"] += 1
        terminated = extract_text(action) == "stop"
        return None, float(state["steps"]), terminated, False, {}


class _ExportedCountingEnv(_CountingEnv):
    checkpoint_mode: ClassVar[str] = "exported"

    def restore_env_state(self, state):
        if not isinstance(state.get("steps"), int):
            raise ValueError("invalid step count")
        return state


def _checkpointed(cls) -> tuple[GymnasiumServer, httpx.AsyncClient]:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = DictConfig({"checkpoint": {"enabled": True, "control_auth_token": "t"}})
    config = BaseResourcesServerConfig(host="", port=0, entrypoint="", name="gymnasium")
    env = cls(config=config, server_client=server_client)
    app = env.setup_webserver()
    return env, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://r")


async def _commit_one_live_session(cls, tmp_path: Path, *, corrupt: bool = False) -> tuple[GymnasiumServer, dict]:
    env, client = _checkpointed(cls)
    async with client:
        await client.post("/ng-rollout/r-a1/reset", json=RESET)
        await client.post("/ng-rollout/r-a1/step", json=_step_body())
        await client.post("/ng-control/v1/checkpoint/prepare", json=_control(), headers=AUTH)
        if corrupt:
            for state in env.session_state.values():
                state["steps"] = "one"
        commit = await client.post(
            "/ng-control/v1/checkpoint/commit", json=_control(checkpoint_dir=str(tmp_path)), headers=AUTH
        )
        assert commit.json()["episode_ids"] == ["r-a1"]
        return env, dict(client.cookies)


async def _restore(client: httpx.AsyncClient, tmp_path: Path) -> httpx.Response:
    scope = [{"rollout_id": "r"}, {"rollout_id": "r", "attempt": 1}]
    return await client.post(
        "/ng-control/v1/checkpoint/restore",
        json=_control("r1", checkpoint_dir=str(tmp_path), episode_ids=scope),
        headers=AUTH,
    )


class TestCheckpointing:
    @pytest.mark.asyncio
    async def test_participant_is_installed_through_the_shared_setup(self):
        env, client = _checkpointed(_CountingEnv)
        async with client:
            status = await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)
        assert env._checkpoint is not None
        assert status.status_code == 200 and status.json()["mode"] == "restart_only"

    @pytest.mark.asyncio
    async def test_restart_only_session_is_a_restart_that_neither_blocks_prepare_nor_waits(self):
        _, client = _checkpointed(_CountingEnv)
        async with client:
            await client.post("/ng-rollout/r-a1/reset", json=RESET)
            await client.post("/ng-rollout/r-a1/step", json=_step_body())
            prepared = await client.post("/ng-control/v1/checkpoint/prepare", json=_control(timeout=0.1), headers=AUTH)
            # Nothing of the session is saved, so it keeps running through the checkpoint.
            stepped = await client.post("/ng-rollout/r-a1/step", json=_step_body("stop"))
            await client.post("/ng-control/v1/checkpoint/resume", json=_control(), headers=AUTH)
        report = prepared.json()["report"]
        assert prepared.json()["phase"] == "prepared"
        assert report["blockers"] == [] and report["restarts"] == ["r-a1"]
        assert stepped.status_code == 200 and stepped.json()["terminated"] is True

    @pytest.mark.asyncio
    async def test_exported_session_continues_in_a_fresh_server(self, tmp_path):
        source, cookies = await _commit_one_live_session(_ExportedCountingEnv, tmp_path)
        [session_id] = source.session_state

        restored, client = _checkpointed(_ExportedCountingEnv)
        async with client:
            client.cookies.update(cookies)
            await _restore(client, tmp_path)
            await client.post("/ng-control/v1/checkpoint/resume", json=_control("r1"), headers=AUTH)
            final = (await client.post("/ng-rollout/r-a2/step", json=_step_body("stop"))).json()
            status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()

        # The uncheckpointed episode ends with the same reward: two steps.
        assert final["terminated"] is True and final["reward"] == 2.0
        assert session_id not in restored.session_state
        assert status["report"]["counts"]["sessions"] == 0

    @pytest.mark.asyncio
    async def test_invalid_state_installs_nothing(self, tmp_path):
        await _commit_one_live_session(_ExportedCountingEnv, tmp_path, corrupt=True)

        restored, client = _checkpointed(_ExportedCountingEnv)
        async with client:
            with pytest.raises(ValueError, match="invalid step count"):
                await _restore(client, tmp_path)
            status = (await client.get("/ng-control/v1/checkpoint/status", headers=AUTH)).json()
        assert restored.session_state == {}
        assert status["phase"] == "idle" and status["report"]["counts"]["sessions"] == 0

    @pytest.mark.asyncio
    async def test_retired_session_is_closed(self):
        env, client = _checkpointed(_ExportedCountingEnv)
        async with client:
            await client.post("/ng-rollout/r-a1/reset", json=RESET)
            await client.post(
                "/ng-control/v1/checkpoint/retire",
                json=_control(episode_ids=[{"rollout_id": "r", "attempt": 1}]),
                headers=AUTH,
            )
            stale = await client.post("/ng-rollout/r-a1/step", json=_step_body())
        # The retire stopped and freed the episode;
        # a late step is refused until the controller forgets the rollout, and does not recreate it.
        assert stale.status_code == 409 and stale.json()["error"]["code"] == "stale_attempt"
        assert env.session_state == {}

    @pytest.mark.asyncio
    async def test_session_the_server_dropped_is_left_out(self, tmp_path):
        env, client = _checkpointed(_ExportedCountingEnv)
        async with client:
            await client.post("/ng-rollout/r-a1/reset", json=RESET)
            env.session_state.clear()
            await client.post("/ng-control/v1/checkpoint/prepare", json=_control(), headers=AUTH)
            commit = await client.post(
                "/ng-control/v1/checkpoint/commit", json=_control(checkpoint_dir=str(tmp_path)), headers=AUTH
            )
        assert commit.json()["manifest"]["record_count"] == 0
