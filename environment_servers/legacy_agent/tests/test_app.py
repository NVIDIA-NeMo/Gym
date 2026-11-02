# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import runpy
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from omegaconf import OmegaConf
from pydantic import ConfigDict

from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.client import ServerClient
from nemo_gym.config import AgentServerRef
from nemo_gym.server_utils import BaseServerConfig, SimpleServer


class _Upstream:
    ok = True

    def __init__(self, body: bytes, *, status: int = 200, headers: CIMultiDict | None = None) -> None:
        self.body = body
        self.status = status
        self.headers = headers or CIMultiDict()

    async def read(self) -> bytes:
        return self.body


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    responses: list[_Upstream]

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Upstream:
        self.calls.append((server_name, url_path, kwargs))
        return self.responses.pop(0)


def _config(**limits: float) -> LegacyAgentEnvironmentServerConfig:
    return LegacyAgentEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        **limits,
    )


def _app(*responses: _Upstream) -> tuple[TestClient, _Client]:
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        calls=[],
        responses=list(responses),
    )
    server = LegacyAgentEnvironmentServer(config=_config(), server_client=client)
    return TestClient(server.setup_webserver()), client


def test_run_relays_body_headers_and_cookies_to_the_agent() -> None:
    body = orjson.dumps({"responses_create_params": {"input": "task"}, "_ng_task_index": 3})
    app, client = _app(_Upstream(b'{"reward": 1.0}'))

    response = app.post(
        "/run",
        content=body,
        headers={"content-type": "application/json", "x-trace": "abc", "cookie": "session=agent-cookie"},
    )

    assert response.status_code == 200
    assert response.content == b'{"reward": 1.0}'
    [(server, path, kwargs)] = client.calls
    assert (server, path) == ("agent", "/run")
    assert kwargs["data"] == body
    assert kwargs["cookies"] == {"session": "agent-cookie"}
    relayed = {name.lower() for name in kwargs["headers"]}
    assert {"content-type", "x-trace"} <= relayed
    assert relayed.isdisjoint({"cookie", "host", "content-length"})


def test_run_returns_upstream_status_and_repeated_set_cookie_headers() -> None:
    headers = CIMultiDict([("set-cookie", "a=1"), ("set-cookie", "b=2"), ("transfer-encoding", "chunked")])
    app, _ = _app(_Upstream(b'{"detail": "boom"}', status=500, headers=headers))

    response = app.post("/run", json={})

    assert response.status_code == 500
    assert response.content == b'{"detail": "boom"}'
    assert response.headers.get_list("set-cookie") == ["a=1", "b=2"]
    assert "transfer-encoding" not in response.headers
    assert response.headers["content-length"] == str(len(response.content))


def test_aggregate_metrics_is_forwarded_to_the_agent() -> None:
    app, client = _app(_Upstream(orjson.dumps({"agent_metrics": {"mean/reward": 0.5}})))

    response = app.post("/aggregate_metrics", json={"verify_responses": [{"_ng_task_index": 0, "reward": 0.5}]})

    assert response.status_code == 200
    assert response.json()["agent_metrics"] == {"mean/reward": 0.5}
    assert [(server, path) for server, path, _ in client.calls] == [("agent", "/aggregate_metrics")]


def test_setting_a_limit_the_relay_does_not_enforce_warns() -> None:
    with pytest.warns(UserWarning, match="queue_timeout_seconds"):
        _config(queue_timeout_seconds=60)


def test_setting_the_episode_deadline_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _config(default_episode_timeout_seconds=60)


class _SlowClient(_Client):
    """Agent `/run` that answers after `delay` seconds and records whether the relay cancelled it."""

    delay: float
    cancelled: list[bool]

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Upstream:
        self.calls.append((server_name, url_path, kwargs))
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled.append(True)
            raise
        return _Upstream(b'{"reward": 1.0}')


def _slow_server(delay: float, **limits: float) -> tuple[LegacyAgentEnvironmentServer, _SlowClient]:
    client = _SlowClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        calls=[],
        responses=[],
        delay=delay,
        cancelled=[],
    )
    return LegacyAgentEnvironmentServer(config=_config(**limits), server_client=client), client


def test_an_episode_past_its_deadline_retains_ambiguous_agent_completion() -> None:
    server, client = _slow_server(delay=30, default_episode_timeout_seconds=0.2)

    started = time.monotonic()
    response = TestClient(server.setup_webserver()).post("/run", json={"_ng_task_index": 0})
    elapsed = time.monotonic() - started

    assert 0.2 <= elapsed < 5
    assert response.status_code == 200
    # The shape and classification single_agent_turn_legacy gives an episode its deadline ended.
    payload = response.json()
    selected = payload.pop("_ng_failure_id")
    events = payload.pop("_ng_failure_events")
    assert events[selected]["operation"] == "episode.deadline"
    assert events[selected]["boundary"] == "environment"
    assert events[selected]["foreign_error"]["exception_type"].endswith("TimeoutError")
    assert payload == {
        "_ng_failure_class": "environment_server_failed",
        "_ng_failure_terminal": False,
        "_ng_failure_message": "Episode timed out",
        "_ng_failure_stage": None,
        "agent_ref": {"name": "agent"},
    }
    assert len(client.calls) == 1
    assert client.cancelled == [True]


def test_an_episode_within_its_deadline_is_relayed_unchanged() -> None:
    server, client = _slow_server(delay=0.1, default_episode_timeout_seconds=10)

    response = TestClient(server.setup_webserver()).post("/run", json={})

    assert response.content == b'{"reward": 1.0}'
    assert client.cancelled == []


def test_without_a_deadline_a_slow_episode_is_relayed() -> None:
    server, client = _slow_server(delay=0.3)

    response = TestClient(server.setup_webserver()).post("/run", json={})

    assert server.config.default_episode_timeout_seconds is None
    assert response.content == b'{"reward": 1.0}'
    assert client.cancelled == []


def test_a_timeout_the_agent_call_raises_itself_is_not_an_episode_timeout() -> None:
    class _TimingOutClient(_Client):
        async def post(self, server_name: str, url_path: str, **kwargs) -> _Upstream:
            raise TimeoutError("agent connection timed out")

    client = _TimingOutClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        calls=[],
        responses=[],
    )
    server = LegacyAgentEnvironmentServer(config=_config(default_episode_timeout_seconds=60), server_client=client)

    with pytest.raises(TimeoutError, match="agent connection timed out"):
        TestClient(server.setup_webserver()).post("/run", json={})


async def test_cancelling_the_relay_cancels_the_agent_call_and_is_not_a_timeout() -> None:
    server, client = _slow_server(delay=30, default_episode_timeout_seconds=60)
    request = SimpleNamespace(body=AsyncMock(return_value=b"{}"), headers={}, cookies={})

    relay = asyncio.create_task(server.run_legacy(request))
    while not client.calls:
        await asyncio.sleep(0)
    relay.cancel()

    with pytest.raises(asyncio.CancelledError):
        await relay
    assert client.cancelled == [True]


def test_legacy_module_exports_app_for_multi_worker_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_app = object()
    monkeypatch.setattr("nemo_gym.utils.server_process.is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))

    namespace = runpy.run_path(str(Path(__file__).parents[1] / "app.py"), run_name="legacy_agent.worker_test")

    assert namespace["app"] is worker_app


async def test_deadline_includes_response_body_read() -> None:
    class SlowBody(_Upstream):
        async def read(self) -> bytes:
            await asyncio.sleep(1)
            return self.body

    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        calls=[],
        responses=[SlowBody(b'{"reward": 1.0}')],
    )
    server = LegacyAgentEnvironmentServer(config=_config(default_episode_timeout_seconds=0.01), server_client=client)
    request = SimpleNamespace(body=AsyncMock(return_value=b"{}"), headers={}, cookies={})
    response = await server.run_legacy(request)
    payload = orjson.loads(response.body)
    event = payload["_ng_failure_events"][payload["_ng_failure_id"]]
    assert event["operation"] == "episode.deadline"
    assert "reward" not in payload
    assert len(client.calls) == 1
    assert server.config.default_episode_timeout_seconds == 0.01
