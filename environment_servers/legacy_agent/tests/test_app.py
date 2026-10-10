# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import runpy
from pathlib import Path

import orjson
import pytest
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from omegaconf import OmegaConf
from pydantic import ConfigDict

import nemo_gym.server_utils
from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.config_types import AgentServerRef
from nemo_gym.server_utils import BaseServerConfig, ServerClient, SimpleServer


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
    with pytest.warns(UserWarning, match="default_episode_timeout_seconds"):
        _config(default_episode_timeout_seconds=60)


def test_legacy_module_exports_app_for_multi_worker_import(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_app = object()
    monkeypatch.setattr(nemo_gym.server_utils, "is_nemo_gym_fastapi_entrypoint", lambda _: True)
    monkeypatch.setattr(SimpleServer, "run_webserver", classmethod(lambda _: worker_app))

    namespace = runpy.run_path(str(Path(__file__).parents[1] / "app.py"), run_name="legacy_agent.worker_test")

    assert namespace["app"] is worker_app
