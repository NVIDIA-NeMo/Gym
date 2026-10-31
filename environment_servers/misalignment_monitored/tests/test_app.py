# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ConfigDict

import environment_servers.misalignment_monitored.app as monitored
from environment_servers.misalignment_monitored.app import (
    MisalignmentMonitoredEnvironmentServer,
    MisalignmentMonitoredEnvironmentServerConfig,
)
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.server_utils import BaseServerConfig, ServerClient


class _Upstream:
    def __init__(self, body: bytes, *, status: int = 200) -> None:
        self.body, self.status, self.ok, self.headers = body, status, status < 400, {}

    async def read(self) -> bytes:
        return self.body


class _Client(ServerClient):
    """Answers each path from its own queue and records every call in order."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    replies: dict[str, list]

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Upstream:
        return self._answer(server_name, url_path, kwargs)

    async def get(self, server_name: str, url_path: str, **kwargs) -> _Upstream:
        return self._answer(server_name, url_path, kwargs)

    def _answer(self, server_name: str, url_path: str, kwargs: dict) -> _Upstream:
        self.calls.append((server_name, url_path, kwargs))
        reply = self.replies[url_path].pop(0) if len(self.replies[url_path]) > 1 else self.replies[url_path][0]
        if isinstance(reply, Exception):
            raise reply
        return reply


def _app(**replies) -> tuple[TestClient, _Client, MisalignmentMonitoredEnvironmentServer]:
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        calls=[],
        replies={"/status": [_Upstream(b'{"pause": false}')], "/monitor": [_Upstream(b'{"accepted": true}')]}
        | {f"/{path}": value for path, value in replies.items()},
    )
    config = MisalignmentMonitoredEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=8002,
        entrypoint="app.py",
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        monitor_server=ResourcesServerRef(type="resources_servers", name="monitor"),
    )
    server = MisalignmentMonitoredEnvironmentServer(config=config, server_client=client)
    return TestClient(server.setup_webserver()), client, server


RUN = {
    "responses_create_params": {"input": "task"},
    "_ng_task_index": 3,
    "_ng_rollout_index": 1,
    "agent_ref": {"name": "agent"},
}
RESULT = {"response": {"output": []}, "reward": 1.0}


def test_a_finished_rollout_is_relayed_unchanged_and_posted_to_the_monitor() -> None:
    app, client, server = _app(run=[_Upstream(orjson.dumps(RESULT))])
    with app:
        response = app.post("/run", json=RUN)
        assert response.status_code == 200 and response.json() == RESULT
    # Leaving the client shuts the server down, which waits for posts in flight.
    [monitor_call] = [call for call in client.calls if call[1] == "/monitor"]
    assert monitor_call[0] == "monitor"
    assert orjson.loads(monitor_call[2]["data"]) == RESULT | {
        "_ng_task_index": 3,
        "_ng_rollout_index": 1,
        "agent_ref": {"name": "agent"},
        "rollout_id": "3-1",
    }
    assert server._counts == {"accepted": 1}


def test_a_failed_rollout_is_not_posted() -> None:
    app, client, _ = _app(run=[_Upstream(b'{"detail": "boom"}', status=500)])
    with app:
        assert app.post("/run", json=RUN).status_code == 500
    assert [path for _, path, _ in client.calls if path == "/monitor"] == []


def test_a_monitor_outage_never_touches_the_rollout() -> None:
    app, _, server = _app(run=[_Upstream(orjson.dumps(RESULT))], monitor=[ConnectionError("monitor down")])
    with app:
        assert app.post("/run", json=RUN).json() == RESULT
    assert server._counts == {"failed": 1}


def test_new_rollouts_wait_while_the_monitor_asks_for_a_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monitored, "STATUS_POLL_INTERVAL_S", 0.01)
    paused = _Upstream(b'{"pause": true, "open_p0": [{"alert_id": "a1"}]}')
    app, client, _ = _app(
        run=[_Upstream(orjson.dumps(RESULT))], status=[paused, paused, _Upstream(b'{"pause": false}')]
    )
    with app:
        assert app.post("/run", json=RUN).status_code == 200
    paths = [path for _, path, _ in client.calls]
    assert paths[:4] == ["/status", "/status", "/status", "/run"]


def test_a_monitor_that_cannot_answer_does_not_stall_the_run() -> None:
    app, client, _ = _app(run=[_Upstream(orjson.dumps(RESULT))], status=[ConnectionError("monitor down")])
    with app:
        assert app.post("/run", json=RUN).status_code == 200


METRICS = {"group_level_metrics": [], "agent_metrics": {"mean/reward": 1.0}, "key_metrics": {}}


def test_the_end_of_a_run_lets_the_monitor_finish_before_metrics() -> None:
    app, client, _ = _app(
        drain=[_Upstream(b'{"counts": {}, "backlog": 0}')], aggregate_metrics=[_Upstream(orjson.dumps(METRICS))]
    )
    with app:
        response = app.post("/aggregate_metrics", json={"verify_responses": []})
        assert response.status_code == 200 and response.json()["agent_metrics"] == {"mean/reward": 1.0}
    calls = [(server, path) for server, path, _ in client.calls if path in ("/drain", "/aggregate_metrics")]
    assert calls == [("monitor", "/drain"), ("agent", "/aggregate_metrics")]


def test_a_monitor_that_cannot_drain_does_not_block_metrics() -> None:
    app, _, _ = _app(drain=[ConnectionError("monitor down")], aggregate_metrics=[_Upstream(orjson.dumps(METRICS))])
    with app:
        assert app.post("/aggregate_metrics", json={"verify_responses": []}).status_code == 200
