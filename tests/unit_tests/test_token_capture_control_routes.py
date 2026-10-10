# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bearer-protected control routes a training framework calls on a capture ledger."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.token_id_capture import control_routes
from nemo_gym.token_id_capture.control_routes import (
    CONTROL_ROUTE_PREFIX,
    RolloutControlClient,
    install_rollout_control_routes,
)
from nemo_gym.token_id_capture.lineage import FileLineageStore
from nemo_gym.token_id_capture.protocols import RolloutRetiredError


TOKEN = "secret"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
RETIRE = f"{CONTROL_ROUTE_PREFIX}/rollouts/retire"
DELETE = f"{CONTROL_ROUTE_PREFIX}/rollouts/delete"


@pytest.fixture
def ledger(tmp_path):
    return FileLineageStore(tmp_path)


@pytest.fixture
def client(ledger):
    app = FastAPI()
    install_rollout_control_routes(app, ledger, auth_token=TOKEN)
    return TestClient(app)


async def _record_failure(ledger, rollout_id: str) -> None:
    await ledger.record_failure(rollout_id, "c1", "worker_capture_failed")


@pytest.mark.asyncio
async def test_retire_route_removes_ledgers_and_fences_them(client, ledger, tmp_path):
    await _record_failure(ledger, "r1")

    response = client.post(RETIRE, json={"rollout_ids": ["r1", "r2"]}, headers=AUTH)

    assert response.status_code == 200
    assert response.json() == {"removed": ["r1"], "absent": ["r2"]}
    assert not (tmp_path / "r1.lineage.jsonl").exists()
    await _record_failure(ledger, "r1")
    with pytest.raises(RolloutRetiredError):
        await ledger.has_rows("r1")


@pytest.mark.asyncio
async def test_delete_route_removes_the_fence(client, ledger):
    await _record_failure(ledger, "r1")
    client.post(RETIRE, json={"rollout_ids": ["r1"]}, headers=AUTH)

    response = client.post(DELETE, json={"rollout_ids": ["r1"]}, headers=AUTH)

    assert response.status_code == 200
    await _record_failure(ledger, "r1")
    assert await ledger.has_rows("r1")


@pytest.mark.parametrize("route", [RETIRE, DELETE])
def test_ledger_routes_require_the_bearer_token(client, route):
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        assert client.post(route, json={"rollout_ids": ["r1"]}, headers=headers).status_code == 401


@pytest.mark.parametrize("route", [RETIRE, DELETE])
@pytest.mark.parametrize(
    "body",
    [
        {"rollout_ids": []},
        {"rollout_ids": [f"r{index}" for index in range(control_routes.MAX_LEDGER_BATCH + 1)]},
        {"rollout_ids": ["r1"], "unknown": True},
    ],
)
def test_ledger_routes_reject_malformed_batches(client, route, body):
    assert client.post(route, json=body, headers=AUTH).status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("route", [RETIRE, DELETE])
async def test_ledger_routes_reject_an_invalid_rollout_id_without_changing_anything(client, ledger, route):
    await _record_failure(ledger, "r1")
    response = client.post(route, json={"rollout_ids": ["r1", "a/b"]}, headers=AUTH)
    assert response.status_code == 400
    assert await ledger.has_rows("r1")


class _Response:
    def __init__(self, status: int, payload: dict) -> None:
        self.status = status
        self._payload = payload

    async def json(self) -> dict:
        return self._payload

    async def text(self) -> str:
        return str(self._payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retire", "delete"])
async def test_client_splits_large_batches(monkeypatch, action):
    monkeypatch.setattr(control_routes, "MAX_LEDGER_BATCH", 2)
    calls = []

    async def fake_request(method, path, **kwargs):
        body = kwargs["json"]
        calls.append((method, path, body))
        return _Response(200, {"removed": body["rollout_ids"][:1], "absent": body["rollout_ids"][1:]})

    client = RolloutControlClient("http://model", auth_token=TOKEN, request_timeout_s=1.0)
    monkeypatch.setattr(client, "_request", fake_request)

    result = await getattr(client, action)(["r1", "r2", "r3"])

    assert calls == [
        ("POST", f"/rollouts/{action}", {"rollout_ids": ["r1", "r2"]}),
        ("POST", f"/rollouts/{action}", {"rollout_ids": ["r3"]}),
    ]
    assert result.removed == ["r1", "r3"]
    assert result.absent == ["r2"]


@pytest.mark.asyncio
async def test_client_raises_on_a_failed_request(monkeypatch):
    async def fake_request(method, path, **kwargs):
        return _Response(400, {"detail": "Invalid rollout id"})

    client = RolloutControlClient("http://model", auth_token=TOKEN, request_timeout_s=1.0)
    monkeypatch.setattr(client, "_request", fake_request)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        await client.retire(["r1"])


def _client_over(client_app, monkeypatch) -> RolloutControlClient:
    """A control client whose requests reach the real routes and ledger."""

    async def request(method, path, **kwargs):
        response = client_app.request(method, f"{CONTROL_ROUTE_PREFIX}{path}", json=kwargs.get("json"), headers=AUTH)
        return _Response(response.status_code, response.json())

    control = RolloutControlClient("http://model", auth_token=TOKEN, request_timeout_s=1.0)
    monkeypatch.setattr(control, "_request", request)
    return control


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retire", "delete"])
async def test_client_rejects_a_bare_rollout_id_before_sending_anything(client, ledger, monkeypatch, action):
    await _record_failure(ledger, "r")
    control = _client_over(client, monkeypatch)

    # A string is a sequence of one-character IDs: "rollout-abc" would remove rollouts "r", "o", "l", ...
    with pytest.raises(TypeError, match="sequence of rollout ids"):
        await getattr(control, action)("rollout-abc")
    assert await ledger.has_rows("r")


@pytest.mark.asyncio
async def test_client_validates_every_batch_before_changing_anything(client, ledger, monkeypatch):
    monkeypatch.setattr(control_routes, "MAX_LEDGER_BATCH", 2)
    await _record_failure(ledger, "r1")
    control = _client_over(client, monkeypatch)

    with pytest.raises(ValueError, match="Invalid rollout id"):
        await control.retire(["r1", "r2", "a/b"])
    assert await ledger.has_rows("r1")


@pytest.mark.asyncio
async def test_client_reports_an_id_repeated_across_batches_once(client, ledger, monkeypatch):
    monkeypatch.setattr(control_routes, "MAX_LEDGER_BATCH", 2)
    await _record_failure(ledger, "r1")
    control = _client_over(client, monkeypatch)

    result = await control.retire(["r1", "r2", "r1"])

    assert result.removed == ["r1"]
    assert result.absent == ["r2"]


@pytest.mark.asyncio
async def test_manifest_route_reports_a_retired_rollout_as_gone(client, ledger, monkeypatch):
    await _record_failure(ledger, "r1")
    client.post(RETIRE, json={"rollout_ids": ["r1"]}, headers=AUTH)

    response = client.get(f"{CONTROL_ROUTE_PREFIX}/rollouts/r1/manifest", headers=AUTH)

    assert response.status_code == 410


@pytest.mark.asyncio
async def test_client_reports_a_retired_manifest_with_the_typed_error(client, ledger, monkeypatch):
    await _record_failure(ledger, "r1")
    client.post(RETIRE, json={"rollout_ids": ["r1"]}, headers=AUTH)

    with pytest.raises(RolloutRetiredError):
        await _client_over(client, monkeypatch).manifest("r1")


@pytest.mark.asyncio
async def test_client_validates_a_manifest_rollout_id_before_sending(monkeypatch):
    control = RolloutControlClient("http://model", auth_token=TOKEN, request_timeout_s=1.0)
    sent = []

    async def request(method, path, **kwargs):
        sent.append(path)
        return _Response(200, {"rollout_id": "r1", "records": [], "failures": []})

    monkeypatch.setattr(control, "_request", request)

    # Dot segments are normalized in the URL, so "../rollouts/r1" would fetch rollout r1's manifest.
    with pytest.raises(ValueError, match="Invalid rollout id"):
        await control.manifest("../rollouts/r1")
    assert sent == []


@pytest.mark.asyncio
async def test_client_rejects_a_manifest_for_a_different_rollout(monkeypatch):
    control = RolloutControlClient("http://model", auth_token=TOKEN, request_timeout_s=1.0)

    async def request(method, path, **kwargs):
        return _Response(200, {"rollout_id": "r2", "records": [], "failures": []})

    monkeypatch.setattr(control, "_request", request)

    with pytest.raises(ValueError, match="r2"):
        await control.manifest("r1")
