# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenSandbox snapshots: start through the SDK, then wait, list and delete over the management API."""

import logging
import time
from types import SimpleNamespace
from typing import Any

import pytest

from nemo_gym.sandbox.providers.base import SandboxHandle, SupportsSandboxSnapshot


pytestmark = pytest.mark.sandbox

pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")

from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


class _ManagementResponse:
    def __init__(self, status: int, payload: Any = None) -> None:
        self.status = status
        self._payload = {} if payload is None else payload

    async def __aenter__(self) -> "_ManagementResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def json(self, content_type: object = None) -> Any:
        return self._payload


class _TextResponse(_ManagementResponse):
    async def json(self, content_type: object = None) -> Any:
        raise ValueError("not JSON")


def _management(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[dict[str, Any]]:
    """Replace Gym's shared client with a scripted management API; returns the calls it received.

    The last response repeats once the others are used up. An exception in the list is raised instead.
    """
    import nemo_gym.server_utils as server_utils

    calls: list[dict[str, Any]] = []

    async def fake_request(method: str, url: str, **kwargs: Any) -> _ManagementResponse:
        calls.append({"method": method, "url": url, **kwargs})
        response = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(server_utils, "request", fake_request)
    return calls


def _provider(**operations: Any) -> Any:
    operations.setdefault("retry_delay_s", 0.0)
    return opensandbox_provider.OpenSandboxProvider(
        connection={"domain": "https://cell.example", "api_key": "k", "tls_verify": False},
        create={"connect_poll_s": 0.01},
        operations=operations,
    )


class _SnapshottingSandbox:
    def __init__(self) -> None:
        self.names: list[str | None] = []

    async def create_snapshot(self, name: str | None = None) -> Any:
        self.names.append(name)
        return SimpleNamespace(id="snap-1", status=SimpleNamespace(state="Creating"))


def test_the_provider_has_the_snapshot_capability() -> None:
    assert isinstance(_provider(), SupportsSandboxSnapshot)


async def test_snapshot_starts_through_the_sdk_and_returns_the_id_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(200)])
    raw = _SnapshottingSandbox()

    snapshot_id = await _provider().snapshot(SandboxHandle("sb-1", "opensandbox", raw), name="ng-s1")

    assert snapshot_id == "snap-1" and raw.names == ["ng-s1"]
    assert calls == [], "starting a snapshot does not wait for it"


@pytest.mark.parametrize("tls_verify", [False, True])
async def test_wait_snapshot_ready_polls_the_management_api_until_ready(
    monkeypatch: pytest.MonkeyPatch, tls_verify: bool
) -> None:
    calls = _management(
        monkeypatch,
        [
            _ManagementResponse(200, {"id": "snap-1", "status": {"state": "Creating"}}),
            _ManagementResponse(200, {"id": "snap-1", "status": {"state": "Ready"}}),
        ],
    )
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"domain": "cell.example", "api_key": "k", "tls_verify": tls_verify},
        create={"connect_poll_s": 0.01},
    )

    await provider.wait_snapshot_ready("snap-1")

    assert [call["method"] for call in calls] == ["GET", "GET"], "polled until Ready"
    # Without a configured protocol the scheme is the SDK's default.
    assert {call["url"] for call in calls} == {"http://cell.example/v1/snapshots/snap-1"}
    assert calls[0]["headers"] == {"OPEN-SANDBOX-API-KEY": "k"}
    # The polls go over Gym's shared client, so they must carry connection.tls_verify themselves.
    assert all((call.get("ssl") is False) == (not tls_verify) for call in calls)


async def test_wait_snapshot_ready_reads_a_top_level_state_and_keeps_a_domain_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(200, {"id": "snap-1", "state": "READY"})])
    await _provider().wait_snapshot_ready("snap-1")
    assert calls[0]["url"] == "https://cell.example/v1/snapshots/snap-1"


async def test_a_failed_snapshot_raises_with_the_servers_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    _management(
        monkeypatch,
        [_ManagementResponse(200, {"id": "snap-1", "status": {"state": "Failed", "message": "registry push failed"}})],
    )
    with pytest.raises(RuntimeError, match="snapshot 'snap-1' failed: registry push failed"):
        await _provider().wait_snapshot_ready("snap-1")


async def test_a_snapshot_that_is_not_ready_in_time_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _management(monkeypatch, [_ManagementResponse(200, {"id": "snap-1", "status": {"state": "Creating"}})])
    with pytest.raises(TimeoutError, match="snapshot 'snap-1' to be ready after 0.05s"):
        await _provider(snapshot_timeout_s=0.05).wait_snapshot_ready("snap-1")


@pytest.mark.parametrize(
    "transient", [_ManagementResponse(429), _ManagementResponse(503), ConnectionResetError("rst")]
)
async def test_transient_lookup_failures_are_waited_out(monkeypatch: pytest.MonkeyPatch, transient: Any) -> None:
    calls = _management(
        monkeypatch,
        [transient, transient, _ManagementResponse(200, {"id": "snap-1", "status": {"state": "Ready"}})],
    )
    await _provider(snapshot_timeout_s=5).wait_snapshot_ready("snap-1")
    assert len(calls) == 3


async def test_a_snapshot_the_api_does_not_know_yet_is_waited_out_until_it_appears(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = _management(
        monkeypatch,
        [
            _ManagementResponse(404, {"code": "SNAPSHOT::NOT_FOUND"}),
            _ManagementResponse(404, {"code": "SNAPSHOT::NOT_FOUND"}),
            _ManagementResponse(200, {"id": "snap-1", "status": {"state": "Creating"}}),
            _ManagementResponse(200, {"id": "snap-1", "status": {"state": "Ready"}}),
        ],
    )

    with caplog.at_level(logging.INFO, logger=opensandbox_provider.__name__):
        await _provider(snapshot_timeout_s=5).wait_snapshot_ready("snap-1")

    assert len(calls) == 4
    [record] = [record for record in caplog.records if "appeared" in record.getMessage()]
    assert "'snap-1'" in record.getMessage() and "2 earlier lookups got 404" in record.getMessage()


async def test_a_snapshot_still_unknown_after_the_grace_fails_before_the_snapshot_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(404, {"code": "SNAPSHOT::NOT_FOUND"})])
    started = time.monotonic()

    with pytest.raises(RuntimeError, match=r"lookup failed -> HTTP 404 \(\d+ lookups over"):
        await _provider(snapshot_timeout_s=30, snapshot_lookup_grace_s=0.05).wait_snapshot_ready("snap-1")

    assert len(calls) > 1, "the 404s within the grace were retried"
    assert time.monotonic() - started < 5, "the grace, not the snapshot timeout, ended the wait"


async def test_a_404_after_the_snapshot_appeared_fails_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(
        monkeypatch,
        [
            _ManagementResponse(200, {"id": "snap-1", "status": {"state": "Creating"}}),
            _ManagementResponse(404, {"code": "SNAPSHOT::NOT_FOUND"}),
            _ManagementResponse(200, {"id": "snap-1", "status": {"state": "Ready"}}),
        ],
    )

    with pytest.raises(RuntimeError, match=r"lookup failed -> HTTP 404$"):
        await _provider(snapshot_timeout_s=5, snapshot_lookup_grace_s=60).wait_snapshot_ready("snap-1")

    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 403])
async def test_client_errors_other_than_404_and_429_fail_at_once(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(status)])

    with pytest.raises(RuntimeError, match=rf"lookup failed -> HTTP {status}$"):
        await _provider(snapshot_timeout_s=5, snapshot_lookup_grace_s=60).wait_snapshot_ready("snap-1")

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (_TextResponse(200), "a body that is not JSON"),
        (_ManagementResponse(200, ["not", "an", "object"]), "a body that is not a JSON object"),
        (_ManagementResponse(200, {"id": "snap-1"}), "returned no state"),
    ],
)
async def test_a_lookup_answer_without_a_state_fails_at_once_instead_of_polling_until_the_timeout(
    monkeypatch: pytest.MonkeyPatch, response: _ManagementResponse, detail: str
) -> None:
    calls = _management(monkeypatch, [response])
    with pytest.raises(RuntimeError, match=detail):
        await _provider(snapshot_timeout_s=5).wait_snapshot_ready("snap-1")
    assert len(calls) == 1


async def test_delete_snapshot_tolerates_a_snapshot_that_is_already_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(404, {"code": "SNAPSHOT::NOT_FOUND"})])

    await _provider().delete_snapshot("snap-1")

    [call] = calls
    assert call["method"] == "DELETE" and call["url"] == "https://cell.example/v1/snapshots/snap-1"


async def test_delete_snapshot_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(
        monkeypatch,
        [_ManagementResponse(503), ConnectionResetError("rst"), _ManagementResponse(429), _ManagementResponse(204)],
    )
    await _provider(retries=3).delete_snapshot("snap-1")
    assert len(calls) == 4


async def test_delete_snapshot_gives_up_after_the_operation_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(500)])
    with pytest.raises(RuntimeError, match="snapshot 'snap-1' delete failed -> HTTP 500"):
        await _provider(retries=2).delete_snapshot("snap-1")
    assert len(calls) == 3
    _management(monkeypatch, [_ManagementResponse(409)])
    with pytest.raises(RuntimeError, match="delete failed -> HTTP 409"):
        await _provider(retries=2).delete_snapshot("snap-1")


async def test_list_snapshots_reads_every_page_and_filters_by_name_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    def page(items: list[dict[str, Any]], more: bool) -> _ManagementResponse:
        return _ManagementResponse(200, {"items": items, "pagination": {"hasNextPage": more}})

    calls = _management(
        monkeypatch,
        [
            page(
                [
                    {"id": "a", "name": "ng-run-srv-1", "sandboxId": "sb-1", "status": {"state": "Ready"}},
                    {"id": "b", "name": "other", "status": {"state": "Ready"}},
                    {"id": "c", "status": {"state": "Ready"}},
                ],
                True,
            ),
            _ManagementResponse(503),
            page([{"id": "d", "name": "ng-run-srv-2", "status": {"state": "Creating"}}], False),
        ],
    )

    found = await _provider(retries=1).list_snapshots("ng-run-srv-")

    assert found == [
        {"id": "a", "name": "ng-run-srv-1", "state": "ready", "sandbox_id": "sb-1"},
        {"id": "d", "name": "ng-run-srv-2", "state": "creating", "sandbox_id": None},
    ]
    assert [call["params"]["page"] for call in calls] == ["1", "2", "2"], "the failed page was asked again"
    assert all(call["url"] == "https://cell.example/v1/snapshots" for call in calls)


@pytest.mark.parametrize(
    "payload",
    [{"items": [{"name": "x"}], "pagination": {"hasNextPage": False}}, {"items": []}],
)
async def test_a_malformed_listing_raises(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    _management(monkeypatch, [_ManagementResponse(200, payload)])
    with pytest.raises(RuntimeError, match="snapshot listing"):
        await _provider().list_snapshots("ng-")


async def test_snapshot_operations_require_a_domain() -> None:
    provider = opensandbox_provider.OpenSandboxProvider(connection={"api_key": "k"})
    with pytest.raises(RuntimeError, match="require connection.domain"):
        await provider.delete_snapshot("snap-1")


@pytest.mark.parametrize(
    ("operations", "match"),
    [
        ({"snapshot_timeout_s": 0}, "snapshot_timeout_s"),
        ({"snapshot_lookup_grace_s": -1}, "snapshot_lookup_grace_s"),
        ({"snapshot_lookup_grace_s": float("nan")}, "snapshot_lookup_grace_s"),
    ],
)
def test_snapshot_budgets_are_validated(operations: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        opensandbox_provider.OpenSandboxOperationConfig(**operations)
    defaults = opensandbox_provider.OpenSandboxOperationConfig()
    assert (defaults.snapshot_timeout_s, defaults.snapshot_lookup_grace_s) == (240.0, 60.0)
