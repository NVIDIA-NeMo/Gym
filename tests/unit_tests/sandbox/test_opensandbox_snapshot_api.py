# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenSandbox snapshots: start through the SDK, then wait, list and delete over the management API."""

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from typing import Any

import aiohttp
import pytest

from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.providers.base import (
    SandboxHandle,
    SandboxSnapshotError,
    SandboxSnapshotFailedError,
    SandboxSnapshotInfo,
    SandboxSnapshotInUseError,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SupportsSandboxSnapshot,
)


pytestmark = pytest.mark.sandbox

pytest.importorskip("tenacity", reason="tenacity optional sandbox dependency is not installed")

from nemo_gym.sandbox.providers.opensandbox import provider as opensandbox_provider


class _ManagementResponse:
    """A management API reply; ``body`` is its text, by default the payload as JSON (empty without a payload)."""

    def __init__(self, status: int, payload: Any = None, *, body: str | None = None) -> None:
        self.status = status
        self._payload = {} if payload is None else payload
        self._body = body if body is not None else json.dumps(payload) if payload else ""

    async def __aenter__(self) -> "_ManagementResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def json(self, content_type: object = None) -> Any:
        return self._payload

    async def text(self, errors: str = "strict") -> str:
        return self._body


class _TextResponse(_ManagementResponse):
    async def json(self, content_type: object = None) -> Any:
        raise ValueError("not JSON")


class _CutOffBody(_ManagementResponse):
    """A reply whose connection drops while its body is read."""

    async def json(self, content_type: object = None) -> Any:
        raise aiohttp.ClientPayloadError("Response payload is not completed")

    async def text(self, errors: str = "strict") -> str:
        raise aiohttp.ClientPayloadError("Response payload is not completed")


def _management(monkeypatch: pytest.MonkeyPatch, responses: list[Any]) -> list[dict[str, Any]]:
    """Replace Gym's shared client with a scripted management API; returns the calls it received.

    The last response repeats once the others are used up. An exception in the list is raised instead.
    """
    import nemo_gym.client.http_client as http_client

    calls: list[dict[str, Any]] = []

    async def fake_request(method: str, url: str, **kwargs: Any) -> _ManagementResponse:
        calls.append({"method": method, "url": url, **kwargs})
        response = responses.pop(0) if len(responses) > 1 else responses[0]
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(http_client, "request", fake_request)
    return calls


class _ScriptedClient:
    """Stands in for Gym's shared aiohttp control client, below the retry loop of ``http_client.request``.

    Each wire attempt takes the next step; the last one repeats. An exception step is raised instead.
    """

    timeout = aiohttp.ClientTimeout(total=5)

    def __init__(self, *script: Any) -> None:
        self.script = list(script)
        self.attempts: list[dict[str, Any]] = []

    async def request(self, **kwargs: Any) -> Any:
        self.attempts.append(kwargs)
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        return step


def _wire(monkeypatch: pytest.MonkeyPatch, *script: Any) -> _ScriptedClient:
    """Script the wire itself, so the shared client's own retries are part of what a test counts."""
    import nemo_gym.client.http_client as http_client

    client = _ScriptedClient(*script)
    monkeypatch.setattr(http_client, "get_global_aiohttp_control_client", lambda: client)
    return client


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
    with pytest.raises(SandboxSnapshotFailedError, match="snapshot 'snap-1' failed: registry push failed") as raised:
        await _provider().wait_snapshot_ready("snap-1")
    assert raised.value.snapshot_id == "snap-1"


async def test_a_snapshot_that_is_not_ready_in_time_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _management(monkeypatch, [_ManagementResponse(200, {"id": "snap-1", "status": {"state": "Creating"}})])
    with pytest.raises(SandboxSnapshotFailedError, match="snapshot 'snap-1' to be ready after 0.05s") as raised:
        await _provider(snapshot_timeout_s=0.05).wait_snapshot_ready("snap-1")
    assert raised.value.snapshot_id == "snap-1"


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

    with pytest.raises(
        SandboxSnapshotNotFoundError, match=r"'snap-1' lookup failed -> HTTP 404 \(\d+ lookups over .*NOT_FOUND"
    ):
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

    # Caught as any snapshot failure, the not-found error still says which snapshot is missing.
    with pytest.raises(SandboxSnapshotError, match=r"'snap-1' lookup failed -> HTTP 404: .*NOT_FOUND") as raised:
        await _provider(snapshot_timeout_s=5, snapshot_lookup_grace_s=60).wait_snapshot_ready("snap-1")

    assert isinstance(raised.value, SandboxSnapshotNotFoundError) and isinstance(raised.value, LookupError)
    assert raised.value.snapshot_id == "snap-1"
    assert len(calls) == 2


@pytest.mark.parametrize("status", [400, 403])
async def test_client_errors_other_than_404_and_429_fail_at_once(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(status, body="key lacks the snapshot scope")])

    with pytest.raises(
        SandboxSnapshotError, match=rf"lookup failed -> HTTP {status}: key lacks the snapshot scope$"
    ) as raised:
        await _provider(snapshot_timeout_s=5, snapshot_lookup_grace_s=60).wait_snapshot_ready("snap-1")

    assert len(calls) == 1
    # Not a verdict on the snapshot: the lookup, not the snapshot, failed.
    assert not isinstance(raised.value, SandboxSnapshotFailedError) and raised.value.snapshot_id == "snap-1"


@pytest.mark.parametrize(
    ("response", "detail", "failed"),
    [
        (_TextResponse(200), "a body that is not JSON", False),
        (_ManagementResponse(200, ["not", "an", "object"]), "a body that is not a JSON object", False),
        (_ManagementResponse(200, {"id": "snap-1"}), "returned no state", True),
    ],
)
async def test_a_lookup_answer_without_a_state_fails_at_once_instead_of_polling_until_the_timeout(
    monkeypatch: pytest.MonkeyPatch, response: _ManagementResponse, detail: str, failed: bool
) -> None:
    calls = _management(monkeypatch, [response])
    with pytest.raises(SandboxSnapshotError, match=detail) as raised:
        await _provider(snapshot_timeout_s=5).wait_snapshot_ready("snap-1")
    assert len(calls) == 1
    assert isinstance(raised.value, SandboxSnapshotFailedError) == failed


_NOT_FOUND = {"code": "SNAPSHOT::NOT_FOUND"}
_SNAPSHOT_URL = "https://cell.example/v1/snapshots/snap-1"


async def test_delete_snapshot_tolerates_a_snapshot_that_a_lookup_confirms_is_already_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(404, _NOT_FOUND), _ManagementResponse(404, _NOT_FOUND)])

    await _provider().delete_snapshot("snap-1")

    assert [(call["method"], call["url"]) for call in calls] == [("DELETE", _SNAPSHOT_URL), ("GET", _SNAPSHOT_URL)]


async def test_a_404_to_the_delete_of_a_snapshot_that_still_exists_is_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _management(
        monkeypatch,
        [
            _ManagementResponse(404, _NOT_FOUND),  # a replica that has not seen the snapshot yet
            _ManagementResponse(200, {"id": "snap-1", "status": {"state": "Ready"}}),
            _ManagementResponse(204),
        ],
    )

    await _provider(retries=1).delete_snapshot("snap-1")

    assert [call["method"] for call in calls] == ["DELETE", "GET", "DELETE"]


async def test_a_snapshot_still_visible_after_every_404_to_its_delete_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    rounds = [_ManagementResponse(404, _NOT_FOUND), _ManagementResponse(200, {"id": "snap-1"})] * 3
    calls = _management(monkeypatch, [*rounds, _ManagementResponse(204)])

    with pytest.raises(
        SandboxSnapshotError, match="snapshot 'snap-1' delete answered 404 but the snapshot still exists"
    ) as raised:
        await _provider(retries=2).delete_snapshot("snap-1")

    assert [call["method"] for call in calls] == ["DELETE", "GET"] * 3, "retries + 1 rounds within one budget"
    assert raised.value.snapshot_id == "snap-1" and not isinstance(raised.value, SandboxSnapshotInUseError)


@pytest.mark.parametrize(
    "unconfirmed", [_ManagementResponse(503), _ManagementResponse(429), aiohttp.ServerDisconnectedError()]
)
async def test_a_404_to_a_delete_that_cannot_be_confirmed_is_retried(
    monkeypatch: pytest.MonkeyPatch, unconfirmed: Any
) -> None:
    calls = _management(
        monkeypatch,
        [_ManagementResponse(404, _NOT_FOUND), unconfirmed, _ManagementResponse(404, _NOT_FOUND)],
    )

    await _provider(retries=1).delete_snapshot("snap-1")

    assert [call["method"] for call in calls] == ["DELETE", "GET", "DELETE", "GET"]


async def test_a_confirming_lookup_that_is_refused_fails_the_delete_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(404, _NOT_FOUND), _ManagementResponse(403, body="denied")])

    with pytest.raises(
        SandboxSnapshotError, match="delete answered 404 and its confirming lookup failed -> HTTP 403: denied"
    ):
        await _provider(retries=2).delete_snapshot("snap-1")

    assert len(calls) == 2


async def test_delete_snapshot_retries_transient_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(
        monkeypatch,
        [_ManagementResponse(503), ConnectionResetError("rst"), _ManagementResponse(429), _ManagementResponse(204)],
    )
    await _provider(retries=3).delete_snapshot("snap-1")
    assert len(calls) == 4


async def test_delete_snapshot_gives_up_after_the_operation_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(500, body="registry unavailable")])
    with pytest.raises(
        SandboxSnapshotError, match="snapshot 'snap-1' delete failed -> HTTP 500: registry unavailable$"
    ) as raised:
        await _provider(retries=2).delete_snapshot("snap-1")
    assert len(calls) == 3
    assert raised.value.snapshot_id == "snap-1" and not isinstance(raised.value, SandboxSnapshotInUseError)


async def test_deleting_a_snapshot_that_is_in_use_raises_the_servers_reason_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(409, {"message": "snapshot is used by sandbox sb-9"})])

    with pytest.raises(
        SandboxSnapshotInUseError, match="'snap-1' delete failed -> HTTP 409: .*snapshot is used by sandbox sb-9"
    ) as raised:
        await _provider(retries=2).delete_snapshot("snap-1")

    assert len(calls) == 1 and raised.value.snapshot_id == "snap-1"


async def test_a_long_error_body_is_cut_to_its_first_500_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    _management(monkeypatch, [_ManagementResponse(403, body="x" * 2000)])
    with pytest.raises(SandboxSnapshotError) as raised:
        await _provider().delete_snapshot("snap-1")
    assert str(raised.value).endswith("delete failed -> HTTP 403: " + "x" * 500)


async def test_an_error_status_whose_body_cannot_be_read_is_still_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(monkeypatch, [_CutOffBody(403)])
    with pytest.raises(SandboxSnapshotError, match="delete failed -> HTTP 403$"):
        await _provider(retries=2).delete_snapshot("snap-1")
    assert len(calls) == 1, "the status, not the unreadable body, decides: a 403 is not retried"


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
        SandboxSnapshotInfo(snapshot_id="a", name="ng-run-srv-1", state="ready", sandbox_id="sb-1"),
        SandboxSnapshotInfo(snapshot_id="d", name="ng-run-srv-2", state="creating", sandbox_id=None),
    ]
    assert [call["params"]["page"] for call in calls] == ["1", "2", "2"], "the failed page was asked again"
    assert all(call["url"] == "https://cell.example/v1/snapshots" for call in calls)


@pytest.mark.parametrize(
    "payload",
    [{"items": [{"name": "x"}], "pagination": {"hasNextPage": False}}, {"items": []}],
)
async def test_a_malformed_listing_raises(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    _management(monkeypatch, [_ManagementResponse(200, payload)])
    with pytest.raises(SandboxSnapshotError, match="snapshot listing"):
        await _provider().list_snapshots("ng-")


@pytest.mark.parametrize("pagination", [{}, {"hasNextPage": None}, {"hasNextPage": "false"}, {"hasNextPage": 0}])
async def test_a_page_with_items_that_does_not_say_whether_another_follows_raises(
    monkeypatch: pytest.MonkeyPatch, pagination: dict[str, Any]
) -> None:
    listing = {"items": [{"id": "s1", "name": "ng-example"}], "pagination": pagination}
    calls = _management(monkeypatch, [_ManagementResponse(200, listing)])

    # Taken for "no more pages", this would leave the later pages' snapshots out of a sweep without a word.
    with pytest.raises(SandboxSnapshotError, match="page 1 does not say whether another page follows"):
        await _provider().list_snapshots("ng-")

    assert len(calls) == 1


async def test_an_empty_page_ends_the_listing_without_a_next_page_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    _management(monkeypatch, [_ManagementResponse(200, {"items": [], "pagination": {}})])
    assert await _provider().list_snapshots("ng-") == []


async def test_a_listed_snapshot_that_reports_no_state_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = {"items": [{"id": "a", "name": "ng-1"}], "pagination": {"hasNextPage": False}}
    _management(monkeypatch, [_ManagementResponse(200, listing)])
    assert await _provider().list_snapshots("ng-") == [SandboxSnapshotInfo(snapshot_id="a", name="ng-1")]


@pytest.mark.parametrize(("status", "requests"), [(403, 1), (503, 3)])
async def test_a_listing_page_that_keeps_failing_raises_instead_of_returning_a_partial_listing(
    monkeypatch: pytest.MonkeyPatch, status: int, requests: int
) -> None:
    first = _ManagementResponse(
        200, {"items": [{"id": "a", "name": "ng-run-srv-1"}], "pagination": {"hasNextPage": True}}
    )
    calls = _management(monkeypatch, [first, _ManagementResponse(status, body="try later")])

    # A partial listing would let a sweep treat the rest as absent; a client error is not retried, a 5xx is.
    with pytest.raises(SandboxSnapshotError, match=rf"snapshot listing failed -> HTTP {status}: try later"):
        await _provider(retries=2).list_snapshots("ng-run-")

    assert [call["params"]["page"] for call in calls] == ["1", *["2"] * requests]


async def test_a_listing_asks_for_the_servers_largest_page(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(200, _EMPTY_LISTING)])
    await _provider().list_snapshots("ng-")
    # The server refuses a pageSize above 200; every page of the store is read, so each one is as large as allowed.
    assert calls[0]["params"] == {"page": "1", "pageSize": "200"}


async def test_an_empty_page_ends_the_listing_even_if_the_server_reports_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = {"items": [{"id": "a", "name": "ng-1"}], "pagination": {"hasNextPage": True}}
    endless = {"items": [], "pagination": {"hasNextPage": True}}
    calls = _management(monkeypatch, [_ManagementResponse(200, first), _ManagementResponse(200, endless)])

    listed = await asyncio.wait_for(_provider().list_snapshots("ng-"), 5)

    assert listed == [SandboxSnapshotInfo(snapshot_id="a", name="ng-1")]
    assert [call["params"]["page"] for call in calls] == ["1", "2"]


async def test_a_listing_that_outlasts_its_budget_raises_instead_of_paging_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each page answers, slowly, so no per-request timeout or retry limit would ever end the listing."""
    import nemo_gym.client.http_client as http_client

    pages: list[str] = []

    async def slow_page(method: str, url: str, **kwargs: Any) -> _ManagementResponse:
        pages.append(kwargs["params"]["page"])
        await asyncio.sleep(0.05)
        item = {"id": f"s{len(pages)}", "name": "ng-1"}
        return _ManagementResponse(200, {"items": [item], "pagination": {"hasNextPage": True}})

    monkeypatch.setattr(http_client, "request", slow_page)

    with pytest.raises(
        SandboxSnapshotError, match=r"listing did not finish within 0\.12s \((\d+) pages read\)"
    ) as raised:
        await asyncio.wait_for(_provider(snapshot_list_timeout_s=0.12).list_snapshots("ng-"), 5)

    # It stopped inside the page it was reading when the budget ran out, a few pages in: the endless listing did
    # not run to the test's own 5 s.
    assert pages == [str(number) for number in range(1, len(pages) + 1)] and len(pages) <= 4
    assert f"({len(pages) - 1} pages read)" in str(raised.value)


async def test_a_delete_that_outlasts_its_budget_raises_and_names_the_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every try is answered with a retryable status, so only the delete's own budget ends it this early."""
    calls = _management(monkeypatch, [_ManagementResponse(503, body="busy")])
    provider = _provider(retries=1000, retry_delay_s=0.02, snapshot_delete_timeout_s=0.1)

    with pytest.raises(SandboxSnapshotError, match=r"'snap-1' delete did not finish within 0\.1s") as raised:
        await asyncio.wait_for(provider.delete_snapshot("snap-1"), 5)

    assert raised.value.snapshot_id == "snap-1"
    assert 1 <= len(calls) < 20, "ended by the budget, long before the 1000 retries"


async def test_a_timeout_inside_a_delete_is_not_mistaken_for_its_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    _management(monkeypatch, [asyncio.TimeoutError("the request timed out")])
    with pytest.raises(SandboxSnapshotError, match="delete failed: TimeoutError"):
        await _provider(retries=1).delete_snapshot("snap-1")


async def test_a_snapshot_the_sdk_returns_without_an_id_raises() -> None:
    class _NoId:
        async def create_snapshot(self, name: str | None = None) -> Any:
            return SimpleNamespace(id=None)

    with pytest.raises(RuntimeError, match="returned no snapshot id for sandbox 'sb-1'"):
        await _provider().snapshot(SandboxHandle("sb-1", "opensandbox", _NoId()), name="ng-s1")


async def test_a_snapshot_id_is_one_escaped_path_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _management(monkeypatch, [_ManagementResponse(200, {"id": "a/b c", "status": {"state": "Ready"}})])
    provider = _provider()

    await provider.wait_snapshot_ready("a/../b c")
    await provider.delete_snapshot("a/../b c")

    assert [(call["method"], call["url"]) for call in calls] == [
        ("GET", "https://cell.example/v1/snapshots/a%2F..%2Fb%20c"),
        ("DELETE", "https://cell.example/v1/snapshots/a%2F..%2Fb%20c"),
    ]


# -- one wire attempt per operation try -------------------------------------------------------------------------

_EMPTY_LISTING = {"items": [], "pagination": {"hasNextPage": False}}
_READY = {"id": "snap-1", "status": {"state": "Ready"}}


def _operation(provider: Any, name: str) -> Any:
    if name == "delete":
        return provider.delete_snapshot("snap-1")
    if name == "list":
        return provider.list_snapshots("ng-")
    return provider.wait_snapshot_ready("snap-1")


@pytest.mark.parametrize(
    "refused",
    [
        aiohttp.ClientOSError(111, "Connect call failed"),
        aiohttp.ClientConnectorError(
            SimpleNamespace(host="cell.example", port=443, ssl=False), OSError(111, "Connect call failed")
        ),
    ],
    ids=["os-error", "connector-error"],
)
@pytest.mark.parametrize("operation", ["delete", "list"])
async def test_a_refused_connection_is_tried_once_per_operation_try_and_then_raises(
    monkeypatch: pytest.MonkeyPatch, operation: str, refused: BaseException
) -> None:
    client = _wire(monkeypatch, refused)

    # The shared client would otherwise resend a refused connection without a bound, under every operation try.
    with pytest.raises(SandboxSnapshotError, match="failed: Client") as raised:
        await asyncio.wait_for(_operation(_provider(retries=2), operation), 30)

    assert len(client.attempts) == 3
    assert raised.value.__cause__ is refused


@pytest.mark.parametrize(
    "transient",
    [aiohttp.ServerDisconnectedError(), asyncio.TimeoutError(), _CutOffBody(200)],
    ids=["disconnected", "request-timeout", "body-cut-off"],
)
@pytest.mark.parametrize("operation", ["delete", "list", "wait"])
async def test_a_transport_failure_is_retried_by_the_operation_alone(
    monkeypatch: pytest.MonkeyPatch, operation: str, transient: Any
) -> None:
    answer = _EMPTY_LISTING if operation == "list" else _READY
    client = _wire(monkeypatch, transient, _ManagementResponse(200, answer))

    await asyncio.wait_for(_operation(_provider(retries=2, snapshot_timeout_s=5), operation), 30)

    assert len(client.attempts) == 2


async def test_a_refused_delete_is_one_wire_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _wire(monkeypatch, _ManagementResponse(403, body="denied"))
    with pytest.raises(SandboxSnapshotError, match="HTTP 403: denied"):
        await _provider(retries=2).delete_snapshot("snap-1")
    assert len(client.attempts) == 1


async def test_a_management_request_has_its_own_short_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _wire(monkeypatch, _ManagementResponse(200, _READY))
    provider = opensandbox_provider.OpenSandboxProvider(
        connection={"domain": "https://cell.example", "api_key": "k", "request_timeout_s": 300},
        operations={"snapshot_request_timeout_s": 7},
    )

    await provider.wait_snapshot_ready("snap-1")

    # Not the connection's request timeout, which is sized for long submits.
    assert [attempt["timeout"].total for attempt in client.attempts] == [7]


@pytest.mark.parametrize(
    ("connection", "url", "headers"),
    [
        # The config leaves both to the environment, as the SDK allows.
        ({}, "http://env.example:8443/v1/snapshots/snap-1", {"OPEN-SANDBOX-API-KEY": "env-key"}),
        # Only the key is left to the environment, to keep it out of the config file.
        ({"domain": "cell.example"}, "http://cell.example/v1/snapshots/snap-1", {"OPEN-SANDBOX-API-KEY": "env-key"}),
        ({"api_key": "k"}, "http://env.example:8443/v1/snapshots/snap-1", {"OPEN-SANDBOX-API-KEY": "k"}),
        # What the config sets wins over the environment.
        (
            {"domain": "https://cell.example", "api_key": "k"},
            "https://cell.example/v1/snapshots/snap-1",
            {"OPEN-SANDBOX-API-KEY": "k"},
        ),
    ],
)
async def test_management_requests_go_to_the_server_and_key_the_sdk_resolves(
    monkeypatch: pytest.MonkeyPatch, connection: dict[str, str], url: str, headers: dict[str, str]
) -> None:
    """The SDK creates sandboxes and submits snapshots with these; the wait, listing and delete must agree."""
    monkeypatch.setenv("OPEN_SANDBOX_DOMAIN", "env.example:8443")
    monkeypatch.setenv("OPEN_SANDBOX_API_KEY", "env-key")
    calls = _management(monkeypatch, [_ManagementResponse(200, _READY)])
    provider = opensandbox_provider.OpenSandboxProvider(connection=connection)

    await provider.wait_snapshot_ready("snap-1")
    await provider.delete_snapshot("snap-1")

    assert [(call["method"], call["url"], call["headers"]) for call in calls] == [
        ("GET", url, headers),
        ("DELETE", url, headers),
    ]


async def test_a_management_request_without_a_key_anywhere_sends_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPEN_SANDBOX_API_KEY", raising=False)
    calls = _management(monkeypatch, [_ManagementResponse(200, _EMPTY_LISTING)])
    provider = opensandbox_provider.OpenSandboxProvider(connection={"domain": "cell.example"})

    assert await provider.list_snapshots("ng-") == []

    assert calls[0]["url"] == "http://cell.example/v1/snapshots" and calls[0]["headers"] == {}


@pytest.mark.parametrize(
    ("operations", "match"),
    [
        ({"snapshot_timeout_s": 0}, "snapshot_timeout_s"),
        ({"snapshot_lookup_grace_s": -1}, "snapshot_lookup_grace_s"),
        ({"snapshot_lookup_grace_s": float("nan")}, "snapshot_lookup_grace_s"),
        ({"snapshot_request_timeout_s": 0}, "snapshot_request_timeout_s"),
        ({"snapshot_request_timeout_s": float("inf")}, "snapshot_request_timeout_s"),
        ({"snapshot_list_timeout_s": 0}, "snapshot_list_timeout_s"),
        ({"snapshot_list_timeout_s": float("inf")}, "snapshot_list_timeout_s"),
        ({"snapshot_delete_timeout_s": -1}, "snapshot_delete_timeout_s"),
        ({"snapshot_delete_timeout_s": float("nan")}, "snapshot_delete_timeout_s"),
    ],
)
def test_snapshot_budgets_are_validated(operations: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        opensandbox_provider.OpenSandboxOperationConfig(**operations)
    defaults = opensandbox_provider.OpenSandboxOperationConfig()
    assert (defaults.snapshot_timeout_s, defaults.snapshot_lookup_grace_s) == (240.0, 60.0)
    assert defaults.snapshot_request_timeout_s == 30.0
    assert (defaults.snapshot_list_timeout_s, defaults.snapshot_delete_timeout_s) == (300.0, 120.0)


def _recording_create(monkeypatch: pytest.MonkeyPatch, provider: Any, outcome: Any) -> list[SandboxSpec]:
    """Replace ``provider.create``: record each spec, then return ``outcome``'s handle or raise it."""
    created: list[SandboxSpec] = []

    async def create(spec: SandboxSpec) -> SandboxHandle:
        created.append(spec)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(provider, "create", create)
    return created


async def test_create_from_snapshot_creates_with_the_snapshot_in_place_of_the_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    created = _recording_create(monkeypatch, provider, SandboxHandle("sb-2", "opensandbox", None))
    spec = SandboxSpec(image="python:3.12", provider_options={"network_policy": {"egress": "none"}})

    handle = await provider.create_from_snapshot("snap-1", spec)

    assert handle.sandbox_id == "sb-2"
    assert created[0].image is None
    assert created[0].provider_options == {"network_policy": {"egress": "none"}, "snapshot_id": "snap-1"}


def _sdk_snapshot_not_found() -> Exception:
    """What the SDK raises for HTTP 404 ``{"code": "SNAPSHOT::NOT_FOUND", "message": "snapshot snap-1 not found"}``."""
    from opensandbox.exceptions import SandboxApiException
    from opensandbox.exceptions.sandbox import SandboxError

    return SandboxApiException(
        message="Create sandbox failed: snapshot snap-1 not found",
        status_code=404,
        error=SandboxError(code="SNAPSHOT::NOT_FOUND", message="snapshot snap-1 not found"),
    )


def _failing_create_once(monkeypatch: pytest.MonkeyPatch, provider: Any, error: Exception) -> list[SandboxSpec]:
    """Fail the SDK create below ``provider.create``, whose retry loop wraps the SDK's error in a typed failure."""
    attempts: list[SandboxSpec] = []

    async def create_once(spec: SandboxSpec) -> SandboxHandle:
        attempts.append(spec)
        raise error

    monkeypatch.setattr(provider, "_create_once", create_once)
    return attempts


async def test_create_from_a_missing_snapshot_raises_snapshot_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider()
    sdk_error = _sdk_snapshot_not_found()
    attempts = _failing_create_once(monkeypatch, provider, sdk_error)

    with pytest.raises(SandboxSnapshotNotFoundError, match="'snap-1'") as raised:
        await provider.create_from_snapshot("snap-1", SandboxSpec(image="python:3.12"))

    assert isinstance(raised.value, SandboxSnapshotError) and raised.value.snapshot_id == "snap-1"
    # The server's code sits under the create path's own failure, and its text is not in the outer message.
    wrapper = raised.value.__cause__
    assert isinstance(wrapper, NgFailureEvent) and wrapper.__cause__ is sdk_error
    assert "SNAPSHOT::NOT_FOUND" not in f"{wrapper!s}"
    assert len(attempts) == 1 and attempts[0].provider_options == {"snapshot_id": "snap-1"}


async def test_a_not_found_code_only_in_an_error_text_is_still_snapshot_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    _recording_create(monkeypatch, provider, RuntimeError("Create sandbox failed: HTTP 404: SNAPSHOT::NOT_FOUND"))

    with pytest.raises(SandboxSnapshotNotFoundError, match="'snap-1'"):
        await provider.create_from_snapshot("snap-1", SandboxSpec(image="python:3.12"))


async def test_a_bare_404_to_a_create_from_a_snapshot_is_snapshot_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a live service answers for a create from a deleted snapshot: HTTP 404 with no error code in the body."""
    from opensandbox.exceptions import SandboxApiException

    provider = _provider()
    bare = SandboxApiException(message="Create sandbox failed: HTTP 404", status_code=404, request_id="33fdb7b4")
    _failing_create_once(monkeypatch, provider, bare)

    with pytest.raises(SandboxSnapshotNotFoundError, match="'snap-1'") as raised:
        await provider.create_from_snapshot("snap-1", SandboxSpec(image="python:3.12"))

    assert raised.value.snapshot_id == "snap-1"
    assert isinstance(raised.value.__cause__, NgFailureEvent) and raised.value.__cause__.__cause__ is bare


async def test_other_failures_to_create_from_a_snapshot_keep_their_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from opensandbox.exceptions import SandboxApiException
    from opensandbox.exceptions.sandbox import SandboxError

    provider = _provider()
    quota = SandboxApiException(
        message="Create sandbox failed: quota exceeded",
        status_code=403,
        error=SandboxError(code="QUOTA::EXCEEDED", message="quota exceeded"),
    )
    _failing_create_once(monkeypatch, provider, quota)

    with pytest.raises(NgFailureEvent) as raised:
        await provider.create_from_snapshot("snap-1", SandboxSpec(image="python:3.12"))
    assert raised.value.__cause__ is quota
    assert not isinstance(raised.value, SandboxSnapshotNotFoundError)
