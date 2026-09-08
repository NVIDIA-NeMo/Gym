# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import subprocess
from pathlib import Path
from typing import Any

import aiohttp
import pytest

from nemo_gym.sandbox.providers.opensandbox import snapshots


SCRIPT = Path(snapshots.__file__)
TEST_ACCESS_KEY = "fixture-access-key"  # pragma: allowlist secret
BASE = "https://sandbox.example/v1"
SANDBOX_ARGS = ["--domain", "sandbox.example", "--api-key", TEST_ACCESS_KEY]
PAGE_PARAMS = [("page", "1"), ("pageSize", "100")]


class Response:
    def __init__(
        self,
        payload: object = "",
        status: int = 200,
        *,
        enter: Any = None,
        exit: Any = None,
        error: BaseException | None = None,
    ) -> None:
        self.payload = payload
        self.status = status
        self.enter = enter
        self.exit = exit
        self.error = error

    async def __aenter__(self) -> "Response":
        if self.error:
            raise self.error
        if self.enter:
            await self.enter()
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self.exit:
            await self.exit()

    async def json(self, *, content_type: None) -> object:
        assert content_type is None
        return self.payload

    async def read(self) -> bytes:
        return b""


class Session:
    def __init__(self, *get_responses: Response, delete_responses: dict[str, Response] | None = None) -> None:
        self.get_responses = iter(get_responses)
        self.delete_responses = delete_responses or {}
        self.requests: list[tuple[str, str, dict[str, object]]] = []
        self.closed = False

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.closed = True

    def get(self, url: str, **kwargs: object) -> Response:
        self.requests.append(("GET", url, kwargs))
        return next(self.get_responses)

    def delete(self, url: str, **kwargs: object) -> Response:
        self.requests.append(("DELETE", url, kwargs))
        return self.delete_responses[url]

    def urls(self, method: str) -> list[str]:
        return [url for request_method, url, _kwargs in self.requests if request_method == method]


def snapshot(snapshot_id: str, *, sandbox_id: str = "sb-1", state: str = "Ready") -> dict[str, object]:
    return {
        "id": snapshot_id,
        "sandboxId": sandbox_id,
        "status": {"state": state},
        "createdAt": "2026-09-01T00:00:00Z",
    }


def sandbox(sandbox_id: str, *, state: str = "Paused") -> dict[str, object]:
    return {"id": sandbox_id, "status": {"state": state}}


def page(items: list[object], *, has_next_page: bool = False) -> Response:
    return Response({"items": items, "pagination": {"hasNextPage": has_next_page}})


def install_session(
    monkeypatch: pytest.MonkeyPatch, session: Session
) -> tuple[list[dict[str, int]], list[dict[str, object]], object]:
    connector_calls: list[dict[str, int]] = []
    session_calls: list[dict[str, object]] = []
    connector = object()

    def make_connector(**kwargs: int) -> object:
        connector_calls.append(kwargs)
        return connector

    def make_session(**kwargs: object) -> Session:
        session_calls.append(kwargs)
        return session

    monkeypatch.setattr(snapshots.aiohttp, "TCPConnector", make_connector)
    monkeypatch.setattr(snapshots.aiohttp, "ClientSession", make_session)
    return connector_calls, session_calls, connector


def run_cleanup(
    *,
    domain: str = "https://sandbox.example",
    protocol: str = "http",
    sandbox_id: str | None = None,
    states: list[str] | None = None,
    snapshot_ids: list[str] | None = None,
    kill_paused: bool = False,
    reap: bool = True,
) -> int:
    return asyncio.run(
        snapshots.cleanup_snapshots(
            domain=domain,
            protocol=protocol,
            access_key=TEST_ACCESS_KEY,
            sandbox_id=sandbox_id,
            states=states,
            snapshot_ids=snapshot_ids,
            kill_paused=kill_paused,
            reap=reap,
        )
    )


def test_audit_lists_matches_without_deleting(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session = Session(
        page([snapshot("snap-a"), snapshot("snap-b", state="Failed")], has_next_page=True),
        page([snapshot("snap/c", sandbox_id="sb-2")]),
        page(
            [
                sandbox("sb-1"),
                sandbox("sb-2", state="PAUSED"),
                sandbox("sb-3", state="Running"),
                {"id": "x", "status": None},
            ]
        ),
    )
    connector_calls, session_calls, connector = install_session(monkeypatch, session)

    assert (
        run_cleanup(
            domain="sandbox.example/",
            protocol="https",
            sandbox_id="sb-1",
            states=["Ready", "Failed"],
            kill_paused=True,
            reap=False,
        )
        == 0
    )

    assert session.closed
    assert connector_calls == [{"limit": snapshots.REAP_CONCURRENCY, "limit_per_host": snapshots.REAP_CONCURRENCY}]
    assert len(session_calls) == 1
    assert session_calls[0]["connector"] is connector
    assert session_calls[0]["headers"] == {"OPEN-SANDBOX-API-KEY": TEST_ACCESS_KEY}
    assert session_calls[0]["timeout"].total == snapshots.REQUEST_TIMEOUT_SECONDS
    filters = [("sandboxId", "sb-1"), ("state", "Ready"), ("state", "Failed")]
    assert session.requests == [
        ("GET", f"{BASE}/snapshots", {"allow_redirects": False, "params": [*filters, *PAGE_PARAMS]}),
        (
            "GET",
            f"{BASE}/snapshots",
            {"allow_redirects": False, "params": [*filters, ("page", "2"), ("pageSize", "100")]},
        ),
        ("GET", f"{BASE}/sandboxes", {"allow_redirects": False, "params": PAGE_PARAMS}),
    ]
    out = capsys.readouterr().out
    assert "Would delete snapshot snap-a (sandbox=sb-1 state=Ready created=2026-09-01T00:00:00Z)" in out
    assert "Would delete snapshot snap-b (sandbox=sb-1 state=Failed" in out
    assert "Would delete snapshot snap/c (sandbox=sb-2" in out
    # --sandbox-id scopes --kill-paused too; other paused sandboxes are left alone.
    assert "Would delete paused sandbox sb-1" in out
    assert "sb-2" not in out.split("Would delete paused sandbox")[-1]
    assert "Would delete 3 OpenSandbox snapshot(s) and 1 paused sandbox(es)" in out


def test_reap_deletes_snapshots_then_paused_sandboxes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot_urls = {f"{BASE}/snapshots/snap-a", f"{BASE}/snapshots/snap%2Fb"}
    sandbox_urls = {f"{BASE}/sandboxes/sb-1", f"{BASE}/sandboxes/sb-3"}
    session = Session(
        page([snapshot("snap-a"), snapshot("snap/b")]),
        page([sandbox("sb-1"), sandbox("sb-2", state="Running"), sandbox("sb-3", state="paused")]),
        delete_responses={url: Response(status=204) for url in snapshot_urls | sandbox_urls},
    )
    install_session(monkeypatch, session)

    assert run_cleanup(kill_paused=True) == 0

    deletes = session.urls("DELETE")
    assert set(deletes[:2]) == snapshot_urls
    assert set(deletes[2:]) == sandbox_urls
    assert all(kwargs == {"allow_redirects": False} for method, _url, kwargs in session.requests if method == "DELETE")
    out = capsys.readouterr().out
    assert "Deleting 2 OpenSandbox snapshot(s) and 2 paused sandbox(es)" in out
    assert "Deleted snapshot snap/b -> HTTP 204" in out
    assert "Deleted paused sandbox sb-3 -> HTTP 204" in out


def test_explicit_snapshot_ids_skip_listing_and_404_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    delete_responses = {
        f"{BASE}/snapshots/snap-a": Response(status=204),
        f"{BASE}/snapshots/gone": Response(status=404),
    }
    session = Session(delete_responses=delete_responses)
    install_session(monkeypatch, session)

    assert run_cleanup(snapshot_ids=["snap-a", "gone"]) == 0

    assert session.urls("GET") == []
    assert set(session.urls("DELETE")) == set(delete_responses)
    out = capsys.readouterr().out
    assert "Deleting snapshot snap-a (sandbox=- state=- created=-)" in out
    assert "snapshot gone was already gone" in out


def test_delete_failures_are_reported_and_do_not_stop_the_sweep(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    session = Session(
        page([snapshot("failed"), snapshot("deleted"), snapshot("disconnected"), snapshot("redirected")]),
        delete_responses={
            f"{BASE}/snapshots/failed": Response(status=500),
            f"{BASE}/snapshots/deleted": Response(status=204),
            f"{BASE}/snapshots/disconnected": Response(error=aiohttp.ClientConnectionError("disconnected")),
            f"{BASE}/snapshots/redirected": Response(status=302),
        },
    )
    install_session(monkeypatch, session)

    assert run_cleanup() == 1

    assert len(session.urls("DELETE")) == 4
    output = capsys.readouterr()
    assert "Deleted snapshot deleted -> HTTP 204" in output.out
    assert "Failed to delete snapshot failed -> HTTP 500" in output.err
    assert "Failed to delete snapshot disconnected -> disconnected" in output.err
    assert "Failed to delete snapshot redirected -> HTTP 302" in output.err
    assert "3 OpenSandbox snapshot(s) or paused sandbox(es) were not deleted" in output.err
    assert TEST_ACCESS_KEY not in output.out + output.err


async def test_reap_limits_concurrent_deletes(monkeypatch: pytest.MonkeyPatch) -> None:
    active = 0
    maximum = 0
    all_started = asyncio.Event()
    release = asyncio.Event()

    async def enter() -> None:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == snapshots.REAP_CONCURRENCY:
            all_started.set()
        await release.wait()

    async def exit() -> None:
        nonlocal active
        active -= 1

    total = snapshots.REAP_CONCURRENCY + 1
    delete_responses = {
        f"{BASE}/snapshots/snap-{index}": Response(status=204, enter=enter, exit=exit) for index in range(total)
    }
    install_session(monkeypatch, Session(delete_responses=delete_responses))
    task = asyncio.create_task(
        snapshots.cleanup_snapshots(
            domain="https://sandbox.example",
            protocol="http",
            access_key=TEST_ACCESS_KEY,
            sandbox_id=None,
            states=None,
            snapshot_ids=[f"snap-{index}" for index in range(total)],
            kill_paused=False,
            reap=True,
        )
    )
    try:
        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert maximum == snapshots.REAP_CONCURRENCY
    finally:
        release.set()
        result = await task
    assert result == 0


def test_list_redirects_and_errors_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    session = Session(Response(status=302))
    install_session(monkeypatch, session)
    with pytest.raises(ValueError, match="snapshots list request failed -> HTTP 302"):
        run_cleanup(reap=False)
    assert session.requests == [("GET", f"{BASE}/snapshots", {"allow_redirects": False, "params": PAGE_PARAMS})]

    install_session(monkeypatch, Session(page([]), Response(status=503)))
    with pytest.raises(ValueError, match="sandboxes list request failed -> HTTP 503"):
        run_cleanup(kill_paused=True, reap=False)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "must be an object"),
        ({"items": {}, "pagination": {"hasNextPage": False}}, "missing items or pagination"),
        ({"items": [], "pagination": {}}, "missing pagination.hasNextPage"),
        ({"items": [{"id": ""}], "pagination": {"hasNextPage": False}}, "item without an id"),
        ({"items": ["snap-a"], "pagination": {"hasNextPage": False}}, "item without an id"),
    ],
)
def test_rejects_malformed_list_responses(monkeypatch: pytest.MonkeyPatch, payload: object, message: str) -> None:
    install_session(monkeypatch, Session(Response(payload)))

    with pytest.raises(ValueError, match=message):
        run_cleanup(reap=False)


def test_rejects_invalid_domain() -> None:
    with pytest.raises(ValueError, match="invalid OpenSandbox domain"):
        run_cleanup(domain="file:///tmp/snapshots", reap=False)


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--domain", "sandbox.example"],
        ["--api-key", TEST_ACCESS_KEY],
        ["--domain", " ", "--api-key", TEST_ACCESS_KEY],
        [*SANDBOX_ARGS, "--protocol", "ftp"],
        ["--connection-config", "env.yaml", *SANDBOX_ARGS],
        [*SANDBOX_ARGS, "--snapshot-id", "snap-a", "--sandbox-id", "sb-1"],
        [*SANDBOX_ARGS, "--snapshot-id", "snap-a", "--state", "Ready"],
        [*SANDBOX_ARGS, "--sandbox-id", " "],
        [*SANDBOX_ARGS, "--state", ""],
        [*SANDBOX_ARGS, "--snapshot-id", "snap-a", "--snapshot-id", " "],
        [*SANDBOX_ARGS, "--unknown"],
    ],
)
def test_cli_rejects_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit, match="2"):
        snapshots.main(argv)


@pytest.mark.parametrize(("passed_protocol", "expected_protocol"), [(None, "http"), ("https", "https")])
def test_cli_forwards_arguments_and_return_codes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    passed_protocol: str | None,
    expected_protocol: str,
) -> None:
    calls = []
    argv = [*SANDBOX_ARGS, "--sandbox-id", "sb-1", "--state", "Ready", "--state", "Failed", "--kill-paused", "--reap"]
    if passed_protocol:
        argv += ["--protocol", passed_protocol]

    async def record_cleanup(**kwargs: object) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(snapshots, "cleanup_snapshots", record_cleanup)
    assert snapshots.main(argv) == 0
    assert calls == [
        {
            "domain": "sandbox.example",
            "protocol": expected_protocol,
            "access_key": TEST_ACCESS_KEY,
            "sandbox_id": "sb-1",
            "states": ["Ready", "Failed"],
            "snapshot_ids": None,
            "kill_paused": True,
            "reap": True,
        }
    ]

    async def failed_cleanup(**_kwargs: object) -> int:
        return 1

    monkeypatch.setattr(snapshots, "cleanup_snapshots", failed_cleanup)
    assert snapshots.main(argv) == 1

    async def raise_cleanup_error(**_kwargs: object) -> int:
        raise OSError("down")

    monkeypatch.setattr(snapshots, "cleanup_snapshots", raise_cleanup_error)
    assert snapshots.main(argv) == 1
    stderr = capsys.readouterr().err
    assert "OpenSandbox snapshot cleanup failed: down" in stderr
    assert TEST_ACCESS_KEY not in stderr


@pytest.mark.parametrize(("configured_protocol", "expected_protocol"), [(None, "http"), ("https", "https")])
def test_cli_uses_standalone_connection_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, configured_protocol: str | None, expected_protocol: str
) -> None:
    calls = []
    config = tmp_path / "env.yaml"
    protocol = f"      protocol: {configured_protocol}\n" if configured_protocol else ""
    config.write_text(
        "decoy:\n"
        "  domain: wrong.example\n"
        "sandbox:\n"
        "  opensandbox:\n"
        "    connection:\n"
        "      domain: sandbox.example\n"
        f"      api_key: {TEST_ACCESS_KEY}\n"
        f"{protocol}"
    )

    async def record_cleanup(**kwargs: object) -> int:
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(snapshots, "cleanup_snapshots", record_cleanup)
    assert (
        snapshots.main(["--connection-config", str(config), "--snapshot-id", "snap-a", "--snapshot-id", "snap-b"]) == 0
    )
    assert calls == [
        {
            "domain": "sandbox.example",
            "protocol": expected_protocol,
            "access_key": TEST_ACCESS_KEY,
            "sandbox_id": None,
            "states": None,
            "snapshot_ids": ["snap-a", "snap-b"],
            "kill_paused": False,
            "reap": False,
        }
    ]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ("[]\n", "must contain a YAML object"),
        ("other: {}\n", "config 'sandbox' is required"),
        ("sandbox:\n  docker: {}\n", "config 'sandbox.opensandbox' is required"),
        ("sandbox:\n  opensandbox: {}\n", "config 'sandbox.opensandbox.connection' is required"),
        (
            f"sandbox:\n  opensandbox:\n    connection:\n      api_key: {TEST_ACCESS_KEY}\n",
            "connection.domain' is required",
        ),
        (
            "sandbox:\n  opensandbox:\n    connection:\n      domain: sandbox.example\n",
            "connection.api_key' is required",
        ),
        (
            "sandbox:\n"
            "  opensandbox:\n"
            "    connection:\n"
            "      domain: sandbox.example\n"
            f"      api_key: {TEST_ACCESS_KEY}\n"
            "      protocol: ftp\n",
            "connection.protocol' must be http or https",
        ),
        (
            "sandbox:\n  opensandbox:\n    connection:\n      api_key: [fixture-access-key\n",
            "invalid YAML connection config",
        ),
    ],
)
def test_cli_rejects_invalid_connection_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path, config: str, message: str
) -> None:
    config_path = tmp_path / "env.yaml"
    config_path.write_text(config)

    async def fail_cleanup(**_kwargs: object) -> int:
        pytest.fail("network request must not be made")

    monkeypatch.setattr(snapshots, "cleanup_snapshots", fail_cleanup)
    assert snapshots.main(["--connection-config", str(config_path)]) == 1
    stderr = capsys.readouterr().err
    assert message in stderr
    assert TEST_ACCESS_KEY not in stderr


def test_cli_reports_a_missing_connection_config(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert snapshots.main(["--connection-config", str(tmp_path / "missing.yaml")]) == 1
    assert "OpenSandbox snapshot cleanup failed: [Errno 2]" in capsys.readouterr().err


def test_script_help_runs_by_direct_path() -> None:
    result = subprocess.run([str(SCRIPT), "--help"], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--kill-paused" in result.stdout
