# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end contracts of the failure path: the collector's written sidecar, the base server's own failures, a probe."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import ClientResponseError
from multidict import CIMultiDict

from nemo_gym.rollout_collection import (
    NG_FAILURE_CLASS_KEY,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _counted_failure_row,
    _failures_path_for,
    _latest_failure_rows,
    is_terminal_failure,
)
from nemo_gym.sandbox.transfer import SandboxProbeError, SandboxUnavailableError, remote_kind
from nemo_gym.server_utils import ServerClient
from tests.unit_tests.test_environment_server import (
    _environment_server,
    _EnvironmentServer,
    _request,
    _Response,
)
from tests.unit_tests.test_rollout_collection import FakeResponse, install_fake_server_client


# Spelled out rather than imported, so the contract is the header name and JSON shape on the wire.
_HEADER = "X-NG-Failure-Event"
_INNER_PATH = [
    {"operation": "judge.request", "exception_type": "nemo_gym.judge.JudgeCredentialsRejected"},
    {"exception_type": "ClientResponseError", "http_status": 401},
]


async def test_a_failed_run_header_reaches_the_written_sidecar_and_its_readers(tmp_path: Path, monkeypatch) -> None:
    """The real writer: /run fails with the header, the sidecar row on disk carries the path, readers ignore it."""
    input_jsonl = tmp_path / "input.jsonl"
    input_jsonl.write_text(
        "\n".join(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my_agent"}, "x": i})
            for i in range(2)
        )
        + "\n"
    )
    output_jsonl = tmp_path / "output.jsonl"
    header = json.dumps({"failure_reason": "x", "terminal": False, "failure_path": _INNER_PATH})

    async def post(server_name: str, url_path: str, json: dict, **kwargs):
        if url_path == "/run":
            if json["x"] == 0:
                request_info = SimpleNamespace(method="POST", url="http://agent/run", real_url="http://agent/run")
                raise ClientResponseError(
                    request_info=request_info, history=(), status=500, headers=CIMultiDict({_HEADER: header})
                )
            return FakeResponse(200, {"reward": 1.0, "response": {"usage": {"total_tokens": 3}}})
        return FakeResponse(200, {})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    monkeypatch.setattr("nemo_gym.rollout_collection.get_global_config_dict", MagicMock(return_value={}))
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(input_jsonl),
        output_jsonl_fpath=str(output_jsonl),
        route_failures_to_sidecar=True,
        disable_health_check=True,
    )
    await RolloutCollectionHelper().run_from_config(config)

    sidecar = _failures_path_for(output_jsonl)
    written = [orjson.loads(line) for line in sidecar.read_bytes().splitlines()]
    assert len(written) == 1 and written[0][NG_FAILURE_CLASS_KEY] == "agent_run_error"
    assert written[0]["_ng_failure_path"] == _INNER_PATH
    # The success is a scored row and carries no path.
    assert ["_ng_failure_path" in orjson.loads(line) for line in output_jsonl.read_bytes().splitlines()] == [False]
    # Readers: the latest-attempt index keeps the row, the metric copy drops the path, retry still reads terminality.
    (row,) = _latest_failure_rows([sidecar]).values()
    assert row["_ng_failure_path"] == _INNER_PATH
    assert "_ng_failure_path" not in _counted_failure_row(row) and is_terminal_failure(row) is False


def _hops(response: _Response) -> list[dict]:
    return response.model_dump(mode="json")["failure"].get("failure_path") or []


def test_the_admission_timeout_failure_carries_its_path() -> None:
    server = _environment_server(max_concurrent_episodes=1, queue_timeout_seconds=0.01)

    async def run() -> _Response:
        async with server._admission:
            return await server.run_request(_request())

    response = asyncio.run(run())
    assert response.failure.stage == "admission" and response.failure.terminal is False
    assert _hops(response)[0] == {"exception_type": "TimeoutError"}


def test_the_episode_deadline_failure_carries_its_path() -> None:
    class _TimedOut(_EnvironmentServer):
        async def run(self, request, cleanup):
            await asyncio.sleep(60)

    server = _TimedOut(
        config=_environment_server(default_episode_timeout_seconds=0.01).config,
        server_client=MagicMock(spec=ServerClient),
    )
    response = asyncio.run(server.run_request(_request()))
    assert response.failure.failure_reason == "Episode timed out" and response.failure.terminal is False
    assert _hops(response) and _hops(response)[0]["exception_type"] == "TimeoutError"


def test_an_unhandled_server_error_carries_its_path() -> None:
    class _Broken(_EnvironmentServer):
        async def run(self, request, cleanup):
            try:
                raise ConnectionResetError("peer reset")
            except ConnectionResetError as err:
                raise ValueError("invalid protocol state") from err

    server = _Broken(config=_environment_server().config, server_client=MagicMock(spec=ServerClient))
    response = asyncio.run(server.run_request(_request()))
    assert response.failure.terminal is True
    assert _hops(response) == [{"exception_type": "ValueError"}, {"exception_type": "ConnectionResetError"}]


@pytest.mark.parametrize(
    "result, raised",
    [
        # The probe ran and failed: the probe's own error.
        ({"return_code": 2, "error_type": None}, SandboxProbeError),
        # The sandbox could not run the command at all: exec_as_root raises before the probe looks.
        ({"return_code": -1, "error_type": "timeout"}, SandboxUnavailableError),
    ],
)
async def test_a_failed_probe_names_its_kind_and_operation(result: dict, raised: type) -> None:
    async def exec(command: str, **_: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="", stderr="no such container", **result)

    with pytest.raises(raised) as info:
        await remote_kind(SimpleNamespace(exec=exec), "/logs/out")
    assert type(info.value) is raised
    assert getattr(info.value, "failure_kind", None) == "provider_unavailable"
    assert getattr(info.value, "operation", None) == "sandbox.exec"
