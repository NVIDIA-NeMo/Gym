# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""A /run whose connection is refused was never delivered: collection resends it within a bounded grace period."""

import asyncio
import json
import socket
import time
from pathlib import Path

import httpx
import pytest
from aiohttp import ClientSession
from omegaconf import OmegaConf

import nemo_gym.client.http_client as transport
import nemo_gym.rollouts.rollout_collection as collection
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.rollouts.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollouts.rollout_recovery import IncompleteEvaluationError
from nemo_gym.rollouts.rollout_store import RolloutStore
from tests.helpers.single_agent_turn import _environment_server, _request


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _config(tmp_path, monkeypatch, port: int, **overrides) -> RolloutCollectionConfig:
    global_config = OmegaConf.create(
        {"environment": {"environment_servers": {"single_agent_turn": {"host": "127.0.0.1", "port": port}}}}
    )
    client = ServerClient(head_server_config=BaseServerConfig(host="head", port=1), global_config_dict=global_config)
    monkeypatch.setattr(collection, "setup_server_client_utils", lambda *args, **kwargs: client)
    source = tmp_path / "input.jsonl"
    source.write_text(_request().task.model_dump_json() + "\n")
    (tmp_path / "artifacts").mkdir()
    return RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(tmp_path / "artifacts" / "rollouts.jsonl"),
        environment_server_routes={"test:source": "environment"},
        require_complete=False,
        disable_aggregation=True,
        disable_health_check=True,
        **overrides,
    )


async def test_a_run_refused_until_its_server_listens_is_delivered_within_the_grace_period(tmp_path, monkeypatch):
    """Refused connections never delivered /run, so collection keeps resending it until the server listens."""
    environment, _ = _environment_server()
    port = _free_port()
    config = _config(tmp_path, monkeypatch, port, route_failures_to_sidecar=False)
    requests = []
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=environment.setup_webserver()), base_url="http://environment"
    ) as endpoint:

        async def handle(reader, writer):
            headers = await reader.readuntil(b"\r\n\r\n")
            length = int(
                next(
                    line.split(b":", 1)[1]
                    for line in headers.split(b"\r\n")
                    if line.lower().startswith(b"content-length:")
                )
            )
            body = json.loads(await reader.readexactly(length))
            requests.append(body)
            payload = (await endpoint.post("/run", json=body)).content
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + payload
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        async def listen_late():
            # Well past the two strict attempts (about 0.5 s) that a refused connection used to get.
            await asyncio.sleep(2.0)
            return await asyncio.start_server(handle, "127.0.0.1", port)

        async with ClientSession() as session:
            monkeypatch.setattr(transport, "get_global_aiohttp_client", lambda: session)
            server = asyncio.create_task(listen_late())
            try:
                await asyncio.wait_for(RolloutCollectionHelper().run_from_config(config), timeout=30)
            finally:
                (await server).close()

    assert len(requests) == 1
    store = RolloutStore.read(Path(config.output_jsonl_fpath))
    assert store.coverage()["measured"] == 1 and store.coverage()["attempts"] == 1


async def test_a_run_refused_past_the_grace_period_is_saved_as_never_sent(tmp_path, monkeypatch):
    """The grace period is bounded and configurable: past it, the refused /run is a not-sent failure."""
    config = _config(tmp_path, monkeypatch, _free_port(), route_failures_to_sidecar=True, dispatch_connect_grace_s=1.5)
    async with ClientSession() as session:
        monkeypatch.setattr(transport, "get_global_aiohttp_client", lambda: session)
        started = time.monotonic()
        with pytest.raises(IncompleteEvaluationError):
            await asyncio.wait_for(RolloutCollectionHelper().run_from_config(config), timeout=30)
        elapsed = time.monotonic() - started

    assert 1.5 <= elapsed < 10
    failures = config.output_jsonl_fpath.replace(".jsonl", "_failures.jsonl")
    [row] = [json.loads(line) for line in open(failures) if line.strip()]
    assert row["_ng_failure_record"]["delivery"] == "not_sent"
