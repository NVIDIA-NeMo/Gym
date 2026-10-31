# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

import aiohttp
import httpx
import pytest
from aiohttp import ClientSession, ClientTimeout, TraceConfig
from omegaconf import OmegaConf

import nemo_gym.client.http_client as transport
import nemo_gym.rollout_collection as collection
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollout_recovery import IncompleteEvaluationError
from nemo_gym.rollout_store import RolloutStore
from tests.single_agent_turn_helpers import _environment_server, _request


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("lost_response", ["disconnect", "timeout"])
async def test_persisted_collection_never_replays_a_completed_episode(tmp_path, monkeypatch, lost_response, routed):
    environment, dependencies = _environment_server()
    dependencies.responses.extend(list(dependencies.responses))
    requests, results = [], []
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
            response = await endpoint.post("/run", json=body)
            assert response.status_code == 200, response.text
            results.append(response.json())
            if len(requests) != 1:
                payload = response.content
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                    + str(len(payload)).encode()
                    + b"\r\nConnection: close\r\n\r\n"
                    + payload
                )
                await writer.drain()
            if len(requests) == 1 and lost_response == "timeout":
                await asyncio.sleep(0.15)
            # The first episode finished, but its entire response was lost.
            writer.close()
            await writer.wait_closed()

        tcp = await asyncio.start_server(handle, "127.0.0.1", 0)
        global_config = OmegaConf.create(
            {
                "environment": {
                    "environment_servers": {
                        "single_agent_turn": {"host": "127.0.0.1", "port": tcp.sockets[0].getsockname()[1]}
                    }
                }
            }
        )
        client = ServerClient(
            head_server_config=BaseServerConfig(host="head", port=1), global_config_dict=global_config
        )
        assert not client.checkpoint_enabled()  # Default configuration; no checkpoint override.
        monkeypatch.setattr(collection, "setup_server_client_utils", lambda *args, **kwargs: client)
        source = tmp_path / "input.jsonl"
        source.write_text(_request().task.model_dump_json() + "\n")
        (tmp_path / "artifacts").mkdir()
        output = tmp_path / "artifacts" / "rollouts.jsonl"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(source),
            output_jsonl_fpath=str(output),
            environment_server_routes={"test:source": "environment"},
            route_failures_to_sidecar=routed,
            require_complete=False,
            disable_aggregation=True,
            disable_health_check=True,
        )
        async with tcp, ClientSession(timeout=ClientTimeout(total=0.08)) as session:
            monkeypatch.setattr(transport, "get_global_aiohttp_client", lambda: session)
            expected_error = IncompleteEvaluationError if routed else (aiohttp.ClientError, asyncio.TimeoutError)
            with pytest.raises(expected_error):
                await asyncio.wait_for(RolloutCollectionHelper().run_from_config(config), timeout=5)
            # A second normal collection invocation must not regenerate the completed but lost episode.
            try:
                await asyncio.wait_for(
                    RolloutCollectionHelper().run_from_config(config.model_copy(update={"resume_from_cache": True})),
                    timeout=5,
                )
            except IncompleteEvaluationError:
                pass
            await asyncio.sleep(0.16)

    agent_calls = [call for call in dependencies.calls if call[1].endswith("/v1/responses")]
    seeds = [call[2]["json"]["agent_session_id"] for call in dependencies.calls if call[1] == "/v1/agent_sessions"]
    # Session seeding is idempotent only for the same ID. A replay creates fresh handles.
    assert len(requests) == len(agent_calls) == len(seeds) == 1
    store = RolloutStore.read(output)
    assert store.coverage()["measured"] == 0
    assert store.coverage()["attempts"] == 1
    assert store.manifest.next_attempt["0-0"] == 1
    assert store.coverage()["max_rollout_attempts"] == 3
    assert store.pending() == []
    failures = [json.loads(line) for line in output.with_name("rollouts_failures.jsonl").read_text().splitlines()]
    assert len(failures) == 1
    assert failures[0]["_ng_failure_type"] == (
        "ServerDisconnectedError" if lost_response == "disconnect" else "TimeoutError"
    )
    record = failures[0]["_ng_failure_record"]
    assert record["delivery"] == "possibly_delivered"
    assert record["source"] == "collector"
    assert record["episode_id"]["attempt"] == 0
    selected = failures[0]["_ng_failure_events"][failures[0]["_ng_failure_id"]]
    assert selected["boundary"] == "collection"
    assert selected["operation"] == "collection.run"
    assert failures[0].get("_ng_failure_counts_as_zero") is not True
    assert failures[0].get("_ng_failure_terminal") is not True


@pytest.mark.parametrize("routed", [False, True])
async def test_persisted_collection_bounds_never_connected_attempts(tmp_path, monkeypatch, routed):
    listener = await asyncio.start_server(lambda *_: None, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    listener.close()
    await listener.wait_closed()
    config_dict = OmegaConf.create(
        {"environment": {"environment_servers": {"single_agent_turn": {"host": "127.0.0.1", "port": port}}}}
    )
    client = ServerClient(head_server_config=BaseServerConfig(host="head", port=1), global_config_dict=config_dict)
    monkeypatch.setattr(collection, "setup_server_client_utils", lambda *args, **kwargs: client)
    source = tmp_path / "input.jsonl"
    source.write_text(_request().task.model_dump_json() + "\n")
    (tmp_path / "artifacts").mkdir()
    output = tmp_path / "artifacts" / "rollouts.jsonl"
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(source),
        output_jsonl_fpath=str(output),
        environment_server_routes={"test:source": "environment"},
        route_failures_to_sidecar=routed,
        require_complete=False,
        disable_aggregation=True,
        disable_health_check=True,
    )
    connections = []
    trace = TraceConfig()

    async def on_connect(*args):
        connections.append(True)

    trace.on_connection_create_start.append(on_connect)
    async with ClientSession(trace_configs=[trace]) as session:
        monkeypatch.setattr(transport, "get_global_aiohttp_client", lambda: session)
        with pytest.raises(IncompleteEvaluationError if routed else aiohttp.ClientConnectorError):
            await asyncio.wait_for(RolloutCollectionHelper().run_from_config(config), timeout=2)
    assert len(connections) == 1
    failure = json.loads(output.with_name("rollouts_failures.jsonl").read_text())
    assert failure["_ng_failure_type"] == "ClientConnectorError"
    assert failure["_ng_failure_record"]["delivery"] == "not_sent"
    assert failure["_ng_failure_record"]["episode_id"]["attempt"] == 0
    assert failure.get("_ng_failure_counts_as_zero") is not True
    assert RolloutStore.read(output).coverage()["attempts"] == 1
    assert RolloutStore.read(output).manifest.next_attempt["0-0"] == 1
    assert len(RolloutStore.read(output).pending()) == 1
    # One transport attempt was reserved, but a proven not-sent interruption spends no model/failure attempt.
    assert RolloutStore.read(output).coverage()["counted_failures"] == 0
