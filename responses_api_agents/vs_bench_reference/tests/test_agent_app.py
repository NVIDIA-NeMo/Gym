# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the Gym /run route through the actual upstream agent process."""

import asyncio
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient

from benchmarks.vs_bench.prepare import convert_row
from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.client import ServerClient
from responses_api_agents.vs_bench_reference.app import VSBenchAgentConfig, VSBenchReferenceAgent


AUTHOR = Path(os.environ.get("VS_BENCH_TEST_UPSTREAM", "/nonexistent"))
DATA = Path(os.environ.get("VS_BENCH_TEST_DATASET", "/nonexistent"))
PYTHON = os.environ.get("VS_BENCH_TEST_AUTHOR_PYTHON", "")
pytestmark = pytest.mark.skipif(
    not AUTHOR.is_dir() or not DATA.is_dir() or not Path(PYTHON).is_file(),
    reason="Pinned author checkout, dataset and 3.10 runtime required",
)


def test_gym_run_routes_author_calls_and_preserves_length_retry():
    raw = next(pq.ParquetFile(DATA / "reasoning/kuhn_poker.parquet").iter_batches(batch_size=1)).to_pylist()[0]
    prepared = convert_row("reasoning", "kuhn_poker", 0, raw)
    legal = json.loads(raw["next_legal_actions"])
    answer = json.dumps({"action": legal[str(raw["next_action"][0])]})
    model_calls = []

    async def policy_post(**kwargs):
        model_calls.append(kwargs)
        truncated = len(model_calls) == 1
        result = {
            "id": f"resp-{len(model_calls)}",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "status": "incomplete" if truncated else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if truncated else None,
            "output": [
                {
                    "id": "message",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "content": [{"type": "output_text", "text": "" if truncated else answer, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
        }
        reply = SimpleNamespace(ok=True, cookies={})
        reply.read = AsyncMock(return_value=json.dumps(result).encode())
        return reply

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client.post = AsyncMock(side_effect=policy_post)
    config = VSBenchAgentConfig(
        host="127.0.0.1",
        port=8080,
        entrypoint="",
        name="vs_bench_reference",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        author_root=str(AUTHOR),
        author_python=PYTHON,
        max_response_tokens=16384,
        max_reasoning_tokens=32000,
    )
    agent = VSBenchReferenceAgent(config=config, server_client=client)
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as agent_http:
        relay_client = MagicMock(spec=ServerClient)
        relay_client.global_config_dict = {}

        async def relay_post(**kwargs):
            path = kwargs["url_path"]
            if path == "/run":
                upstream = await asyncio.to_thread(
                    agent_http.post, path, content=kwargs["data"], headers={"content-type": "application/json"}
                )
            else:
                assert path == "/aggregate_metrics"
                upstream = await asyncio.to_thread(agent_http.post, path, json=kwargs["json"].model_dump(mode="json"))
            reply = SimpleNamespace(status=upstream.status_code, headers=upstream.headers)
            reply.ok = upstream.is_success
            reply.read = AsyncMock(return_value=upstream.content)
            return reply

        relay_client.post = AsyncMock(side_effect=relay_post)
        relay = LegacyAgentEnvironmentServer(
            config=LegacyAgentEnvironmentServerConfig(
                name="environment",
                host="127.0.0.1",
                port=8081,
                entrypoint="",
                agent_server={"type": "responses_api_agents", "name": "vs_bench_reference"},
            ),
            server_client=relay_client,
        )
        with TestClient(relay.setup_webserver(), raise_server_exceptions=False) as http:
            response = http.post("/run", json=prepared)
            assert response.status_code == 200, response.text
            aggregate = http.post("/aggregate_metrics", json={"verify_responses": [response.json()]})
            assert aggregate.status_code == 200, aggregate.text
            metrics = aggregate.json()["agent_metrics"]
            assert metrics["vs_bench/reasoning/kuhn_poker/count"] == 1
            assert metrics["vs_bench/reasoning/kuhn_poker/accuracy"] == 1.0
            assert metrics["vs_bench/reasoning/full_coverage"] is False
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["reward"] == 1.0
    assert result["author_source_variant"] == agent._author_source_variant
    assert result["action"] in raw["next_action"]
    assert result["author_model_calls"] == 2
    assert result["author_exception_types"] == ["MaxTokenLimit"]
    requests = [call["json"].model_dump(mode="json", exclude_none=True) for call in model_calls]
    assert [req["max_output_tokens"] for req in requests] == [24192, 48384]
    assert requests[0]["input"] == requests[1]["input"]
    assert [req["temperature"] for req in requests] == [1.0, 1.0]


def test_perception_agent_routes_unmodified_author_answer_to_scorer(monkeypatch):
    from resources_servers.vs_bench_reference.app import VSBenchReferenceConfig, VSBenchReferenceServer

    raw = next(pq.ParquetFile(DATA / "perception/kuhn_poker.parquet").iter_batches(batch_size=1)).to_pylist()[0]
    prepared = convert_row("perception", "kuhn_poker", 0, raw)

    async def score_fixture_only(_server, metadata, answer):
        # This bypass is exclusively for the known constant fixture; the HSG
        # integration executes the actual scorer under Bubblewrap instead.
        assert answer == "{}"
        worker = Path(__file__).resolve().parents[3] / "benchmarks/vs_bench/reference_perception_score.py"
        proc = await asyncio.create_subprocess_exec(
            PYTHON,
            str(worker),
            str(AUTHOR),
            str(DATA),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate(
            json.dumps({"game": metadata["game"], "row_index": metadata["row_index"], "answer": answer}).encode()
        )
        assert proc.returncode == 0, err.decode()
        return json.loads(out)["reward"]

    monkeypatch.setattr(VSBenchReferenceServer, "score_perception", score_fixture_only)
    resource_client = MagicMock(spec=ServerClient)
    resource_client.global_config_dict = {}
    resource = VSBenchReferenceServer(
        config=VSBenchReferenceConfig(
            name="resource", host="127.0.0.1", port=8002, entrypoint="", author_root=str(AUTHOR), author_python=PYTHON
        ),
        server_client=resource_client,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    agent = VSBenchReferenceAgent(
        config=VSBenchAgentConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="",
            name="vs_bench_reference",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            resources_server={"type": "resources_servers", "name": "resource"},
            author_root=str(AUTHOR),
            author_python=PYTHON,
            max_response_tokens=16384,
            max_reasoning_tokens=32000,
        ),
        server_client=client,
    )
    with TestClient(resource.setup_webserver(), raise_server_exceptions=False) as scorer:

        async def post(**kwargs):
            if kwargs["server_name"] == "resource":
                path = kwargs["url_path"]
                response = await asyncio.to_thread(scorer.post, path, json=kwargs["json"])
                reply = SimpleNamespace(
                    ok=response.is_success, cookies={}, status=response.status_code, headers=response.headers
                )
                reply.read = AsyncMock(return_value=response.content)
                return reply
            result = {
                "id": "fixture",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "status": "completed",
                "output": [
                    {
                        "id": "message",
                        "role": "assistant",
                        "status": "completed",
                        "type": "message",
                        "content": [{"type": "output_text", "text": "{}", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            }
            reply = SimpleNamespace(ok=True, cookies={})
            reply.read = AsyncMock(return_value=json.dumps(result).encode())
            return reply

        client.post = AsyncMock(side_effect=post)
        with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as http:
            response = http.post("/run", json=prepared)
            resource._author_source_variant = (
                "upstream" if agent._author_source_variant == "kaz_numpy_labels_v1" else "kaz_numpy_labels_v1"
            )
            mismatched = http.post("/run", json=prepared)
    assert response.status_code == 200, response.text
    assert response.json()["reward"] == 0.0
    assert response.json()["author_model_calls"] == 1
    assert response.json()["author_source_variant"] == agent._author_source_variant
    assert mismatched.status_code >= 400
