# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Gym's gameplay endpoint with the original ten-episode poker loop."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientConnectionError
from fastapi.testclient import TestClient

from benchmarks.vs_bench.prepare import AUTHOR_REVISION
from nemo_gym.client import ServerClient
from responses_api_agents.vs_bench_gameplay.app import GameplayConfig, VSBenchGameplayAgent


AUTHOR = Path(os.environ.get("VS_BENCH_TEST_UPSTREAM", "/nonexistent"))
PYTHON = os.environ.get("VS_BENCH_TEST_AUTHOR_PYTHON", "")
pytestmark = pytest.mark.skipif(
    not AUTHOR.is_dir() or not Path(PYTHON).is_file(), reason="Author checkout and runtime required"
)


def test_gameplay_runs_actual_author_suite_with_fixture_policy(tmp_path):
    calls = []
    action = '{"action":"<PASS>"}'

    async def policy(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ClientConnectionError("Transient fixture failure; the author must retry")
        selected = action
        if "a1a2" in action:
            prompt = "\n".join(
                part.get("text", "")
                for message in kwargs["json"].model_dump(mode="json")["input"]
                for part in message["content"]
            )
            legal = prompt.split("LEGAL ACTIONS:\n", 1)[1].split(".", 1)[0].split(", ")
            selected = json.dumps({"action": legal[0]})
        result = {
            "id": f"r{len(calls)}",
            "created_at": 0,
            "model": "fixture",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "id": "message",
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                    "content": [{"type": "output_text", "text": selected, "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        reply = SimpleNamespace(ok=True, cookies={})
        reply.read = AsyncMock(return_value=json.dumps(result).encode())
        return reply

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client.post = AsyncMock(side_effect=policy)
    config = GameplayConfig(
        name="gameplay",
        host="127.0.0.1",
        port=8002,
        entrypoint="",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        author_root=str(AUTHOR),
        author_python=PYTHON,
        episode_root=str(tmp_path),
        max_response_tokens=8192,
        max_reasoning_tokens=32000,
    )
    agent = VSBenchGameplayAgent(config=config, server_client=client)
    row = {
        "responses_create_params": {"input": "VS-Bench gameplay"},
        "verifier_metadata": {"track": "gameplay", "game": "kuhn_poker", "author_revision": AUTHOR_REVISION},
    }
    with TestClient(agent.setup_webserver(), raise_server_exceptions=True) as http:
        response = http.post("/run", json=row)
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result["terminal_returns"]) == 10
        assert result["model_seats"] == [1]
        assert result["author_model_calls"] == len(calls) > 0
        assert Path(result["source_results_path"]).is_file()
        assert result["response_scope"] == "suite_summary"
        assert result["response"]["usage"]["input_tokens"] == (len(calls) - 1) * 10
        assert result["response"]["usage"]["output_tokens"] == (len(calls) - 1) * 5
        assert result["reward"] == sum(row[1] for row in result["terminal_returns"]) / 10
        aggregated = http.post("/aggregate_metrics", json={"verify_responses": [result]})
        assert aggregated.status_code == 200, aggregated.text
        assert aggregated.json()["agent_metrics"]["vs_bench/gameplay/kuhn_poker/episodes"] == 10
        assert aggregated.json()["agent_metrics"]["vs_bench/gameplay/kuhn_poker/normalized_approx"] == pytest.approx(
            100 * (result["reward"] + 0.1) / 0.1
        )
        assert not aggregated.json().get("key_metrics")
        paper_row = {
            "responses_create_params": {"input": "VS-Bench gameplay"},
            "verifier_metadata": {
                "track": "gameplay",
                "game": "kuhn_poker",
                "author_revision": AUTHOR_REVISION,
                "protocol": "paper_competitive_v1",
                "paper_episode": {
                    "game": "kuhn_poker",
                    "block": 0,
                    "index": 0,
                    "environment_seed": 0,
                    "model_seat": 0,
                },
            },
        }
        paper_response = http.post("/run", json=paper_row)
        assert paper_response.status_code == 200, paper_response.text
        paper_result = paper_response.json()
        assert paper_result["model_seats"] == [0]
        assert len(paper_result["terminal_returns"]) == 1
        partial = http.post("/aggregate_metrics", json={"verify_responses": [paper_result]})
        assert partial.status_code == 200, partial.text
        assert partial.json()["agent_metrics"]["vs_bench/gameplay/kuhn_poker/episodes"] == 1
        assert "vs_bench/gameplay/kuhn_poker/normalized_approx" not in partial.json()["agent_metrics"]
        paper_row["verifier_metadata"]["paper_episode"].update(index=60, model_seat=1)
        seat_one = http.post("/run", json=paper_row)
        assert seat_one.status_code == 200, seat_one.text
        assert seat_one.json()["model_seats"] == [1]
        action = '{"action":"a1a2"}'
        for index, seat in ((0, 0), (10, 1)):
            paper_row["verifier_metadata"]["game"] = "breakthrough"
            paper_row["verifier_metadata"]["paper_episode"].update(game="breakthrough", index=index, model_seat=seat)
            bt = http.post("/run", json=paper_row)
            assert bt.status_code == 200, bt.text
            assert bt.json()["model_seats"] == [seat]
            assert bt.json()["reward"] in (-1, 1)
