# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise Gym's gameplay endpoint with the original ten-episode poker loop."""

import asyncio
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
from responses_api_agents.vs_bench_gameplay import app as gameplay_app
from responses_api_agents.vs_bench_gameplay.app import GameplayConfig, GameplayRequest, VSBenchGameplayAgent


AUTHOR = Path(os.environ.get("VS_BENCH_TEST_UPSTREAM", "/nonexistent"))
PYTHON = os.environ.get("VS_BENCH_TEST_AUTHOR_PYTHON", "")
pytestmark = pytest.mark.skipif(
    not AUTHOR.is_dir() or not Path(PYTHON).is_file(), reason="Author checkout and runtime required"
)


@pytest.fixture
def gameplay_agent(tmp_path):
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
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
    return VSBenchGameplayAgent(config=config, server_client=client)


def test_gameplay_runs_actual_author_suite_with_fixture_policy(gameplay_agent):
    calls = []
    action = '{"action":"<PASS>"}'
    missing_usage_at = None

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
        if len(calls) == missing_usage_at:
            result["usage"] = None
        reply = SimpleNamespace(ok=True, cookies={})
        reply.read = AsyncMock(return_value=json.dumps(result).encode())
        return reply

    agent = gameplay_agent
    agent.server_client.post = AsyncMock(side_effect=policy)
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
        # Keep known usage before and after one missing record. A known subset
        # must not appear as the suite's complete token count.
        missing_usage_at = len(calls) + 2
        mixed_usage = http.post("/run", json=row)
        assert mixed_usage.status_code == 200, mixed_usage.text
        assert len(calls) > missing_usage_at
        assert mixed_usage.json()["response"]["usage"] is None
        assert len(mixed_usage.json()["terminal_returns"]) == 10
        source_log = json.loads(Path(mixed_usage.json()["source_results_path"]).read_text())
        assert any(
            info.get("token_info") is None and "reasoning" in info
            for name, episode in source_log.items()
            if name.startswith("episode_")
            for step, details in episode.items()
            if step.startswith("step_")
            for agent_name, info in details.items()
            if agent_name.startswith("agent_")
        )
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


@pytest.mark.parametrize("cancel_request", [False, True], ids=["route_failure", "request_cancelled"])
async def test_taskgroup_cancels_sibling_calls_and_reaps_worker(gameplay_agent, monkeypatch, cancel_request):
    original_spawn = asyncio.create_subprocess_exec
    children = []
    active, cancelled = set(), set()
    two_calls = asyncio.Event()
    call_count = 0

    async def spawn(*args, **kwargs):
        child = await original_spawn(*args, **kwargs)
        children.append(child)
        return child

    async def policy(**kwargs):
        nonlocal call_count
        call_count += 1
        call_id = call_count
        active.add(call_id)
        if call_count == 2:
            two_calls.set()
        try:
            if not cancel_request and call_id == 1:
                await two_calls.wait()
                raise RuntimeError("Unexpected fixture routing defect")
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.add(call_id)
            raise
        finally:
            active.remove(call_id)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    gameplay_agent.server_client.post = AsyncMock(side_effect=policy)
    body = GameplayRequest.model_validate(
        {
            "responses_create_params": {"input": "VS-Bench gameplay"},
            "verifier_metadata": {"track": "gameplay", "game": "kuhn_poker", "author_revision": AUTHOR_REVISION},
        }
    )
    run = asyncio.create_task(gameplay_agent.run(SimpleNamespace(cookies={}), body))
    try:
        await asyncio.wait_for(two_calls.wait(), 45)
        if cancel_request:
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(run, 10)
        else:
            with pytest.raises(ExceptionGroup, match="TaskGroup") as caught:
                await asyncio.wait_for(run, 10)
            assert any("Unexpected fixture routing defect" in str(error) for error in caught.value.exceptions)
        assert cancelled
        assert not active
        assert len(children) == 1 and children[0].returncode is not None
    finally:
        if not run.done():
            run.cancel()
        await asyncio.gather(run, return_exceptions=True)


# Test-only bootstrap: exercise the real author's fallback at one attempt
# without changing source files, production retry counts, or waiting for backoff.
SINGLE_ATTEMPT_BOOTSTRAP = """
import asyncio, runpy, sys, types
from pathlib import Path
worker, author = map(Path, sys.argv[1:])
sys.path[:0] = [str(worker.parent), str(author)]
for name in ('agents', 'envs', 'models'):
    package = types.ModuleType(name)
    package.__path__ = [str(author / name)]
    sys.modules[name] = package
from agents.prompt_agent import PromptAgent
original_init = PromptAgent.__init__
def initialize(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.max_errors = 1
async def no_backoff(*args, **kwargs):
    pass
PromptAgent.__init__ = initialize
asyncio.sleep = no_backoff
sys.argv = [str(worker), str(author)]
runpy.run_path(str(worker), run_name='__main__')
"""


async def test_exhausted_author_transport_retry_rejects_suite_with_successful_calls(gameplay_agent, monkeypatch):
    original_spawn = asyncio.create_subprocess_exec
    original_read = gameplay_app.read_frame
    complete = []
    calls = 0

    async def spawn(executable, worker, author, **kwargs):
        return await original_spawn(executable, "-c", SINGLE_ATTEMPT_BOOTSTRAP, worker, author, **kwargs)

    async def read_frame(reader):
        frame = await original_read(reader)
        if frame["type"] == "complete":
            complete.append(frame)
        return frame

    async def policy(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ClientConnectionError("Exhaust this action's transport retry allowance")
        result = {
            "id": f"r{calls}",
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
                    "content": [{"type": "output_text", "text": '{"action":"<PASS>"}', "annotations": []}],
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

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(gameplay_app, "read_frame", read_frame)
    gameplay_agent.server_client.post = AsyncMock(side_effect=policy)
    body = GameplayRequest.model_validate(
        {
            "responses_create_params": {"input": "VS-Bench gameplay"},
            "verifier_metadata": {"track": "gameplay", "game": "kuhn_poker", "author_revision": AUTHOR_REVISION},
        }
    )
    with pytest.raises(RuntimeError, match="policy transport failure"):
        await asyncio.wait_for(gameplay_agent.run(SimpleNamespace(cookies={}), body), 60)
    assert calls > 1
    assert len(complete) == 1
    assert complete[0]["transport_fallbacks"] == 1
    assert len(complete[0]["returns"]) == 10
