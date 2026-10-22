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
"""The rds tools over MCP, driven the way Claude Code drives them: seed, tools/list, tools/call, verify.

The calls run the real tools and the real scorer. Tests that fold need EternaFold and skip without it."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.rna_design.app import RnaDesignResourcesServer, RnaDesignResourcesServerConfig


TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _suite_target(name: str) -> dict:
    import yaml

    suite = (
        Path(__file__).resolve().parents[3]
        / "responses_api_agents/rna_design_agent/suites/rnainvbench_eterna100_v2.yaml"
    )
    return next(t for t in yaml.safe_load(suite.read_text())["targets"] if t["name"] == name)


TARGET = _suite_target("eterna100_v2_157")


def _have_eternafold() -> bool:
    try:
        import arnie  # noqa: F401
    except ImportError:
        return False
    return Path(os.environ.get("ARNIEFILE", "/nonexistent")).is_file()


needs_eternafold = pytest.mark.skipif(not _have_eternafold(), reason="EternaFold and arnie are not configured")


def _client(tmp_path: Path) -> TestClient:
    from nemo_gym.mcp_auto_exposure import maybe_auto_expose

    config = RnaDesignResourcesServerConfig(
        host="0.0.0.0", port=8080, entrypoint="", name="rna_design", expose_tools_over_mcp=True, work_dir=str(tmp_path)
    )
    server = RnaDesignResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    return TestClient(app, base_url="http://127.0.0.1:8000")


def _rpc(client: TestClient, method: str, params: dict | None = None, token: str | None = None, rid: int = 1) -> dict:
    headers = dict(RPC_HEADERS)
    if token:
        headers[TOKEN_HEADER] = token
    body = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        body["params"] = params
    return client.post("/mcp", headers=headers, json=body, follow_redirects=False).json()


def _seed(client: TestClient) -> str:
    seeded = client.post(
        "/seed_session",
        json={"responses_create_params": {"input": []}, "suite": "rnainvbench_eterna100_v2", "target": TARGET},
    )
    assert seeded.status_code == 200, seeded.text
    _rpc(
        client,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
    )
    client.post("/mcp", headers=RPC_HEADERS, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return seeded.json()["mcp"]["headers"][TOKEN_HEADER]


def _call(client: TestClient, token: str, name: str, arguments: dict, rid: int) -> dict:
    result = _rpc(client, "tools/call", {"name": name, "arguments": arguments}, token=token, rid=rid)["result"]
    return json.loads(result["content"][0]["text"])


def _verify(client: TestClient, usage: dict | None = None) -> dict:
    body = {
        "responses_create_params": {"input": []},
        "response": {
            "id": "r",
            "created_at": 0,
            "model": "m",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            **({"usage": usage} if usage else {}),
        },
        "suite": "rnainvbench_eterna100_v2",
        "target": TARGET,
    }
    reply = client.post("/verify", json=body)
    assert reply.status_code == 200, reply.text
    return reply.json()


def test_every_rds_tool_is_listed_with_its_own_schema(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        tools = {t["name"]: t for t in _rpc(client, "tools/list", {}, token=token, rid=2)["result"]["tools"]}
        _verify(client)
    assert {"generate_grnade", "score_fold_match", "score_constraints", "fold_eternafold"} <= set(tools)
    assert "sequences" in tools["score_fold_match"]["inputSchema"]["properties"]
    # operator knobs such as a write path stay off the schema, as they do for the agent's model
    assert "out_dir" not in tools["report_rules"]["inputSchema"]["properties"]
    # the suite withholds these, so the harness never sees them, as the agent's model never does
    assert not {"generate_inverse_fold", "pipeline_design_and_rank"} & set(tools)


def test_a_late_reply_from_an_abandoned_call_is_not_taken_for_the_next(tmp_path: Path) -> None:
    """A harness gave up on a slow call. Its reply arrives after the next request was sent and is skipped."""
    import asyncio

    from resources_servers.rna_design.app import ToolWorker

    script = tmp_path / "echo.py"
    script.write_text(
        "import json, sys\n"
        "for line in sys.stdin:\n"
        "    r = json.loads(line)\n"
        "    if r['op'] == 'slow':\n"
        "        continue\n"
        "    if r['op'] == 'score':\n"
        "        print(json.dumps({'ok': True, 'result': 'late', 'id': r['id'] - 1}), flush=True)\n"
        "    print(json.dumps({'ok': True, 'result': r['op'], 'id': r['id']}), flush=True)\n"
    )

    async def scenario() -> dict:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE
        )
        worker = ToolWorker(proc, timeout_s=0.5)
        with pytest.raises(asyncio.TimeoutError):
            await worker.ask({"op": "slow"})
        worker.timeout_s = 10
        try:
            return await worker.ask({"op": "score"})
        finally:
            await worker.close()

    assert asyncio.run(scenario())["result"] == "score"


@needs_eternafold
def test_a_tool_the_task_does_not_offer_is_refused(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        # this suite withholds the CPU generator from every arm, so the session's MCP token refuses it
        reply = _rpc(
            client,
            "tools/call",
            {"name": "generate_inverse_fold", "arguments": {"n": 5, "target": TARGET["name"]}},
            token=token,
            rid=2,
        )
        _verify(client)
    assert reply["result"]["isError"] and "not allowed for this session" in reply["result"]["content"][0]["text"]


@needs_eternafold
def test_designs_written_by_the_harness_are_scored_like_the_agents(tmp_path: Path) -> None:
    """A harness may write its own sequences and score them. They enter the pool, and verify scores the pool."""
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        state = _call(
            client,
            token,
            "score_fold_match",
            {"sequences": ["GGGGAAACCCCA", "AAAAAAAAAAAA"], "target": TARGET["name"]},
            rid=2,
        )
        _call(client, token, "score_constraints", {"sequences": "@all", "target": TARGET["name"]}, rid=3)
        result = _verify(client)
    assert state["population"]["n"] == 2
    assert result["pool_size"] == 2 and result["tool_calls"] == 2 and result["tool_failures"] == 0
    assert result["reward"] == (1.0 if result["solved"] else 0.0) and result["mask_sample"] is False


@needs_eternafold
def test_a_rollout_whose_every_call_failed_is_masked(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        _call(client, token, "score_fold_match", {"sequences": "@unscored", "target": TARGET["name"]}, rid=2)
        result = _verify(client)
    assert result["tool_calls"] == 1 and result["tool_failures"] == 1 and result["pool_size"] == 0
    assert result["mask_sample"] is True and result["failure_kind"] == "agent_run_error"


@needs_eternafold
def test_reading_the_campaign_state_is_free(tmp_path: Path) -> None:
    """The agent's model sees the state every round for free. A harness reads it with campaign_state."""
    pytest.importorskip("mcp")
    with _client(tmp_path) as client:
        token = _seed(client)
        names = {t["name"] for t in _rpc(client, "tools/list", {}, token=token, rid=2)["result"]["tools"]}
        state = _call(client, token, "campaign_state", {}, rid=3)
        again = _call(client, token, "campaign_state", {}, rid=4)
        result = _verify(client)
    assert "campaign_state" in names
    assert state["objective"]["target"] == TARGET["name"]
    # wall_seconds is a clock reading, not a charge: it advances between any two snapshots. Every axis
    # a call could spend must be unchanged.
    free = lambda b: {k: v for k, v in b["spend"].items() if k != "wall_seconds"}  # noqa: E731
    assert free(state["budget"]) == free(again["budget"])
    assert state["budget"]["limits"] == again["budget"]["limits"]
    assert result["tool_calls"] == 0 and result["rounds"] == 0


def test_a_rollout_that_never_reached_the_model_is_masked_not_scored(tmp_path: Path) -> None:
    """A dead model endpoint fails the harness before any tool call. That is no measurement of the model, so it is
    masked. A model that did answer and then chose to stop is a real result and scores 0."""
    pytest.importorskip("mcp")
    answered = {
        "input_tokens": 1200,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": 40,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": 1240,
    }
    with _client(tmp_path) as client:
        _seed(client)
        unreached = _verify(client)
    with _client(tmp_path) as client:
        _seed(client)
        stopped = _verify(client, usage=answered)
    assert unreached["tool_calls"] == 0 and unreached["mask_sample"] is True
    assert unreached["failure_reason"] == "the harness made no model call and no tool call"
    assert stopped["tool_calls"] == 0 and stopped["mask_sample"] is False and stopped["reward"] == 0.0
