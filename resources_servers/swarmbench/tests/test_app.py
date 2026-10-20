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
import asyncio
import json
import os
import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from nemo_gym.mcp_auto_exposure import maybe_auto_expose
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.swarmbench.app import (
    CANARY,
    LATEST_VERSION,
    TRACKS,
    TYPE_DIRS,
    SwarmBenchResourcesServer,
    SwarmBenchResourcesServerConfig,
    SwarmBenchSeedSessionRequest,
    SwarmBenchVerifyRequest,
    final_answer,
    row_tools,
)
from resources_servers.swarmbench.tracks.corpus import track as corpus
from resources_servers.swarmbench.tracks.schedule.sched_world import slot_name


TOKEN_HEADER = "X-NeMo-Gym-Session-Token"
RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


def _server(tmp_path: Path) -> SwarmBenchResourcesServer:
    config = SwarmBenchResourcesServerConfig(
        host="127.0.0.1",
        port=12345,
        entrypoint="app.py",
        name="swarmbench",
        expose_tools_over_mcp=True,
        worlds_dir=str(tmp_path),
    )
    return SwarmBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _request(session_id: str) -> Request:
    request = MagicMock(spec=Request)
    request.session = {SESSION_ID_KEY: session_id}
    return request


def _meta(track: str, n: int = 16, seed: int = 0, **extra) -> dict:
    return {"track": track, "n": n, "seed": seed, **extra}


def _message(text: str) -> NeMoGymResponseOutputMessage:
    content = [NeMoGymResponseOutputText(text=text, annotations=[])]
    return NeMoGymResponseOutputMessage(
        id="msg", content=content, role="assistant", status="completed", type="message"
    )


def _verify_body(meta: dict, answer: str, output: list | None = None) -> SwarmBenchVerifyRequest:
    response = NeMoGymResponse(
        id="resp_1",
        created_at=0,
        model="test",
        object="response",
        output=output if output is not None else [_message(answer)],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
    return SwarmBenchVerifyRequest(
        responses_create_params={"input": [{"role": "user", "content": "task"}]},
        response=response,
        verifier_metadata=meta,
    )


async def _seed(server: SwarmBenchResourcesServer, session_id: str, meta: dict):
    await server.seed_session(_request(session_id), SwarmBenchSeedSessionRequest(verifier_metadata=meta))
    return await server._world(session_id)


def _gold(world) -> dict:
    with open(world.task["gold_path"]) as f:
        return json.load(f)


def _oracle(world) -> str:
    """The exact answer each track's scorer expects, built from its gold file (the code track edits the repo)."""
    track = world.meta.track
    if track == "code":
        gold = Path(world.task["gold_path"])
        for name in json.loads((gold / "meta.json").read_text())["units"]:
            shutil.copyfile(
                gold / "mods" / f"{name}.py", Path(world.task["world_dir"]) / "acme" / "mods" / f"{name}.py"
            )
        return "done"
    g = _gold(world)
    if track == "corpus":
        return corpus.oracle_answer(g["table"])
    if track == "search":
        return "\n".join(f"{c['id']} | {c['label']}" + (f" | {c['value']}" if c["value"] else "") for c in g["claims"])
    if track == "finance":
        return json.dumps(
            {
                "per_company": g["per_company"],
                "top20": [[c, round(v, 2)] for c, v in g["top20"]],
                "sector_medians": {s: round(v, 2) for s, v in g["sector_medians"].items()},
            }
        )
    if track == "dag":
        return json.dumps({t: g["values"][t] for t in g["targets"]})
    if track == "entities":
        return "\n".join(
            " ".join(e["records"]) + "".join(f" | {k}={v}" for k, v in e["profile"].items()) for e in g["entities"]
        )
    if track == "schedule":
        return json.dumps({it["id"]: [slot_name(it["slot"]), it["room"]] for it in g["items"]})
    if track == "rootcause":
        lines = []
        for inc in g["incidents"]:
            lines += [f"{inc['ticket']} | root | {inc['root']}", f"{inc['ticket']} | change | {inc['change']}"]
            lines += [f"{inc['ticket']} | edge | {a} -> {b}" for a, b in inc["edges"]]
        return "\n".join(lines)
    raise AssertionError(track)


BASE_TRACKS = sorted(set(TRACKS) - set(TYPE_DIRS))  # the types have their own tests (test_types.py)
VERSIONS = sorted({(t, 1) for t in BASE_TRACKS} | set(LATEST_VERSION.items()))


@pytest.mark.parametrize("track,version", VERSIONS)
@pytest.mark.asyncio
async def test_oracle_scores_one_and_empty_scores_zero(track: str, version: int, tmp_path: Path) -> None:
    server = _server(tmp_path)
    meta = _meta(track, track_version=version)
    oracle_world = await _seed(server, "oracle", meta)
    result = await server.verify(_request("oracle"), _verify_body(meta, _oracle(oracle_world)))
    assert result.reward == result.score_partial == result.score_exact == 1.0, result.score_detail
    assert (result.track, result.n, result.seed, result.track_version) == (track, 16, 0, version)
    assert not os.path.exists(oracle_world.root)  # verify releases the world

    await _seed(server, "empty", meta)
    result = await server.verify(_request("empty"), _verify_body(meta, ""))
    assert result.reward == result.score_exact == 0.0, result.score_detail


@pytest.mark.parametrize("track", BASE_TRACKS)
@pytest.mark.asyncio
async def test_worlds_are_deterministic_by_seed(track: str, tmp_path: Path) -> None:
    server = _server(tmp_path)
    a = await _seed(server, "a", _meta(track, seed=1))
    b = await _seed(server, "b", _meta(track, seed=1))
    c = await _seed(server, "c", _meta(track, seed=2))
    assert a.root != b.root
    assert a.task["prompt"] == b.task["prompt"]
    assert _gold_bytes(a) == _gold_bytes(b) != _gold_bytes(c)


def _gold_bytes(world) -> bytes:
    path = Path(world.task["gold_path"])
    return (path / "hidden_cases.json" if path.is_dir() else path).read_bytes()


@pytest.mark.asyncio
async def test_canary_is_in_the_world_and_the_prompt(tmp_path: Path) -> None:
    server = _server(tmp_path)
    world = await _seed(server, "a", _meta("dag", canary=True))
    assert world.task["prompt"].startswith(CANARY + "\n\n")
    assert (Path(world.task["world_dir"]) / "CANARY.txt").read_text() == CANARY + "\n"
    plain = await _seed(server, "b", _meta("dag"))
    assert not (Path(plain.task["world_dir"]) / "CANARY.txt").exists()


@pytest.mark.asyncio
async def test_concurrent_sessions_are_isolated(tmp_path: Path) -> None:
    server = _server(tmp_path)
    a, b = await asyncio.gather(_seed(server, "a", _meta("code")), _seed(server, "b", _meta("code")))
    module = Path(a.task["world_dir"]) / "UNITS.txt"
    unit = module.read_text().split()[0]
    path = f"acme/mods/{unit}.py"
    out = await server.call_tool("code_write_file", {"path": path, "content": "x = 1\n"}, _request("a"))
    assert out.startswith("wrote")
    assert (Path(a.task["world_dir"]) / path).read_text() == "x = 1\n"
    assert (Path(b.task["world_dir"]) / path).read_text() != "x = 1\n"

    # Scratch stores are per world too: a write in one schedule session is invisible to another.
    await asyncio.gather(_seed(server, "s1", _meta("schedule")), _seed(server, "s2", _meta("schedule")))
    await server.call_tool("schedule_scratch_write", {"key": "plan/a", "value": "{}"}, _request("s1"))
    assert "plan/a" in await server.call_tool("schedule_scratch_list", {}, _request("s1"))
    assert "plan/a" not in await server.call_tool("schedule_scratch_list", {}, _request("s2"))


@pytest.mark.asyncio
async def test_reseeding_a_session_keeps_its_world(tmp_path: Path) -> None:
    # OpenCode seeds its MCP tool server in the rollout's session; the edits made there are what verify scores.
    server = _server(tmp_path)
    world = await _seed(server, "a", _meta("code"))
    assert await _seed(server, "a", _meta("code")) is world
    _oracle(world)
    assert (await server.verify(_request("a"), _verify_body(_meta("code"), "done"))).reward == 1.0
    assert server._worlds == {}


@pytest.mark.asyncio
async def test_tool_errors_go_back_to_the_model(tmp_path: Path) -> None:
    server = _server(tmp_path)
    world = await _seed(server, "a", _meta("corpus"))
    assert "unknown tool" in await server.call_tool("search_search", {"query": "x"}, _request("a"))
    world.tools["grep"]["fn"] = MagicMock(side_effect=RuntimeError("boom"))
    assert await server.call_tool("corpus_grep", {"pattern": "x"}, _request("a")) == "error: RuntimeError: boom"
    world.tools["grep"]["fn"] = MagicMock(return_value="y" * 9000)
    assert "[truncated 1000 chars]" in await server.call_tool("corpus_grep", {"pattern": "x"}, _request("a"))
    with pytest.raises(HTTPException):
        await server.call_tool("corpus_list_docs", {}, _request("never-seeded"))


@pytest.mark.asyncio
async def test_budget_closes_the_tools_and_is_stamped(tmp_path: Path) -> None:
    server = _server(tmp_path)
    meta = _meta("dag", budget_s=0.2, grace_s=1)
    await _seed(server, "a", meta)
    first = await server.call_tool("dag_list_services", {}, _request("a"))
    assert first.endswith("[time used: 0 s of 0 s]") and not first.startswith("Time is up")
    time.sleep(0.3)
    closed = await server.call_tool("dag_list_services", {}, _request("a"))
    assert closed.startswith("Time is up") and "Tools are closed" in closed
    result = await server.verify(_request("a"), _verify_body(meta, ""))
    assert (result.tool_calls, result.tool_calls_after_budget, result.budget_s) == (2, 1, 0.2)
    assert result.elapsed_s >= 0.3

    # Without a budget nothing is stamped or closed, and a rollout without tool calls has no elapsed time.
    await _seed(server, "b", _meta("dag"))
    assert "time used" not in await server.call_tool("dag_list_services", {}, _request("b"))
    await _seed(server, "c", _meta("dag"))
    assert (await server.verify(_request("c"), _verify_body(_meta("dag"), ""))).elapsed_s is None


def test_final_answer_is_the_longest_text_after_the_last_tool_call() -> None:
    call = NeMoGymResponseFunctionToolCall(arguments="{}", call_id="c", name="dag_list_services", type="function_call")
    body = _verify_body(
        _meta("dag"), "", [_message("a long draft before tools"), call, _message("ANSWER"), _message("ok")]
    )
    assert final_answer(body) == "ANSWER"
    assert final_answer(_verify_body(_meta("dag"), "", [call])) == ""


@pytest.mark.asyncio
async def test_close_session_releases_world(tmp_path: Path) -> None:
    server = _server(tmp_path)
    world = await _seed(server, "a", _meta("dag"))
    await server.close_resources_session(_request("a"), MagicMock(resources_session_id="r"))
    assert not os.path.exists(world.root)
    assert not any(k.startswith(world.root) for k in TRACKS["dag"]._STORES)
    await server.close_resources_session(_request("a"), MagicMock(resources_session_id="r"))  # idempotent


def _rpc(client: TestClient, method: str, params: dict, token: str, rid: int) -> dict:
    headers = dict(RPC_HEADERS, **{TOKEN_HEADER: token})
    body = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}
    return client.post("/mcp", headers=headers, json=body).json()["result"]


@pytest.mark.parametrize("version,extra", [(1, []), (2, ["search_list_claims"])])
def test_mcp_lists_only_the_rollout_track_tools(version: int, extra: list, tmp_path: Path) -> None:
    server = _server(tmp_path)
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app, base_url="http://127.0.0.1:8000") as client:
        seed = client.post("/seed_session", json={"verifier_metadata": _meta("search", track_version=version)})
        assert seed.status_code == 200
        token = seed.json()["mcp"]["headers"][TOKEN_HEADER]
        init = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}}
        _rpc(client, "initialize", init, token, 1)
        names = sorted(t["name"] for t in _rpc(client, "tools/list", {}, token, 2)["tools"])
        assert names == sorted(["search_find_in_page", "search_open_page", "search_search"] + extra)

        result = _rpc(client, "tools/call", {"name": "search_search", "arguments": {"query": "founded"}}, token, 3)
        assert result.get("isError") is not True, result
        assert result["content"][0]["text"].strip() and not result["content"][0]["text"].startswith("error")

        denied = _rpc(client, "tools/call", {"name": "corpus_list_docs", "arguments": {}}, token, 4)
        assert denied.get("isError") is True


def test_compute_metrics_by_tier_and_track_with_cost(tmp_path: Path) -> None:
    server = _server(tmp_path)
    child = {"kind": "agent_invocation", "invocation_id": "c", "parent_invocation_id": "r"}
    root = {"kind": "agent_invocation", "invocation_id": "r", "parent_invocation_id": None}

    def row(tier: str | None, track: str, score: float, children: int = 0) -> dict:
        return {
            "tier": tier,
            "track": track,
            "n": 64,
            "reward": score,
            "score_partial": score,
            "score_exact": float(score == 1.0),
            "tool_calls": 10,
            "elapsed_s": 5.0,
            "response": {"usage": {"input_tokens": 100, "output_tokens": 20}},
            "ng_agent_observations": {"records": [root] + [child] * children},
        }

    # Each inner list is one instance with its repeats.
    metrics = server.compute_metrics(
        [
            [row("easy", "corpus", 1.0, 2), row("easy", "corpus", 0.5, 4)],
            [row("easy", "dag", 0.25)],
            [row("hard", "dag", 0.0)],
            [row(None, "corpus", 1.0)],  # a v1 row groups by size
        ]
    )
    assert metrics["easy/corpus/score_partial"] == 0.75 and metrics["easy/corpus/score_exact"] == 0.5
    assert metrics["easy/score_partial"] == pytest.approx((0.75 + 0.25) / 2)
    assert metrics["n64/corpus/score_partial"] == 1.0
    assert metrics["swarmbench/score_partial"] == pytest.approx((0.5 + 0.0 + 1.0) / 3)
    assert metrics["swarmbench/score_partial_ci95_lo"] <= metrics["swarmbench/score_partial"]
    assert metrics["swarmbench/score_partial_ci95_hi"] >= metrics["swarmbench/score_partial"]
    assert metrics["easy/corpus/subagents"] == 3 and metrics["easy/dag/subagents"] == 0
    assert metrics["easy/corpus/input_tokens"] == 100 and metrics["easy/corpus/tool_calls"] == 10
    assert metrics["swarmbench/score"] == metrics["swarmbench/score_partial"] and metrics["easy/corpus/score"] == 0.75
    assert set(server.get_key_metrics(metrics)) >= {"swarmbench/score", "swarmbench/score_exact", "easy/score"}


@pytest.mark.asyncio
async def test_row_tools_are_routed_and_declare_agent_ids(tmp_path: Path) -> None:
    # Agents that read tools from the row (swarm_agent) call them by these names and fill in declared `_agent`.
    tools = {t["name"]: t for t in row_tools("dag", 1)}
    assert "_agent" in tools["dag_post_value"]["parameters"]["properties"]
    assert (
        "_agent"
        in {t["name"]: t for t in row_tools("entities", 1)}["entities_scratch_write"]["parameters"]["properties"]
    )
    assert "_agent" not in tools["dag_get_value"]["parameters"]["properties"]

    server = _server(tmp_path)
    world = await _seed(server, "a", _meta("dag"))
    sid = (await server.call_tool("dag_list_services", {}, _request("a"))).splitlines()[1].split()[0]
    posted = await server.call_tool(
        "dag_post_value", {"service_id": sid, "value": 1, "_agent": "agent-7"}, _request("a")
    )
    assert not posted.startswith("error"), posted
    assert "agent-7" in await server.call_tool("dag_get_value", {"service_id": sid}, _request("a"))
    assert set(tools) == {f"dag_{name}" for name in world.tools}
