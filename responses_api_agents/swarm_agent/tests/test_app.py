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
import re
from copy import deepcopy
from itertools import count
from unittest.mock import MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import ServerClient
from responses_api_agents.swarm_agent.app import SwarmAgent, SwarmAgentConfig, SwarmAgentRunRequest


VALUES = {f"u{i}": 100 + i for i in range(4)}
PROMPT = "Find the value of every unit: u0 u1 u2 u3. Answer with unit=value pairs."
TOOLS = [
    {
        "type": "function",
        "name": "lookup",
        "description": "Value of one unit.",
        "parameters": {"type": "object", "properties": {"unit_id": {"type": "string"}}, "required": ["unit_id"]},
        "strict": False,
    }
]
PAIR = re.compile(r"u\d+=\d+")


def text(item: dict) -> str:
    content = item.get("content")
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content or [])


class FakeResponse:
    def __init__(self, payload: object, cookies: dict | None = None) -> None:
        self.ok, self.status, self.cookies = True, 200, cookies or {}
        self.body = (payload if isinstance(payload, str) else json.dumps(payload)).encode()
        self.content = MagicMock()
        self.content.read = self.read

    async def read(self) -> bytes:
        return self.body


class ToyEnv:
    """A mock model server (scripted policy) and a mock resources server with a toy lookup track."""

    def __init__(self, policy=None, delay=None) -> None:
        self.policy = policy or toy_policy
        self.delay = delay  # (agent request) -> seconds to wait before replying
        self.ids = count()
        self.model_calls: list[tuple[str, dict]] = []
        self.tool_calls: list[tuple[dict, dict]] = []
        self.verify_calls: list[tuple[dict, dict]] = []

    async def post(self, server_name: str, url_path: str, json: dict, cookies: object = None) -> FakeResponse:
        if url_path.endswith("/v1/responses"):
            self.model_calls.append((server_name, deepcopy(json)))
            if self.delay:
                await asyncio.sleep(self.delay(json))
            output = self.policy(json)
            output = output if isinstance(output, dict) else {"items": output}
            n = next(self.ids)
            return FakeResponse(
                {
                    "id": f"resp_{n}",
                    "model": "toy",
                    "output": [dict(item, id=f"item_{n}_{i}") for i, item in enumerate(output["items"])],
                    "incomplete_details": {"reason": "max_output_tokens"} if output.get("cut") else None,
                    "usage": output.get("usage", {"input_tokens": 10, "output_tokens": 5}),
                }
            )
        if url_path == "/seed_session":
            return FakeResponse({}, {"session": "seeded"})
        if url_path == "/verify":
            self.verify_calls.append((json, cookies))
            answer = text([o for o in json["response"]["output"] if o.get("role") == "assistant"][-1])
            reward = sum(f"{k}={v}" in answer for k, v in VALUES.items()) / len(VALUES)
            return FakeResponse(json | {"reward": reward})
        self.tool_calls.append((json, cookies))
        unit = json.get("unit_id")
        return FakeResponse(f"{unit}={VALUES[unit]}" if unit in VALUES else f"error: no unit {unit!r}")


def call(name: str, **args: object) -> dict:
    return {"type": "function_call", "call_id": f"c_{name}_{json.dumps(args, sort_keys=True)}", "name": name,
            "arguments": json.dumps(args)}  # fmt: skip


def say(content: str) -> dict:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}


def found(items: list[dict]) -> str:
    seen = PAIR.findall(" ".join(str(i.get("output") or text(i)) for i in items if i.get("type") != "reasoning"))
    return ", ".join(sorted(set(seen)))


def toy_policy(req: dict) -> list[dict]:
    """A deterministic agent: split units across sub-agents when it can, else look them up, then report."""
    items, tools = req["input"], {t["name"] for t in req.get("tools", [])}
    system = text(items[0])
    task = text(next(i for i in items if i.get("role") == "user"))
    outputs = [i for i in items if i.get("type") == "function_call_output"]
    units = sorted(set(re.findall(r"u\d+", task)))
    if not tools or req.get("tool_choice") == "none":
        return [say(found(items))]
    if "claim" in tools:  # a peer
        k = int(re.search(r"You are peer-(\d+)", system).group(1))
        unit = f"u{k % 2}"
        if not outputs:
            recruit = [call("recruit", n=2, brief="take a unit")] if "recruit" in tools and k == 0 else []
            return [call("claim", unit_id=unit), *recruit]
        last = outputs[-1]["output"]
        if last == "ok" or last.startswith("recruited"):
            return [call("lookup", unit_id=unit)]
        if PAIR.fullmatch(last):
            return [call("post", to="all", subject="result", body=last)]
        return [say("done")]
    if "spawn_pool" in tools:  # board root
        if not outputs:
            return [call("spawn_pool", instructions="Claim one unit, look it up and post unit=value.")]
        if len(outputs) == 1:
            return [call("board_read", since_id=0)]
        return [say(found(items))]
    if "board_claim" in tools:  # board worker
        i = int(re.search(r"worker (\d+) of", task).group(1))
        unit = f"u{i % len(VALUES)}"
        if not outputs:
            return [call("board_claim", unit_id=unit)]
        last = outputs[-1]["output"]
        if last == "ok":
            return [call("lookup", unit_id=unit)]
        if PAIR.fullmatch(last):
            return [call("board_post", text=last)]
        return [say("done")]
    if not outputs:
        if "spawn_agents" in tools and len(units) > 1:
            # Roots whose children can spawn again split in halves. Everyone else gives one unit per sub-agent.
            half = (len(units) + 1) // 2
            parts = [units[:half], units[half:]] if "can spawn" in system else [[u] for u in units]
            return [call("spawn_agents", tasks=[f"Find {' '.join(p)}." for p in parts])]
        return [call("lookup", unit_id=u) for u in units]
    return [say(found(items))]


def make_agent(env: ToyEnv, **overrides: object) -> SwarmAgent:
    config = SwarmAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="swarm",
        resources_server=ResourcesServerRef(type="resources_servers", name="toy"),
        model_server=ModelServerRef(type="responses_api_models", name="root_model"),
        worker_model_server=ModelServerRef(type="responses_api_models", name="worker_model"),
        **overrides,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    client.post = env.post
    return SwarmAgent(config=config, server_client=client)


async def run(env: ToyEnv, **overrides: object) -> dict:
    row = {"responses_create_params": {"input": [{"role": "user", "content": PROMPT}], "tools": TOOLS}}
    row |= {"verifier_metadata": {"budget_s": overrides.pop("budget_s", 600)}}
    overrides.setdefault("child_margin_s", 1)
    body = SwarmAgentRunRequest.model_validate(row)
    result = await make_agent(env, **overrides).run(MagicMock(cookies={}), body)
    return result.model_dump(mode="json")


@pytest.mark.parametrize(
    "topology,max_agents,n_agents",
    [
        ("solo", 0, 1),
        ("flat", 16, 5),  # root + one worker per unit
        ("hier", 16, 7),  # root + 2 leads + 4 workers
        ("recursive", 16, 7),
        ("board", 4, 5),  # root + 4 pool workers
        ("peer", 4, 5),  # 4 peers + the reporter
        ("peer-dyn", 4, 5),  # 2 starting peers + 2 recruits + the reporter
    ],
)
async def test_every_topology_solves_the_toy_track(topology: str, max_agents: int, n_agents: int) -> None:
    env = ToyEnv()
    if topology in ("peer", "peer-dyn"):
        # Peers only know u0 and u1 (they claim u{k % 2}), so the full answer is half the units.
        expected = 0.5
    else:
        expected = 1.0
    result = await run(env, topology=topology, max_agents=max_agents, peer_start=2)

    swarm = result["swarm"]
    assert result["reward"] == expected
    assert swarm["n_agents"] == n_agents
    assert swarm["n_errors"] == 0
    assert len(env.verify_calls) == 1
    # Every tool call carried the seeded session cookies.
    assert env.tool_calls and all(cookies == {"session": "seeded"} for _, cookies in env.tool_calls)


async def test_spawned_agents_use_the_worker_model_server() -> None:
    env = ToyEnv()
    await run(env, topology="flat")
    servers = [(server, text(req["input"][0]).startswith("You are the root")) for server, req in env.model_calls]
    assert {s for s, root in servers if root} == {"root_model"}
    assert {s for s, root in servers if not root} == {"worker_model"}


async def test_agent_cap_is_never_exceeded() -> None:
    env = ToyEnv()
    result = await run(env, topology="flat", max_agents=2)
    assert result["swarm"]["n_agents"] == 3
    spawn_output = next(i["output"] for i in result["response"]["output"] if i.get("type") == "function_call_output")
    answers = [r["answer"] for r in json.loads(spawn_output)["results"]]
    assert answers.count("not run: agent cap reached") == 2
    assert result["reward"] == 0.5


async def test_peer_dyn_recruits_up_to_the_population_cap() -> None:
    env = ToyEnv()
    result = await run(env, topology="peer-dyn", max_agents=3, peer_start=2)
    peers = [a for a in result["swarm"]["agents"] if a["role"] == "peer"]
    assert len(peers) == 3
    recruit_out = [
        i["output"]
        for inv in result["ng_agent_observations"]["records"]
        for i in inv["conversation"]
        if i.get("type") == "function_call_output" and i["output"].startswith("recruited")
    ]
    assert recruit_out == ["recruited 1: peer-2 .. peer-2 (0 more allowed)"]


async def test_claims_are_atomic() -> None:
    # Four peers claim u0/u1: exactly one owner per unit, the other two see the owner.
    env = ToyEnv()
    result = await run(env, topology="peer", max_agents=4)
    swarm = result["swarm"]
    assert swarm["claims"] == 2
    assert swarm["duplicate_claims"] == 2
    claim_replies = sorted(
        i["output"]
        for inv in result["ng_agent_observations"]["records"]
        for i in inv["conversation"]
        if i.get("type") == "function_call_output" and ("owned by" in i["output"] or i["output"] == "ok")
    )
    assert claim_replies == ["ok", "ok", "owned by peer-0", "owned by peer-1"]


async def test_turn_cap_forces_a_tool_free_final_answer() -> None:
    def stubborn(req: dict) -> list[dict]:
        if req.get("tool_choice") == "none":
            return [say(found(req["input"]))]
        return [call("lookup", unit_id=f"u{len(req['input']) % 4}")]

    env = ToyEnv(stubborn)
    result = await run(env, topology="solo", root_turns=3, max_tokens=111, final_max_tokens=999)
    reqs = [req for _, req in env.model_calls]
    assert len(reqs) == 4
    assert [r["max_output_tokens"] for r in reqs] == [111, 111, 111, 999]
    assert reqs[-1]["tool_choice"] == "none" and reqs[-1]["tools"]  # tools stay in the prompt
    assert text(reqs[-1]["input"][-1]).startswith("Turn limit reached.")
    assert result["swarm"]["agents"][0]["stop_reason"] == "max_turns"
    assert PAIR.search(text(result["response"]["output"][-1]))


async def test_truncated_reply_is_not_accepted_as_final() -> None:
    def rambler(req: dict) -> object:
        nudged = any("cut off" in text(i) for i in req["input"] if i.get("role") == "user")
        if not nudged:
            return {"items": [say("u0=1 and I keep thinking")], "cut": True}
        return [say("u0=100, u1=101, u2=102, u3=103")]

    env = ToyEnv(rambler)
    result = await run(env, topology="solo")
    root = result["swarm"]["agents"][0]
    assert root["length_stops"] == 2  # the cut reply was resampled once, then nudged
    assert len(env.model_calls) == 3
    assert result["reward"] == 1.0


async def test_context_guard_ends_with_a_final_call() -> None:
    def big_context(req: dict) -> object:
        if req.get("tool_choice") == "none":
            return [say("u0=100")]
        return {"items": [call("lookup", unit_id="u0")], "usage": {"input_tokens": 400, "output_tokens": 100}}

    env = ToyEnv(big_context)
    result = await run(env, topology="solo", root_ctx=3000, max_tokens=500)
    assert len(env.model_calls) == 2
    assert result["swarm"]["agents"][0]["stop_reason"] == "context"
    assert text(env.model_calls[-1][1]["input"][-1]).startswith("Context limit reached.")


async def test_deadline_stops_children_first_and_root_still_answers() -> None:
    def slow_workers(req: dict) -> float:
        return 5.0 if text(req["input"][0]).startswith("You are a sub-agent") else 0.0

    env = ToyEnv(delay=slow_workers)
    result = await run(env, topology="flat", budget_s=1.0, child_margin_s=0.5, final_grace_s=1.0)
    swarm = result["swarm"]
    assert swarm["deadline_s"] == 1.0  # taken from the row's budget_s
    assert swarm["deadline_hit"]
    workers = [a for a in swarm["agents"] if a["role"] == "worker"]
    assert workers and all(w["stop_reason"] == "deadline" and w["wall_s"] < 1.0 for w in workers)
    spawn_output = next(i["output"] for i in result["response"]["output"] if i.get("type") == "function_call_output")
    assert all(r["answer"].startswith("[stopped at deadline]") for r in json.loads(spawn_output)["results"])
    assert len(env.verify_calls) == 1


async def test_stuck_call_is_retried_once_then_the_agent_stops() -> None:
    def hang(req: dict) -> float:
        return 0.0 if req.get("tool_choice") == "none" else 5.0

    env = ToyEnv(delay=hang)
    result = await run(env, topology="solo", call_timeout_s=0.1)
    swarm = result["swarm"]
    assert swarm["call_timeouts"] == 2
    assert swarm["agents"][0]["stop_reason"] == "deadline"
    assert len(env.model_calls) == 3  # two timed-out tries and the final answer call


async def test_transcripts_and_tree_are_exported() -> None:
    env = ToyEnv()
    result = await run(env, topology="hier")
    bundle = AgentObservationBundle.model_validate(result["ng_agent_observations"])
    invocations = {r.invocation_id: r for r in bundle.records}
    swarm_agents = result["swarm"]["agents"]
    assert len(invocations) == len(swarm_agents) == 7
    root = invocations["agent-0"]
    leads = [r for r in invocations.values() if r.parent_invocation_id == "agent-0"]
    workers = [r for r in invocations.values() if r.parent_invocation_id in {lead.invocation_id for lead in leads}]
    assert root.parent_invocation_id is None and len(leads) == 2 and len(workers) == 4
    spawn_ids = {i.call_id for i in root.conversation if i.type == "function_call"}
    assert {lead.spawned_by_tool_call_id for lead in leads} <= spawn_ids
    assert all(r.status == "completed" and r.model_calls and r.conversation for r in invocations.values())
    # Per-agent token usage adds up to the response usage, and the response output is the root's transcript.
    usage = result["response"]["usage"]
    assert sum(a["input_tokens"] for a in swarm_agents) == usage["input_tokens"] == 10 * len(env.model_calls)
    assert [i.get("type") for i in result["response"]["output"]] == [
        "function_call",
        "function_call_output",
        "message",
    ]


async def test_verify_gets_the_final_answer_and_seeded_cookies() -> None:
    env = ToyEnv()
    result = await run(env, topology="flat")
    payload, cookies = env.verify_calls[0]
    assert cookies == {"session": "seeded"}
    assert payload["verifier_metadata"] == {"budget_s": 600}
    assert text(payload["response"]["output"][-1]) == "u0=100, u1=101, u2=102, u3=103"
    assert result["reward"] == 1.0


async def test_child_margin_scales_with_a_short_row_budget() -> None:
    # 300 s of budget with the 3600 s reference margin would leave sub-agents no time at all.
    env = ToyEnv()
    result = await run(env, topology="flat", budget_s=300, child_margin_s=None)
    assert result["swarm"]["n_agents"] == 5
    assert result["reward"] == 1.0


async def test_failed_worker_reports_to_its_parent_and_failed_root_fails_the_rollout() -> None:
    def flaky(req: dict) -> object:
        if text(req["input"][1]) == "Find u1.":
            raise RuntimeError("model server down")
        return toy_policy(req)

    result = await run(ToyEnv(flaky), topology="flat")
    failed = [a for a in result["swarm"]["agents"] if a["error"]]
    assert [a["stop_reason"] for a in failed] == ["error"] and result["reward"] == 0.75

    def broken(req: dict) -> object:
        raise RuntimeError("model server down")

    with pytest.raises(RuntimeError, match="model server down"):
        await run(ToyEnv(broken), topology="solo")
