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
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional
from uuid import uuid4

from fastapi import Request, Response
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ModelCallRef
from nemo_gym.server_utils import get_response_json, is_nemo_gym_fastapi_entrypoint, raise_for_status


LOG = logging.getLogger(__name__)

Topology = Literal["solo", "flat", "hier", "recursive", "board", "peer", "peer-dyn"]
DEADLINE_MSG = "deadline reached, answer now"
# Text that means the model tried to call a tool but the server's parser did not extract it.
MALFORMED = re.compile(r"<tool_call>|<function=|<\|tool_call")
INBOX_CHARS = 8000


class SwarmAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef
    model_server: ModelServerRef = Field(description="Model server of the root, the peers and the peer-0 reporter.")
    worker_model_server: Optional[ModelServerRef] = Field(
        default=None, description="Model server for spawned agents (leads and workers). Defaults to the root's."
    )
    topology: Topology = "flat"
    max_agents: int = Field(default=16, description="Cap on agents started beyond the root (peers: population cap).")
    max_depth: int = Field(default=4, description="recursive: deepest level that may still spawn.")
    peer_start: int = Field(default=4, description="peer-dyn: peers alive at t=0.")
    deadline_s: Optional[float] = Field(
        default=None, description="Wall-clock budget per rollout. Unset: the row's budget_s metadata, else 3600."
    )
    final_grace_s: Optional[float] = Field(
        default=None, description="Extra time for the root's final answer call. Unset: the row's grace_s, else 1800."
    )
    child_margin_s: Optional[float] = Field(
        default=None,
        description="Each depth level stops this much before its parent. Unset: deadline / 12 (300 s of 3600 s).",
    )
    root_turns: int = 400
    worker_turns: int = 30
    root_ctx: int = Field(default=262144, description="Context length of the root model, for the context guard.")
    worker_ctx: int = Field(default=262144, description="Context length of the worker model.")
    max_tokens: int = Field(default=16384, description="max_output_tokens of every normal call, root included.")
    final_max_tokens: int = Field(default=65536, description="max_output_tokens of the root's final answer call.")
    call_timeout_s: Optional[float] = Field(
        default=1800, description="Per-call timeout on normal turns. One retry, then the agent stops."
    )
    tool_chars: int = Field(default=8000, description="Tool results longer than this are truncated.")
    answer_chars: int = Field(default=1500, description="Each sub-agent answer returned to its parent.")
    digest_posts: int = Field(default=20, description="peer: newest posts to all shown per read or turn.")


class SwarmAgentRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class SwarmAgentVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")


class Deadline(Exception):
    pass


def trunc(text: Any, limit: int) -> str:
    text = "" if text is None else str(text)
    return text if len(text) <= limit else text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def fn_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
        "strict": False,
    }


def user_msg(text: str) -> dict:
    return {"type": "message", "role": "user", "content": text}


def message_text(item: dict) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    return "".join(p.get("text") or "" for p in content or [] if isinstance(p, dict))


SPAWN_TOOL = fn_tool(
    "spawn_agents",
    "Run one sub-agent per task string, concurrently. Each sub-agent sees only its task string (make it "
    "self-contained) and has the same task tools. Returns each sub-agent's final answer.",
    {"tasks": {"type": "array", "items": {"type": "string"}}},
    ["tasks"],
)
POOL_TOOL = fn_tool(
    "spawn_pool",
    "Start the whole worker pool. Every worker gets these same instructions plus its worker number and "
    "shares the message board. Returns when all workers finish; then read their results from the board.",
    {"instructions": {"type": "string"}},
    ["instructions"],
)
BOARD_TOOLS = [
    fn_tool("board_post", "Post a message to the shared board.", {"text": {"type": "string"}}, ["text"]),
    fn_tool(
        "board_read",
        "Read board posts with id > since_id (paged; follow next_since_id).",
        {"since_id": {"type": "integer"}},
        [],
    ),
    fn_tool(
        "board_claim",
        "Atomically claim a work unit. Returns 'ok' if you got it, else 'already-claimed'.",
        {"unit_id": {"type": "string"}},
        ["unit_id"],
    ),
]
PEER_TOOLS = [
    fn_tool(
        "post",
        "Post a message to the board, to one peer (e.g. 'peer-3') or to 'all'.",
        {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
        ["to", "subject", "body"],
    ),
    fn_tool(
        "read",
        "Read the board: posts to you plus posts to all. Without since_id: a digest of the newest posts. "
        "With since_id: the next page after that id (up to limit posts, ~8k chars; follow next_since_id).",
        {"since_id": {"type": "integer"}, "limit": {"type": "integer"}},
        [],
    ),
    fn_tool(
        "claim",
        "Atomically claim a work unit before working on it. Returns 'ok', or the peer that owns it.",
        {"unit_id": {"type": "string"}},
        ["unit_id"],
    ),
]
RECRUIT_TOOL = fn_tool(
    "recruit",
    "Add n new peers to the board. Each gets the task plus your brief as its first message; they are "
    "ordinary peers, not your workers, and coordinate only through the board.",
    {"n": {"type": "integer"}, "brief": {"type": "string"}},
    ["n", "brief"],
)


@dataclass
class Agent:
    id: int
    parent: Optional[int]
    depth: int
    role: str  # root, lead, worker, peer or reporter
    server: str
    deadline: float
    name: str = ""
    spawned_by: Optional[str] = None  # call_id of the spawn/pool/recruit call that started this agent
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    length_stops: int = 0
    ctx_tokens: int = 0  # input + output of the latest call
    cut: bool = False  # the latest reply was cut at max_output_tokens with no tool call
    answer: str = ""
    error: Optional[str] = None
    stop_reason: Optional[str] = None
    items: list = field(default_factory=list, repr=False)  # full transcript in Responses input items
    prompt_len: int = 0  # items before the agent's first reply
    response_ids: list = field(default_factory=list, repr=False)
    cookies: Any = field(default=None, repr=False)  # model server cookies of this agent's conversation
    inbox_seen: int = 0  # board length when this peer was last shown new posts
    t0: float = field(default_factory=time.monotonic, repr=False)
    wall_s: float = 0.0


class Swarm:
    """One rollout's agents, board and claims. Board ops run on the event loop, so check-and-set needs no lock."""

    def __init__(
        self,
        agent: "SwarmAgent",
        params: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_path: str,
        cookies: Any,
        deadline_s: float,
        grace_s: float,
    ) -> None:
        self.cfg = agent.config
        self.grace_s = grace_s
        self.client = agent.server_client
        self.model_path = model_path
        self.cookies = {k: getattr(v, "value", v) for k, v in (cookies or {}).items()}
        raw = params.model_dump(mode="json", exclude_unset=True)
        task = raw.pop("input")
        self.task_items = [user_msg(task)] if isinstance(task, str) else task
        self.task_tools = raw.pop("tools", [])
        raw.pop("metadata", None)
        raw.pop("tool_choice", None)
        self.call_params = raw
        self.agent_aware = {
            t["name"] for t in self.task_tools if "_agent" in (t.get("parameters") or {}).get("properties", {})
        }
        self.agents: list[Agent] = []
        self.spawned = 0  # agents started beyond the root, reserved before they run so the cap is never exceeded
        self.t0 = time.monotonic()
        self.deadline = self.t0 + deadline_s
        self.margin = self.cfg.child_margin_s if self.cfg.child_margin_s is not None else deadline_s / 12
        self.deadline_hit = False
        self.call_timeouts = 0
        self.board: list[dict] = []
        self.claims: dict[str, str] = {}
        self.duplicate_claims = 0
        self.peer_tasks: list[asyncio.Task] = []
        self.model = None

    @property
    def peers(self) -> bool:
        return self.cfg.topology in ("peer", "peer-dyn")

    def expired(self, a: Optional[Agent] = None) -> bool:
        if time.monotonic() >= (a.deadline if a else self.deadline):
            self.deadline_hit = True
            return True
        return False

    def child_deadline(self, depth: int) -> float:
        # Children stop earlier than their parent so their partial answers arrive before the parent's own deadline.
        return self.deadline - depth * self.margin

    def can_spawn(self, a: Agent) -> bool:
        t = self.cfg.topology
        return (
            (t == "flat" and a.depth == 0)
            or (t == "hier" and a.depth < 2)
            or (t == "recursive" and a.depth < self.cfg.max_depth)
        )

    def tools_for(self, a: Agent) -> list[dict]:
        tools = list(self.task_tools)
        if self.can_spawn(a):
            tools.append(SPAWN_TOOL)
        if self.cfg.topology == "board":
            tools += BOARD_TOOLS + ([POOL_TOOL] if a.depth == 0 else [])
        if a.role == "peer":
            tools += PEER_TOOLS + ([RECRUIT_TOOL] if self.cfg.topology == "peer-dyn" else [])
        return tools

    def system_prompt(self, a: Agent) -> str:
        c = self.cfg
        if a.role == "peer":
            who = (
                f"one of {c.max_agents} peers (peer-0 .. peer-{c.max_agents - 1})"
                if c.topology == "peer"
                else f"one of a growing group of peers (up to {c.max_agents}; recruit(n, brief) adds more)"
            )
            return (
                f"You are {a.name}, {who} working on the same task at the same time. There is no coordinator. "
                "Coordinate through the shared board: claim(unit_id) a unit before working on it (atomic; returns ok "
                "or its owner), post(to, subject, body) results and messages to a peer name or 'all', read(since_id) "
                "to catch up. New posts for you or for all are shown at the start of each turn. Post every result to "
                "'all' with a clear subject: peer-0 writes the final answer from the board alone, after all peers "
                "stop at the deadline. When you have nothing left to do, reply with a short summary and no tool "
                "call; that ends your work."
                + (
                    " You are peer-0, the reporter: work like the others; at the end you get one extra call with the "
                    "whole board to write the final answer."
                    if a.name == "peer-0"
                    else ""
                )
            )
        base = (
            "You are the root agent. Solve the user's task and reply with the final answer as plain text with no "
            "tool call."
            if a.depth == 0
            else "You are a sub-agent working on one part of a larger task. Do your part with the tools, then reply "
            "with a concise, complete final answer (plain text, no tool call); it is passed back to your parent."
        )
        if c.topology in ("flat", "hier", "recursive") and self.can_spawn(a):
            child = (
                "leads, which can spawn their own workers"
                if c.topology == "hier" and a.depth == 0
                else "sub-agents" + (", which can spawn further" if c.topology == "recursive" else "")
            )
            base += (
                f" You can delegate with spawn_agents (it runs {child} concurrently). The whole run may spawn at "
                f"most {c.max_agents} agents in total, so plan how to split the work. Give each sub-agent a "
                "self-contained task and ask for a compact, structured answer."
            )
        if c.topology == "board":
            base += (
                " A shared message board connects all agents: board_post, board_read, and board_claim to take a "
                "work unit exclusively before working on it."
            )
            if a.depth == 0:
                base += (
                    f" Call spawn_pool once with instructions for a pool of {c.max_agents} workers; tell them how "
                    "to pick and claim units and what to post. Then read the board and answer."
                )
        return base

    def new_agent(
        self, role: str, parent: Optional[Agent] = None, name: str = "", spawned_by: Optional[str] = None
    ) -> Agent:
        # Peers keep a child's time budget. The reporter is the root of the peer modes.
        depth = parent.depth + 1 if parent else (1 if role == "peer" else 0)
        worker = self.cfg.worker_model_server or self.cfg.model_server
        server = worker.name if role in ("lead", "worker") else self.cfg.model_server.name
        deadline = self.deadline if role == "reporter" else self.child_deadline(depth)
        a = Agent(len(self.agents), parent.id if parent else None, depth, role, server, deadline, name, spawned_by)
        self.agents.append(a)
        return a

    def budget(self, a: Agent, final: bool = False) -> int:
        # Root turns are capped like workers' turns: one runaway 65k-token reasoning call ate a whole flat-16 hour.
        return self.cfg.final_max_tokens if final and a.depth == 0 else self.cfg.max_tokens

    def ctx_left(self, a: Agent, since: int) -> int:
        # Tokens left before the model's context limit, estimating new items at ~3 chars per token.
        limit = self.cfg.worker_ctx if a.role in ("lead", "worker") else self.cfg.root_ctx
        new = sum(len(json.dumps(item)) for item in a.items[since:])
        return limit - a.ctx_tokens - new // 3 - 2000

    async def call(self, a: Agent, tools: list[dict], *, final: bool = False, max_tokens: int = 0) -> list[dict]:
        body = {**self.call_params, "input": a.items, "max_output_tokens": max_tokens or self.budget(a, final)}
        if tools:
            # Final calls keep the tools in the prompt so the cached prefix still matches.
            body |= {"tools": tools, "tool_choice": "none" if final else "auto"}
        resampled = stuck = False
        while True:
            now = time.monotonic()
            if not final:
                timeout = a.deadline - now
                if self.cfg.call_timeout_s:
                    timeout = min(timeout, self.cfg.call_timeout_s)
            elif a.depth == 0:
                timeout = max(a.deadline - now, 0) + self.grace_s
            else:
                timeout = min(self.margin / 2, self.deadline - now)
            if timeout <= 0:
                raise Deadline()
            try:
                data = await asyncio.wait_for(self.post_model(a, body), timeout)
            except asyncio.TimeoutError:
                if not final and time.monotonic() < a.deadline:  # the per-call timeout fired, not the deadline
                    self.call_timeouts += 1
                    if not stuck:
                        stuck = True  # one stuck call: retry once, then stop as if the time were up
                        continue
                self.deadline_hit = True
                raise Deadline()
            a.turns += 1
            usage = data.get("usage") or {}
            a.input_tokens += usage.get("input_tokens") or 0
            a.output_tokens += usage.get("output_tokens") or 0
            a.ctx_tokens = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
            output = data.get("output") or []
            has_calls = any(o.get("type") == "function_call" for o in output)
            cut = (data.get("incomplete_details") or {}).get("reason") == "max_output_tokens" and not has_calls
            # A reply cut at max_output_tokens with no tool call is usually a reasoning loop: resample once.
            if not cut:
                break
            a.length_stops += 1
            if resampled:
                break
            resampled = True
        a.cut = cut
        return output

    async def post_model(self, a: Agent, body: dict) -> dict:
        response = await self.client.post(server_name=a.server, url_path=self.model_path, json=body, cookies=a.cookies)
        await raise_for_status(response)
        data = await get_response_json(response)
        a.cookies = response.cookies
        if data.get("id"):
            a.response_ids.append(data["id"])
        self.model = data.get("model") or self.model
        return data

    async def run_agent(
        self,
        items: list[dict],
        role: str = "root",
        parent: Optional[Agent] = None,
        name: str = "",
        by: Optional[str] = None,
    ) -> str:
        a = self.new_agent(role, parent, name, by)
        tools = self.tools_for(a)
        a.items = merge_system(self.system_prompt(a), items)
        a.prompt_len = len(a.items)
        max_turns = self.cfg.root_turns if a.depth == 0 or role == "peer" else self.cfg.worker_turns
        last, since, stop = "", 0, None
        try:
            try:
                while a.turns < max_turns and not self.expired(a) and self.ctx_left(a, since) > self.budget(a):
                    if role == "peer":
                        self.inject_inbox(a)
                    output = await self.call(a, tools)
                    since = len(a.items)
                    a.items += output
                    calls = [o for o in output if o.get("type") == "function_call"]
                    content = "".join(message_text(o) for o in output if o.get("type") == "message")
                    last = content or last
                    if not calls:
                        if MALFORMED.search(content):
                            a.items.append(
                                user_msg(
                                    "Your tool call could not be parsed. Use the tool-calling format with valid "
                                    "JSON arguments, or give the final answer."
                                )
                            )
                            continue
                        if a.cut:  # cut twice at max_output_tokens: working notes, not an answer
                            a.items.append(
                                user_msg(
                                    "Your reply was cut off at the token limit. Do not restate your work; give the "
                                    "final answer now in the required format."
                                )
                            )
                            continue
                        if not content.strip():  # e.g. reasoning used up max_output_tokens
                            a.items.append(
                                user_msg(
                                    "Your reply was empty. Think briefly, then call a tool or write the final answer."
                                )
                            )
                            continue
                        a.answer, a.stop_reason = content, "answered"
                        return a.answer
                    results = await asyncio.gather(*(self.run_tool(a, c) for c in calls))
                    a.tool_calls += len(calls)
                    a.items += [
                        {"type": "function_call_output", "call_id": c["call_id"], "output": r}
                        for c, r in zip(calls, results)
                    ]
            except Deadline:
                stop = "deadline"
            # Out of turns, context or time: one final tool-free call. Sub-agents make it only while the run has
            # time left. Peers that reached their stop time just stop, since peer-0 reports from the board.
            a.stop_reason = stop or (
                "deadline" if self.expired(a) else "max_turns" if a.turns >= max_turns else "context"
            )
            if a.depth == 0 or not (self.expired() or (role == "peer" and self.expired(a))):
                note = {
                    "deadline": DEADLINE_MSG,
                    "max_turns": "Turn limit reached.",
                    "context": "Context limit reached.",
                }
                a.items.append(user_msg(f"{note[a.stop_reason]} Give your final answer now, without tool calls."))
                room = min(self.budget(a, final=True), max(1024, self.ctx_left(a, since)))
                try:
                    output = await self.call(a, tools, final=True, max_tokens=room)
                    a.items += output
                    a.answer = "".join(message_text(o) for o in output if o.get("type") == "message") or last
                except Deadline:
                    a.answer = last if a.depth == 0 else f"[stopped at deadline] {last}"
            else:
                a.answer = f"[stopped at deadline] {last}"
            return a.answer
        except Exception as e:  # A failed sub-agent or peer reports to its parent. A failed root fails the rollout.
            LOG.exception("swarm agent %s failed", a.id)
            a.error = f"{type(e).__name__}: {e}"[:1000]
            a.stop_reason = "error"
            if role == "root":
                raise
            a.answer = a.answer or f"[agent failed: {a.error[:300]}] {last}"
            return a.answer
        finally:
            a.wall_s = round(time.monotonic() - a.t0, 3)

    async def run_tool(self, a: Agent, call: dict) -> str:
        name = call.get("name") or ""
        raw = call.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            return f"error: could not parse arguments for {name}: {e}. Got: {str(raw)[:300]}"
        if self.expired(a):
            return DEADLINE_MSG
        try:
            if name == "spawn_agents" and self.can_spawn(a):
                return await self.spawn(a, args.get("tasks"), call["call_id"])
            if name == "spawn_pool" and self.cfg.topology == "board" and a.depth == 0:
                return await self.spawn_pool(a, str(args.get("instructions", "")), call["call_id"])
            if name in ("post", "read", "claim", "recruit") and a.role == "peer":
                return trunc(getattr(self, "peer_" + name)(a, args, call["call_id"]), self.cfg.tool_chars)
            if name.startswith("board_") and self.cfg.topology == "board":
                return trunc(getattr(self, name)(a, args), self.cfg.tool_chars)
            if any(t["name"] == name for t in self.task_tools):
                if name in self.agent_aware:
                    args = {**args, "_agent": f"agent-{a.id}"}
                return trunc(await self.resources_tool(name, args), self.cfg.tool_chars)
            return f"error: unknown tool {name!r}; available: {[t['name'] for t in self.tools_for(a)]}"
        except Exception as e:
            return f"error: {name} raised {type(e).__name__}: {str(e)[:500]}"

    async def resources_tool(self, name: str, args: dict) -> str:
        # Resources-server errors are valid model-visible tool outputs, as in simple_agent.
        response = await self.client.post(
            server_name=self.cfg.resources_server.name, url_path=f"/{name}", json=args, cookies=self.cookies
        )
        self.cookies.update({k: getattr(v, "value", v) for k, v in response.cookies.items()})
        return (await response.content.read()).decode(errors="replace")

    def reserve(self, k: int) -> int:
        n = max(0, min(k, self.cfg.max_agents - self.spawned))
        self.spawned += n
        return n

    async def spawn(self, a: Agent, tasks: Any, call_id: str) -> str:
        if not isinstance(tasks, list) or not tasks:
            return "error: tasks must be a non-empty list of strings"
        tasks = [t if isinstance(t, str) else json.dumps(t) for t in tasks]
        if time.monotonic() >= self.child_deadline(a.depth + 1):
            return "error: too little time left to start sub-agents; answer with what you have"
        n = self.reserve(len(tasks))
        role = "lead" if self.cfg.topology == "hier" and a.depth == 0 else "worker"
        answers = await asyncio.gather(*(self.run_agent([user_msg(t)], role, a, by=call_id) for t in tasks[:n]))
        out = [{"task": i, "answer": trunc(ans, self.cfg.answer_chars)} for i, ans in enumerate(answers)]
        out += [{"task": i, "answer": "not run: agent cap reached"} for i in range(n, len(tasks))]
        return json.dumps({"results": out, "agents_left": self.cfg.max_agents - self.spawned})

    async def spawn_pool(self, a: Agent, instructions: str, call_id: str) -> str:
        if time.monotonic() >= self.child_deadline(1):
            return "error: too little time left to start workers; read the board and answer"
        n = self.reserve(self.cfg.max_agents)
        if not n:
            return "error: agent cap reached, no workers started"
        await asyncio.gather(
            *(
                self.run_agent([user_msg(f"{instructions}\n\nYou are worker {i} of {n}.")], "worker", a, by=call_id)
                for i in range(n)
            )
        )
        return json.dumps(
            {
                "workers_run": n,
                "board_posts": len(self.board),
                "note": "workers finished; read their results with board_read",
            }
        )

    async def run_peers(self) -> str:
        # Fixed peers all start at t=0. peer-dyn starts peer_start peers that grow the population with recruit().
        start = self.cfg.peer_start if self.cfg.topology == "peer-dyn" else self.cfg.max_agents
        self.launch_peers(self.reserve(start), None, "", None)
        while not all(t.done() for t in self.peer_tasks):  # recruits add tasks while we wait
            await asyncio.gather(*self.peer_tasks)
        return await self.report(self.peer_tasks[0].result() if self.peer_tasks else "")

    def launch_peers(self, k: int, by: Optional[str], brief: str, call_id: Optional[str]) -> list[str]:
        names = [f"peer-{len(self.peer_tasks) + i}" for i in range(k)]
        items = self.task_items + ([user_msg(f"# Brief from {by}, who recruited you\n{brief}")] if by else [])
        for name in names:
            self.peer_tasks.append(asyncio.create_task(self.run_agent(items, "peer", name=name, by=call_id)))
        return names

    def peer_recruit(self, a: Agent, args: dict, call_id: str) -> str:
        if time.monotonic() >= self.child_deadline(1) - 60:
            return "error: too close to the stop time to recruit"
        n = self.reserve(max(0, int(args.get("n") or 0)))
        if not n:
            return f"error: population cap {self.cfg.max_agents} reached, nobody recruited"
        names = self.launch_peers(n, a.name, trunc(args.get("brief", ""), 4000), call_id)
        return f"recruited {n}: {names[0]} .. {names[-1]} ({self.cfg.max_agents - self.spawned} more allowed)"

    async def report(self, notes: str) -> str:
        # peer-0's extra call: the task, the whole board (most recent part if it is too long) and its own notes.
        a = self.new_agent("reporter", name="peer-0")
        cap = max(20000, min(600000, (self.cfg.root_ctx - self.cfg.final_max_tokens - 20000) * 3))
        board = "\n\n".join(self.fmt_post(p) for p in self.board)
        if len(board) > cap:
            board = f"[earliest {len(board) - cap} chars of the board omitted]\n" + board[-cap:]
        task = "\n\n".join(
            message_text(i) for i in self.task_items if i.get("role") in ("user", "system", "developer")
        )
        # One reporter once planned more lookups instead of answering (scored 0), so the prompt rules that out.
        a.items = [
            {
                "type": "message",
                "role": "system",
                "content": "You are peer-0, the reporter. All peers have stopped. This is your only reply: you have no "
                "tools, cannot read anything more, and get no further turn. Write the complete final answer to the "
                "task from the board posts and your own notes, in the format the task asks for. Where something is "
                "missing or disputed, give your best estimate instead of describing what you would do.",
            },
            user_msg(
                f"{task}\n\n# The board ({len(self.board)} posts)\n{board or '(empty)'}\n\n"
                f"# Your own notes as peer-0\n{notes or '(none)'}\n\nWrite the complete final answer now."
            ),
        ]
        try:
            output = await self.call(a, [], final=True)
            a.items += output
            a.answer = "".join(message_text(o) for o in output if o.get("type") == "message") or notes
            a.stop_reason = "answered"
        except Exception as e:
            a.error = f"{type(e).__name__}: {e}"[:1000]
            a.stop_reason = "error"
            a.answer = notes
        finally:
            a.wall_s = round(time.monotonic() - a.t0, 3)
        return a.answer

    def fmt_post(self, p: dict) -> str:
        return f"## {p['ts']} [FROM {p['from']} / TO {p['to']}] {p['subject']}\n{p['body']}"

    def visible(self, a: Agent, p: dict) -> bool:
        return p["from"] != a.name and p["to"] in ("all", a.name)

    def digest(self, a: Agent, posts: list[dict], limit: int, chars: int) -> str:
        # Direct messages first (all of them, newest kept if too long), then the most recent `limit` posts to all.
        broad = [p for p in posts if p["to"] == "all"]
        shown = [p for p in posts if p["to"] == a.name] + broad[-limit:]
        dropped = len(broad) - min(len(broad), limit)
        out, size = [], 0
        for p in reversed(shown):  # keep the newest when over the char cap
            text = self.fmt_post(p)
            if out and size + len(text) > chars:
                dropped += 1
                continue
            out.append((p["id"], text))
            size += len(text)
        out.sort()
        note = f"[{dropped} older posts not shown; page with read(since_id, limit)]\n" if dropped else ""
        return note + "\n\n".join(t for _, t in out)

    def inject_inbox(self, a: Agent) -> None:
        new = [p for p in self.board[a.inbox_seen :] if self.visible(a, p)]
        a.inbox_seen = len(self.board)
        if not new:
            return
        text = "New board posts for you since your last turn:\n" + self.digest(
            a, new, self.cfg.digest_posts, INBOX_CHARS
        )
        last = a.items[-1]
        if last.get("type") == "message" and last.get("role") == "user" and isinstance(last.get("content"), str):
            a.items[-1] = {**last, "content": last["content"] + "\n\n" + text}
        else:
            a.items.append(user_msg(text))

    def peer_post(self, a: Agent, args: dict, call_id: str) -> str:
        to = str(args.get("to") or "all").strip() or "all"
        self.board.append(
            {
                "id": len(self.board) + 1,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "from": a.name,
                "to": to,
                "subject": trunc(args.get("subject", ""), 200),
                "body": trunc(args.get("body", ""), 4000),
            }
        )
        return f"posted id={len(self.board)} to {to}"

    def peer_read(self, a: Agent, args: dict, call_id: str) -> str:
        # No since_id: a digest of the newest posts. With since_id: the next page, oldest first, up to `limit` posts.
        limit = max(1, min(int(args.get("limit") or self.cfg.digest_posts), 100))
        if args.get("since_id") is None:
            a.inbox_seen = len(self.board)
            posts = [p for p in self.board if self.visible(a, p)]
            return self.digest(a, posts, limit, self.cfg.tool_chars - 300) + f"\n[newest id={len(self.board)}]"
        since = max(0, int(args.get("since_id") or 0))
        out, size, last = [], 0, since
        for p in self.board[since:]:
            if self.visible(a, p):
                text = self.fmt_post(p)
                if out and (len(out) >= limit or size + len(text) > self.cfg.tool_chars - 300):
                    break
                out.append(text)
                size += len(text)
            last = p["id"]
        a.inbox_seen = max(a.inbox_seen, last)
        return "\n\n".join(out + [f"[next_since_id={last}; {len(self.board) - last} newer posts]"])

    def peer_claim(self, a: Agent, args: dict, call_id: str) -> str:
        unit = str(args.get("unit_id", "")).strip()
        owner = self.claims.setdefault(unit, a.name)
        if owner == a.name:
            return "ok"
        self.duplicate_claims += 1
        return f"owned by {owner}"

    def board_post(self, a: Agent, args: dict) -> str:
        self.board.append({"id": len(self.board) + 1, "agent": a.id, "text": trunc(args.get("text", ""), 4000)})
        return f"posted id={len(self.board)}"

    def board_read(self, a: Agent, args: dict) -> str:
        since = max(0, int(args.get("since_id") or 0))
        posts, size = [], 0
        for p in self.board[since:]:
            size += len(p["text"]) + 40
            if posts and size > self.cfg.tool_chars - 200:
                break
            posts.append(p)
        nxt = posts[-1]["id"] if posts else since
        return json.dumps({"posts": posts, "next_since_id": nxt, "remaining": len(self.board) - nxt})

    def board_claim(self, a: Agent, args: dict) -> str:
        unit = str(args.get("unit_id", ""))
        owner = self.claims.setdefault(unit, a.id)
        if owner == a.id:
            return "ok"
        self.duplicate_claims += 1
        return f"already-claimed (by agent {owner})"

    async def run(self) -> tuple[NeMoGymResponse, dict, AgentObservationBundle]:
        if self.peers:
            await self.run_peers()
            # The output is peer-0's transcript followed by the reporter's answer, which verify scores.
            peer0 = next((a for a in self.agents if a.role == "peer" and a.name == "peer-0"), None)
            output = (peer0.items[peer0.prompt_len :] if peer0 else []) + self.agents[-1].items[2:]
        else:
            await self.run_agent(self.task_items)
            output = self.agents[0].items[self.agents[0].prompt_len :]
        return self.response(output), self.summary(), self.observations()

    def response(self, output: list[dict]) -> NeMoGymResponse:
        usage = {
            "input_tokens": sum(a.input_tokens for a in self.agents),
            "output_tokens": sum(a.output_tokens for a in self.agents),
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return NeMoGymResponse.model_validate(
            {
                "id": f"resp_swarm_{uuid4().hex}",
                "created_at": time.time(),
                "model": self.model or self.cfg.model_server.name,
                "object": "response",
                "output": output,
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": self.task_tools,
                "usage": usage,
            }
        )

    def summary(self) -> dict:
        keys = ("id", "parent", "depth", "role", "name", "server", "turns", "tool_calls", "input_tokens")
        keys += ("output_tokens", "length_stops", "wall_s", "stop_reason", "error")
        return {
            "topology": self.cfg.topology,
            "n_agents": len(self.agents),
            "spawned": self.spawned,
            "max_depth": max((a.depth for a in self.agents), default=0),
            "wall_s": round(time.monotonic() - self.t0, 3),
            "deadline_s": round(self.deadline - self.t0, 3),
            "deadline_hit": self.deadline_hit,
            "call_timeouts": self.call_timeouts,
            "board_posts": len(self.board),
            "claims": len(self.claims),
            "duplicate_claims": self.duplicate_claims,
            "n_errors": sum(a.error is not None for a in self.agents),
            "agents": [{k: getattr(a, k) for k in keys} for a in self.agents],
        }

    def observations(self) -> AgentObservationBundle:
        def invocation(a: Agent) -> AgentInvocation:
            status = "failed" if a.error else "completed" if a.stop_reason == "answered" else "incomplete"
            return AgentInvocation(
                invocation_id=f"agent-{a.id}",
                parent_invocation_id=None if a.parent is None else f"agent-{a.parent}",
                spawned_by_tool_call_id=a.spawned_by,
                status=status,
                stop_reason=a.stop_reason,
                duration_ms=a.wall_s * 1000,
                error_type=a.error.split(":")[0] if a.error else None,
                model_calls=[
                    ModelCallRef(model_ref=ModelServerRef(type="responses_api_models", name=a.server), response_id=r)
                    for r in a.response_ids
                ],
                conversation=a.items,
            )

        return AgentObservationBundle(source="swarm_agent", records=[invocation(a) for a in self.agents])


def merge_system(system: str, items: list[dict]) -> list[dict]:
    # Chat templates take one leading system message, so a row's own system prompt follows the swarm's.
    first = items[0] if items else {}
    if first.get("role") in ("system", "developer") and isinstance(first.get("content"), str):
        return [{"type": "message", "role": "system", "content": f"{system}\n\n{first['content']}"}, *items[1:]]
    return [{"type": "message", "role": "system", "content": system}, *items]


def row_time(key: str, *sources: Any) -> float:
    # Row metadata such as SwarmBench's verifier_metadata.budget_s and grace_s, where 0 means none.
    return next((float(s[key]) for s in sources if isinstance(s, dict) and s.get(key)), 0.0)


class SwarmAgent(SimpleResponsesAPIAgent):
    config: SwarmAgentConfig

    async def _swarm(
        self,
        params: NeMoGymResponseCreateParamsNonStreaming,
        *,
        model_path: str,
        cookies: Any,
        meta: list[Any],
    ) -> tuple[NeMoGymResponse, dict, AgentObservationBundle, dict]:
        deadline_s = self.config.deadline_s or row_time("budget_s", *meta) or 3600
        grace_s = self.config.final_grace_s if self.config.final_grace_s is not None else row_time("grace_s", *meta)
        swarm = Swarm(
            self, params, model_path=model_path, cookies=cookies, deadline_s=deadline_s, grace_s=grace_s or 1800
        )
        response, summary, observations = await swarm.run()
        return response, summary, observations, swarm.cookies

    async def responses(
        self, request: Request, response: Response, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        result, _, _, cookies = await self._swarm(
            body,
            model_path=self.url_path_for_request("/v1/responses", request),
            cookies=request.cookies,
            meta=[body.metadata],
        )
        for k, v in cookies.items():
            response.set_cookie(k, v)
        return result

    async def run(self, request: Request, body: SwarmAgentRunRequest) -> SwarmAgentVerifyResponse:
        seed = await self.server_client.post(
            server_name=self.config.resources_server.name,
            url_path="/seed_session",
            json=body.model_dump(),
            cookies=request.cookies,
        )
        await raise_for_status(seed)
        extra = body.model_extra or {}
        response, summary, observations, cookies = await self._swarm(
            body.responses_create_params,
            model_path=self.url_path_for_run("/v1/responses", body),
            cookies=seed.cookies,
            meta=[extra.get("verifier_metadata"), body.responses_create_params.metadata],
        )
        response_json = response.model_dump(mode="json")
        if self.config.skip_verification:
            result = body.model_dump() | {
                "response": response_json,
                "reward": float(self.config.skip_verification_reward),
                "verification_skipped": True,
            }
        else:
            verify = await self.server_client.post(
                server_name=self.config.resources_server.name,
                url_path="/verify",
                json=body.model_dump() | {"response": response_json},
                cookies=cookies,
            )
            await raise_for_status(verify)
            result = await get_response_json(verify)
        return SwarmAgentVerifyResponse.model_validate(
            result | {"swarm": summary, "ng_agent_observations": observations.model_dump(mode="json")}
        )


if __name__ == "__main__":
    SwarmAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SwarmAgent.run_webserver()  # noqa: F401
