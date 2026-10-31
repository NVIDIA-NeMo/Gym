# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test support: a server whose model servers are scripted fakes, request builders, and Responses items."""

import copy
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from resources_servers.agentic_turn_simulation.app import (
    AgenticTurnSimulationResourcesServer,
    AgenticTurnSimulationResourcesServerConfig,
    AgenticTurnSimulationRunRequest,
    JudgeSpec,
    SimulateToolCallRequest,
)
from resources_servers.agentic_turn_simulation.workspace import FileToolSpec, FileToolWorkspace


SERVER = "agentic_turn_simulation"
# A judge verdict that finds the turn correct.
VERDICT = {
    "analysis": "fine",
    "addresses_concern": "yes",
    "correct": "yes",
    "blocked_by": "none",
    "request_questionable": "ok",
    "discard": "no",
    "discard_reason": "none",
    "reason": "ok",
}

Reply = MagicMock | Exception


def http(payload: Any = None, *, status: int = 200, body: str = "") -> MagicMock:
    """An aiohttp-like response: get_response_json reads .read(); error paths read .content.read()."""
    resp = MagicMock(status=status, cookies={})
    resp.read = AsyncMock(return_value=json.dumps(payload if payload is not None else {}).encode())
    resp.content.read = AsyncMock(return_value=body.encode())
    return resp


def chat(content: str | None, *, finish: str = "stop", **message: Any) -> MagicMock:
    return http(
        {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content, **message}}]}
    )


def judge_reply(verdict: dict[str, Any] | None = None, *, text: str | None = None, finish: str = "stop") -> MagicMock:
    return chat(text if text is not None else json.dumps(verdict), finish=finish)


def make_server(n_judges: int = 1, **overrides: Any) -> AgenticTurnSimulationResourcesServer:
    """A server whose model calls fail the test until `script` gives them replies."""
    cfg: dict[str, Any] = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name=SERVER,
        tool_simulator_model_server=ModelServerRef(type="responses_api_models", name="simulator"),
        tool_simulator_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[], max_tokens=1000, temperature=1.0, reasoning_effort="high"
        ),
        judges=[
            JudgeSpec(model_server=ModelServerRef(type="responses_api_models", name=f"judge_{i}"))
            for i in range(n_judges)
        ],
        judge_chat_completions_create_params=NeMoGymChatCompletionCreateParamsNonStreaming(
            messages=[], max_tokens=2000, temperature=1.0, reasoning_effort="high"
        ),
    )
    cfg.update(overrides)
    server = AgenticTurnSimulationResourcesServer(
        config=AgenticTurnSimulationResourcesServerConfig(**cfg), server_client=MagicMock(spec=ServerClient)
    )
    server.server_client.post = AsyncMock(side_effect=AssertionError("unexpected model call"))
    return server


def script(
    server: AgenticTurnSimulationResourcesServer, replies: list[Reply] | dict[str, list[Reply]]
) -> list[dict[str, Any]]:
    """Queue replies (per server name when a dict) and record a deep copy of every request body sent."""
    sent: list[dict[str, Any]] = []

    async def post(*, server_name: str, url_path: str, json: dict[str, Any], **_: Any) -> MagicMock:
        sent.append({"server_name": server_name, "url_path": url_path, "json": copy.deepcopy(json)})
        queue = replies[server_name] if isinstance(replies, dict) else replies
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    server.server_client.post = AsyncMock(side_effect=post)
    return sent


def req(session_id: str | None = "s1") -> SimpleNamespace:
    """A request in this session (None: a request without one)."""
    return SimpleNamespace(session={} if session_id is None else {SESSION_ID_KEY: session_id})


def function_call(name: str, call_id: str, arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments)
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def function_output(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id, "output": output}


def raw_call(name: str, arguments_text: str = "{}", call_id: str = "c1") -> dict[str, Any]:
    """A function call whose arguments are this text as it is (e.g. not JSON)."""
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments_text}


def raw_output(call_id: str, value: Any) -> dict[str, Any]:
    """A function call output whose output is this value as it is (e.g. not a string)."""
    return {"type": "function_call_output", "call_id": call_id, "output": value}


def text_message(role: str, text: str) -> dict[str, Any]:
    return {"type": "message", "role": role, "content": text}


def synthetic_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": f"Synthetic description of {name}.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        "strict": False,
    }


def harness_row(harness: str, version: str, tools: list[str], history: list[dict[str, Any]], **metadata: Any):
    """(responses_create_params, verifier_metadata) of a synthetic row for this harness."""
    rcp = NeMoGymResponseCreateParamsNonStreaming(
        input=[text_message("system", f"You are a synthetic {harness} agent."), *history],
        tools=[synthetic_tool(n) for n in tools],
    )
    return rcp, {"harness": harness, "harness_version": version, "record_kind": "interactive", **metadata}


async def seed_row(server: AgenticTurnSimulationResourcesServer, row) -> tuple[Any, Any]:
    """(what /harness then tells the agent, the session) for a harness_row seeded in session "s1"."""
    rcp, metadata = row
    await server.seed_session(
        req(), AgenticTurnSimulationRunRequest(responses_create_params=rcp, verifier_metadata=metadata)
    )
    return await server.describe_harness(req()), server._sessions["s1"]


def sim(name: str, call_id: str, arguments: Any, turn: list[dict[str, Any]] | None = None):
    """A /simulate_tool_call request for a call made at the end of the turn so far."""
    call = function_call(name, call_id, arguments)
    return SimulateToolCallRequest(call_id=call_id, name=name, arguments=call["arguments"], turn=(turn or []) + [call])


def workspace_after(spec: FileToolSpec, version: str | None, *pairs) -> FileToolWorkspace:
    """A workspace that has seen these (tool name, arguments, output) calls."""
    ws = FileToolWorkspace(spec, version)
    items = []
    for k, (name, args, output) in enumerate(pairs):
        items += [function_call(name, f"h{k}", args), function_output(f"h{k}", output)]
    ws.replay(items)
    return ws


def answer(spec: FileToolSpec, ws: FileToolWorkspace, name: str, args: dict[str, Any]) -> str | None:
    """The workspace's exact answer to a call, the row listing every file and shell tool."""
    return ws.answer(function_call(name, "c1", args), frozenset(spec.file_tools) | frozenset(spec.shell_tools))
