# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI service wrapping the deepagents agent (./langchain_agent) as a Gym remote_agent.

    uv run uvicorn service:app --host 0.0.0.0 --port 9000

Contract: fern/versions/latest/pages/agent-server/remote-agent-frameworks.mdx. `params["tools"]`
is ignored on purpose — deepagents brings its own tools (TavilySearch) and runs them internally,
so this service never asks Gym to execute anything; it always answers in one call.

Bare minimum on purpose: only the final assistant message is reported. deepagents' internal tool
calls (tavily_search, etc.) are invisible to `gym eval profile` as a result — see the "optional"
paired function_call/function_call_output pattern in the doc above if that legibility is wanted.
"""

import time
import uuid

from agent import agent, model
from fastapi import FastAPI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


app = FastAPI()


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def to_langchain(input_items: list) -> list:
    roles = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage, "developer": SystemMessage}
    return [
        roles.get(item.get("role", "user"), HumanMessage)(content=_text(item.get("content", "")))
        for item in input_items
        if item.get("type", "message") == "message"
    ]


def to_responses(new_messages: list) -> dict:
    final_text = next((m.content for m in reversed(new_messages) if isinstance(m, AIMessage) and m.content), "")
    return {
        "id": f"resp_{uuid.uuid4().hex}",
        "created_at": time.time(),
        "model": model.model_name,
        "object": "response",
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": _text(final_text), "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tools": [],
        "tool_choice": "auto",
    }


@app.post("/v1/responses")
async def responses(params: dict) -> dict:
    input_messages = to_langchain(params["input"])
    result = await agent.ainvoke({"messages": input_messages})
    new_messages = result["messages"][len(input_messages) :]
    return to_responses(new_messages)
