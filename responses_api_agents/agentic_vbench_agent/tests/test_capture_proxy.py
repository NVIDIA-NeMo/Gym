# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from nemo_gym.token_id_capture.completion import completion_items, output_item_fingerprint
from responses_api_agents.agentic_vbench_agent.capture_proxy import (
    classify_agent,
    CaptureProxy,
    capture_output,
    load_records,
    redact_images,
)


def completion(turn: int, message: dict, finish: str, usage: int = 7) -> dict:
    return {
        "id": f"chatcmpl-{turn}",
        "object": "chat.completion",
        "created": 1,
        "model": "policy",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage, "completion_tokens": 3, "total_tokens": usage + 3},
    }


class Upstream:
    """Stands in for Gym's rollout-scoped policy route."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, dict, dict]] = []
        self.responses: list = []
        self.runner: web.AppRunner | None = None
        self.url = ""

    async def chat(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.requests.append((request.path, body, dict(request.headers)))
        item = self.responses.pop(0)
        if isinstance(item, int):
            return web.json_response({"detail": "boom"}, status=item)
        return web.json_response(item)

    async def models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [], "path": request.path})

    async def start(self) -> None:
        app = web.Application()
        app.router.add_post("/ng-rollout/{rollout}/training-token-capture/v1/chat/completions", self.chat)
        app.router.add_get("/ng-rollout/{rollout}/training-token-capture/v1/models", self.models)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", 0).start()
        self.url = f"http://127.0.0.1:{self.runner.addresses[0][1]}/ng-rollout/r-7/training-token-capture"


def parse_sse(text: str) -> list:
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data: ")
        payload = block[len("data: ") :]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


@pytest.mark.asyncio
async def test_proxy_records_and_replays_streamed_calls(tmp_path: Path) -> None:
    upstream = Upstream()
    await upstream.start()
    tool_call = {"id": "call_a1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}
    served = [
        completion(
            0,
            {"role": "assistant", "content": None, "tool_calls": [tool_call], "reasoning_content": "think"},
            "tool_calls",
        ),
        completion(1, {"role": "assistant", "content": "Project renamed"}, "stop"),  # OpenCode's title call
        completion(2, {"role": "assistant", "content": "## Goal\n- do it"}, "stop"),  # compaction summary
        completion(3, {"role": "assistant", "content": "Done."}, "stop"),
    ]
    upstream.responses = list(served)
    proxy = CaptureProxy(advertise_host="127.0.0.1")
    await proxy.start()
    try:
        endpoint = proxy.register("ep1", upstream.url, tmp_path / "calls")
        assert endpoint.startswith("http://127.0.0.1:") and endpoint.endswith("/avb/ep1/v1")
        async with aiohttp.ClientSession() as client:
            # Streamed call with usage requested, as OpenCode's openai-compatible provider sends.
            async with client.post(
                f"{endpoint}/chat/completions",
                json={
                    "model": "policy",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hi"},
                                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                            ],
                        }
                    ],
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
                },
                headers={"x-opencode-agent": "build", "x-opencode-session": "ses_1"},
            ) as response:
                assert response.status == 200
                assert response.content_type == "text/event-stream"
                events = parse_sse(await response.text())
            assert events[-1] == "[DONE]"
            assert any(e.get("usage") for e in events if isinstance(e, dict)), "usage chunk requested"
            assert any(
                e["choices"] and e["choices"][0]["delta"].get("tool_calls") for e in events if isinstance(e, dict)
            )
            for agent in ("title", "compaction"):
                async with client.post(
                    f"{endpoint}/chat/completions",
                    json={"model": "policy", "messages": [{"role": "user", "content": "x"}], "stream": True},
                    headers={"x-opencode-agent": agent},
                ) as response:
                    assert response.status == 200
            async with client.post(
                f"{endpoint}/chat/completions",
                json={"model": "policy", "messages": [{"role": "user", "content": "x"}]},
                headers={"x-opencode-agent": "build"},
            ) as response:
                assert response.status == 200 and response.content_type == "application/json"
                assert (await response.json())["id"] == "chatcmpl-3"
            async with client.get(f"{endpoint}/models") as response:
                assert response.status == 200
                assert (await response.json())["path"].endswith("/training-token-capture/v1/models")
        proxy.unregister("ep1")
        async with aiohttp.ClientSession() as client:
            async with client.post(f"{endpoint}/chat/completions", json={}) as response:
                assert response.status == 404
    finally:
        await proxy.stop()
        await upstream.runner.cleanup()
    # Upstream saw non-streaming requests on the rollout-prefixed route, with the agent tag.
    paths = {path for path, _, _ in upstream.requests}
    assert paths == {"/ng-rollout/r-7/training-token-capture/v1/chat/completions"}
    assert all("stream" not in body and "stream_options" not in body for _, body, _ in upstream.requests)
    assert upstream.requests[0][2].get("x-opencode-agent") == "build"
    records = load_records(tmp_path / "calls")
    assert [r["turn"] for r in records] == [0, 1, 2, 3]
    assert [r["agent"] for r in records] == ["build", "title", "compaction", "build"]
    assert records[0]["streamed"] and not records[3]["streamed"]
    assert records[0]["request"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;sha256,")
    items, stats = capture_output(records)
    assert stats.model_calls == 4 and stats.ignored_calls == 1 and stats.compaction_summaries == 1
    assert stats.terminal_response_id == "chatcmpl-3" and stats.complete and stats.generated_tokens == 9
    # Parity with the capture ledger's conversion of the same served completions.
    expected = [item for index in (0, 2, 3) for item in completion_items(served[index])]
    assert len(items) == len(expected) == 3
    assert [output_item_fingerprint(i) for i in items] == [output_item_fingerprint(i) for i in expected]
    assert items[0]["type"] == "function_call" and items[0]["id"] == "call_a1" and items[0]["call_id"] == "call_a1"
    without, stats = capture_output(records, include_compaction_summaries=False)
    assert len(without) == 2 and stats.compaction_summaries == 1


@pytest.mark.asyncio
async def test_proxy_forwards_upstream_errors_and_marks_failed_calls(tmp_path: Path) -> None:
    upstream = Upstream()
    await upstream.start()
    upstream.responses = [422, completion(1, {"role": "assistant", "content": "Done."}, "stop")]
    proxy = CaptureProxy(advertise_host="127.0.0.1")
    await proxy.start()
    try:
        endpoint = proxy.register("ep2", upstream.url, tmp_path / "calls")
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"{endpoint}/chat/completions", json={"model": "policy", "messages": [], "stream": True}
            ) as response:
                assert response.status == 422
                assert (await response.json()) == {"detail": "boom"}
            async with client.post(
                f"{endpoint}/chat/completions", json={"model": "policy", "messages": []}
            ) as response:
                assert response.status == 200
    finally:
        await proxy.stop()
        await upstream.runner.cleanup()
    records = load_records(tmp_path / "calls")
    assert records[0]["status"] == 422 and "boom" in records[0]["error"] and "response" not in records[0]
    items, stats = capture_output(records)
    assert stats.failed_calls == 1 and len(items) == 1 and stats.terminal_response_id == "chatcmpl-1"


def test_capture_output_requires_records_and_ids(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="No model calls"):
        load_records(tmp_path)
    record = {
        "turn": 0,
        "agent": "build",
        "status": 200,
        "response": completion(0, {"role": "assistant", "content": "x"}, "stop"),
    }
    del record["response"]["id"]
    with pytest.raises(ValueError, match="missing its id"):
        capture_output([record])
    assert redact_images("not a list") == "not a list"


def test_classify_agent_recognizes_untagged_compaction_requests() -> None:
    # Stock OpenCode 1.14.39 sends no x-opencode-agent header: a task turn carries
    # the tool set, the compaction request carries none and one of two fixed prompts.
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    task = {"messages": [{"role": "system", "content": "You are opencode"}, {"role": "user", "content": "Do it"}], "tools": tools}
    fresh = {
        "messages": [
            {"role": "system", "content": "You are an anchored context summarization assistant"},
            {"role": "user", "content": "Create a new anchored summary from the conversation history above.\nOutput..."},
        ]
    }
    update = {
        "messages": [
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "text", "text": "Update the anchored summary below using the conversation history above. Keep"}]},
        ],
        "tools": [],
    }
    assert classify_agent(None, task) is None
    assert classify_agent(None, fresh) == "compaction"
    assert classify_agent(None, update) == "compaction"
    # A tool-less plain user turn (e.g. the post-compaction continue prompt with tools stripped) is not a summary.
    assert classify_agent(None, {"messages": [{"role": "user", "content": "Continue if you have next steps"}]}) is None
    # The header still wins when OpenCode sends one.
    assert classify_agent("title", fresh) == "title"
    assert classify_agent("build", task) == "build"
