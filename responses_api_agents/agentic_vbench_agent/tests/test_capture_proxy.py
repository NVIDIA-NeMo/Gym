# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web


# Gym's completion fingerprint helpers ship with the framework-owned capture route
# (Gym PR #3368); branches without it still run the proxy tests that do not compare
# fingerprints.
_completion = pytest.importorskip(
    "nemo_gym.token_id_capture.completion", reason="framework-owned capture helpers not on this Gym"
)
completion_items = _completion.completion_items
output_item_fingerprint = _completion.output_item_fingerprint
from responses_api_agents.agentic_vbench_agent.capture_proxy import (
    CaptureProxy,
    capture_output,
    classify_agent,
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
    routes = "encoded-native-routes" * 100_000
    for index in (0, 3):
        served[index]["choices"][0]["routed_experts"] = routes
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
                forwarded = await response.json()
                assert forwarded == served[3]
                assert forwarded["choices"][0]["routed_experts"] == routes
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
    for index in (0, 3):
        choice = records[index]["response"]["choices"][0]
        assert "routed_experts" not in choice
        assert choice["routed_experts_summary"] == {
            "encoded_chars": len(routes),
            "sha256_utf8": hashlib.sha256(routes.encode()).hexdigest(),
        }
        assert (tmp_path / "calls" / f"{index:04d}.json").stat().st_size < 10_000
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
    task = {
        "messages": [{"role": "system", "content": "You are opencode"}, {"role": "user", "content": "Do it"}],
        "tools": tools,
    }
    fresh = {
        "messages": [
            {"role": "system", "content": "You are an anchored context summarization assistant"},
            {
                "role": "user",
                "content": "Create a new anchored summary from the conversation history above.\nOutput...",
            },
        ]
    }
    update = {
        "messages": [
            {"role": "user", "content": "Do it"},
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Update the anchored summary below using the conversation history above. Keep",
                    }
                ],
            },
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


def test_step_cap_request_detection_and_local_completion() -> None:
    from responses_api_agents.agentic_vbench_agent.capture_proxy import (
        STEP_CAP_AGENT,
        is_step_cap_request,
        step_cap_completion,
    )

    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    cap = {
        "model": "policy",
        "tools": tools,
        "messages": [
            {"role": "user", "content": "Assemble the video"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
            {
                "role": "assistant",
                "content": "CRITICAL - MAXIMUM STEPS REACHED\n\nThe maximum number of steps allowed for this task has been reached.",
            },
        ],
    }
    ordinary = {"model": "policy", "tools": tools, "messages": cap["messages"][:3]}
    assert is_step_cap_request(cap)
    assert not is_step_cap_request(ordinary)
    # A served assistant turn that merely mentions the phrase mid-way is not the cap.
    assert not is_step_cap_request(
        {
            "messages": [
                {"role": "assistant", "content": "Note: CRITICAL - MAXIMUM STEPS REACHED is what OpenCode says"}
            ][:0]
            + [{"role": "user", "content": "x"}]
        }
    )
    completion = step_cap_completion("policy")
    assert completion["choices"][0]["finish_reason"] == "stop" and completion["model"] == "policy"
    # The local reply is an ignored side call, never training data.
    records = [{"turn": 0, "agent": STEP_CAP_AGENT, "status": 200, "response": completion}]
    items, stats = capture_output(records)
    assert items == [] and stats.ignored_calls == 1


@pytest.mark.parametrize("system_as_parts", [False, True])
def test_sessions_keep_parent_and_child_identity_across_midnight(tmp_path: Path, system_as_parts: bool) -> None:
    from responses_api_agents.agentic_vbench_agent.capture_proxy import EpisodeRoute, observe_session

    def request(date: str, prompt: str, instructions: str) -> dict:
        text = f"{instructions}\n<env>\n  Today's date: {date}\n</env>"
        content = [{"type": "text", "text": text}] if system_as_parts else text
        return {"messages": [{"role": "system", "content": content}, {"role": "user", "content": prompt}]}

    route = EpisodeRoute(upstream="http://u", record_dir=tmp_path)
    parent_before = request("Sat Oct 17 2026", "Assemble the video", "Parent instructions")
    parent = observe_session(route, parent_before)
    route.task_prompts["Prepare contact sheets"] = (parent.session_id, "call_task_1")
    child = observe_session(route, request("Sat Oct 17 2026", "Prepare contact sheets", "Child instructions"))

    parent_after = request("Sun Oct 18 2026", "Assemble the video", "Parent instructions")
    child_after = request("Sun Oct 18 2026", "Prepare contact sheets", "Child instructions")
    parent_after["messages"].append({"role": "assistant", "content": "The child finished"})
    expected_bodies = json.dumps([parent_before, parent_after, child_after], sort_keys=True)

    assert observe_session(route, parent_after) is parent
    assert observe_session(route, child_after) is child
    assert parent.role == "parent" and parent.parent_session_id is None
    assert child.role == "child" and child.parent_session_id == parent.session_id
    assert child.spawn_call_id == "call_task_1"
    assert len(route.sessions) == 2
    assert json.dumps([parent_before, parent_after, child_after], sort_keys=True) == expected_bodies


def test_session_date_normalization_preserves_other_system_and_user_context() -> None:
    from responses_api_agents.agentic_vbench_agent.capture_proxy import session_key

    def request(instructions: str, date: str, user: str = "Assemble the video") -> dict:
        return {
            "messages": [
                {"role": "system", "content": f"{instructions}\n<env>\n  Today's date: {date}\n</env>"},
                {"role": "user", "content": user},
            ]
        }

    base = request("Parent instructions", "Sat Oct 17 2026")
    assert session_key(base) == session_key(request("Parent instructions", "Sun Oct 18 2026"))
    assert session_key(base) != session_key(request("Child instructions", "Sat Oct 17 2026"))
    assert session_key(base) != session_key(
        request("Parent instructions", "Sat Oct 17 2026", "Prepare contact sheets")
    )
    outside_before = request("Today's date: task-specific date A", "Sat Oct 17 2026")
    outside_after = request("Today's date: task-specific date B", "Sat Oct 17 2026")
    assert session_key(outside_before) != session_key(outside_after)


def test_sessions_are_keyed_by_leading_context_and_children_attributed_to_task_calls(tmp_path: Path) -> None:
    from responses_api_agents.agentic_vbench_agent.capture_proxy import (
        EpisodeRoute,
        observe_session,
        session_key,
        session_traces,
        task_tool_prompts,
    )

    system = {"role": "system", "content": "You are OpenCode."}
    parent_open = {"role": "user", "content": "Assemble the video"}
    spawn = completion(
        1,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_task_1",
                    "type": "function",
                    "function": {
                        "name": "task",
                        "arguments": json.dumps(
                            {"description": "probe", "prompt": "Inspect clip A", "subagent_type": "general"}
                        ),
                    },
                },
                {
                    "id": "call_task_2",
                    "type": "function",
                    "function": {
                        "name": "Task",
                        "arguments": json.dumps(
                            {"description": "probe", "prompt": "Inspect clip B", "subagent_type": "general"}
                        ),
                    },
                },
                {"id": "call_bash", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
            ],
        },
        "tool_calls",
    )
    assert task_tool_prompts(spawn) == [
        ("call_task_1", "Inspect clip A"),
        ("call_task_2", "Inspect clip B"),
    ]

    route = EpisodeRoute(upstream="http://u", record_dir=tmp_path)
    parent_first = {"messages": [system, parent_open]}
    parent_later = {
        "messages": [system, parent_open, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "go on"}]
    }
    child_first = {"messages": [system, {"role": "user", "content": "Inspect clip A"}]}

    parent = observe_session(route, parent_first)
    assert parent.role == "parent" and parent.session_id == session_key(parent_first)
    # Same session on later calls: history grows, leading context does not.
    assert observe_session(route, parent_later) is parent
    for call_id, prompt in task_tool_prompts(spawn):
        route.task_prompts[prompt.strip()] = (parent.session_id, call_id)
    child = observe_session(route, child_first)
    assert (
        child.role == "child" and child.parent_session_id == parent.session_id and child.spawn_call_id == "call_task_1"
    )
    assert child.session_id != parent.session_id
    # A child whose opening turn matches no task prompt still hangs off the parent.
    orphan = observe_session(route, {"messages": [system, {"role": "user", "content": "unrelated"}]})
    assert orphan.role == "child" and orphan.parent_session_id == parent.session_id and orphan.spawn_call_id is None

    records = [
        {
            "turn": 0,
            "agent": "build",
            "status": 200,
            "session_id": parent.session_id,
            "session_role": "parent",
            "parent_session_id": None,
            "spawn_call_id": None,
            "response": completion(0, {"role": "assistant", "content": "a"}, "stop"),
        },
        {
            "turn": 1,
            "agent": "build",
            "status": 200,
            "session_id": parent.session_id,
            "session_role": "parent",
            "parent_session_id": None,
            "spawn_call_id": None,
            "response": spawn,
        },
        {
            "turn": 2,
            "agent": "build",
            "status": 200,
            "session_id": child.session_id,
            "session_role": "child",
            "parent_session_id": parent.session_id,
            "spawn_call_id": "call_task_1",
            "response": completion(2, {"role": "assistant", "content": "b"}, "stop"),
        },
        {
            "turn": 3,
            "agent": "build",
            "status": 500,
            "session_id": child.session_id,
            "session_role": "child",
            "error": "boom",
        },
        {
            "turn": 4,
            "agent": "build",
            "status": 200,
            "response": completion(4, {"role": "assistant", "content": "untagged"}, "stop"),
        },
    ]
    traces = session_traces(records)
    assert [t["session_role"] for t in traces] == ["parent", "child"]
    assert traces[0]["response_ids"] == ["chatcmpl-0", "chatcmpl-1"]
    assert traces[1]["response_ids"] == ["chatcmpl-2"] and traces[1]["spawn_call_id"] == "call_task_1"
    _, stats = capture_output(records[:3])
    assert stats.session_traces == session_traces(records[:3])


@pytest.mark.asyncio
async def test_compaction_and_rewritten_context_keep_task_session_ownership(tmp_path: Path) -> None:
    from responses_api_agents.agentic_vbench_agent.capture_proxy import session_key, session_traces

    def tool(call_id: str, name: str, arguments: dict) -> dict:
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

    main_user = {"role": "user", "content": "Assemble the video"}
    main_system = {"role": "system", "content": "You are the main OpenCode agent."}
    child_user = {"role": "user", "content": "Inspect clip A"}
    child_system = {"role": "system", "content": "You are a general subagent."}
    summary_system = {"role": "system", "content": "You are an anchored context summarization assistant."}
    compact_user = {"role": "user", "content": "Create a new anchored summary from the conversation history above."}
    rewritten_user = {"role": "user", "content": "What did we do so far?"}
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    messages = [
        {"role": "assistant", "content": "Inspecting inputs.", "tool_calls": [tool("call_main", "bash", {})]},
        {"role": "assistant", "content": "Main summary: inputs inspected; delegate preparation next."},
        {
            "role": "assistant",
            "content": "Delegate clip inspection.",
            "tool_calls": [tool("call_task", "task", {"prompt": "Inspect clip A", "subagent_type": "general"})],
        },
        {"role": "assistant", "content": "Inspecting clip A.", "tool_calls": [tool("call_child", "bash", {})]},
        {"role": "assistant", "content": "Child summary: clip A inspected; report its dimensions."},
        {"role": "assistant", "content": "Clip A is 1280 by 720."},
        {"role": "assistant", "content": "Integrated and verified the child report."},
    ]
    child_tool_result = {"role": "tool", "tool_call_id": "call_child", "content": "1280x720"}
    main_tool_result = {"role": "tool", "tool_call_id": "call_main", "content": "clip A found"}
    task_result = {"role": "tool", "tool_call_id": "call_task", "content": messages[5]["content"]}
    histories = [
        [main_system, main_user],
        [summary_system, main_user, messages[0], main_tool_result, compact_user],
        [main_system, rewritten_user, messages[1]],
        # Even replaying a parent's summary cannot override an explicit Task spawn.
        [child_system, child_user, messages[1]],
        [summary_system, child_user, messages[3], child_tool_result, compact_user],
        # Deliberately use the same rewritten prefix as the main session. The
        # actually served summary, not the shared prefix, identifies its owner.
        [main_system, rewritten_user, messages[4]],
        [main_system, rewritten_user, messages[1], messages[2], task_result],
    ]
    bodies = [
        {"model": "policy", "messages": history, **({} if i in (1, 4) else {"tools": tools})}
        for i, history in enumerate(histories)
    ]
    served = [completion(i, m, "tool_calls" if m.get("tool_calls") else "stop") for i, m in enumerate(messages)]
    upstream = Upstream()
    await upstream.start()
    upstream.responses = list(served)
    proxy = CaptureProxy(advertise_host="127.0.0.1")
    await proxy.start()
    try:
        endpoint = proxy.register("compacted-sessions", upstream.url, tmp_path)
        async with aiohttp.ClientSession() as client:
            for body, expected in zip(bodies, served):
                async with client.post(f"{endpoint}/chat/completions", json=body) as response:
                    assert response.status == 200
                    assert await response.json() == expected
        # Neither the model's input history nor its returned completion changes.
        assert [body for _, body, _ in upstream.requests] == bodies
    finally:
        await proxy.stop()
        await upstream.runner.cleanup()

    records = load_records(tmp_path)
    main_id, child_id = session_key(bodies[0]), session_key(bodies[3])
    assert main_id != child_id
    assert [r["session_id"] for r in records] == [main_id, main_id, main_id, child_id, child_id, child_id, main_id]
    assert [r["session_role"] for r in records] == ["parent", "parent", "parent", "child", "child", "child", "parent"]
    for r in records:
        assert r["spawn_call_id"] == ("call_task" if r["session_role"] == "child" else None)
        assert r["parent_session_id"] == (main_id if r["session_role"] == "child" else None)
    traces = session_traces(records)
    assert len(traces) == 2
    assert traces[0]["response_ids"] == [served[i]["id"] for i in (0, 1, 2, 6)]
    assert traces[1]["response_ids"] == [served[i]["id"] for i in (3, 4, 5)]
    output, stats = capture_output(records)
    expected = [item for response in served for item in completion_items(response)]
    assert stats.compaction_summaries == 2
    assert [output_item_fingerprint(i) for i in output] == [output_item_fingerprint(i) for i in expected]


def test_ambiguous_compaction_text_cannot_claim_another_session(tmp_path: Path) -> None:
    from responses_api_agents.agentic_vbench_agent.capture_proxy import (
        COMPACTION_AGENT,
        EpisodeRoute,
        _remember_session_messages,
        observe_session,
    )

    route = EpisodeRoute(upstream="http://u", record_dir=tmp_path)
    parent_body = {"messages": [{"role": "user", "content": "Main task"}]}
    parent = observe_session(route, parent_body)
    route.task_prompts["Child task"] = (parent.session_id, "call_task")
    child_body = {"messages": [{"role": "user", "content": "Child task"}]}
    child = observe_session(route, child_body)
    summary = {"role": "assistant", "content": "The work is complete."}
    for session in (parent, child):
        _remember_session_messages(route, session, completion(0, summary, "stop"), agent=COMPACTION_AGENT)
    # The shared content is ambiguous. Retain the independently known parent
    # prefix and the child's explicit Task binding, rather than choosing an owner.
    assert observe_session(route, {"messages": [*parent_body["messages"], summary]}) is parent
    assert observe_session(route, {"messages": [*child_body["messages"], summary]}) is child
    compact_child = {"messages": [{"role": "system", "content": "Summary agent"}, *child_body["messages"], summary]}
    assert observe_session(route, compact_child, agent=COMPACTION_AGENT) is child
