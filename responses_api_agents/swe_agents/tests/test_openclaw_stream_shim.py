# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web

from responses_api_agents.swe_agents.openclaw.stream_shim import StreamShim, _extract_prefix_token_ids


async def _start_fake_upstream(handler, *, host: str = "127.0.0.1") -> tuple[str, web.AppRunner]:
    app = web.Application()
    app.router.add_post("/v1/responses", handler)
    app.router.add_post("/v1/chat/completions", handler)
    app.router.add_get("/v1/models", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=0)
    await site.start()
    sock = next(iter(runner.sites))._server.sockets[0]
    port = sock.getsockname()[1]
    return f"http://{host}:{port}", runner


@pytest.fixture
async def fake_upstream_basic():
    async def handler(request):
        body = await request.json() if request.method == "POST" else {}
        return web.json_response({"id": "resp_1", "echo": body, "output": []})

    base_url, runner = await _start_fake_upstream(handler)
    yield base_url + "/v1"
    await runner.cleanup()


@pytest.fixture
async def shim(tmp_path, fake_upstream_basic):
    port_file = tmp_path / "shim.port"
    pid_file = tmp_path / "shim.pid"
    log_file = tmp_path / "proxy.jsonl"
    s = StreamShim(
        upstream_base_url=fake_upstream_basic,
        port_file=str(port_file),
        pid_file=str(pid_file),
        jsonl_log=str(log_file),
        max_turns=100,
    )
    await s.start()
    yield s
    await s.stop()


async def test_start_writes_port_and_pid_files(shim, tmp_path):
    port = int(Path(tmp_path / "shim.port").read_text())
    assert 0 < port < 65536
    pid = int(Path(tmp_path / "shim.pid").read_text())
    assert pid > 0


@pytest.fixture
async def fake_upstream_models():
    # A distinct models-shaped body (NOT the POST echo body) plus a record of the
    # exact path/method the shim hit, so the test asserts a real GET /v1/models
    # passthrough rather than a canned value shared with the POST handler.
    seen: list[tuple[str, str]] = []

    async def handler(request):
        seen.append((request.method, request.path))
        return web.json_response({"object": "list", "data": [{"id": "model-X"}]})

    base_url, runner = await _start_fake_upstream(handler)
    yield base_url + "/v1", seen
    await runner.cleanup()


@pytest.fixture
async def shim_models(tmp_path, fake_upstream_models):
    base_url, seen = fake_upstream_models
    s = StreamShim(
        upstream_base_url=base_url,
        port_file=str(tmp_path / "shim.port"),
        pid_file=str(tmp_path / "shim.pid"),
        jsonl_log=str(tmp_path / "proxy.jsonl"),
        max_turns=100,
    )
    await s.start()
    yield s, seen
    await s.stop()


async def test_get_models_forwarded(shim_models):
    s, seen = shim_models
    async with aiohttp.ClientSession() as sess:
        async with sess.get(f"http://127.0.0.1:{s.port}/v1/models") as r:
            assert r.status == 200
            body = await r.json()
            assert body["object"] == "list"
            assert body["data"][0]["id"] == "model-X"
    # The shim reached the upstream via GET on the /v1/models path (real passthrough).
    assert seen == [("GET", "/v1/models")]


async def test_unknown_path_returns_404(shim):
    async with aiohttp.ClientSession() as sess:
        async with sess.get(f"http://127.0.0.1:{shim.port}/admin") as r:
            assert r.status == 404


async def test_post_responses_forwards_body_with_stream_stripped(shim, tmp_path):
    async with aiohttp.ClientSession() as sess:
        body = {"input": [{"role": "user", "content": "hi"}], "model": "X", "stream": True}
        async with sess.post(f"http://127.0.0.1:{shim.port}/v1/responses", json=body) as r:
            assert r.status == 200
            await r.read()
    log_lines = Path(tmp_path / "proxy.jsonl").read_text().splitlines()
    assert len(log_lines) == 1
    entry = json.loads(log_lines[0])
    assert "stream" not in entry["request"] or entry["request"]["stream"] is False
    assert entry["endpoint"] == "/v1/responses"
    assert entry["upstream_status"] == 200


# top_p can't be set via openclaw.json (the openai-responses transport doesn't wire it), so
# the shim injects it onto the forwarded body — the choke point before the vllm_model proxy.
@pytest.fixture
async def shim_top_p(tmp_path, fake_upstream_basic):
    s = StreamShim(
        upstream_base_url=fake_upstream_basic,
        port_file=str(tmp_path / "shim.port"),
        pid_file=str(tmp_path / "shim.pid"),
        jsonl_log=str(tmp_path / "proxy.jsonl"),
        max_turns=100,
        top_p=0.95,
    )
    await s.start()
    yield s
    await s.stop()


async def test_top_p_injected_into_forwarded_body(shim_top_p, tmp_path):
    async with aiohttp.ClientSession() as sess:
        body = {"input": [{"role": "user", "content": "hi"}], "model": "X", "stream": True}
        async with sess.post(f"http://127.0.0.1:{shim_top_p.port}/v1/responses", json=body) as r:
            assert r.status == 200
            await r.read()
    entry = json.loads(Path(tmp_path / "proxy.jsonl").read_text().splitlines()[0])
    assert entry["request"]["top_p"] == 0.95


async def test_top_p_not_injected_when_unset(shim, tmp_path):
    async with aiohttp.ClientSession() as sess:
        body = {"input": [{"role": "user", "content": "hi"}], "model": "X", "stream": True}
        async with sess.post(f"http://127.0.0.1:{shim.port}/v1/responses", json=body) as r:
            await r.read()
    entry = json.loads(Path(tmp_path / "proxy.jsonl").read_text().splitlines()[0])
    assert "top_p" not in entry["request"]


async def test_chat_completions_emits_data_done(shim):
    async with aiohttp.ClientSession() as sess:
        body = {"messages": [{"role": "user", "content": "hi"}], "model": "X", "stream": True}
        async with sess.post(f"http://127.0.0.1:{shim.port}/v1/chat/completions", json=body) as r:
            assert r.status == 200
            text = await r.text()
    assert "data: [DONE]" in text
    # The upstream body must ride in a preceding data: frame, not just the sentinel:
    # a regression dropping the body but keeping [DONE] would still pass the line above.
    frames = [line[len("data: ") :] for line in text.splitlines() if line.startswith("data: ")]
    assert any(frame != "[DONE]" and json.loads(frame).get("id") == "resp_1" for frame in frames)


def _parse_sse(text: str) -> list[tuple[str | None, dict | None]]:
    """Parse an SSE stream into (event, data-json) pairs."""
    events: list[tuple[str | None, dict | None]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        ev = None
        data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        events.append((ev, json.loads(data) if data and data != "[DONE]" else None))
    return events


@pytest.fixture
async def fake_upstream_tool_call():
    # A realistic /v1/responses body: one assistant message followed by a
    # function_call. OpenClaw materializes output items from the granular SSE
    # events (output_item.added/done, *.delta), not from response.completed.
    async def handler(request):
        await request.read()
        return web.json_response(
            {
                "id": "resp_tc",
                "object": "response",
                "created_at": 1,
                "model": "X",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Let me look.", "annotations": []}],
                    },
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "exec",
                        "arguments": '{"command": "ls"}',
                        "status": "completed",
                    },
                ],
            }
        )

    base_url, runner = await _start_fake_upstream(handler)
    yield base_url + "/v1"
    await runner.cleanup()


@pytest.fixture
async def shim_tool_call(tmp_path, fake_upstream_tool_call):
    s = StreamShim(
        upstream_base_url=fake_upstream_tool_call,
        port_file=str(tmp_path / "shim.port"),
        pid_file=str(tmp_path / "shim.pid"),
        jsonl_log=str(tmp_path / "proxy.jsonl"),
        max_turns=100,
    )
    await s.start()
    yield s
    await s.stop()


async def _post_responses_sse(shim, body=None) -> str:
    body = body or {"input": [{"role": "user", "content": "hi"}], "model": "X", "stream": True}
    async with aiohttp.ClientSession() as sess:
        async with sess.post(f"http://127.0.0.1:{shim.port}/v1/responses", json=body) as r:
            assert r.status == 200
            return await r.text()


async def test_responses_sse_emits_required_event_types(shim_tool_call):
    text = await _post_responses_sse(shim_tool_call)
    names = [ev for ev, _ in _parse_sse(text)]
    for required in (
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.output_text.delta",
        "response.function_call_arguments.delta",
        "response.output_item.done",
        "response.completed",
    ):
        assert required in names, f"missing SSE event {required}; got {names}"


async def test_responses_sse_output_item_added_once_per_output_item(shim_tool_call):
    events = _parse_sse(await _post_responses_sse(shim_tool_call))
    added = [d for ev, d in events if ev == "response.output_item.added"]
    assert len(added) == 2  # one message + one function_call
    assert added[0]["output_index"] == 0
    assert added[1]["output_index"] == 1


async def test_responses_sse_text_delta_carries_full_message_text(shim_tool_call):
    events = _parse_sse(await _post_responses_sse(shim_tool_call))
    deltas = [d for ev, d in events if ev == "response.output_text.delta"]
    assert deltas, "no output_text.delta event emitted"
    assert "".join(d["delta"] for d in deltas) == "Let me look."


async def test_responses_sse_function_call_arguments_delta_carries_args(shim_tool_call):
    events = _parse_sse(await _post_responses_sse(shim_tool_call))
    deltas = [d for ev, d in events if ev == "response.function_call_arguments.delta"]
    assert deltas, "no function_call_arguments.delta event emitted"
    assert "".join(d["delta"] for d in deltas) == '{"command": "ls"}'


async def test_responses_sse_output_item_done_carries_complete_items(shim_tool_call):
    events = _parse_sse(await _post_responses_sse(shim_tool_call))
    done = [d["item"] for ev, d in events if ev == "response.output_item.done"]
    assert len(done) == 2
    assert done[0]["type"] == "message"
    assert done[1]["type"] == "function_call"
    assert done[1]["name"] == "exec"
    assert done[1]["arguments"] == '{"command": "ls"}'


async def test_responses_sse_terminal_event_is_completed_with_full_body(shim_tool_call):
    events = _parse_sse(await _post_responses_sse(shim_tool_call))
    last_ev, last_data = events[-1]
    assert last_ev == "response.completed"
    assert last_data["response"]["id"] == "resp_tc"
    assert len(last_data["response"]["output"]) == 2


@pytest.fixture
async def fake_upstream_unknown_item():
    # An output item of an unhandled type (reasoning) — exercises the else arm of
    # _responses_sse_payload that surfaces it as a plain added/done pair rather than
    # silently dropping it.
    async def handler(request):
        await request.read()
        return web.json_response(
            {
                "id": "resp_unk",
                "object": "response",
                "created_at": 1,
                "model": "X",
                "status": "completed",
                "output": [{"id": "rs_1", "type": "reasoning", "summary": []}],
            }
        )

    base_url, runner = await _start_fake_upstream(handler)
    yield base_url + "/v1"
    await runner.cleanup()


@pytest.fixture
async def shim_unknown_item(tmp_path, fake_upstream_unknown_item):
    s = StreamShim(
        upstream_base_url=fake_upstream_unknown_item,
        port_file=str(tmp_path / "shim.port"),
        pid_file=str(tmp_path / "shim.pid"),
        jsonl_log=str(tmp_path / "proxy.jsonl"),
        max_turns=100,
    )
    await s.start()
    yield s
    await s.stop()


async def test_responses_sse_unknown_item_type_surfaced_as_added_done_pair(shim_unknown_item):
    events = _parse_sse(await _post_responses_sse(shim_unknown_item))
    added = [d for ev, d in events if ev == "response.output_item.added"]
    done = [d for ev, d in events if ev == "response.output_item.done"]
    assert len(added) == 1
    assert len(done) == 1
    assert added[0]["item"]["type"] == "reasoning"
    assert done[0]["item"]["type"] == "reasoning"
    assert added[0]["item"]["id"] == "rs_1"


# ----------------------------------------------------------------------------
# On-policy token-ID carry-forward: when vllm_model returns prompt_token_ids /
# generation_token_ids on a turn's output (return_token_id_information=true),
# the shim must re-attach the prior turn's token IDs onto the last assistant
# item of the NEXT request. NeMo-RL's vLLM server reverse-scans request
# messages for those fields to build `required_prefix_token_ids` and run its
# on-policy splice. Pi (the openclaw agent) drops the fields when it
# re-serializes history, so the shim is the only place that can carry them.
# ----------------------------------------------------------------------------
@pytest.fixture
async def fake_upstream_token_ids():
    async def handler(request):
        await request.read()
        return web.json_response(
            {
                "id": "resp_tok",
                "object": "response",
                "created_at": 1,
                "model": "X",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                    },
                    {
                        "id": "fc_1",
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "exec",
                        "arguments": "{}",
                        "status": "completed",
                        # vllm_model attaches the triple to the turn's LAST output item.
                        "prompt_token_ids": [1, 2, 3],
                        "generation_token_ids": [4, 5],
                        "generation_log_probs": [-0.1, -0.2],
                    },
                ],
            }
        )

    base_url, runner = await _start_fake_upstream(handler)
    yield base_url + "/v1"
    await runner.cleanup()


@pytest.fixture
async def shim_token_ids(tmp_path, fake_upstream_token_ids):
    s = StreamShim(
        upstream_base_url=fake_upstream_token_ids,
        port_file=str(tmp_path / "shim.port"),
        pid_file=str(tmp_path / "shim.pid"),
        jsonl_log=str(tmp_path / "proxy.jsonl"),
        max_turns=100,
    )
    await s.start()
    yield s
    await s.stop()


# A subsequent turn's input as Pi sends it: prior turn's output items echoed
# verbatim (NO token IDs) followed by the tool result.
_TURN2_INPUT = [
    {"role": "system", "content": "s"},
    {"role": "user", "content": "u"},
    {"id": "msg_1", "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
    {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "exec", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
]


async def test_prior_turn_token_ids_injected_into_next_request(shim_token_ids, tmp_path):
    url = f"http://127.0.0.1:{shim_token_ids.port}/v1/responses"
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            url,
            json={
                "input": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                "model": "X",
                "stream": True,
            },
        ) as r1:
            await r1.read()
        async with sess.post(url, json={"input": _TURN2_INPUT, "model": "X", "stream": True}) as r2:
            await r2.read()

    lines = Path(tmp_path / "proxy.jsonl").read_text().splitlines()
    req2_input = json.loads(lines[1])["request"]["input"]
    # The last assistant-turn item (the function_call, just before the trailing
    # function_call_output) carries the prior turn's ground-truth token IDs.
    fc = req2_input[3]
    assert fc["type"] == "function_call"
    assert fc.get("prompt_token_ids") == [1, 2, 3]
    assert fc.get("generation_token_ids") == [4, 5]
    assert fc.get("generation_log_probs") == [-0.1, -0.2]
    # The tool result (an input, not a generation) must NOT be tagged.
    assert "prompt_token_ids" not in req2_input[4]


async def test_first_request_has_no_injected_token_ids(shim_token_ids, tmp_path):
    url = f"http://127.0.0.1:{shim_token_ids.port}/v1/responses"
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            url, json={"input": [{"role": "user", "content": "u"}], "model": "X", "stream": True}
        ) as r1:
            await r1.read()
    req0_input = json.loads(Path(tmp_path / "proxy.jsonl").read_text().splitlines()[0])["request"]["input"]
    for item in req0_input:
        assert "prompt_token_ids" not in item


async def test_no_injection_when_upstream_returns_no_token_ids(shim, tmp_path):
    # `shim` uses fake_upstream_basic, which returns output: [] (eval mode,
    # return_token_id_information=false). Nothing to carry → nothing injected.
    url = f"http://127.0.0.1:{shim.port}/v1/responses"
    async with aiohttp.ClientSession() as sess:
        async with sess.post(
            url, json={"input": [{"role": "user", "content": "u"}], "model": "X", "stream": True}
        ) as r1:
            await r1.read()
        async with sess.post(url, json={"input": _TURN2_INPUT, "model": "X", "stream": True}) as r2:
            await r2.read()
    req2_input = json.loads(Path(tmp_path / "proxy.jsonl").read_text().splitlines()[1])["request"]["input"]
    for item in req2_input:
        assert "prompt_token_ids" not in item


# Direct unit tests for the carry-forward helpers (defensive branches).
def _bare_shim() -> StreamShim:
    # __init__ opens no files (start() does), so this is safe to build without cleanup.
    return StreamShim(
        upstream_base_url="http://x/v1",
        port_file="/dev/null",
        pid_file="/dev/null",
        jsonl_log="/dev/null",
        max_turns=100,
    )


def test_extract_prefix_token_ids_picks_last_bearing_item():
    up = {
        "output": [
            {"type": "message", "prompt_token_ids": [9], "generation_token_ids": [8], "generation_log_probs": [-1.0]},
            {
                "type": "function_call",
                "prompt_token_ids": [1, 2],
                "generation_token_ids": [3],
                "generation_log_probs": [-0.5],
            },
        ]
    }
    assert _extract_prefix_token_ids(up) == {
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3],
        "generation_log_probs": [-0.5],
    }


def test_extract_prefix_token_ids_none_when_absent_or_malformed():
    assert _extract_prefix_token_ids({"output": [{"type": "message"}]}) is None
    assert _extract_prefix_token_ids({"output": "not-a-list"}) is None
    assert _extract_prefix_token_ids("not-a-dict") is None


def test_inject_prefix_token_ids_noop_without_prior_turn():
    shim = _bare_shim()  # _last_prefix_token_ids is None
    body = {"input": [{"type": "function_call", "name": "x"}]}
    shim._inject_prefix_token_ids(body)
    assert "prompt_token_ids" not in body["input"][0]


def test_inject_prefix_token_ids_noop_when_no_input_list():
    shim = _bare_shim()
    shim._last_prefix_token_ids = {
        "prompt_token_ids": [1],
        "generation_token_ids": [2],
        "generation_log_probs": [-0.1],
    }
    body = {"messages": []}  # chat-completions shape — no "input"; must not raise
    shim._inject_prefix_token_ids(body)
    assert "input" not in body


def test_inject_prefix_token_ids_handles_non_dict_input_item():
    shim = _bare_shim()
    shim._last_prefix_token_ids = {
        "prompt_token_ids": [1],
        "generation_token_ids": [2],
        "generation_log_probs": [-0.1],
    }
    body = {"input": ["not-a-dict"]}
    shim._inject_prefix_token_ids(body)  # must not raise
    assert body["input"] == ["not-a-dict"]


def test_inject_prefix_token_ids_skips_when_last_item_not_assistant():
    shim = _bare_shim()
    shim._last_prefix_token_ids = {
        "prompt_token_ids": [1],
        "generation_token_ids": [2],
        "generation_log_probs": [-0.1],
    }
    body = {"input": [{"role": "user", "content": "u"}]}
    shim._inject_prefix_token_ids(body)
    assert "prompt_token_ids" not in body["input"][0]


@pytest.mark.parametrize(
    "last_item",
    [
        {"type": "reasoning", "summary": []},
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]},
    ],
    ids=["reasoning", "assistant_message"],
)
def test_inject_prefix_token_ids_tags_non_function_call_assistant_items(last_item):
    # The function_call branch is covered by the e2e test; cover the reasoning and
    # assistant-message branches of is_assistant_generation directly.
    shim = _bare_shim()
    prefix = {"prompt_token_ids": [1], "generation_token_ids": [2], "generation_log_probs": [-0.1]}
    shim._last_prefix_token_ids = prefix
    body = {"input": [{"role": "user", "content": "u"}, last_item]}
    shim._inject_prefix_token_ids(body)
    tagged = body["input"][-1]
    assert tagged["prompt_token_ids"] == [1]
    assert tagged["generation_token_ids"] == [2]
    assert tagged["generation_log_probs"] == [-0.1]


@pytest.fixture
async def fake_upstream_with_cookie():
    async def handler(request):
        body = await request.json() if request.method == "POST" else {}
        resp = web.json_response({"id": "resp", "echo": body, "output": []})
        if "X-Force-Cookie" in request.headers:
            resp.headers["Set-Cookie"] = f"SESSION_ID={request.headers['X-Force-Cookie']}; Path=/"
        # Echo back whatever cookie we got so we can assert the shim resent it.
        resp.headers["X-Echo-Cookie"] = request.headers.get("Cookie", "")
        return resp

    base_url, runner = await _start_fake_upstream(handler)
    yield base_url + "/v1"
    await runner.cleanup()


@pytest.fixture
async def shim_cookie(tmp_path, fake_upstream_with_cookie):
    s = StreamShim(
        upstream_base_url=fake_upstream_with_cookie,
        port_file=str(tmp_path / "shim.port"),
        pid_file=str(tmp_path / "shim.pid"),
        jsonl_log=str(tmp_path / "proxy.jsonl"),
        max_turns=100,
    )
    await s.start()
    yield s
    await s.stop()


async def test_session_cookie_captured_and_reused(shim_cookie, tmp_path):
    url = f"http://127.0.0.1:{shim_cookie.port}/v1/responses"
    body = {"input": [], "model": "X", "stream": True}

    # Turn 1: upstream returns Set-Cookie with our forced value.
    async with aiohttp.ClientSession() as sess:
        async with sess.post(url, json=body, headers={"X-Force-Cookie": "abc-123"}) as r:
            await r.read()
    # Turn 2: shim should send Cookie: SESSION_ID=abc-123 upstream.
    async with aiohttp.ClientSession() as sess:
        async with sess.post(url, json=body) as r2:
            await r2.read()

    log_lines = Path(tmp_path / "proxy.jsonl").read_text().splitlines()
    e1, e2 = (json.loads(line) for line in log_lines)
    # First turn captured Set-Cookie.
    assert "SESSION_ID=abc-123" in e1["session_cookie_out"]
    # Second turn forwarded it upstream.
    assert "SESSION_ID=abc-123" in e2["session_cookie_in"]


@pytest.fixture
async def shim_one_turn(tmp_path, fake_upstream_basic):
    s = StreamShim(
        upstream_base_url=fake_upstream_basic,
        port_file=str(tmp_path / "shim.port"),
        pid_file=str(tmp_path / "shim.pid"),
        jsonl_log=str(tmp_path / "proxy.jsonl"),
        max_turns=1,
    )
    await s.start()
    yield s
    await s.stop()


async def test_max_turns_second_call_refused_with_max_iteration(shim_one_turn, tmp_path):
    url = f"http://127.0.0.1:{shim_one_turn.port}/v1/responses"
    async with aiohttp.ClientSession() as sess:
        async with sess.post(url, json={"input": [], "model": "X"}) as r1:
            assert r1.status == 200
            await r1.read()
        async with sess.post(url, json={"input": [], "model": "X"}) as r2:
            assert r2.status == 400
            body = await r2.json()
            assert body["error"]["type"] == "max_iteration"
            assert "1" in body["error"]["message"]  # references the configured max

    # The refusal is recorded in the JSONL.
    log_lines = Path(tmp_path / "proxy.jsonl").read_text().splitlines()
    assert len(log_lines) == 2
    last = json.loads(log_lines[-1])
    assert last["upstream_status"] is None
    assert last["error"] == "max_iteration"


@pytest.fixture
async def fake_upstream_context_overflow():
    async def handler(request):
        await request.read()
        return web.json_response(
            {
                "error": {
                    "message": "This model's maximum context length is 65536 tokens",
                    "type": "invalid_request_error",
                }
            },
            status=400,
        )

    base_url, runner = await _start_fake_upstream(handler)
    yield base_url + "/v1"
    await runner.cleanup()


@pytest.fixture
async def shim_overflow(tmp_path, fake_upstream_context_overflow):
    s = StreamShim(
        upstream_base_url=fake_upstream_context_overflow,
        port_file=str(tmp_path / "shim.port"),
        pid_file=str(tmp_path / "shim.pid"),
        jsonl_log=str(tmp_path / "proxy.jsonl"),
        max_turns=100,
    )
    await s.start()
    yield s
    await s.stop()


async def test_upstream_4xx_propagated_and_logged(shim_overflow, tmp_path):
    url = f"http://127.0.0.1:{shim_overflow.port}/v1/responses"
    async with aiohttp.ClientSession() as sess:
        async with sess.post(url, json={"input": [], "model": "X"}) as r:
            assert r.status == 400
            body = await r.text()
            assert "maximum context length" in body

    entry = json.loads(Path(tmp_path / "proxy.jsonl").read_text().splitlines()[-1])
    assert entry["upstream_status"] == 400
    assert "maximum context length" in json.dumps(entry["response"])
