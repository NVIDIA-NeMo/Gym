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
"""Tests for the sandbox-bound capture interceptor, store, and trajectory assembly."""

import asyncio
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from starlette.responses import JSONResponse, Response

from nemo_gym.adapters.capture_store import (
    CaptureStore,
    _content_text,
    _response_message,
    _token_fields,
    assemble_trajectory,
    has_token_ids,
)
from nemo_gym.adapters.pipeline import AdapterPipeline
from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.sandbox_capture import SandboxCaptureProxy, start_capture_proxy
from nemo_gym.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext


def test_capture_registered():
    assert "capture" in InterceptorRegistry.available()


def test_store_roundtrip_is_session_keyed(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("rollout-a", {"request_id": "1", "response": {"choices": []}})
    store.record("rollout-a", {"request_id": "2", "response": {"choices": []}})
    store.record("rollout-b", {"request_id": "3", "response": {"choices": []}})

    assert len(store.read("rollout-a")) == 2
    assert len(store.read("rollout-b")) == 1
    assert store.read("missing") == []
    # one file per session, so a reaped box cannot lose another box's turns
    assert store.path_for("rollout-a") != store.path_for("rollout-b")
    assert store.path_for("rollout-a").exists()


def _fake_upstream(response_body):
    async def _upstream(req: AdapterRequest) -> AdapterResponse:
        # echo what the agent would have sent so the test can assert injection
        req.ctx.extra["_seen_request_body"] = dict(req.body)
        return AdapterResponse(status_code=200, headers={}, body=response_body, latency_ms=4.2, ctx=req.ctx)

    return _upstream


def test_capture_records_exchange_and_injects_flag(tmp_path):
    interceptor = InterceptorRegistry.create(
        "capture",
        {
            "store_dir": str(tmp_path),
            "session_id": "rollout-1",
            "inject_extra_body": {"return_token_id_information": True},
        },
    )
    pipeline = AdapterPipeline([interceptor])

    response_body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi", "generation_token_ids": [5, 6, 7]},
                "prompt_token_ids": [1, 2, 3],
            }
        ]
    }
    req = AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        body={"model": "policy", "messages": [{"role": "user", "content": "hello"}]},
        ctx=InterceptorContext(),
    )

    resp = asyncio.run(pipeline.process(req, upstream_call=_fake_upstream(response_body)))
    assert resp.status_code == 200
    # the token-id flag was injected into the upstream request
    assert resp.ctx.extra["_seen_request_body"]["return_token_id_information"] is True

    exchanges = CaptureStore(tmp_path).read("rollout-1")
    assert len(exchanges) == 1
    recorded = exchanges[0]
    assert recorded["session_id"] == "rollout-1"
    assert recorded["status"] == 200
    assert recorded["response"]["choices"][0]["message"]["generation_token_ids"] == [5, 6, 7]
    assert recorded["request"]["return_token_id_information"] is True


def test_assemble_trajectory_interleaves_tools_and_keeps_token_ids():
    exchanges = [
        {
            "request": {"messages": [{"role": "user", "content": "fix the bug"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "let me look",
                            "generation_token_ids": [10, 11],
                            "tool_calls": [
                                {"id": "call_1", "function": {"name": "bash", "arguments": '{"cmd":"ls"}'}}
                            ],
                        },
                        "prompt_token_ids": [1, 2],
                    }
                ]
            },
        },
        {
            "request": {
                "messages": [
                    {"role": "user", "content": "fix the bug"},
                    {"role": "assistant", "content": "let me look"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "file.py"},
                ]
            },
            "response": {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done", "generation_token_ids": [20, 21, 22]},
                        "prompt_token_ids": [3, 4, 5],
                    }
                ]
            },
        },
    ]

    items = assemble_trajectory(exchanges)
    kinds = [item.type for item in items]
    assert kinds == ["message", "function_call", "function_call_output", "message"]

    first_assistant = items[0]
    assert first_assistant.generation_token_ids == [10, 11]
    assert first_assistant.prompt_token_ids == [1, 2]
    assert items[1].name == "bash"
    assert items[2].call_id == "call_1"
    assert items[2].output == "file.py"
    assert items[3].generation_token_ids == [20, 21, 22]
    assert has_token_ids(items)


def test_assemble_trajectory_tolerates_empty_and_malformed():
    assert assemble_trajectory([]) == []
    assert assemble_trajectory([{"request": {}, "response": {}}]) == []
    assert assemble_trajectory([{"response": {"choices": []}}]) == []


def test_assemble_trajectory_responses_wire_interleaves_tool_outputs():
    exchanges = [
        {
            "request": {"input": "fix it"},
            "response": {
                "output": [
                    {"type": "reasoning", "content": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "on it"}],
                        "generation_token_ids": [9, 8, 7],
                    },
                    {
                        "type": "function_call",
                        "id": "c1",
                        "call_id": "c1",
                        "name": "shell",
                        "arguments": '{"cmd":"ls"}',
                    },
                ]
            },
        },
        {
            # the tool result of c1 arrives in the next request's input
            "request": {
                "input": [
                    {"type": "function_call_output", "call_id": "c1", "output": "file.py"},
                ]
            },
            "response": {
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
                ]
            },
        },
    ]
    items = assemble_trajectory(exchanges, wire="responses")
    assert [it.type for it in items] == ["message", "function_call", "function_call_output", "message"]
    assert items[0].generation_token_ids == [9, 8, 7]
    assert items[1].name == "shell"
    assert items[2].call_id == "c1"
    assert items[2].output == "file.py"


def test_capture_stamps_upstream_api_key_on_forwarded_headers(tmp_path):
    """``upstream_api_key`` is stamped on the *forwarded* request headers (so the
    in-box agent only holds a dummy) and never persisted in the recorded body."""
    interceptor = InterceptorRegistry.create(
        "capture",
        {"store_dir": str(tmp_path), "session_id": "rollout-key", "upstream_api_key": "secret-key"},
    )

    seen: dict[str, str | None] = {}

    async def _upstream(req: AdapterRequest) -> AdapterResponse:
        seen["auth"] = req.headers.get("Authorization")
        return AdapterResponse(status_code=200, headers={}, body={"choices": []}, latency_ms=1.0, ctx=req.ctx)

    req = AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        body={"model": "policy", "messages": []},
        ctx=InterceptorContext(),
    )

    resp = asyncio.run(AdapterPipeline([interceptor]).process(req, upstream_call=_upstream))
    assert resp.status_code == 200
    assert seen["auth"] == "Bearer secret-key"

    recorded = CaptureStore(tmp_path).read("rollout-key")[0]
    assert "Authorization" not in (recorded.get("request") or {})


# ---------------------------------------------------------------------------
# capture_store edge branches (store IO + trajectory assembly)
# ---------------------------------------------------------------------------


def test_store_root_property_and_read_skips_blank_and_malformed_lines(tmp_path):
    """``read`` streams line-by-line, skipping blank/whitespace-only lines and any
    line that is not valid JSON (a partially-flushed tail of a reaped box)."""
    store = CaptureStore(tmp_path)
    assert store.root == Path(tmp_path)

    path = store.path_for("s")
    path.write_text(
        '{"request_id": "1"}\n\n   \nnot-json-at-all\n{"request_id": "2"}\n',
        encoding="utf-8",
    )

    rows = store.read("s")
    assert [r["request_id"] for r in rows] == ["1", "2"]


def test_token_fields_and_response_message_degenerate_inputs():
    # _token_fields: non-dict input yields no fields; only list-valued token keys are kept.
    assert _token_fields("not-a-dict") == {}
    assert _token_fields({"prompt_token_ids": [1, 2], "generation_token_ids": "nope"}) == {"prompt_token_ids": [1, 2]}

    # _response_message: a non-dict response, or a non-dict first-choice message, yields {}.
    assert _response_message("weird-non-dict") == {}
    assert _response_message({"choices": [{"message": "weird-message"}]}) == {}

    # normal path hoists token-ids from both the choice and the message onto the merged message.
    merged = _response_message(
        {"choices": [{"message": {"content": "hi", "generation_token_ids": [9]}, "prompt_token_ids": [1]}]}
    )
    assert merged["content"] == "hi"
    assert merged["generation_token_ids"] == [9]
    assert merged["prompt_token_ids"] == [1]


def test_content_text_handles_list_and_non_string_scalars():
    assert _content_text("plain") == "plain"
    # list parts: dicts read ``.text`` via ``.get``; non-dicts fall back to ``getattr`` (empty for a str).
    assert _content_text([{"text": "a"}, {"no_text": "b"}, "loose"]) == "a"
    # neither str nor list: None -> "", everything else -> str(...).
    assert _content_text(None) == ""
    assert _content_text(123) == "123"


def test_assemble_chat_dedups_tool_results_and_skips_functionless_tool_calls():
    """A tool result seen in an earlier request is not re-emitted, and assistant
    ``tool_calls`` lacking a ``function`` (or not even a dict) are skipped."""
    exchanges = [
        {
            "request": {"messages": [{"role": "tool", "tool_call_id": "call_1", "content": "o1"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "a",
                            "tool_calls": [{"id": "x"}, "loose-not-dict"],
                        }
                    }
                ]
            },
        },
        {
            "request": {"messages": [{"role": "tool", "tool_call_id": "call_1", "content": "o1-dup"}]},
            "response": {"choices": [{"message": {"role": "assistant", "content": "b"}}]},
        },
    ]

    items = assemble_trajectory(exchanges)
    fco = [it for it in items if it.type == "function_call_output"]
    assert len(fco) == 1
    assert fco[0].output == "o1"
    assert all(it.type != "function_call" for it in items)


def test_assemble_responses_skips_noise_and_dedups_tool_outputs():
    """Responses wire: non-``function_call_output`` (and non-dict) request-input
    items are skipped, non-dict response-output items are skipped, and a repeated
    ``function_call_output`` call id is only emitted once."""
    exchanges = [
        {
            "request": {
                "input": [
                    {"type": "message", "role": "user", "content": "hi"},
                    "loose-input-string",
                    {"type": "function_call_output", "call_id": "c1", "output": "r1"},
                ]
            },
            "response": {
                "output": [
                    "loose-output-string",
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "a"}]},
                ]
            },
        },
        {
            "request": {"input": [{"type": "function_call_output", "call_id": "c1", "output": "r1-dup"}]},
            "response": {
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "b"}]},
                ]
            },
        },
    ]

    items = assemble_trajectory(exchanges, wire="responses")
    fco = [it for it in items if it.type == "function_call_output"]
    assert len(fco) == 1
    assert fco[0].output == "r1"
    assert [it.type for it in items] == ["function_call_output", "message", "message"]


# ---------------------------------------------------------------------------
# sandbox_capture: a per-rollout, session-keyed capture proxy fronting an
# upstream model. Mirrors test_adapter_proxy.py: a localhost uvicorn stub
# upstream + a real proxy thread, asserted against, then stopped.
# ---------------------------------------------------------------------------


class _StubModelUpstream:
    """Minimal localhost model upstream returning a canned OpenAI chat completion.

    Echoes the request body it received so a test can assert the capture proxy
    forwarded (and injected into) the request.
    """

    def __init__(self, response_body: dict) -> None:
        self._response_body = response_body
        self.received: list[dict] = []
        self.app = FastAPI()
        self.port: int | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

        @self.app.api_route("/{path:path}", methods=["GET", "POST"])
        async def echo(path: str, request: Request) -> Response:
            if request.method == "POST":
                try:
                    body = json.loads(await request.body() or b"{}")
                except Exception:
                    body = None
                self.received.append({"path": "/" + path, "body": body})
                return JSONResponse(self._response_body, status_code=200)
            return JSONResponse({"ok": True})

    def start(self) -> None:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        cfg = uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning", access_log=False)
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/healthcheck", timeout=1) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("stub upstream did not become healthy")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=3)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


@pytest.fixture
def model_upstream():
    stub = _StubModelUpstream(
        {
            "id": "stub-cc",
            "object": "chat.completion",
            "model": "policy",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "captured-ok", "generation_token_ids": [11, 22, 33]},
                    "prompt_token_ids": [1, 2],
                    "finish_reason": "stop",
                }
            ],
        }
    )
    stub.start()
    try:
        yield stub
    finally:
        stub.stop()


def _post_json(url: str, body: dict) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _get(url: str) -> int:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_sandbox_capture_proxy_records_exchange_then_stops(tmp_path, model_upstream):
    """Start a session-keyed capture proxy in front of a stub model, drive one
    chat completion through it, and assert the exchange (with token-ids) was
    durably recorded under the session — then stop the proxy."""
    proxy = start_capture_proxy(
        model_base_url=model_upstream.url,
        session_id="rollout-xyz",
        store_dir=str(tmp_path),
        inject_extra_body={"return_token_id_information": True},
    )
    try:
        assert isinstance(proxy, SandboxCaptureProxy)
        assert proxy.session_id == "rollout-xyz"
        assert proxy.store_dir == str(tmp_path)
        assert proxy.handle.url.startswith("http://127.0.0.1:")

        status, body = _post_json(
            f"{proxy.handle.url}/v1/chat/completions",
            {"model": "policy", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert status == 200, body
        assert json.loads(body)["choices"][0]["message"]["content"] == "captured-ok"
        # the injected token-id flag reached the upstream request
        assert model_upstream.received[-1]["body"]["return_token_id_information"] is True
    finally:
        proxy.stop()

    exchanges = CaptureStore(tmp_path).read("rollout-xyz")
    assert len(exchanges) == 1
    assert exchanges[0]["session_id"] == "rollout-xyz"
    assert exchanges[0]["status"] == 200

    items = assemble_trajectory(exchanges)
    assert [it.type for it in items] == ["message"]
    assert items[0].generation_token_ids == [11, 22, 33]
    assert has_token_ids(items)


def test_sandbox_capture_proxy_translate_anthropic_branch_starts_and_stops(tmp_path, model_upstream):
    """The ``translate_anthropic`` flag wires the translation adapter ahead of
    capture. We only assert the proxy comes up healthy and stops cleanly — the
    translation behavior itself is covered by the translate_anthropic tests."""
    proxy = start_capture_proxy(
        model_base_url=model_upstream.url,
        session_id="rollout-translate",
        store_dir=str(tmp_path),
        translate_anthropic=True,
        translate_model_override="policy-model",
    )
    try:
        assert proxy.handle.url.startswith("http://127.0.0.1:")
        assert _get(f"{proxy.handle.url}/_proxy_health") == 200
    finally:
        proxy.stop()
