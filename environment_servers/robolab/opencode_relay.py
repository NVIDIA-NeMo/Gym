# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenCode Responses API compatibility relay owned by the RoboLab environment server.

OpenCode's EmberBridge adapter speaks the OpenAI Responses API over SSE via
`@ai-sdk/openai`, which has two wire-dialect expectations that stock buffered
Gym model servers do not assume:
1. OpenCode omits the optional ``detail`` field on ``input_image`` parts,
   whereas Gym's strict Responses request schema requires it.
2. OpenCode builds streamed text, reasoning, and tool calls from
   ``response.output_item.added`` and ``*.delta`` events rather than only the
   terminal ``response.output_item.done`` event synthesized by buffered servers.

This relay runs on the RoboLab environment server's FastAPI app and forwards
every request through NeMo Gym's global aiohttp client to Gym's configured
``policy_model`` server so model-call capture and trajectory recording stay
intact without modifying core ``nemo_gym/models``.
"""

import codecs
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

from aiohttp import ClientTimeout, StreamReader
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from nemo_gym.client import request as gym_http_request


_SKIP_REQUEST_HEADERS = frozenset({"host", "content-length", "connection", "transfer-encoding", "accept-encoding"})
_STREAM_TIMEOUT = ClientTimeout(total=None, sock_connect=60.0)


def normalize_opencode_responses_body(body: dict[str, Any]) -> dict[str, Any]:
    """Fill optional ``detail: "auto"`` on OpenCode ``input_image`` parts."""
    items = body.get("input")
    if not isinstance(items, list):
        return body

    changed = False
    normalized_items: list[Any] = []
    for raw_item in items:
        if not isinstance(raw_item, dict):
            normalized_items.append(raw_item)
            continue
        item = raw_item
        parts_key = None
        if (
            item.get("type") in (None, "message")
            and item.get("role") is not None
            and isinstance(item.get("content"), list)
        ):
            parts_key = "content"
        elif item.get("type") == "function_call_output" and isinstance(item.get("output"), list):
            parts_key = "output"

        if parts_key is not None:
            parts = item[parts_key]
            item_changed = False
            new_parts: list[Any] = []
            for part in parts:
                if isinstance(part, dict) and part.get("type") == "input_image" and "detail" not in part:
                    new_parts.append({**part, "detail": "auto"})
                    item_changed = True
                else:
                    new_parts.append(part)
            if item_changed:
                item = {**item, parts_key: new_parts}
                changed = True
        normalized_items.append(item)
    return {**body, "input": normalized_items} if changed else body


def _sse_event(payload: dict[str, Any]) -> str:
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"


def response_item_events(index: int, item: dict[str, Any]) -> Iterator[str]:
    """Synthesize opening and delta SSE events for a buffered Responses item."""
    kind = item.get("type")
    if kind not in {"reasoning", "message", "function_call"}:
        return

    opening = dict(item)
    if kind == "reasoning":
        opening["summary"] = []
    elif kind == "message":
        opening.update(content=[], status="in_progress")
    elif kind == "function_call":
        opening.update(arguments="", status="in_progress")
    yield _sse_event({"type": "response.output_item.added", "output_index": index, "item": opening})

    item_fields = {"output_index": index, "item_id": item.get("id")}
    if kind == "function_call":
        arguments = item.get("arguments", "")
        yield _sse_event({"type": "response.function_call_arguments.delta", **item_fields, "delta": arguments})
        yield _sse_event({"type": "response.function_call_arguments.done", **item_fields, "arguments": arguments})
        return

    is_reasoning = kind == "reasoning"
    parts = item.get("summary" if is_reasoning else "content") or []
    part_event = "reasoning_summary_part" if is_reasoning else "content_part"
    index_key = "summary_index" if is_reasoning else "content_index"
    for part_index, part in enumerate(parts):
        if not isinstance(part, dict) or "type" not in part:
            continue
        fields = {**item_fields, index_key: part_index}
        part_type = part["type"]
        value_key = "refusal" if part_type == "refusal" else "text"
        text_event = "reasoning_summary_text" if is_reasoning else part_type
        value = part.get(value_key, "")
        opening_part = {**part, value_key: ""}
        yield _sse_event({"type": f"response.{part_event}.added", **fields, "part": opening_part})
        yield _sse_event({"type": f"response.{text_event}.delta", **fields, "delta": value})
        yield _sse_event({"type": f"response.{text_event}.done", **fields, value_key: value})
        yield _sse_event({"type": f"response.{part_event}.done", **fields, "part": part})


async def expand_responses_sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Forward SSE blocks, inserting item added/delta events before bare item.done blocks."""
    seen_added: set[int] = set()
    pending: list[str] = []

    def _flush(block: list[str]) -> list[str]:
        if not block:
            return []
        out: list[str] = []
        data_lines = [line[5:].lstrip() for line in block if line.startswith("data:")]
        if data_lines:
            raw_data = "\n".join(data_lines)
            if raw_data != "[DONE]":
                try:
                    payload = json.loads(raw_data)
                except ValueError:
                    payload = None
                if isinstance(payload, dict):
                    event_type = payload.get("type")
                    output_index = payload.get("output_index")
                    if event_type == "response.output_item.added" and isinstance(output_index, int):
                        seen_added.add(output_index)
                    elif (
                        event_type == "response.output_item.done"
                        and isinstance(output_index, int)
                        and output_index not in seen_added
                        and isinstance(payload.get("item"), dict)
                    ):
                        out.extend(response_item_events(output_index, payload["item"]))
        out.append("\n".join(block) + "\n\n")
        return out

    async for line in lines:
        if line == "":
            for chunk in _flush(pending):
                yield chunk
            pending.clear()
        else:
            pending.append(line)
    if pending:
        for chunk in _flush(pending):
            yield chunk


async def _iter_aiohttp_lines(content: StreamReader) -> AsyncIterator[str]:
    """Decode an aiohttp byte stream into newline-delimited text lines without buffering whole responses."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    buf = ""
    async for chunk in content.iter_any():
        if not chunk:
            continue
        buf += decoder.decode(chunk)
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            yield line.rstrip("\r")
    buf += decoder.decode(b"", final=True)
    if buf:
        yield buf.rstrip("\r")


def mount_opencode_relay(app: FastAPI, relays: dict[str, str], *, server_name: str = "policy_model") -> None:
    """Register the per-episode OpenCode Responses compatibility route on `app`."""

    @app.post("/_opencode_relay/{relay_id}/v1/responses")
    async def opencode_responses_relay(relay_id: str, request: Request) -> Response:
        upstream_base = relays.get(relay_id)
        if upstream_base is None:
            raise HTTPException(404, "Unknown or closed OpenCode relay session")

        raw_body = await request.body()
        try:
            parsed = json.loads(raw_body)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            normalized = normalize_opencode_responses_body(parsed)
            if normalized is not parsed:
                raw_body = json.dumps(normalized).encode("utf-8")

        headers = {k: v for k, v in request.headers.items() if k.lower() not in _SKIP_REQUEST_HEADERS}
        target = f"{upstream_base.rstrip('/')}/responses"
        upstream = await gym_http_request(
            "POST",
            target,
            data=raw_body,
            headers=headers,
            timeout=_STREAM_TIMEOUT,
            _internal=True,
            _server_name=server_name,
            _resend_after_disconnect=False,
            _retry_other_errors=False,
        )

        content_type = upstream.headers.get("content-type", "application/json")
        if "text/event-stream" not in content_type.lower():
            try:
                body = await upstream.read()
            finally:
                upstream.release()
            return Response(content=body, status_code=upstream.status, media_type=content_type)

        async def _stream() -> AsyncIterator[str]:
            try:
                async for chunk in expand_responses_sse_lines(_iter_aiohttp_lines(upstream.content)):
                    yield chunk
            finally:
                upstream.release()

        return StreamingResponse(_stream(), status_code=upstream.status, media_type="text/event-stream")
