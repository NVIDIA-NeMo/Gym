# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted Messages responses, including deliberately absent usage."""

import json
from collections.abc import Iterator


def messages_sse(response: dict) -> Iterator[str]:
    def event(kind: str, **fields) -> str:
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **fields})}\n\n"

    shell = {**response, "content": [], "stop_reason": None}
    if "usage" in response:
        shell["usage"] = {**response["usage"], "output_tokens": 0}
    yield event("message_start", message=shell)
    for index, block in enumerate(response["content"]):
        if block["type"] == "tool_use":
            initial = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        else:
            initial = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        yield event("content_block_start", index=index, content_block=initial)
        yield event("content_block_delta", index=index, delta=delta)
        yield event("content_block_stop", index=index)
    fields = {"delta": {"stop_reason": response["stop_reason"]}}
    if "usage" in response:
        fields["usage"] = {"output_tokens": response["usage"]["output_tokens"]}
    yield event("message_delta", **fields)
    yield event("message_stop")


def messages_response(model: str, response_id: str, content: dict, *, usage: bool, index: int) -> dict:
    response = {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [content],
        "stop_reason": "tool_use" if content["type"] == "tool_use" else "end_turn",
    }
    if usage:
        response["usage"] = {
            "input_tokens": 20 + index,
            "output_tokens": 8,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        }
    return response


def is_title_request(body: dict) -> bool:
    """Recognize Claude's structured auxiliary request without using prompt wording."""
    schema = body.get("output_config", {}).get("format", {}).get("schema", {})
    return not body.get("tools") and schema.get("properties") == {"title": {"type": "string"}}
