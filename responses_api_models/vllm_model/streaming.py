# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read upstream SSE through Gym's aiohttp client and retain the complete completion."""

import json

from openai._streaming import SSEDecoder
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionChunk


async def streamed_chat_completion(client, body):
    """Only transport streams; callers still receive a complete non-streaming response.

    A disconnected or errored stream is never converted into a successful partial answer.
    The SDK accumulator preserves reasoning extensions and fragmented tool arguments.
    """
    response = await client._request(method="POST", url=client.base_url + "/chat/completions", json=body)
    try:
        await client._raise_for_status(response, {"json": body})
        state = ChatCompletionStreamState()
        done = False
        async for event in SSEDecoder().aiter_bytes(response.content.iter_any()):
            if event.data == "[DONE]":
                done = True
                break
            if not event.data:
                continue
            chunk = json.loads(event.data)
            if event.event == "error" or chunk.get("error"):
                raise RuntimeError("Upstream model stream failed: " + event.data)
            state.handle_chunk(ChatCompletionChunk.model_validate(chunk))
        if not done:
            raise RuntimeError("Upstream model stream ended without [DONE]")
        result = state.get_final_completion().model_dump()
        if not result.get("choices") or any(not choice.get("finish_reason") for choice in result["choices"]):
            raise RuntimeError("Upstream model stream has no completed choice")
        if not result.get("usage"):
            raise RuntimeError("Upstream model stream omitted requested token usage")
        for choice in result["choices"]:
            message = choice["message"]
            message.pop("parsed", None)
            for tool in message.get("tool_calls") or []:
                if tool.get("function"):
                    tool["function"].pop("parsed_arguments", None)
        return result
    finally:
        response.release()
