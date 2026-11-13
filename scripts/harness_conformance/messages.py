# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scripted Messages responses, including deliberately absent usage."""

from nemo_gym.core import NeMoGymResponse
from nemo_gym.models import AnthropicConverter


def messages_response(
    model: str,
    response_id: str,
    output: list[dict],
    *,
    usage: bool,
    index: int,
    input_tokens: int | None = None,
    output_tokens: int = 8,
) -> dict:
    response = AnthropicConverter().responses_to_anthropic_response(
        NeMoGymResponse(
            id=response_id,
            created_at=1,
            model=model,
            object="response",
            output=output,
            parallel_tool_calls=False,
            tool_choice="auto",
            tools=[],
        ),
        model=model,
    )
    # Prescribed provider usage is test data, including its deliberate absence.
    if usage:
        response["usage"] = {
            "input_tokens": 20 + index if input_tokens is None else input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
        }
    else:
        response.pop("usage", None)
    return response


def is_title_request(body: dict) -> bool:
    """Recognize Claude's structured auxiliary request without using prompt wording."""
    schema = body.get("output_config", {}).get("format", {}).get("schema", {})
    return not body.get("tools") and schema.get("properties") == {"title": {"type": "string"}}
