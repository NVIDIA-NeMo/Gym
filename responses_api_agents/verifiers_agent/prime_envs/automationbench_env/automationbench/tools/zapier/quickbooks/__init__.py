# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks Online tools for AutomationBench.

Tools map 1:1 with Zapier QuickBooks V3 actions.
"""

from automationbench.tools.zapier.quickbooks.bills import (
    quickbooks_create_bill,
    quickbooks_create_bill_v2,
    quickbooks_update_bill,
)
from automationbench.tools.zapier.quickbooks.customers import (
    quickbooks_create_customer,
    quickbooks_find_customer,
    quickbooks_update_customer,
)
from automationbench.tools.zapier.quickbooks.estimates import (
    quickbooks_create_estimate,
    quickbooks_create_sales_receipt,
    quickbooks_find_estimate,
    quickbooks_find_sales_receipt,
    quickbooks_send_estimate,
    quickbooks_send_sales_receipt,
)
from automationbench.tools.zapier.quickbooks.invoices import (
    quickbooks_create_invoice,
    quickbooks_create_invoice_by_name,
    quickbooks_create_invoice_v2,
    quickbooks_create_invoice_v3,
    quickbooks_create_invoice_v4,
    quickbooks_find_invoice,
    quickbooks_send_invoice,
    quickbooks_void_invoice,
)
from automationbench.tools.zapier.quickbooks.items import (
    quickbooks_create_item,
    quickbooks_find_product,
    quickbooks_update_item,
)
from automationbench.tools.zapier.quickbooks.payments import (
    quickbooks_create_bill_payment,
    quickbooks_create_payment,
    quickbooks_find_payment,
    quickbooks_query,
)
from automationbench.tools.zapier.quickbooks.vendors import (
    quickbooks_create_vendor,
    quickbooks_find_vendor,
    quickbooks_update_vendor,
)

__all__ = [
    # Customer tools
    "quickbooks_create_customer",
    "quickbooks_update_customer",
    "quickbooks_find_customer",
    # Invoice tools
    "quickbooks_create_invoice",
    "quickbooks_create_invoice_v2",
    "quickbooks_create_invoice_v3",
    "quickbooks_create_invoice_v4",
    "quickbooks_create_invoice_by_name",
    "quickbooks_send_invoice",
    "quickbooks_void_invoice",
    "quickbooks_find_invoice",
    # Bill tools
    "quickbooks_create_bill",
    "quickbooks_create_bill_v2",
    "quickbooks_update_bill",
    # Payment tools
    "quickbooks_create_payment",
    "quickbooks_create_bill_payment",
    "quickbooks_find_payment",
    # Query tools
    "quickbooks_query",
    # Vendor tools
    "quickbooks_create_vendor",
    "quickbooks_update_vendor",
    "quickbooks_find_vendor",
    # Item tools
    "quickbooks_create_item",
    "quickbooks_update_item",
    "quickbooks_find_product",
    # Estimate tools
    "quickbooks_create_estimate",
    "quickbooks_send_estimate",
    "quickbooks_find_estimate",
    # Sales receipt tools
    "quickbooks_create_sales_receipt",
    "quickbooks_send_sales_receipt",
    "quickbooks_find_sales_receipt",
]
