# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Xero tools for AutomationBench.

Tools map 1:1 with Zapier Xero actions.
"""

from automationbench.tools.zapier.xero.contacts import (
    xero_create_contact,
    xero_find_contact,
    xero_update_contact,
)
from automationbench.tools.zapier.xero.financial import (
    xero_allocate_credit_note,
    xero_create_account,
    xero_create_bank_transaction,
    xero_create_bank_transfer,
    xero_create_credit_note,
    xero_create_payment,
    xero_find_bank_transaction,
    xero_find_credit_note,
    xero_find_payment,
)
from automationbench.tools.zapier.xero.invoices import (
    xero_create_bill,
    xero_create_sales_invoice,
    xero_email_invoice,
    xero_find_invoice,
    xero_update_sales_invoice,
)
from automationbench.tools.zapier.xero.resources import (
    xero_create_employee,
    xero_create_inventory_item,
    xero_create_purchase_order,
    xero_create_quote,
    xero_find_employee,
    xero_find_item,
    xero_find_purchase_order,
    xero_find_quote,
    xero_update_employee,
    xero_update_purchase_order,
)

__all__ = [
    # Contact tools
    "xero_create_contact",
    "xero_find_contact",
    "xero_update_contact",
    # Invoice tools
    "xero_create_sales_invoice",
    "xero_update_sales_invoice",
    "xero_email_invoice",
    "xero_find_invoice",
    # Bill tools
    "xero_create_bill",
    # Payment tools
    "xero_create_payment",
    "xero_find_payment",
    # Bank tools
    "xero_create_bank_transaction",
    "xero_create_bank_transfer",
    "xero_find_bank_transaction",
    # Credit note tools
    "xero_create_credit_note",
    "xero_find_credit_note",
    "xero_allocate_credit_note",
    # Account tools
    "xero_create_account",
    # Item tools
    "xero_create_inventory_item",
    "xero_find_item",
    # Employee tools
    "xero_create_employee",
    "xero_update_employee",
    "xero_find_employee",
    # Purchase order tools
    "xero_create_purchase_order",
    "xero_find_purchase_order",
    "xero_update_purchase_order",
    # Quote tools
    "xero_create_quote",
    "xero_find_quote",
]
