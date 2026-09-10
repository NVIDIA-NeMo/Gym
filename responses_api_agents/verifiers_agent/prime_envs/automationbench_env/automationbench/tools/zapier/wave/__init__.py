# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Wave tools for AutomationBench.

Tools map 1:1 with Zapier Wave actions.
"""

from automationbench.tools.zapier.wave.actions import (
    wave_create_customer,
    wave_create_invoice,
    wave_create_product,
    wave_find_customer,
    wave_find_product,
    wave_list_invoices,
    wave_record_sale,
    wave_send_invoice,
    wave_update_product,
)

__all__ = [
    # Customer tools
    "wave_create_customer",
    "wave_find_customer",
    # Invoice tools
    "wave_create_invoice",
    "wave_list_invoices",
    "wave_send_invoice",
    # Product tools
    "wave_create_product",
    "wave_find_product",
    "wave_update_product",
    # Sale tools
    "wave_record_sale",
]
