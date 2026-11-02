# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenAI function-calling schemas for the 33 banking tools plus transfer_to_human_agents
(the latter is a terminal world flag in engine.py, not a banking_tools call)."""

from __future__ import annotations


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "name": "search_knowledge_base",
        "description": "Search the bank's knowledge corpus (RBI circulars, product docs, scheme\n"
        "details, FAQs, how-to guides) with a natural-language query.\n"
        "\n"
        "Only ``query`` is required. Optionally narrow with ``bank_id`` (a "
        "datapoint id\n"
        'like "BNK-0001"), ``category`` (a search type like "Regulatory"), or '
        "``top_k``\n"
        "(number of results). Returns the most relevant chunks, each with its\n"
        "``source_url`` and a relevance ``score``.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "bank_id": {"type": "string"},
                "category": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_account_balance",
        "description": "Return current available balance and hold/lien amount for one or more\nlinked accounts.",
        "parameters": {
            "type": "object",
            "properties": {"account_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["account_ids"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_account_details",
        "description": "Return account metadata — bank name, type, status, home branch, IFSC,\n"
        "holders, nominee, open date. Does not return balances or transactions.",
        "parameters": {
            "type": "object",
            "properties": {"account_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["account_ids"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_transaction_history",
        "description": "Return statement entries for a single account, newest first, with\n"
        "date/direction/channel filters and pagination.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "from_date": {"type": "string"},
                "to_date": {"type": "string"},
                "type": {"type": "string", "enum": ["debit", "credit", "all"]},
                "channel": {"type": "string", "enum": ["upi", "neft", "imps", "rtgs", "atm", "pos", "cheque", "all"]},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
            "required": ["account_id"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_fd_details",
        "description": "Return full details of the customer's Fixed Deposits.",
        "parameters": {
            "type": "object",
            "properties": {"deposit_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["deposit_ids"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_rd_details",
        "description": "Return full details of the customer's Recurring Deposits, including\n"
        "installment amount, installments paid, and total accumulated to date.",
        "parameters": {
            "type": "object",
            "properties": {"deposit_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["deposit_ids"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_deposit_loan_rates",
        "description": "Return the current rate card for a deposit or loan product, by\n"
        "tenure/amount slab. The sole authoritative source for any rate used in a\n"
        "calculator or a booking.",
        "parameters": {
            "type": "object",
            "properties": {
                "product_type": {"type": "string", "enum": ["fd", "rd", "home_loan", "personal_loan", "gold_loan"]},
                "tenure_months": {"type": "integer"},
                "amount": {"type": "number"},
            },
            "required": ["product_type"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "calculate_fd_maturity",
        "description": "Compute maturity value and interest earned for a Fixed Deposit.\n"
        "Compounding is quarterly (bank standard).",
        "parameters": {
            "type": "object",
            "properties": {
                "principal_amount": {"type": "number"},
                "rate": {"type": "number"},
                "tenure_months": {"type": "integer"},
                "payout_type": {"type": "string", "enum": ["cumulative", "monthly", "quarterly"]},
            },
            "required": ["principal_amount", "rate", "tenure_months"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "calculate_rd_maturity",
        "description": "Compute maturity value and interest earned for a Recurring Deposit.\n"
        "Interest is paid at maturity; compounding is quarterly.",
        "parameters": {
            "type": "object",
            "properties": {
                "monthly_installment": {"type": "number"},
                "rate": {"type": "number"},
                "tenure_months": {"type": "integer"},
            },
            "required": ["monthly_installment", "rate", "tenure_months"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "create_fd",
        "description": "Book a new Fixed Deposit, debiting principal_amount from the source\n"
        "account as a one-time transaction.",
        "parameters": {
            "type": "object",
            "properties": {
                "principal_amount": {"type": "number"},
                "tenure_months": {"type": "integer"},
                "source_account": {"type": "string"},
                "payout_type": {"type": "string", "enum": ["cumulative", "monthly", "quarterly"]},
                "maturity_instruction": {
                    "type": "string",
                    "enum": ["no_renewal", "renew_principal", "renew_principal_interest"],
                },
                "nominee": {"type": "object"},
            },
            "required": ["principal_amount", "tenure_months", "source_account"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "create_rd",
        "description": "Book a new Recurring Deposit. Authorizes a recurring monthly debit for "
        "the\n"
        "full tenure; the first installment is debited at booking.",
        "parameters": {
            "type": "object",
            "properties": {
                "monthly_installment": {"type": "number"},
                "tenure_months": {"type": "integer"},
                "source_account": {"type": "string"},
                "maturity_instruction": {
                    "type": "string",
                    "enum": ["no_renewal", "renew_principal", "renew_principal_interest"],
                },
                "nominee": {"type": "object"},
            },
            "required": ["monthly_installment", "tenure_months", "source_account"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_deposit_closure_quote",
        "description": "Quote the payout if an FD or RD were closed today. Before maturity a\n"
        "penalty applies (rate recalculated at the penalised effective rate); "
        "after\n"
        "maturity no penalty applies. Read-only.",
        "parameters": {
            "type": "object",
            "properties": {"deposit_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["deposit_ids"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "close_deposit",
        "description": "Close an FD or RD and credit proceeds to the payout account. Works for\n"
        "premature (penalty) and post-maturity (no penalty) closure. For an RD "
        "this\n"
        "also permanently stops all future installment debits.",
        "parameters": {
            "type": "object",
            "properties": {"deposit_id": {"type": "string"}, "payout_account": {"type": "string"}},
            "required": ["deposit_id"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "update_deposit_renewal",
        "description": "Set the maturity instruction on an FD or RD. Takes effect on the maturity\n"
        "date, not immediately.",
        "parameters": {
            "type": "object",
            "properties": {
                "deposit_id": {"type": "string"},
                "instruction": {
                    "type": "string",
                    "enum": ["no_renewal", "renew_principal", "renew_principal_interest"],
                },
            },
            "required": ["deposit_id", "instruction"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_loan_details",
        "description": "Return current loan state: outstanding balance, EMI, rate, remaining\ntenure, next due date.",
        "parameters": {
            "type": "object",
            "properties": {"loan_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["loan_ids"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_loan_foreclosure_quote",
        "description": "Quote the total payoff to close a loan early as of a given date.\n"
        "Informational only. Also use this when a customer asks to 'pause', 'stop', "
        "or\n"
        "'reduce' EMIs, since no such capability exists.",
        "parameters": {
            "type": "object",
            "properties": {
                "loan_ids": {"type": "array", "items": {"type": "string"}},
                "as_of_date": {"type": "string"},
            },
            "required": ["loan_ids"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "calculate_emi",
        "description": "Compute the fixed monthly EMI, total interest, total repayment, and full\n"
        "amortisation schedule for a loan.",
        "parameters": {
            "type": "object",
            "properties": {
                "principal": {"type": "number"},
                "rate": {"type": "number"},
                "tenure_months": {"type": "integer"},
            },
            "required": ["principal", "rate", "tenure_months"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_gold_rate",
        "description": "Return today's gold rate per gram for a given purity.",
        "parameters": {
            "type": "object",
            "properties": {"purity_karat": {"type": "integer", "enum": [18, 22, 24]}},
            "required": ["purity_karat"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "calculate_gold_loan_ltv",
        "description": "Value gold collateral (grams x rate) and apply the bank's current LTV cap\n"
        "to compute the maximum eligible loan amount. The LTV cap is applied\n"
        "internally by the backend.",
        "parameters": {
            "type": "object",
            "properties": {
                "gold_grams": {"type": "number"},
                "purity_karat": {"type": "integer", "enum": [18, 22, 24]},
                "gold_rate_per_gram": {"type": "number"},
            },
            "required": ["gold_grams", "purity_karat", "gold_rate_per_gram"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_card_details",
        "description": "Return card metadata, status, credit/available limits, and per-channel\n"
        "usage controls. Use to diagnose card declines.",
        "parameters": {
            "type": "object",
            "properties": {"card_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["card_ids"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "toggle_card_freeze",
        "description": "Temporarily freeze or unfreeze a card. Reversible. Cannot unfreeze a card\n"
        "whose status is blocked or expired.",
        "parameters": {
            "type": "object",
            "properties": {"card_id": {"type": "string"}, "state": {"type": "string", "enum": ["freeze", "unfreeze"]}},
            "required": ["card_id", "state"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "block_card",
        "description": "Permanently and irreversibly block a card. A blocked card cannot be\n"
        "unblocked; request a replacement via raise_request.",
        "parameters": {
            "type": "object",
            "properties": {
                "card_id": {"type": "string"},
                "reason": {"type": "string", "enum": ["lost", "stolen", "fraud", "damaged"]},
            },
            "required": ["card_id", "reason"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "set_card_controls",
        "description": "Update per-channel card controls. Each channel (atm, online, pos) has\n"
        "independent domestic and international enable/limit settings. Only "
        "provided\n"
        "fields change. A daily_limit supplied with enabled: false is ignored.",
        "parameters": {
            "type": "object",
            "properties": {
                "card_id": {"type": "string"},
                "atm": {"type": "object"},
                "online": {"type": "object"},
                "pos": {"type": "object"},
            },
            "required": ["card_id"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "show_mandates",
        "description": "List standing mandates (e-NACH, UPI AutoPay, standing instructions) on "
        "the\n"
        "customer's profile.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "status": {"type": "string", "enum": ["active", "paused", "cancelled", "all"]},
            },
            "required": [],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "cancel_mandate",
        "description": "Permanently cancel a mandate, stopping all future debits under it.\n"
        "Already-processed debits are unaffected.",
        "parameters": {"type": "object", "properties": {"mandate_id": {"type": "string"}}, "required": ["mandate_id"]},
        "strict": False,
    },
    {
        "type": "function",
        "name": "stop_cheque_payment",
        "description": "Place a stop-payment instruction on one or more cheques issued from an\n"
        "account. Only affects cheques that have not already cleared.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "cheque_numbers": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string", "enum": ["lost", "dispute", "other"]},
            },
            "required": ["account_id", "cheque_numbers"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "request_cheque_book",
        "description": "Order a new cheque book for a savings or current account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "leaves": {"type": "integer"},
                "delivery_address": {"type": "string"},
            },
            "required": ["account_id"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "request_duplicate_statement",
        "description": "Order an official duplicate account statement for a date range, delivered\n"
        "by email or made available for download.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "from_date": {"type": "string"},
                "to_date": {"type": "string"},
                "delivery_mode": {"type": "string", "enum": ["email", "download"]},
            },
            "required": ["account_id", "from_date", "to_date"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "update_address",
        "description": "Log a request to update the communication or permanent address. The "
        "change\n"
        "is effective only after bank verification — never state it is already\n"
        "updated.",
        "parameters": {
            "type": "object",
            "properties": {
                "address_type": {"type": "string", "enum": ["communication", "permanent"]},
                "line1": {"type": "string"},
                "city": {"type": "string"},
                "state": {"type": "string"},
                "pincode": {"type": "string"},
                "line2": {"type": "string"},
            },
            "required": ["address_type", "line1", "city", "state", "pincode"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "raise_request",
        "description": "Register a complaint or general service request — the single escalation\n"
        "channel for disputes, service issues, and any action no other tool can\n"
        "perform.\n"
        "\n"
        "Agent-only: infer `category` and `related_transaction_id` from the\n"
        "conversation. Do not ask the customer for backend field names or category\n"
        "codes (e.g. unauthorized_debit); confirm the facts in plain language "
        "first.",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": [
                        "failed_transaction",
                        "unauthorized_debit",
                        "service_quality",
                        "app_issue",
                        "charges_dispute",
                        "general_query",
                        "other",
                    ],
                },
                "related_transaction_id": {"type": "string"},
                "account_id": {"type": "string"},
            },
            "required": ["description"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_request_status",
        "description": "Return the current status of a previously raised service request.",
        "parameters": {"type": "object", "properties": {"request_id": {"type": "string"}}, "required": ["request_id"]},
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_insurance_details",
        "description": "Return policy details for the customer's bancassurance policies. No\n"
        "premium-payment or claim tools exist — route those via raise_request.",
        "parameters": {
            "type": "object",
            "properties": {"policy_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["policy_ids"],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "get_products_and_offers",
        "description": "Return a combined list of the bank's products and the customer's\n"
        "personalised offers, optionally filtered by type. Offers are indicative —\n"
        "never state one is guaranteed. Never quote rates from here; use\n"
        "get_deposit_loan_rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["deposit", "loan", "card", "insurance", "offer", "all"]}
            },
            "required": [],
        },
        "strict": False,
    },
    {
        "type": "function",
        "name": "transfer_to_human_agents",
        "description": "Transfer the user to a human agent, with a summary of the user's issue. "
        "Only transfer if the user explicitly asks for a human agent, or if given "
        "the policy and available tools you cannot solve the user's issue.",
        "parameters": {
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "A summary of the user's issue."}},
            "required": ["summary"],
        },
        "strict": False,
    },
]

TOOL_NAMES: list[str] = [s["name"] for s in TOOL_SCHEMAS]

# Mutating tools, mirrors engine.WRITE_TOOLS exactly.
# Handoff and read-only/soft-write tools are excluded: they do not mutate financial state.
WRITE_TOOLS: set[str] = {
    "block_card",
    "create_fd",
    "update_deposit_renewal",
    "stop_cheque_payment",
    "raise_request",
    "cancel_mandate",
    "close_deposit",
    "set_card_controls",
    "create_rd",
    "update_address",
    "toggle_card_freeze",
}
