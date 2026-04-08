#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate JSONL data for customer_service_env.

Usage:
  python scripts/generate_customer_service_data.py --num 500 --seed 42 --output data/train.jsonl
  python scripts/generate_customer_service_data.py --num 5 --seed 0 --output resources_servers/customer_service_env/data/example.jsonl
"""

import argparse
import json
import random

SYSTEM_PROMPT = (
    "You are a customer service agent for TechShop. Help the customer resolve their issue. "
    "Be polite, ask clarifying questions if needed, and take appropriate action. "
    "You can offer refunds, replacements, order cancellations, or status updates as needed."
)

PRODUCTS = [
    {"id": "P001", "name": "Wireless Headphones", "price": 79.99},
    {"id": "P002", "name": "USB-C Cable", "price": 12.99},
    {"id": "P003", "name": "Laptop Stand", "price": 45.00},
    {"id": "P004", "name": "Mechanical Keyboard", "price": 129.99},
    {"id": "P005", "name": "Mouse Pad", "price": 19.99},
    {"id": "P006", "name": "Webcam HD", "price": 59.99},
    {"id": "P007", "name": "Monitor Light Bar", "price": 34.99},
    {"id": "P008", "name": "Desk Organizer", "price": 24.99},
    {"id": "P009", "name": "Phone Charger", "price": 15.99},
    {"id": "P010", "name": "Bluetooth Speaker", "price": 49.99},
]

FIRST_NAMES = ["Alex", "Jordan", "Sam", "Morgan", "Casey", "Riley", "Quinn", "Avery", "Taylor", "Drew"]
LAST_NAMES = ["Chen", "Patel", "Kim", "Garcia", "Nguyen", "Smith", "Lee", "Brown", "Wilson", "Singh"]

ISSUES = {
    "refund": {
        "order_status": "delivered",
        "openers": [
            "I'd like a refund for order {order_id}. The {product} isn't what I expected.",
            "Can I get my money back for order {order_id}? The {product} doesn't work well.",
            "I want to return the {product} from order {order_id}.",
        ],
        "resolution_keywords": ["refund", "processed", "credited", "returned"],
    },
    "order_status": {
        "order_status": "in_transit",
        "openers": [
            "Where is my order {order_id}? I placed it {days} days ago.",
            "I haven't received order {order_id} yet. It's been {days} days.",
            "Can you check the status of order {order_id}?",
        ],
        "resolution_keywords": ["transit", "tracking", "arrive", "delivered", "shipping"],
    },
    "wrong_item": {
        "order_status": "delivered",
        "openers": [
            "I received the wrong item. I ordered a {product} but got something else. Order {order_id}.",
            "Order {order_id} has the wrong product. I wanted a {product}.",
        ],
        "resolution_keywords": ["replacement", "reship", "correct item", "new order", "send"],
    },
    "damaged": {
        "order_status": "delivered",
        "openers": [
            "My {product} arrived damaged. Order {order_id}.",
            "The {product} from order {order_id} was broken when it arrived.",
        ],
        "resolution_keywords": ["replacement", "refund", "sorry", "new one"],
    },
    "cancel": {
        "order_status": "processing",
        "openers": [
            "I need to cancel order {order_id}. I changed my mind about the {product}.",
            "Please cancel order {order_id}.",
        ],
        "resolution_keywords": ["cancel", "cancelled", "stopped"],
    },
}

REFUND_POLICY = "Refunds are available within 30 days of delivery for unused items. Damaged items are eligible for immediate refund or replacement."
CANCEL_POLICY = "Orders in 'processing' status can be cancelled. Shipped orders cannot be cancelled but may be returned after delivery."

USER_TOOLS = [
    {
        "type": "function",
        "name": "lookup_order",
        "description": "Look up order details by order ID",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "type": "function",
        "name": "check_account",
        "description": "Check account details by email",
        "parameters": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
    {
        "type": "function",
        "name": "get_policy",
        "description": "Get company policy on refunds, cancellations, etc.",
        "parameters": {
            "type": "object",
            "properties": {"policy_type": {"type": "string", "enum": ["refund", "cancel"]}},
            "required": ["policy_type"],
        },
    },
]


def generate_entry(rng: random.Random) -> dict:
    issue_type = rng.choice(list(ISSUES.keys()))
    config = ISSUES[issue_type]

    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    email = f"{first.lower()}.{last.lower()}@example.com"

    product = rng.choice(PRODUCTS)
    order_id = f"ORD-{rng.randint(1000, 9999)}"
    days = rng.randint(2, 14)

    opener = rng.choice(config["openers"]).format(
        order_id=order_id,
        product=product["name"],
        days=days,
    )

    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": opener},
            ]
        },
        "customer": {"name": name, "email": email},
        "order": {
            "order_id": order_id,
            "product_name": product["name"],
            "price": product["price"],
            "status": config["order_status"],
            "days_since_order": days,
        },
        "issue_type": issue_type,
        "opener": opener,
        "resolution_keywords": config["resolution_keywords"],
        "policies": {"refund": REFUND_POLICY, "cancel": CANCEL_POLICY},
        "user_tools": USER_TOOLS,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    with open(args.output, "w") as f:
        for _ in range(args.num):
            f.write(json.dumps(generate_entry(rng)) + "\n")

    print(f"Generated {args.num} entries -> {args.output}")


if __name__ == "__main__":
    main()
