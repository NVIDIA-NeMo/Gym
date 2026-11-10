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
"""Generate tasks whose required nullable tool arguments should be JSON null."""

import argparse
import json
import random
from pathlib import Path


OUTPUT_FPATH = Path(__file__).parent / "data" / "nullable_tool_args_benchmark.jsonl"

FIRST = ["John", "Priya", "Wei", "Amara", "Lucas", "Sofia", "Kenji", "Fatima", "Mateo", "Olga", "Noah", "Aisha"]
LAST = ["Smith", "Patel", "Chen", "Okafor", "Silva", "Rossi", "Tanaka", "Haddad", "Garcia", "Ivanova", "Brown", "Khan"]
AIRPORTS = ["SFO", "JFK", "LHR", "NRT", "CDG", "SEA", "ORD", "BOS", "DFW", "AMS"]
TITLES = ["Weekly sync", "Budget review", "1:1", "Design review", "Standup", "Retro"]
BUGS = ["Login page returns 500", "Export button does nothing", "Search is slow", "Typo on pricing page"]
STATUS = ["pending", "shipped", "delivered", "cancelled"]

# (name, description, {param: description}, nullable params, request templates, correct call arguments)
TOOLS = [
    (
        "create_contact",
        "Create an address book contact.",
        {
            "first_name": "First name.",
            "middle_name": "Middle name, or null if none.",
            "last_name": "Last name.",
            "email": "Email address.",
        },
        ["middle_name"],
        [
            "Add a contact named {first} {last} with email {email}.",
            "Save {first} {last}, {email}, to my contacts.",
            "New contact: {first} {last} ({email}). No middle name.",
        ],
        {"first_name": "{first}", "last_name": "{last}", "email": "{email}"},
    ),
    (
        "create_event",
        "Create a calendar event.",
        {"title": "Event title.", "start": "Start time in ISO 8601.", "location": "Location, or null if none."},
        ["location"],
        [
            "Put '{title}' on my calendar for {date}T{hour:02d}:00. It's a video call, no location.",
            "Schedule {title} at {date}T{hour:02d}:00 with no location.",
        ],
        {"title": "{title}", "start": "{date}T{hour:02d}:00"},
    ),
    (
        "create_ticket",
        "Open a support ticket.",
        {
            "title": "Short summary.",
            "body": "Details.",
            "assignee": "Username to assign, or null to leave unassigned.",
        },
        ["assignee"],
        [
            "Open a ticket titled '{bug}' with body 'Reported by {first}.' Leave it unassigned.",
            "File an unassigned ticket: '{bug}'. Body: reported by {first}.",
        ],
        {"title": "{bug}", "body": "Reported by {first}."},
    ),
    (
        "search_orders",
        "Search orders.",
        {
            "customer_id": "Customer ID.",
            "status": "Order status filter, or null for any status.",
            "since": "ISO date lower bound, or null for no bound.",
        },
        ["status", "since"],
        ["Show me all orders for customer C-{num}.", "List every order customer C-{num} has ever placed."],
        {"customer_id": "C-{num}"},
    ),
    (
        "search_orders",
        "Search orders.",
        {
            "customer_id": "Customer ID.",
            "status": "Order status filter, or null for any status.",
            "since": "ISO date lower bound, or null for no bound.",
        },
        ["since"],
        [
            "Find {status} orders for customer C-{num}, any date.",
            "Which orders for C-{num} are {status}? No date limit.",
        ],
        {"customer_id": "C-{num}", "status": "{status}"},
    ),
    (
        "send_email",
        "Send an email.",
        {
            "to": "Recipient address.",
            "subject": "Subject line.",
            "body": "Body text.",
            "cc": "CC address, or null for none.",
        },
        ["cc"],
        [
            "Email {email} with subject '{title}' and body 'See you there.'",
            "Send {first} ({email}) a note titled '{title}' saying 'See you there.'",
        ],
        {"to": "{email}", "subject": "{title}", "body": "See you there."},
    ),
    (
        "update_profile",
        "Update the user's profile. Pass null to clear a field.",
        {"user_id": "User ID.", "phone": "New phone number, or null to clear it."},
        ["phone"],
        ["Remove the phone number from user u_{num}'s profile.", "Clear the phone field for u_{num}."],
        {"user_id": "u_{num}"},
    ),
    (
        "book_flight",
        "Book a one-way or round-trip flight.",
        {
            "origin": "Origin airport code.",
            "destination": "Destination airport code.",
            "depart_date": "Departure date.",
            "return_date": "Return date, or null for one-way.",
        },
        ["return_date"],
        ["Book me a one-way flight from {a} to {b} on {date}.", "One-way {a} to {b}, leaving {date}, please."],
        {"origin": "{a}", "destination": "{b}", "depart_date": "{date}"},
    ),
]


def make_tool(rng: random.Random, spec: tuple) -> dict:
    name, description, params, nullable, _, _ = spec
    as_any_of = rng.random() < 0.5
    properties = {}
    for param, param_description in params.items():
        if param not in nullable:
            properties[param] = {"type": "string", "description": param_description}
        elif as_any_of:
            properties[param] = {"anyOf": [{"type": "string"}, {"type": "null"}], "description": param_description}
        else:
            properties[param] = {"type": ["string", "null"], "description": param_description}
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": rng.random() < 0.5,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(params),
            "additionalProperties": False,
        },
    }


def make_request(rng: random.Random, spec: tuple) -> tuple[str, dict]:
    """Return a user request for `spec` and the correct call arguments."""
    first, last = rng.choice(FIRST), rng.choice(LAST)
    a, b = rng.sample(AIRPORTS, 2)
    values = dict(
        first=first,
        last=last,
        email=f"{first}.{last}@example.com".lower(),
        title=rng.choice(TITLES),
        bug=rng.choice(BUGS),
        status=rng.choice(STATUS),
        num=rng.randint(100, 9999),
        a=a,
        b=b,
        date=f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
        hour=rng.randint(8, 17),
    )
    _, _, _, nullable, prompts, args = spec
    arguments = {k: v.format(**values) for k, v in args.items()} | dict.fromkeys(nullable)
    return rng.choice(prompts).format(**values), arguments


def make_task(rng: random.Random, position: str) -> dict:
    """`first`: the model's first tool call. `mid`: an earlier correct call with null arguments is in the history."""
    spec = rng.choice(TOOLS)
    tools = [make_tool(rng, spec)]
    history = []
    if position == "mid":
        earlier = rng.choice(TOOLS)
        if earlier[0] == spec[0]:
            earlier = spec
        else:
            tools.append(make_tool(rng, earlier))
        prompt, arguments = make_request(rng, earlier)
        history = [
            {"role": "user", "content": prompt},
            {"type": "function_call", "call_id": "call_0", "name": earlier[0], "arguments": json.dumps(arguments)},
            {"type": "function_call_output", "call_id": "call_0", "output": json.dumps({"status": "ok"})},
            {"role": "assistant", "content": "Done."},
        ]
    prompt, _ = make_request(rng, spec)
    return {
        "responses_create_params": {"input": history + [{"role": "user", "content": prompt}], "tools": tools},
        "verifier_metadata": {"null_args": spec[3], "position": position},
    }


def prepare(num_samples: int = 2000, seed: int = 0) -> Path:
    rng = random.Random(seed)
    OUTPUT_FPATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w") as f:
        for i in range(num_samples):
            f.write(json.dumps(make_task(rng, "first" if i % 2 == 0 else "mid")) + "\n")
    print(f"Wrote {num_samples} samples to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(args.num_samples, args.seed)
