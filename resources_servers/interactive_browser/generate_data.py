# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Generate example tasks for the interactive_browser environment.

Each row is a NeMo-Gym rollout input: `responses_create_params` (the system +
user messages and the browser tool schemas the policy may call) plus the extra
fields this env reads — `initial_url` (in seed_session) and `verifier_metadata`
(in verify). Run:

    python generate_data.py          > data/example.jsonl         # offline site/, any backend
    python generate_data.py --remote > data/example_remote.jsonl  # live web, remote backends
"""

import json
import sys


TOOLS = [
    {
        "type": "function",
        "name": "browser_observe",
        "description": "Return the current page: URL, title, and a numbered list of interactive elements ([id] role: name).",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "type": "function",
        "name": "browser_navigate",
        "description": "Navigate to a URL.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    },
    {
        "type": "function",
        "name": "browser_click",
        "description": "Click the interactive element with the given id from the latest observation.",
        "parameters": {
            "type": "object",
            "properties": {"element_id": {"type": "integer"}},
            "required": ["element_id"],
        },
    },
    {
        "type": "function",
        "name": "browser_type",
        "description": "Type text into the element with the given id.",
        "parameters": {
            "type": "object",
            "properties": {"element_id": {"type": "integer"}, "text": {"type": "string"}},
            "required": ["element_id", "text"],
        },
    },
    {
        "type": "function",
        "name": "browser_finish",
        "description": "End the episode, reporting an answer ('' if none).",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": []},
    },
]

# Responses-API strict function tools: additionalProperties:false, strict:true,
# and every property listed in `required`.
for _t in TOOLS:
    _p = _t["parameters"]
    _p["additionalProperties"] = False
    _p["required"] = list(_p.get("properties", {}).keys())
    _t["strict"] = True

SYSTEM = (
    "You are a web agent operating a live browser through tools. Call browser_observe to see the "
    "page (its URL, title, and a numbered list of interactive elements as `[id] role: name`). "
    "Use browser_navigate / browser_click / browser_type to act — element_id values come from the "
    "most recent observation. Call browser_finish when the task is complete."
)

# Offline tasks against the bundled `site/` — deterministic and ToS-safe, and
# the ones CI grades. `initial_url` is resolved relative to this directory.
TASKS = [
    ("Navigate from the home page to the About page, then finish.", {"url_contains": "about.html"}),
    ("Go to the form page, type 'neo' as the username, submit, then finish.", {"dom_contains": "Welcome neo"}),
    ("Open the About page, read the secret code, report it via finish(answer=...).", {"dom_contains": "ALPHA-42"}),
    ("From the home page, reach the form page, then finish there.", {"url_contains": "form.html"}),
    ("Stay on the home page and finish immediately.", {"url_contains": "index.html"}),
]

# Live-web tasks for backends that cannot open this repo's local file:// pages
# (any remote browser). Kept to IANA's example domains: stable, static, and
# explicitly published for this kind of use.
REMOTE_INITIAL_URL = "https://example.com/"
REMOTE_TASKS = [
    ("Report the page's main heading via finish(answer=...).", {"dom_contains": "Example Domain"}),
    ("Follow the 'More information' link, then finish.", {"url_contains": "iana.org"}),
    ("Navigate to https://www.iana.org/help/example-domains and finish there.", {"url_contains": "example-domains"}),
]


def main() -> None:
    remote = "--remote" in sys.argv[1:]
    tasks = REMOTE_TASKS if remote else TASKS
    initial_url = REMOTE_INITIAL_URL if remote else "site/index.html"
    for prompt, verifier_metadata in tasks:
        row = {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "tools": TOOLS,
            },
            "initial_url": initial_url,
            "verifier_metadata": verifier_metadata,
        }
        print(json.dumps(row))


if __name__ == "__main__":
    main()
