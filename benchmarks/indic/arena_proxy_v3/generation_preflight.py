# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check that the served model separates reasoning from its visible answer."""

import argparse
import json
from urllib.request import Request, urlopen


def validate_preflight(*, content: str, reasoning: str) -> None:
    """Reject missing answers or reasoning that leaked into the answer field."""
    if "<think>" in content or "</think>" in content:
        raise ValueError("Thinking preflight failed: reasoning delimiters leaked into content")
    if not content.strip() or not reasoning.strip():
        raise ValueError("Thinking preflight failed: a separate reasoning field and final answer are required")


def main() -> None:
    """Probe the chat endpoint before any benchmark rows are generated."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=-1)
    args = parser.parse_args()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "भारत की राजधानी क्या है? एक वाक्य में उत्तर दें।"}],
        "temperature": 1.0,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": 42,
        "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    request = Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=900) as response:
        message = json.load(response)["choices"][0]["message"]
    validate_preflight(
        content=message.get("content") or "",
        reasoning=message.get("reasoning_content") or message.get("reasoning") or "",
    )
    print("Thinking preflight passed.")


if __name__ == "__main__":
    main()
