# Reference: https://gitlab-master.nvidia.com/pritamb/saas-bench
# Base revision: ad48579c1af66bb7100501f45460bb3e9ec5eaae (Apache-2.0).
# Source: saas_bench/opencode/training.py; local reference file SHA-256:
# 73dcf8e240a89bee3f3df2334a0fc01362ef9ecd4312e7fb80dc25de045ccf7e
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join the final native actor action to an actual served model completion."""

from __future__ import annotations

import json


def terminal_response_id(events: list[dict], calls: list[dict], rollout_id: str) -> str:
    """Return a witnessed completion ID; ambiguity or missing evidence is an error.

    Token IDs, sampled log probabilities and image tensors stay in Gym's ledger.
    This helper only identifies the native turn whose outcome was verified.
    """
    actions = [event for event in events if event.get("type") in {"text", "tool_use"}]
    if not actions:
        raise ValueError("No native actor output to bind to training capture")
    last = actions[-1]
    part = last["part"]
    matches = set()
    for call in calls:
        if call["rollout_id"] != rollout_id:
            raise ValueError("Training capture contains a different rollout ID")
        response = call["response"]
        if len(response["choices"]) != 1:
            raise ValueError("Training capture requires exactly one completion")
        message = response["choices"][0]["message"]
        if last["type"] == "text":
            matched = bool(part["text"].strip()) and message.get("content") == part["text"]
        else:
            matched = any(tool["id"] == part["callID"] for tool in message.get("tool_calls") or [])
        if matched:
            matches.add(response["id"])
    if len(matches) != 1:
        raise ValueError(f"Native terminal output matches {len(matches)} served completions")
    return matches.pop()


def read_calls(path):
    """Read complete, append-only model journal entries."""
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]
