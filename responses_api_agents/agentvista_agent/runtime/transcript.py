# Reference: https://gitlab-master.nvidia.com/pritamb/saas-bench
# Base revision: ad48579c1af66bb7100501f45460bb3e9ec5eaae (Apache-2.0).
# Source: saas_bench/opencode/core.py; local reference file SHA-256:
# c951f983457b09d884c89a05cd78e8d3a37494acc73892cfc860ee330d76c059
"""Native OpenCode events to Gym typed items. Adapted from SaaS harness 2.1.1."""

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json


def response_items(events: list[dict]) -> list[dict]:
    """Translate native CLI events into Gym's typed message/tool transcript."""
    items = []
    for index, event in enumerate(events):
        part = event.get("part") or {}
        event_type = event.get("type")
        if event_type == "text" and part.get("text"):
            items.append(
                {
                    "id": part.get("id", f"msg_{index}"),
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": part["text"], "annotations": []}],
                }
            )
        elif event_type == "reasoning" and part.get("text"):
            items.append(
                {
                    "id": part.get("id", f"reasoning_{index}"),
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": part["text"]}],
                }
            )
        elif event_type == "tool_use":
            state = part.get("state") or {}
            call_id = part.get("callID", f"call_{index}")
            items.append(
                {
                    "type": "function_call",
                    "id": call_id,
                    "call_id": call_id,
                    "name": part.get("tool", "unknown"),
                    "arguments": json.dumps(state.get("input", {})),
                    "status": "completed" if state.get("status") == "completed" else "incomplete",
                }
            )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": str(state.get("output", state.get("error", ""))),
                }
            )
    return items
