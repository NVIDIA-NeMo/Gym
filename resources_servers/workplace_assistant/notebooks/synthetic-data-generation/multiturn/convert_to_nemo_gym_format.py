# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for converting generated multi-turn records to NeMo Gym JSONL format."""

from __future__ import annotations

import json
from typing import Any, Callable

import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy types when serializing to JSON."""

    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        return super().default(o)


def _convert_conversation_to_messages(
    conversation: dict[str, Any],
    system_prompt: str,
) -> list[dict[str, Any]]:
    """Convert conversation trace to standard message format."""
    messages = [{"role": "system", "content": system_prompt}]

    for turn in conversation.get("conversation_trace", []):
        role = turn.get("role", "")
        content = turn.get("content", "")

        # Parse tool calls
        tool_calls = turn.get("tool_calls")
        if tool_calls is None:
            tool_calls = turn.get("tool_call")
        if tool_calls is None:
            tool_calls = []
        if hasattr(tool_calls, "tolist"):
            tool_calls = tool_calls.tolist()
        if not isinstance(tool_calls, list):
            tool_calls = []

        if role == "user":
            messages.append({"role": "user", "content": content})

        elif role == "agent":
            if tool_calls:
                # Agent turn with tool calls
                formatted_tool_calls = []
                for tc in tool_calls:
                    formatted_tool_calls.append(
                        {
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": tc.get("arguments", "{}"),
                            }
                        }
                    )
                messages.append(
                    {
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": formatted_tool_calls,
                    }
                )
                # Add simulated tool results
                for tc in tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps({"status": "success"}),
                        }
                    )
            else:
                # Agent turn without tool calls (clarification question)
                messages.append({"role": "assistant", "content": content})

    return messages


def convert_to_nemo_gym_format(
    row: dict[str, Any],
    idx: int,
    environment_name: str = "workplace_assistant",
) -> dict[str, Any]:
    """Convert a generated multi-turn row to standard conversation format."""
    # Parse fields if they're JSON strings
    ambiguity_query = row.get("ambiguity_query", {})
    if isinstance(ambiguity_query, str):
        ambiguity_query = json.loads(ambiguity_query)

    conversation = row.get("conversation", {})
    if isinstance(conversation, str):
        conversation = json.loads(conversation)

    # Parse ground truth tool calls
    ground_truth = row.get("ground_truth", [])
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    # Parse tools and system prompt from seed data
    tools = json.loads(row["tools_json"]) if isinstance(row["tools_json"], str) else row["tools_json"]
    system_prompt = row.get("system_prompt", "")

    # Convert conversation to standard message format
    conversation_messages = _convert_conversation_to_messages(conversation, system_prompt)

    # Extract just the type name from ambiguity_type JSON
    ambiguity_type_raw = row.get("ambiguity_type", "")
    if isinstance(ambiguity_type_raw, str):
        try:
            ambiguity_type_raw = json.loads(ambiguity_type_raw)
        except (json.JSONDecodeError, TypeError):
            pass
    if isinstance(ambiguity_type_raw, dict):
        ambiguity_type = ambiguity_type_raw.get("type", "")
    else:
        ambiguity_type = str(ambiguity_type_raw)

    return {
        "responses_create_params": {
            "input": conversation_messages,
            "tools": tools,
        },
        "verifier_metadata": {
            "ground_truth": ground_truth,
            "ambiguity_type": ambiguity_type,
            "original_user_query": row.get("user_query", ""),
            "ambiguous_message": ambiguity_query.get("ambiguous_message", ""),
            "removed_info": ambiguity_query.get("removed_info", ""),
            "clarification_targets": ambiguity_query.get("clarification_targets", []),
            "clarification_requirement": ambiguity_query.get("clarification_requirement", ""),
        },
        "category": row.get("category", f"{environment_name}_general"),
        "environment_name": environment_name,
    }


def save_for_nemo_gym(
    df: pd.DataFrame,
    output_path: str,
    convert_fn: Callable[[dict[str, Any], int], dict[str, Any]],
) -> None:
    """Save records as JSONL for NeMo Gym."""
    with open(output_path, "w") as f:
        for idx, row in df.iterrows():
            record = convert_fn(row.to_dict(), idx)
            f.write(json.dumps(record, cls=_NumpyEncoder) + "\n")

    print(f"Saved {len(df)} examples to {output_path}")
