#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Convert raw Perplexity dataset JSONL files to NeMo Gym format.

Supported raw formats:
  Pre-baked trajectory datasets (model receives full trajectory, generates summary only):
  - perplexity_user_if:               messages (full trajectory) + question + instruction
  - perplexity_search:                messages (full trajectory) + reference_answer + tools
  - perplexity_chat:                  messages (system + user only) + reference_answer
  - perplexity_abstention:            messages (full trajectory) + question + instruction

  Fresh rollout datasets (model starts from scratch, makes its own tool calls):
  - perplexity_frames (HuggingFace):  Prompt + Answer (+ wiki_content, etc.)
  - perplexity_facts_grounding (HuggingFace): prompt + response + ground_truth (varies by HF schema)

Usage:
  python preprocess_to_gym.py \\
      --input /path/to/raw.jsonl \\
      --output /path/to/output.jsonl \\
      --dataset_name perplexity_user_if

  # For HuggingFace FRAMES:
  python preprocess_to_gym.py \\
      --input /path/to/frames.jsonl \\
      --output /path/to/output.jsonl \\
      --dataset_name perplexity_frames
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SEARCH_WEB_TOOL = {
    "type": "function",
    "name": "search_web",
    "strict": True,
    "description": 'Searches the web for current and factual information to answer user queries, returning relevant results with titles, URLs, and content snippets, similar to Google or Bing. Intended for questions about up-to-date or externally verified information beyond your knowledge cutoff. The tool works best with an array of short, keyword-focused queries. Complex queries that require multi-step reasoning are not supported. Time-sensitive queries are supported if the date is included in the query.\n\nBest practices for using this tool:\n- Limit the number of queries in each request to a maximum of three to maintain efficiency.\n- For multi-entity questions, break them into separate, single-entity queries:\n  - Preferred:\n    [\n      "Brand A protein powder review",\n      "Brand B protein powder review"\n    ]\n  - Not recommended:\n    [\n      "Brand A vs Brand B protein powder review"\n    ]\n\n- For simple queries, keep each query straightforward and focused:\n  - Preferred: ["inflation rate Canada"]\n  - Not recommended: ["What is the inflation rate in Canada?"]\n\nEach query should be short to ensure optimal tool performance. Make sure all provided examples and generated queries follow this guideline.',
    "parameters": {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "description": "An array of keyword-based search queries. Each query should be short, as longer queries may reduce performance. Do not provide more than three queries to maintain efficiency.",
                "items": {"type": "string"},
            }
        },
        "required": ["queries"],
        "additionalProperties": False,
    },
}

from resources_servers.perplexity_summarizer.prompts import (
    SYSTEM_PROMPT_PERPLEXITY_FACTS_GROUNDING,
    SYSTEM_PROMPT_PERPLEXITY_FRAMES,
)


def _extract_first_user_query(messages: list[dict]) -> str:
    """Extract the first user message content from a message list."""
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def _messages_to_input(messages: list[dict]) -> list[dict]:
    """Convert Chat Completions messages to Responses API input format.

    Chat Completions uses role-based messages with tool_calls/tool_call_id.
    Responses API uses typed items: message, function_call, function_call_output.
    """
    items = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role in ("system", "user"):
            items.append({"role": role, "content": content, "type": "message"})
        elif role == "assistant":
            # Emit text content as a message (if non-empty)
            if content and content.strip():
                items.append({"role": "assistant", "content": content, "type": "message"})
            # Emit each tool call as a separate function_call item
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                arguments = fn.get("arguments", "")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                items.append(
                    {
                        "type": "function_call",
                        "name": fn.get("name", ""),
                        "arguments": arguments,
                        "call_id": tc.get("id", ""),
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content,
                }
            )
    return items


def convert_perplexity_user_if(row: dict, idx: int) -> dict:
    """Convert a perplexity_user_if raw row. Preserves full pre-baked trajectory."""
    messages = row["messages"]
    query_text = row.get("question", _extract_first_user_query(messages))
    return {
        "responses_create_params": {
            "input": _messages_to_input(messages),
            "tools": [SEARCH_WEB_TOOL],
        },
        "dataset_name": "perplexity_user_if",
        "example_id": f"perplexity_user_if_{idx:04d}",
        "query": query_text,
        "instruction": row.get("instruction"),
    }


def convert_perplexity_search(row: dict, idx: int) -> dict:
    """Convert a perplexity_search raw row. Preserves full pre-baked trajectory."""
    messages = row["messages"]
    query = _extract_first_user_query(messages)
    return {
        "responses_create_params": {
            "input": _messages_to_input(messages),
            "tools": [SEARCH_WEB_TOOL],
        },
        "dataset_name": "perplexity_search",
        "example_id": f"perplexity_search_{idx:04d}",
        "query": query,
        "reference_answer": row.get("reference_answer"),
    }


def convert_perplexity_chat(row: dict, idx: int) -> dict:
    """Convert a perplexity_chat raw row. System + user only (no tools, no trajectory)."""
    messages = row["messages"]
    query = _extract_first_user_query(messages)
    return {
        "responses_create_params": {
            "input": _messages_to_input(messages),
            "tools": [],
        },
        "dataset_name": "perplexity_chat",
        "example_id": f"perplexity_chat_{idx:04d}",
        "query": query,
        "reference_answer": row.get("reference_answer"),
    }


def convert_perplexity_abstention(row: dict, idx: int) -> dict:
    """Convert a perplexity_abstention raw row. Preserves full pre-baked trajectory."""
    messages = row["messages"]
    query_text = row.get("question", _extract_first_user_query(messages))
    return {
        "responses_create_params": {
            "input": _messages_to_input(messages),
            "tools": [SEARCH_WEB_TOOL],
        },
        "dataset_name": "perplexity_abstention",
        "example_id": f"perplexity_abstention_{idx:04d}",
        "query": query_text,
        "instruction": row.get("instruction"),
        "abstention_answer": row.get("abstention_answer"),
        "original_answer": row.get("original_answer"),
    }


def convert_perplexity_frames_hf(row: dict, idx: int) -> dict:
    """Convert a HuggingFace FRAMES row (google/frames-benchmark)."""
    query = row.get("Prompt", row.get("prompt", ""))
    ground_truth = row.get("Answer", row.get("answer", row.get("ground_truth", "")))
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT_PERPLEXITY_FRAMES, "type": "message"},
                {"role": "user", "content": query, "type": "message"},
            ],
            "tools": [SEARCH_WEB_TOOL],
        },
        "dataset_name": "perplexity_frames",
        "example_id": f"perplexity_frames_{idx:04d}",
        "query": query,
        "ground_truth": ground_truth,
    }


def convert_perplexity_facts_grounding_hf(row: dict, idx: int) -> dict:
    """Convert a HuggingFace FACTS-grounding row.

    Schema: system_instruction, user_request, context_document, full_prompt.
    The user_request is the query sent to the model. The context_document is
    used only as ground truth for grading — NOT included in the model input.
    This matches lotus's SingleFactEval which sends only the question and
    expects the model to use search_web to find the answer.
    """
    user_request = row.get("user_request", row.get("prompt", ""))
    context_document = row.get("context_document", row.get("ground_truth", ""))
    return {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT_PERPLEXITY_FACTS_GROUNDING, "type": "message"},
                {"role": "user", "content": user_request, "type": "message"},
            ],
            "tools": [SEARCH_WEB_TOOL],
        },
        "dataset_name": "perplexity_facts_grounding",
        "example_id": f"perplexity_facts_grounding_{idx:04d}",
        "query": user_request,
        "ground_truth": context_document,
    }


def convert_perplexity_language_mismatch(row: dict, idx: int) -> dict:
    """Convert a perplexity_language_mismatch raw row. Preserves full pre-baked trajectory."""
    messages = row["messages"]
    query_text = row.get("question", _extract_first_user_query(messages))
    return {
        "responses_create_params": {
            "input": _messages_to_input(messages),
            "tools": [SEARCH_WEB_TOOL],
        },
        "dataset_name": "perplexity_language_mismatch",
        "example_id": f"perplexity_language_mismatch_{idx:04d}",
        "query": query_text,
        "instruction": row.get("instruction"),
    }


CONVERTERS = {
    "perplexity_user_if": convert_perplexity_user_if,
    "perplexity_search": convert_perplexity_search,
    "perplexity_chat": convert_perplexity_chat,
    "perplexity_abstention": convert_perplexity_abstention,
    "perplexity_frames": convert_perplexity_frames_hf,
    "perplexity_facts_grounding": convert_perplexity_facts_grounding_hf,
    "perplexity_language_mismatch": convert_perplexity_language_mismatch,
}


def main():
    parser = argparse.ArgumentParser(description="Convert raw dataset JSONL to NeMo Gym format.")
    parser.add_argument("--input", required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", required=True, help="Path to output JSONL file.")
    parser.add_argument(
        "--dataset_name",
        required=True,
        choices=list(CONVERTERS.keys()),
        help="Dataset name (determines conversion logic).",
    )
    args = parser.parse_args()

    converter = CONVERTERS[args.dataset_name]
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for idx, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: skipping line {idx} (JSON error: {e.msg} at pos {e.pos})", file=sys.stderr)
                skipped += 1
                continue
            converted = converter(row, count + 1)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            count += 1

    print(f"Converted {count} rows from {args.dataset_name} -> {output_path}")
    if skipped:
        print(f"  Skipped {skipped} malformed lines.", file=sys.stderr)


if __name__ == "__main__":
    main()
