# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copied from https://github.com/katerynaCh/Gym at
# 168e95ee80780737273dda011d31d1d5c58f35ce, Apache-2.0:
# responses_api_agents/opencode_multimodal_agent/media.py (final_answer only).

import re


def final_answer(output: list[dict], *, strip_reasoning: bool = False) -> str:
    """Extract the scoring copy; optional training cleanup never mutates history."""
    for item in reversed(output):
        if item.get("type") in {"reasoning", "function_call"}:
            break
        if item.get("type") == "message" and item.get("role") == "assistant":
            text = "".join(part["text"] for part in item["content"] if part["type"] == "output_text")
            if not strip_reasoning:
                return text
            text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL)
            text = re.split(r"<(?:think|thinking)>", text, maxsplit=1)[0]
            return re.split(r"</(?:think|thinking)>", text)[-1].strip()
    return ""
