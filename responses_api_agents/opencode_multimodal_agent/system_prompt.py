# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the stock OpenCode plugin for an exact system-prompt override."""

import json


def system_prompt_plugin(prompt: str) -> str:
    """Replace the build prompt and appended environment text, preserving other agents."""
    return (
        f"const prompt = {json.dumps(prompt)};\n"
        "export default async () => ({\n"
        '  "experimental.chat.system.transform": async (_input, output) => {\n'
        "    if (output.system.some(text => text.includes(prompt))) {\n"
        "      output.system.splice(0, output.system.length, prompt);\n"
        "    }\n"
        "  }\n"
        "});\n"
    )
