# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference-independent selection of the final LaTeX boxed answer."""

import re


_BOX_COMMAND = re.compile(r"\\boxed(?![A-Za-z])")


def _last_box_command_end(response: str) -> int | None:
    last_end = None
    for match in _BOX_COMMAND.finditer(response):
        preceding_slashes = 0
        index = match.start() - 1
        while index >= 0 and response[index] == "\\":
            preceding_slashes += 1
            index -= 1
        if preceding_slashes % 2 == 0:
            last_end = match.end()
    return last_end


def extract_last_boxed_answer(response: str) -> str | None:
    r"""Return the contents of the last ``\boxed{...}``, or None if it is invalid.

    Whitespace between the command and its argument is accepted. Escaped braces
    are literal content, and nested argument braces remain balanced. An empty or
    malformed last box never falls back to a preceding complete box.
    """
    last_end = _last_box_command_end(response)
    if last_end is None:
        return None
    opening = last_end
    while opening < len(response) and response[opening].isspace():
        opening += 1
    if opening == len(response) or response[opening] != "{":
        return None
    index, depth = opening + 1, 1
    while index < len(response):
        character = response[index]
        if character == "\\":
            index += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                selected = response[opening + 1 : index]
                return selected if selected.strip() else None
        index += 1
    return None
