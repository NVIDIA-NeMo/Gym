# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from extract_answer in mmrl/artifacts.py at revision b0d39e87f3a30391ae5aeaa02a08ff3cc539f9cc of NVIDIA's
# visual-coding RL research code, an NVIDIA-internal repository with no public URL. NVIDIA releases this file under
# Apache-2.0.
import re


def extract_answer(text: str, language: str) -> str:
    """Use final answer only; never salvage an unfinished thinking/code fragment."""
    text = text.strip()
    if "<think>" in text:
        match = re.fullmatch(r"<think>[\s\S]*?</think>\s*([\s\S]*)", text)
        if not match:
            raise ValueError("Unfinished thinking")
        text = match[1].strip()
    # [^\S\n]* (whitespace other than a newline) stays on one line, so a long run of blank lines cannot make this
    # match take cubic time. Leading blank lines are removed after the match: an SVG whose <?xml declaration
    # follows a blank line does not parse.
    fence = re.fullmatch(r"```([\w+-]*)[^\S\n]*\n([\s\S]*?)\n[^\S\n]*```", text)
    if fence:
        aliases = {
            "python": {"python", "py"},
            "svg": {"svg", "xml"},
            "html": {"html"},
            "diff": {"diff", "patch"},
            "latex": {"latex", "tex"},
        }
        if fence[1].lower() not in aliases[language]:
            raise ValueError("Wrong code fence")
        text = re.sub(r"\A\s*\n", "", fence[2])
    if not text.strip() or "```" in text or "<think>" in text or "</think>" in text:
        raise ValueError("Expected one complete artifact")
    return text


# The grading sandbox has no copy of the input image, so an answer that shows it must carry its bytes in the source.
# Runs of 32+ base64 or hex characters that only up to 16 quotes, whitespace, commas, plus signs or backslashes
# separate (how long literals are wrapped) are chained. 1024 characters hold 768 bytes of base64 or 512 of hex: less
# than the smallest example reference (958 bytes), and more than drawing code writes in such a chain.
ENCODED_RUN = re.compile(r"[A-Za-z0-9+/]{32,}")
WRAP_GAP = re.compile(r"""[\s"'+,\\]{0,16}""")
# SVG path data written without spaces ("M0 0h1v1h1v1...") forms the same runs, but it uses only the command letters,
# digits, "+" and the exponent "e". The base64 or hex of 1024 characters of image data has other characters as well.
PATH_DATA = re.compile(r"[MmZzLlHhVvCcSsQqTtAaEe0-9+]+")


def embeds_image_data(source: str) -> bool:
    """Whether the source carries base64 or hex data long enough to hold an image."""
    chain, encoded, end = 0, False, None
    for run in ENCODED_RUN.finditer(source):
        wrapped = end is not None and WRAP_GAP.fullmatch(source, end, run.start()) is not None
        chain = chain + len(run[0]) if wrapped else len(run[0])
        encoded = (wrapped and encoded) or PATH_DATA.fullmatch(run[0]) is None
        if encoded and chain >= 1024:
            return True
        end = run.end()
    return False
