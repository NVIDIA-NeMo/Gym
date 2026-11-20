# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Route visual-coding rows to a harness, and optionally give the image to it as a workspace file.

Every row gets ``input_images``, which the resources server writes into the lent sandbox as
``/workspace/input-<i>.<ext>``. With ``--file-input`` the prompt carries those paths in place of the inline image,
for a harness (Claude Code) whose sandbox adapter takes only text.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path


HARNESSES = ("opencode", "pi", "codex", "claude_code", "openclaw")
WORKDIR = "/workspace"
EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}


def _extension(url: str) -> str:
    mime = url.partition(";base64,")[0].removeprefix("data:")
    if mime not in EXTENSIONS:
        raise ValueError("Visual-coding images must be base64 data URLs of a PNG, JPEG, WebP or GIF")
    return EXTENSIONS[mime]


def prepare_row(row: dict, *, harness: str, file_input: bool) -> dict:
    row = deepcopy(row)
    params = row["responses_create_params"]
    messages = [{"role": "user", "content": params["input"]}] if isinstance(params["input"], str) else params["input"]
    if len(messages) != 1 or messages[0]["role"] != "user" or isinstance(messages[0]["content"], str):
        raise ValueError("Visual-coding rows carry one user message of text and image parts")
    parts = messages[0]["content"]
    images = [part["image_url"] for part in parts if part["type"] == "input_image"]
    if not images:
        raise ValueError("Visual-coding rows need an image")
    if any(part["type"] not in ("input_image", "input_text") for part in parts):
        raise ValueError("Visual-coding rows carry only text and image parts")
    row["input_images"] = images
    if file_input:
        texts = [part["text"] for part in parts if part["type"] == "input_text"]
        paths = [f"{WORKDIR}/input-{index}.{_extension(url)}" for index, url in enumerate(images)]
        params["input"] = [{"role": "user", "content": "\n\n".join([*texts, "Image files:\n" + "\n".join(paths)])}]
    row["agent_ref"] = {"name": f"visual_coding_{harness}_agent"}
    row["_ng_environment_server"] = f"visual_coding_{harness}_environment_server"
    return row


def prepare(source: Path, output: Path, *, harness: str = "opencode", file_input: bool = False) -> int:
    if harness not in HARNESSES:
        raise ValueError(f"Unsupported harness: {harness}")
    if harness == "claude_code" and not file_input:
        raise ValueError("Claude Code takes only text in its prompt; use --file-input")
    if source.resolve() == output.resolve():
        raise ValueError("Use a separate output file")
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open() as src, output.open("x") as dst:
        for line in src:
            dst.write(json.dumps(prepare_row(json.loads(line), harness=harness, file_input=file_input)) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--harness", choices=HARNESSES, default="opencode")
    parser.add_argument("--file-input", action="store_true", help="Give the image to the prompt as workspace paths")
    args = parser.parse_args()
    print(f"Prepared {prepare(args.source, args.output, harness=args.harness, file_input=args.file_input)} rows")
