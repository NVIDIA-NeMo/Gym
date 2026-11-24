# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers the commands share: message text, argv quoting, newlines, dotted row fields, JSONL and task folders."""

import json
from collections.abc import Iterator
from pathlib import Path


# nemo_gym.data.tasks.harbor.task.MAX_DISCOVERY_DEPTH: how many levels down Gym looks for task folders.
MAX_DISCOVERY_DEPTH = 6


class InputError(Exception):
    """A usage or input problem: the command prints it to stderr and exits 2."""


def task_folders(root: Path) -> list[Path]:
    """The task folders of a dataset, found the way Gym's `iter_task_folders` finds them.

    `root` itself when it has `task.toml`; else every folder with `task.toml` under it, up to
    `MAX_DISCOVERY_DEPTH` levels down, sorted by path, skipping hidden folders and never following a symbolic link.
    Two task folders of the same name raise InputError, as Gym refuses them.
    """
    if (root / "task.toml").is_file():
        return [root]
    found: list[Path] = []

    def walk(folder: Path, depth: int) -> None:
        for child in sorted(folder.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            if (child / "task.toml").is_file():
                found.append(child)
            elif depth < MAX_DISCOVERY_DEPTH and not child.is_symlink():
                walk(child, depth + 1)

    if root.is_dir():
        walk(root, 1)
    seen: dict[str, Path] = {}
    for folder in found:
        if folder.name in seen:
            raise InputError(f"two task folders are named {folder.name}: {seen[folder.name]} and {folder}")
        seen[folder.name] = folder
    return found


def message_text(content: object) -> str:
    """The text of a chat message's `content`: a string, or the joined `text` of a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"] for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return ""


def unquote_argv(text: str) -> str | None:
    """Undo the quoting OpenCode's `run` applies to an argv prompt.

    The prompt arrives wrapped in one pair of `"` with every inner `"` escaped as `\\"`; other backslashes are left
    as they are (the math pilot's recorded requests keep a LaTeX `\\\\cdots` unchanged). Returns the inner text with
    `\\"` turned back into `"`, or None when `text` is not quoted that way.
    """
    if len(text) < 2 or text[0] != '"' or text[-1] != '"':
        return None
    inner = text[1:-1]
    if '"' in inner.replace('\\"', ""):
        return None
    return inner.replace('\\"', '"')


def lf_newlines(text: str) -> str:
    """CRLF and CR turned into LF, as Python's text mode (and so Gym's instruction.md reader) does."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


_MISSING = object()


def field_value(row: dict, dotted: str, index: int) -> object:
    """The value at a dotted path of `row`, or `index` (the 0-based line) for `_index`; raises KeyError if absent."""
    if dotted == "_index":
        return index
    value: object = row
    for part in dotted.split("."):
        value = value.get(part, _MISSING) if isinstance(value, dict) else _MISSING
        if value is _MISSING:
            raise KeyError(dotted)
    return value


def id_text(value: object) -> str:
    """A row id as text: strings as they are, numbers and other JSON values in their JSON form."""
    return value if isinstance(value, str) else json.dumps(value)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    """(0-based line, row) for each non-empty line of a JSONL file, streamed."""
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    with handle:
        for index, raw in enumerate(handle):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise InputError(f"{path} line {index + 1}: not UTF-8 ({exc})") from exc
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path} line {index + 1}: not JSON ({exc})") from exc
            if not isinstance(row, dict):
                raise InputError(f"{path} line {index + 1}: not a JSON object")
            yield index, row


def read_id_list(path: Path) -> list[str]:
    """Ids from a text file, one per line; blank lines and `#` comments are skipped."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError(f"cannot read {path}: {exc}") from exc
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
