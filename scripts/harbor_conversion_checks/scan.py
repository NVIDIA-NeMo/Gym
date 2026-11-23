# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What `check` looks for and where: the hidden values of a native row, and a file's bytes with its base64 bundles.

Hidden values are the string and number leaves of a field: the whole leaf if it has `WHOLE_VALUE` or more
characters, each line of `VALUE_LINE` or more characters of a multi-line leaf, and only the added lines of
`PATCH_LINE` or more characters of a patch. A string holding a JSON array (scale_swe's `FAIL_TO_PASS`) also counts
each of its items, since each one alone gives a hidden test away. A string holding a JSON object (workplace_assistant's
call `arguments`) counts only whole: its fields are values the agent is meant to look up in the environment's data.
`VALUE_LINE` keeps the boilerplate any script carries (`if __name__ == "__main__":`) from counting as hidden.
"""

import base64
import binascii
import io
import json
import re
import tarfile
import zlib
from collections.abc import Iterator
from pathlib import Path


WHOLE_VALUE = 8
VALUE_LINE = 40
PATCH_LINE = 20
# 1024+ characters of the base64 alphabet, possibly wrapped into lines (`base64` wraps at 76 columns).
BASE64_RUN = re.compile(rb"[A-Za-z0-9+/][A-Za-z0-9+/\r\n]{1022,}[A-Za-z0-9+/]={0,2}")
# Decoded bytes per file, over every layer, and the layers (a base64 run, a gunzip, a tar member) one inside another.
MAX_DECODED = 256 << 20
MAX_NESTING = 8
GZIP_CHUNK = 1 << 16


def leaves(value: object) -> Iterator[str]:
    """The string and number leaves of a JSON value; a string holding a JSON array yields its leaves too."""
    if isinstance(value, dict):
        for item in value.values():
            yield from leaves(item)
    elif isinstance(value, list):
        for item in value:
            yield from leaves(item)
    elif isinstance(value, str):
        yield value
        if value.lstrip().startswith("["):
            try:
                inner = json.loads(value)
            except json.JSONDecodeError:
                return
            if isinstance(inner, list):
                yield from leaves(inner)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield str(value)


def candidates(value: object) -> tuple[set[str], int]:
    """The hidden strings to scan for, and the number of leaves too short to scan."""
    found, short = set(), 0
    for leaf in leaves(value):
        if leaf.startswith("diff --git") or "\n@@ " in leaf:
            added = (
                line[1:].strip() for line in leaf.splitlines() if line.startswith("+") and not line.startswith("+++")
            )
            found.update(line for line in added if len(line) >= PATCH_LINE)
            continue
        if len(leaf.strip()) >= WHOLE_VALUE:
            found.add(leaf.strip())
        elif leaf.strip():
            short += 1
        if "\n" in leaf:
            found.update(line.strip() for line in leaf.splitlines() if len(line.strip()) >= VALUE_LINE)
    return found, short


def _decode(lines: list[bytes]) -> bytes | None:
    """A base64 run decoded; a trailing line that is not part of it (a heredoc's `EOF`) is dropped once."""
    for kept in (lines, lines[:-1]):
        try:
            return base64.b64decode(b"".join(kept), validate=True) if kept else None
        except (binascii.Error, ValueError):
            continue
    return None


def _wrapped(lines: list[bytes]) -> bool:
    """Whether a run is laid out as base64 is: one line, or lines of one width but the last."""
    return len({len(line) for line in lines[:-1]}) <= 1


def _archive_kind(data: bytes) -> str | None:
    """The kind of an archive or compressed file `contents` does not unpack, by its magic bytes. A gzip is one when
    it does not gunzip."""
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[:4] == b"PK\x03\x04":
        return "zip"
    if data[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if data[:3] == b"BZh" and data[3:4].isdigit():
        return "bz2"
    return None


def contents(path: Path, base: Path, problems: list[str]) -> Iterator[tuple[str, bytes]]:
    """A file's bytes and every layer inside them, each with its label: each base64 run decoded, a gzip gunzipped
    (every member of the stream) and the members of a tar, then the same again in each of those.

    One traversal handles every layer, and they share `MAX_NESTING` layers of depth and `MAX_DECODED` decoded bytes
    in all. What cannot be scanned in full is appended to `problems`: a zip, xz or bz2 (a tar compressed with xz or
    bz2 is untarred), a gzip that does not gunzip, ends early or has other bytes after it, a run laid out as base64
    that does not decode, a layer nested deeper than `MAX_NESTING`, and what is past `MAX_DECODED` decoded bytes (only
    the start is scanned). Lines of mixed widths that happen to use only the base64 alphabet (a word list) and do not
    decode are not base64 and are skipped.
    """
    yield from _walk(str(path.relative_to(base)), path.read_bytes(), problems, [MAX_DECODED], 0)


def _walk(label: str, blob: bytes, problems: list[str], budget: list[int], depth: int) -> Iterator[tuple[str, bytes]]:
    """`blob`, then each layer inside it walked the same way one level deeper. `budget` is the decoded bytes left."""
    if depth > MAX_NESTING:
        problems.append(f"{label}: nested over {MAX_NESTING} layers deep: not scanned")
        return
    yield label, blob
    for match in BASE64_RUN.finditer(blob):
        inner = f"{label}[base64 at {match.start()}]"
        lines = match.group().split()
        decoded = _decode(lines)
        if decoded is None:
            if _wrapped(lines):
                problems.append(f"{inner}: a {len(match.group())}-character base64 run does not decode: not scanned")
        elif len(decoded) > budget[0]:
            problems.append(f"{inner}: over {MAX_DECODED} decoded bytes in all: not scanned")
        else:
            budget[0] -= len(decoded)
            yield from _walk(inner, decoded, problems, budget, depth + 1)
    if blob[:2] == b"\x1f\x8b":
        members = _gunzip(label, blob, problems, budget)
        if members is None:
            problems.append(f"{label}: a gzip archive: not scanned")
            return
        # The members together (a tar split across members), then each later member alone: the first is the start
        # of the whole, but a later one's own gzip, tar or text start is only seen alone.
        yield from _walk(label, b"".join(members), problems, budget, depth + 1)
        for number, member in enumerate(members[1:], 2):
            yield from _walk(f"{label}[gzip member {number}]", member, problems, budget, depth + 1)
        return
    is_tar = False
    try:
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            is_tar = True
            for member in tar:
                handle = tar.extractfile(member) if member.isfile() else None
                if handle is None:
                    continue
                name = f"{label}:{member.name}"
                data = handle.read(budget[0])
                budget[0] -= len(data)
                if member.size > len(data):
                    problems.append(f"{name}: over {MAX_DECODED} decoded bytes in all: only the start is scanned")
                yield from _walk(name, data, problems, budget, depth + 1)
    except (tarfile.TarError, EOFError, OSError, zlib.error):
        pass
    kind = None if is_tar else _archive_kind(blob)
    if kind:
        problems.append(f"{label}: a {kind} archive: not scanned")


def _gunzip(label: str, blob: bytes, problems: list[str], budget: list[int]) -> list[bytes] | None:
    """Each member of a gzip stream gunzipped, within `budget`; None when one does not gunzip. Output past the
    budget, a stream that ends early and bytes after the stream (other than zero padding) go to `problems`. Input is
    fed in `GZIP_CHUNK` pieces, so a stream of many members costs time in proportion to its size."""
    members, view, position = [], memoryview(blob), 0
    while True:
        stream, parts = zlib.decompressobj(wbits=31), []
        while not stream.eof and budget[0]:
            data = stream.unconsumed_tail
            if not data:
                data = view[position : position + GZIP_CHUNK]
                position += len(data)
            if not data:
                break
            try:
                part = stream.decompress(data, budget[0])
            except zlib.error:
                return None
            budget[0] -= len(part)
            parts.append(part)
        members.append(b"".join(parts))
        if not stream.eof:
            if budget[0]:
                problems.append(f"{label}: a gzip stream that ends early: only the start is scanned")
            else:
                problems.append(f"{label}: over {MAX_DECODED} decoded bytes in all: only the start is scanned")
            return members
        position -= len(stream.unused_data)
        if view[position : position + 2] != b"\x1f\x8b":
            rest = bytes(view[position:])
            if rest.strip(b"\0"):
                problems.append(f"{label}: {len(rest)} bytes after the gzip stream: not scanned")
            return members
