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
import itertools
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
MAX_DECODED = 256 << 20
# Tar members are unpacked in turn (a .tar.gz inside a tar) down to this many archives deep.
MAX_NESTING = 8


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
    it is still a gzip after one gunzip, or does not gunzip."""
    if data[:2] == b"\x1f\x8b":
        return "gzip"
    if data[:4] == b"PK\x03\x04":
        return "zip"
    if data[:6] == b"\xfd7zXZ\x00":
        return "xz"
    if data[:3] == b"BZh" and data[3:4].isdigit():
        return "bz2"
    return None


def _unpack(
    label: str, blob: bytes, problems: list[str], *, keep: bool, depth: int = 0
) -> Iterator[tuple[str, bytes]]:
    """`blob` gunzipped when it is a gzip (yielded when it was, or when `keep`), then the members of a tar it holds,
    each unpacked the same way. A zip, xz or bz2 that `blob` is or gunzips to, or a member nested over `MAX_NESTING`
    archives deep, is appended to `problems` as not scanned, and so is a gzip inside a gzip."""
    if blob[:2] == b"\x1f\x8b":
        gunzip = zlib.decompressobj(wbits=31)
        try:
            blob = gunzip.decompress(blob, MAX_DECODED)
        except zlib.error:
            pass
        else:
            keep = True
            if gunzip.unconsumed_tail:
                problems.append(f"{label}: gunzips to more than {MAX_DECODED} bytes: only the start is scanned")
    if keep:
        yield label, blob
    is_tar = False
    try:
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            is_tar = True
            for member in tar:
                handle = tar.extractfile(member) if member.isfile() else None
                if handle is None:
                    continue
                name = f"{label}:{member.name}"
                if depth >= MAX_NESTING:
                    problems.append(f"{name}: archives nested over {MAX_NESTING} deep: not scanned")
                    continue
                if member.size > MAX_DECODED:
                    problems.append(f"{name}: over {MAX_DECODED} bytes: only the start is scanned")
                yield from _unpack(name, handle.read(MAX_DECODED), problems, keep=True, depth=depth + 1)
    except (tarfile.TarError, EOFError, OSError, zlib.error):
        pass
    kind = None if is_tar else _archive_kind(blob)
    if kind:
        problems.append(f"{label}: a {kind} archive: not scanned")


def contents(path: Path, base: Path, problems: list[str]) -> Iterator[tuple[str, bytes]]:
    """A file's bytes, gunzipped and untarred when it is a gzip or a tar, then each base64 run in the file, in what it
    gunzips to and in its tar members decoded (and gunzipped and untarred the same way).

    What cannot be scanned in full is appended to `problems`: a zip, xz or bz2 file, base64 payload or tar member (a
    tar compressed with xz or bz2 is untarred), a gzip inside a gzip, a run laid out as base64 that does not decode,
    a gzip or a tar member over `MAX_DECODED` bytes (only its start is scanned). Lines of mixed widths that happen to
    use only the base64 alphabet (a word list) and do not decode are not base64 and are skipped.
    """
    label = str(path.relative_to(base))
    data = path.read_bytes()
    for item_label, blob in itertools.chain([(label, data)], _unpack(label, data, problems, keep=False)):
        yield item_label, blob
        yield from _base64_payloads(item_label, blob, problems)


def _base64_payloads(label: str, data: bytes, problems: list[str]) -> Iterator[tuple[str, bytes]]:
    """Each base64 run in `data` decoded, gunzipped and untarred; a run laid out as base64 that does not decode is
    appended to `problems`."""
    for match in BASE64_RUN.finditer(data):
        inner = f"{label}[base64 at {match.start()}]"
        lines = match.group().split()
        blob = _decode(lines)
        if blob is None:
            if _wrapped(lines):
                problems.append(f"{inner}: a {len(match.group())}-character base64 run does not decode: not scanned")
            continue
        yield from _unpack(inner, blob, problems, keep=True)
