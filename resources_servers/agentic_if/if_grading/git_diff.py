# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository paths in Git binary/unified diff sections."""

import re


_C_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "a": "\a", "b": "\b", "f": "\f", "r": "\r", "v": "\v"}


def _unquote(s: str) -> str:
    """Decode Git's C-quoted paths, including octal-escaped UTF-8 bytes."""
    s = s.rstrip("\t")
    if len(s) < 2 or not (s.startswith('"') and s.endswith('"')):
        return s
    body, out, i = s[1:-1], bytearray(), 0
    while i < len(body):
        if body[i] == "\\" and i + 1 < len(body):
            if re.match(r"[0-7]{3}", body[i + 1 : i + 4]):
                out.append(int(body[i + 1 : i + 4], 8))
                i += 4
                continue
            out += _C_ESCAPES.get(body[i + 1], body[i + 1]).encode("utf-8")
            i += 2
            continue
        out += body[i].encode("utf-8")
        i += 1
    return out.decode("utf-8", "replace")


def _side(p: str, prefix: str) -> str | None:
    p = _unquote(p)
    if p == "/dev/null":
        return None
    return p[2:] if p.startswith(prefix + "/") else p


def _header_path(head: str) -> str | None:
    """A mode-only header repeats one path; renames name their paths separately."""
    head = head.rstrip()
    if head.startswith('"'):
        m = re.match(r'"(?:[^"\\]|\\.)*"', head)
        return _side(m.group(0), "a") if m else head
    k, rest = divmod(len(head) - 5, 2)
    if (
        not rest
        and k > 0
        and head.startswith("a/")
        and head[2 + k : 5 + k] == " b/"
        and head[2 : 2 + k] == head[5 + k :]
    ):
        return head[2 : 2 + k]
    i = head.find(" b/")
    return head[2:i] if head.startswith("a/") and i > 2 else head


def patch_section_paths(section: str) -> tuple[str | None, str | None]:
    """Old/new repository-relative paths in one complete ``diff --git`` section.

    Rename/copy records and content headers are authoritative. A missing side of
    an addition/deletion is None, as are both paths for a non-diff preamble.
    """
    if not section.startswith("diff --git "):
        return None, None
    lines = section.removeprefix("diff --git ").split("\n")
    found = {}
    for ln in lines[1:]:
        if ln.startswith(("@@", "Binary files", "GIT binary patch")):
            break
        for key in ("rename from ", "rename to ", "copy from ", "copy to ", "--- ", "+++ "):
            if ln.startswith(key):
                found.setdefault(key, ln[len(key) :])
    if "rename from " in found or "copy from " in found:
        return (
            _unquote(found.get("rename from ") or found.get("copy from ") or ""),
            _unquote(found.get("rename to ") or found.get("copy to ") or ""),
        )
    if "--- " in found or "+++ " in found:
        return _side(found.get("--- ", "/dev/null"), "a"), _side(found.get("+++ ", "/dev/null"), "b")
    p = _header_path(lines[0])
    head = "\n".join(lines[1:6])
    if "new file mode" in head:
        return None, p
    if "deleted file mode" in head:
        return p, None
    return p, p
