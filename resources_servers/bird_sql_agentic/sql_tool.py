# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only SQL execution used by the ``execute_sql`` tool.

Kept free of any server/framework code so it can be tested directly.
"""

import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional


# Matches queries that start with SELECT, WITH, or PRAGMA table_info (any case, leading spaces ok).
ALLOWED_PREFIX = re.compile(r"^\s*(select|with|pragma\s+table_info)\b", re.IGNORECASE)


def check_sql_allowed(sql: str) -> Optional[str]:
    """Return an error message if ``sql`` is not allowed, else ``None``."""
    s = _strip_sql_comments(sql).strip()
    if s.endswith(";"):
        s = s[:-1].strip()

    if not s:
        return "Empty query."

    if _has_unquoted_semicolon(s):
        return "Only one statement per call is allowed."

    if not ALLOWED_PREFIX.match(s):
        return "Only read-only SELECT, WITH, or PRAGMA table_info(...) queries are allowed."

    return None


def _has_unquoted_semicolon(sql: str) -> bool:
    """True if ``sql`` contains a ';' that is not inside '...', "..." or `...`."""
    quote: Optional[str] = None
    for ch in sql:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch == ";":
            return True
    return False


def _strip_sql_comments(sql: str) -> str:
    """Replace ``--`` and ``/* */`` comments with a space, leaving quoted text untouched."""
    out = []
    i = 0
    state = "NORMAL"
    quote: Optional[str] = None
    while i < len(sql):
        ch = sql[i]
        two = sql[i : i + 2]

        if state == "NORMAL":
            if ch in ("'", '"', "`"):
                state = "STRING"
                quote = ch
                out.append(ch)
                i += 1
            elif two == "--":
                state = "LINE_COMMENT"
                out.append(" ")
                i += 2
            elif two == "/*":
                state = "BLOCK_COMMENT"
                out.append(" ")
                i += 2
            else:
                out.append(ch)
                i += 1

        elif state == "STRING":
            out.append(ch)
            if ch == quote:
                state = "NORMAL"
            i += 1

        elif state == "LINE_COMMENT":
            if ch == "\n":
                state = "NORMAL"
                out.append(ch)
            i += 1

        else:  # BLOCK_COMMENT
            if two == "*/":
                state = "NORMAL"
                i += 2
            else:
                i += 1

    return "".join(out)


def format_rows(
    columns: list[str], rows: list[tuple[Any, ...]], total_rows: int, max_rows: int, max_cell_chars: int
) -> dict[str, Any]:
    """Turn fetched rows into the (truncated) dict returned to the model."""
    result = {"columns": columns}
    rows = rows[:max_rows]
    new_rows = []
    for row in rows:
        new_row = []
        for cell in row:
            if isinstance(cell, bytes):
                cell = cell.decode(errors="replace")
            if isinstance(cell, str) and len(cell) > max_cell_chars:
                cell = cell[:max_cell_chars] + "..."
            new_row.append(cell)
        new_rows.append(new_row)
    result["rows"] = new_rows
    result["total_rows"] = total_rows
    result["truncated"] = total_rows > len(new_rows)
    result["error"] = None
    return result


def run_readonly_query(
    db_path: Path, sql: str, timeout_s: float, max_rows: int, max_cell_chars: int
) -> dict[str, Any]:
    """Run one allowed query on a read-only connection. Never raises for SQL problems."""
    error = check_sql_allowed(sql)
    if error:
        return {"error": error}

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as e:
        return {"error": f"Could not open database: {e}"}

    try:
        deadline = time.monotonic() + timeout_s
        conn.text_factory = lambda b: b.decode(errors="replace")
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)

        cur = conn.execute(sql)
        columns = [d[0] for d in cur.description or []]
        rows = cur.fetchmany(max_rows)
        # Count the remaining rows without storing them.
        count = 0
        for _ in cur:
            count += 1
        total_rows = len(rows) + count
    except sqlite3.Error as e:
        if "interrupted" in str(e):
            return {"error": f"Query timed out after {timeout_s:g} seconds."}
        return {"error": f"SQL error: {e}"}
    finally:
        conn.close()

    return format_rows(columns, rows, total_rows, max_rows, max_cell_chars)
