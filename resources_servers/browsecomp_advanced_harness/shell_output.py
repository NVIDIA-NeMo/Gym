# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identical rendering for sandbox and explicitly enabled development execution."""

_BASH_HEAD_BYTES = 8192
_BASH_TAIL_BYTES = 4096


def _bash_truncate(s: str) -> str:
    if len(s) <= _BASH_HEAD_BYTES + _BASH_TAIL_BYTES + 64:
        return s
    omitted = len(s) - _BASH_HEAD_BYTES - _BASH_TAIL_BYTES
    return (
        f"{s[:_BASH_HEAD_BYTES]}\n"
        f"[...truncated {omitted} bytes; head shown above, tail follows...]\n"
        f"{s[-_BASH_TAIL_BYTES:]}"
    )


def format_shell_output(stdout: str, stderr: str, exit_code: int) -> str:
    """Preserve the read-only tool's historical output, including truncation markers."""
    truncated = (len(stdout) + len(stderr)) > (_BASH_HEAD_BYTES + _BASH_TAIL_BYTES) * 2
    parts = []
    if stdout:
        parts.append(_bash_truncate(stdout))
    if stderr:
        parts.extend(("--- stderr ---", _bash_truncate(stderr)))
    parts.append(f"[exit_code={exit_code}{' truncated' if truncated else ''}]")
    return "\n".join(parts)
