# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide the Python helpers needed by OpenCode without modifying task packages."""

from shlex import quote
from uuid import uuid4

from nemo_gym.sandbox.bootstrap import python_asset


async def ensure_python(sandbox, exec_options) -> str:
    probe = await sandbox.exec(
        "if command -v python3 >/dev/null 2>&1 && python3 -c 'import sqlite3' >/dev/null 2>&1; "
        "then command -v python3; else uname -m; "
        "if test -f /etc/alpine-release || ls /lib/ld-musl-*.so* >/dev/null 2>&1; "
        "then echo musl; else echo gnu; fi; fi",
        timeout_s=30,
        **exec_options,
    )
    lines = (probe.stdout or "").strip().splitlines()
    if probe.return_code or probe.error_type:
        raise RuntimeError(f"Python bootstrap probe failed: {probe.stdout} {probe.stderr}")
    if len(lines) == 1 and lines[0].startswith("/"):
        return lines[0]
    if len(lines) != 2:
        raise RuntimeError(f"Unexpected Python bootstrap probe output: {probe.stdout}")
    archive = await python_asset(*lines)
    directory = f"/tmp/nemo-gym-python-{uuid4().hex}"
    remote_archive = directory + ".tar.gz"
    await sandbox.upload(archive, remote_archive)
    executable = directory + "/python/bin/python3"
    result = await sandbox.exec(
        f"mkdir -p {quote(directory)} && tar -xzf {quote(remote_archive)} -C {quote(directory)} "
        f"&& {quote(executable)} -c 'import sqlite3'",
        timeout_s=120,
        **exec_options,
    )
    if result.return_code or result.error_type:
        raise RuntimeError(f"Python bootstrap failed: {result.stdout} {result.stderr}")
    return executable
