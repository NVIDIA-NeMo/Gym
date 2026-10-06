# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional pinned Python bootstrap for native sandbox process supervision."""

import re
from shlex import quote
from urllib.parse import urlsplit

from nemo_gym.sandbox.api import AsyncSandbox


async def ensure_python(
    sandbox: AsyncSandbox,
    *,
    runtime_url: str | None = None,
    runtime_sha256: str | None = None,
    timeout_s: float = 180,
) -> str:
    """Reuse Python >=3.8, or unpack a verified install-only standalone archive.

    The bootstrap uses shell utilities only and writes outside the task repository.
    The owner must make the pinned archive URL reachable under its egress policy.
    """
    check = "import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)"
    probe = await sandbox.exec("python3 -c " + quote(check), timeout_s=15)
    if probe.return_code == 0 and not probe.error_type:
        return "python3"
    if not runtime_url or not runtime_sha256:
        raise RuntimeError("Sandbox needs Python >=3.8 or a pinned python_runtime_url/python_runtime_sha256")
    if urlsplit(runtime_url).scheme not in {"http", "https"} or not re.fullmatch("[a-f0-9]{64}", runtime_sha256):
        raise ValueError("Python runtime requires an HTTP(S) URL and a lowercase SHA-256 digest")
    root = "/tmp/nemo-gym-python-" + runtime_sha256
    executable = root + "/python/bin/python3"
    # Concurrent users unpack into distinct directories then publish atomically.
    # Only the expected hash can populate this path.
    command = (
        f"set -eu; if [ ! -x {quote(executable)} ]; then "
        f"tmp=$(mktemp -d {quote(root + '.XXXXXX')}); trap 'rm -rf -- \"$tmp\"' EXIT; "
        f'curl -fL --retry 2 --connect-timeout 30 -o "$tmp/archive.tar.gz" -- {quote(runtime_url)}; '
        f"printf '%s  %s\\n' {quote(runtime_sha256)} \"$tmp/archive.tar.gz\" | sha256sum -c -; "
        'mkdir "$tmp/unpacked"; tar -xzf "$tmp/archive.tar.gz" -C "$tmp/unpacked"; '
        'test -x "$tmp/unpacked/python/bin/python3"; '
        f'mv -T "$tmp/unpacked" {quote(root)} 2>/dev/null || test -x {quote(executable)}; fi; '
        f"{quote(executable)} -c {quote(check)}"
    )
    result = await sandbox.exec(command, timeout_s=timeout_s)
    if result.return_code or result.error_type:
        raise RuntimeError("Pinned Python bootstrap failed: " + (result.stderr or ""))
    return executable
