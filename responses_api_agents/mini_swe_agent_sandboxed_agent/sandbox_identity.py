# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional per-command check for stale Kubernetes sandbox endpoints.

This checks the hostname in the same exec as the action, without changing global
provider methods. It detects endpoint reuse; it is not a security boundary.
"""

import re
import shlex


class SandboxIdentityMismatch(RuntimeError):
    pass


async def checked_exec(sandbox, command, expected_hostname=None, **kwargs):
    marker = "MINISWE_SANDBOX_IDENTITY_MISMATCH"
    if expected_hostname is not None:
        expected = shlex.quote(expected_hostname)
        command = (
            "miniswe_actual_hostname=$(/bin/uname -n)\n"
            f'if [ "$miniswe_actual_hostname" != {expected} ]; then\n'
            f"  printf '{marker} expected=%s actual=%s\\n' {expected} "
            '"$miniswe_actual_hostname" >&2\n'
            "  exit 97\n"
            "fi\n" + command
        )
    result = await sandbox.exec(command, **kwargs)
    if expected_hostname is not None and result.return_code == 97:
        pattern = re.escape(f"{marker} expected={expected_hostname}") + r" actual=\S+"
        for line in ((result.stdout or "") + "\n" + (result.stderr or "")).splitlines():
            if re.fullmatch(pattern, line):
                raise SandboxIdentityMismatch(line)
    return result
