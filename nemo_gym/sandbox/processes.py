# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stop only process groups recorded by one borrowed-sandbox activation."""

from shlex import quote

from nemo_gym.sandbox import AsyncSandbox


async def stop_process_groups(sandbox: AsyncSandbox, *, session_id: str, user: str | int | None) -> None:
    """Quiesce the harness before releasing its borrowed sandbox."""
    pidfile = quote(f"/tmp/{session_id}.pids")
    result = await sandbox.exec(
        f"if [ -f {pidfile} ]; then groups=$(cat {pidfile}); for p in $groups; do "
        "case $p in ''|*[!0-9]*) exit 1;; esac; "
        'kill -TERM -- -"$p" 2>/dev/null || true; done; sleep 1; '
        'for p in $groups; do if kill -0 -- -"$p" 2>/dev/null; then '
        'kill -KILL -- -"$p" 2>/dev/null || exit 1; fi; done; '
        f"sleep 1; rm -f {pidfile}; fi",
        timeout_s=30,
        user=user,
    )
    if result.return_code:
        raise RuntimeError("Could not stop the agent's sandbox process groups")
