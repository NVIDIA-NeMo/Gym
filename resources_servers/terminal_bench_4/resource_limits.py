# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize explicit provider reports of sandbox resource-limit deaths."""

import re

from resources_servers.terminal_bench_4.models import SandboxResourceLimit


def sandbox_resource_limit(detail: str | None) -> SandboxResourceLimit | None:
    """Require a sandbox death report, not a command's exit code or stderr.

    Node-pressure eviction, ENOSPC, and SIGKILL alone do not establish that the
    sandbox exceeded its own allocation. OpenSandbox's backend-loss diagnostic
    and SANDBOX::ENDED responses include the control plane's termination cause.
    """
    if not detail or not re.search(
        r"\[SANDBOX::ENDED\]|[\"']code[\"']\s*:\s*[\"']SANDBOX::ENDED[\"']"
        r"|Sandbox was OOM-killed\.|Sandbox is dead\. OpenSandbox status:",
        detail,
    ):
        return None
    if re.search(r"\boom[\s_-]*killed\b", detail, re.IGNORECASE):
        return SandboxResourceLimit(resource="memory", detail=detail)
    if re.search(
        r"\bPod ephemeral local storage usage exceeds the total limit of containers\b"
        r"|\bContainer \S+ exceeded its local ephemeral storage limit\b",
        detail,
        re.IGNORECASE,
    ):
        return SandboxResourceLimit(resource="storage", detail=detail)
    return None
