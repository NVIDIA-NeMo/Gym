# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognize explicit provider reports of sandbox resource-limit deaths."""

import re

from resources_servers.terminal_bench_4.models import SandboxResourceLimit


# The kubelet's eviction of a pod over the total ephemeral-storage limit of its containers. OpenSandbox's lifecycle
# report cuts the pod detail at 256 characters, which can end it inside "of containers", so the match stops at "limit".
_POD_STORAGE_LIMIT = r"\bPod ephemeral local storage usage exceeds the total limit\b"


def sandbox_resource_limit(detail: str | None) -> SandboxResourceLimit | None:
    """Require a sandbox death report, not a command's exit code or stderr.

    Node-pressure eviction, ENOSPC, and SIGKILL alone do not establish that the
    sandbox exceeded its own allocation. OpenSandbox's backend-loss diagnostic
    and SANDBOX::ENDED responses include the control plane's termination cause.
    OpenSandbox's lifecycle report of an unavailable sandbox names no affected
    container, so it counts only with the pod's own storage-limit eviction.
    """
    if not detail:
        return None
    if "Sandbox is unavailable; lifecycle status:" in detail and re.search(_POD_STORAGE_LIMIT, detail, re.IGNORECASE):
        return SandboxResourceLimit(resource="storage", detail=detail)
    if not re.search(
        r"\[SANDBOX::ENDED\]|[\"']code[\"']\s*:\s*[\"']SANDBOX::ENDED[\"']"
        r"|Sandbox was OOM-killed\.|Sandbox is dead\. OpenSandbox status:",
        detail,
    ):
        return None
    if re.search(r"\boom[\s_-]*killed\b", detail, re.IGNORECASE):
        return SandboxResourceLimit(resource="memory", detail=detail)
    if re.search(
        _POD_STORAGE_LIMIT + r"|\bContainer \S+ exceeded its local ephemeral storage limit\b",
        detail,
        re.IGNORECASE,
    ):
        return SandboxResourceLimit(resource="storage", detail=detail)
    return None
