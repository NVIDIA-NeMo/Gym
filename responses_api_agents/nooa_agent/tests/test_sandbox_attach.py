# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import shutil
import subprocess
from uuid import uuid4

import pytest

from responses_api_agents.nooa_agent.sandbox_attach import attach_docker_sandbox
from responses_api_agents.nooa_agent.sandbox_shell import SandboxShellTools


_DOCKER_TEST_IMAGE = os.getenv("NOOA_TEST_DOCKER_IMAGE", "ghcr.io/laude-institute/t-bench/python-3-13:20250620")


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=check,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    return _docker("info", check=False).returncode == 0


@pytest.mark.asyncio
@pytest.mark.skipif(not _docker_ready(), reason="Docker daemon is unavailable")
async def test_attached_seeded_docker_sandbox_executes_edits_without_owning_lifecycle() -> None:
    if _docker("image", "inspect", _DOCKER_TEST_IMAGE, check=False).returncode != 0:
        pytest.skip(f"seeded test image is not available locally: {_DOCKER_TEST_IMAGE}")

    container = f"nooa-seeded-acceptance-{uuid4().hex[:10]}"
    started = _docker(
        "run",
        "--rm",
        "--detach",
        "--name",
        container,
        _DOCKER_TEST_IMAGE,
        "bash",
        "-lc",
        "mkdir -p /app && printf seeded >/app/source.txt && exec sleep infinity",
    )
    assert started.stdout.strip()
    try:
        sandbox = attach_docker_sandbox(container)
        initial = await sandbox.exec("cat /app/source.txt", timeout_s=10)
        assert initial.return_code == 0
        assert initial.stdout == "seeded"

        shell = SandboxShellTools(sandbox, cwd="/app")
        write = await shell.write_file("result.txt", "changed through NOOA\n")
        assert write.path == "result.txt"
        stored = await shell.read("result.txt")
        assert stored.text == "changed through NOOA\n"

        # The resources server owns the seeded container. Closing the agent-side shell
        # must not stop or remove it before verifier execution.
        await shell._session.close()
        running = _docker("inspect", "--format", "{{.State.Running}}", container)
        assert running.stdout.strip() == "true"
        verifier_view = _docker("exec", container, "cat", "/app/result.txt")
        assert verifier_view.stdout == "changed through NOOA\n"
    finally:
        _docker("rm", "--force", container, check=False)
