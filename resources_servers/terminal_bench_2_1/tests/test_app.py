# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TEST_SH_PATCHES,
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
)


class TestApp:
    @pytest.mark.parametrize(
        ("task", "packages"),
        [("qemu-startup", "curl expect"), ("qemu-alpine-ssh", "curl sshpass"), ("code-from-image", "curl")],
    )
    def test_verifier_bootstrap_patch_preserves_grading_and_package_requirements(
        self, tmp_path: Path, task: str, packages: str
    ) -> None:
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="", sandbox_config={}, host="", port=0, entrypoint="", name=""
        )
        server = TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        grading = "\nuvx pytest /tests/test_outputs.py\nif [ $? -eq 0 ]; then echo 1; else echo 0; fi\n"
        original = f"#!/bin/bash\napt-get update\napt-get install -y {packages}\n" + grading
        path = tmp_path / "test.sh"
        path.write_text(original)
        with server._patch_golden_patch_solve_sh(f"terminal-bench/{task}", path, TEST_SH_PATCHES) as patched:
            content = Path(patched).read_text()
            subprocess.run(["bash", "-n", str(patched)], check=True)
            assert content.endswith(grading)
            assert f"install -y {packages} || exit $?" in content
            assert "DPkg::Lock::Timeout=300" in content
            assert "https://deb.debian.org" in content
        assert path.read_text() == original

    def test_sanity(self) -> None:
        config = TerminalBench21ResourcesServerConfig(
            sandbox_provider="",
            sandbox_config=dict(),
            host="",
            port=0,
            entrypoint="",
            name="",
        )
        TerminalBench21ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
