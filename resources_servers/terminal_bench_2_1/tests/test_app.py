# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.terminal_bench_2_1.app import (
    TEST_SH_PATCHES,
    VERIFIER_COMMAND,
    TerminalBench21ResourcesServer,
    TerminalBench21ResourcesServerConfig,
)


class TestApp:
    def test_verifier_receives_apt_lock_wait_and_preserves_exit_status(self, tmp_path: Path) -> None:
        script = tmp_path / "test.sh"
        script.write_text('cat "$APT_CONFIG"\nrm "$APT_CONFIG"\nexit 7\n')
        result = subprocess.run(
            ["bash", "-c", VERIFIER_COMMAND.replace("/tests/test.sh", str(script))],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 7
        assert result.stdout == 'DPkg::Lock::Timeout "300";\n'
        assert result.stderr == ""

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
            if task.startswith("qemu-"):
                assert content.count('Dir::Etc::sourcelist="$verifier_apt_sources"') == 2
                assert "https://snapshot.debian.org/archive/debian-security/20260831T235959Z/" in content
                assert "[check-valid-until=no]" in content
                assert "trusted=yes" not in content
                assert "--allow-unauthenticated" not in content
            else:
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
