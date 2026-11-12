# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import shutil
import subprocess

import pytest

from responses_api_agents.codex_agent.app import _sandbox_prepare_command


@pytest.mark.parametrize("package_manager", ["apk", "apt-get"])
@pytest.mark.parametrize("failure", [None, "nonroot", "install", "unavailable"])
def test_posix_bootstrap_installs_missing_bash(tmp_path, package_manager, failure):
    """Exercise the actual preparation through sh without invoking a host package manager."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    bash = shutil.which("bash")
    (bindir / "mkdir").symlink_to(shutil.which("mkdir"))
    (bindir / "ln").symlink_to(shutil.which("ln"))
    uid = bindir / "id"
    uid.write_text(f"#!/bin/sh\necho {1000 if failure == 'nonroot' else 0}\n")
    uid.chmod(0o755)
    manager = bindir / package_manager
    manager.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_ROOT/packages.log"\n'
        + ("exit 100\n" if failure == "install" else "")
        + f'[ "$*" = "${{*%bash}}" ] || ln -s {shlex.quote(bash)} "$TEST_ROOT/bin/bash"\n'
    )
    manager.chmod(0o755)
    if failure == "unavailable":
        manager.unlink()
    task = tmp_path / "task"
    task.mkdir()
    command = _sandbox_prepare_command(str(task), str(tmp_path / "session"), str(tmp_path / "runtime"))
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env=os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    if failure:
        assert result.returncode == (100 if failure == "install" else 1), result.stderr
        assert not (tmp_path / "session").exists()
        if failure != "install":
            assert "preinstall" in result.stderr
            assert not (tmp_path / "packages.log").exists()
        return
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "session/home/.codex").is_dir()
    assert (tmp_path / "packages.log").read_text().splitlines() == (
        ["add --no-cache bash"] if package_manager == "apk" else ["update", "install -y --no-install-recommends bash"]
    )
    bash = subprocess.run(
        ["/bin/sh", "-c", "bash -c 'printf ready'"],
        env=os.environ | {"PATH": str(bindir)},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    assert bash.returncode == 0, bash.stderr
    assert bash.stdout == "ready"
