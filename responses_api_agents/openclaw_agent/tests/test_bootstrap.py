# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import shutil
import subprocess
import sys

import pytest

from responses_api_agents.openclaw_agent.app import _sandbox_prepare_command


@pytest.mark.parametrize("package_manager", ["apk", "apt-get"])
@pytest.mark.parametrize("missing", [(), ("bash",), ("python3",), ("python3", "bash")])
@pytest.mark.parametrize("failure", [None, "nonroot", "install", "unavailable"])
def test_posix_bootstrap_installs_only_missing_prerequisites(tmp_path, package_manager, missing, failure):
    """Exercise the actual preparation through sh without invoking a host package manager."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binaries = {"python3": sys.executable, "bash": shutil.which("bash"), "mkdir": shutil.which("mkdir")}
    assert all(binaries.values())
    for name, executable in binaries.items():
        if name not in missing:
            (bindir / name).symlink_to(executable)
    (bindir / "ln").symlink_to(shutil.which("ln"))
    uid = bindir / "id"
    uid.write_text(f"#!/bin/sh\necho {1000 if failure == 'nonroot' else 0}\n")
    uid.chmod(0o755)
    manager = bindir / package_manager
    manager.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_ROOT/packages.log"\n'
        + ("exit 100\n" if failure == "install" else "")
        + 'for name in "$@"; do\n'
        + f'case "$name" in\n python3) ln -s {shlex.quote(sys.executable)} "$TEST_ROOT/bin/python3" ;;\n'
        + f' bash) ln -s {shlex.quote(binaries["bash"])} "$TEST_ROOT/bin/bash" ;;\n'
        + "esac\ndone\n"
    )
    manager.chmod(0o755)
    if failure == "unavailable":
        manager.unlink()
    directory = tmp_path / "sessions" / "session with spaces"
    task = tmp_path / "task"
    task.mkdir()
    command = _sandbox_prepare_command(str(task), str(directory), str(tmp_path / "runtime"))
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env=os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    if failure and missing:
        assert result.returncode == (100 if failure == "install" else 1), result.stderr
        assert not directory.exists()
        if failure != "install":
            assert "preinstall" in result.stderr
            assert not (tmp_path / "packages.log").exists()
        return
    assert result.returncode == 0, result.stderr
    assert (directory / "home/.openclaw").is_dir()
    packages = " ".join(missing)
    if missing:
        assert (tmp_path / "packages.log").read_text().splitlines() == (
            [f"add --no-cache {packages}"]
            if package_manager == "apk"
            else ["update", f"install -y --no-install-recommends {packages}"]
        )
    else:
        assert not (tmp_path / "packages.log").exists()
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
