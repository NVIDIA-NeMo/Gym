# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

from responses_api_agents.kilocode_agent.sandbox import _sandbox_prepare_command


def prepare(command: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/sh", "-c", command], env=env, capture_output=True, text=True, errors="replace", timeout=10
    )


@pytest.mark.parametrize("package_manager", ["apk", "apt-get"])
@pytest.mark.parametrize("missing", [(), ("bash",)])
@pytest.mark.parametrize("failure", [None, "nonroot", "install", "unavailable"])
def test_posix_bootstrap_installs_only_missing_prerequisites(tmp_path, package_manager, missing, failure):
    """Exercise the actual preparation through sh, with no Python, without invoking a host package manager."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binaries = {"bash": shutil.which("bash"), "mkdir": shutil.which("mkdir")}
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
        + f'case "$name" in\n bash) ln -s {shlex.quote(binaries["bash"])} "$TEST_ROOT/bin/bash" ;;\n'
        + "esac\ndone\n"
    )
    manager.chmod(0o755)
    if failure == "unavailable":
        manager.unlink()
    directory = tmp_path / "sessions" / "session with spaces"
    task = tmp_path / "task"
    task.mkdir()
    result = prepare(
        _sandbox_prepare_command(str(task), str(directory)),
        os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)},
    )
    if failure and missing:
        assert result.returncode == (100 if failure == "install" else 1), result.stderr
        assert not directory.exists()
        if failure != "install":
            assert "preinstall" in result.stderr
            assert not (tmp_path / "packages.log").exists()
        return
    assert result.returncode == 0, result.stderr
    assert directory.is_dir()
    packages = " ".join(missing)
    if missing:
        assert (tmp_path / "packages.log").read_text().splitlines() == (
            [f"add --no-cache {packages}"]
            if package_manager == "apk"
            else ["update", f"install -y --no-install-recommends {packages}"]
        )
    else:
        assert not (tmp_path / "packages.log").exists()
    bash = prepare("bash -c 'printf ready'", os.environ | {"PATH": str(bindir)})
    assert bash.returncode == 0, bash.stderr
    assert bash.stdout == "ready"


@pytest.mark.parametrize("layout", ["inside", "same", "contains", "symlink", "root_workdir", "missing"])
def test_session_storage_must_not_overlap_the_task_workdir(tmp_path: Path, layout: str) -> None:
    task = tmp_path / "task"
    task.mkdir()
    workdir, directory = task, tmp_path / "sessions" / "id"
    if layout == "inside":
        directory = task / "sessions" / "id"
    elif layout == "same":
        directory = task / "id"
    elif layout == "contains":
        workdir = tmp_path / "sessions" / "task"
        workdir.mkdir(parents=True)
    elif layout == "symlink":
        # Lexically separate, but the session root resolves into the task workdir.
        (tmp_path / "link").symlink_to(task)
        directory = tmp_path / "link" / "sessions" / "id"
    elif layout == "root_workdir":
        workdir = Path("/")
    else:
        workdir = tmp_path / "absent"
    result = prepare(_sandbox_prepare_command(str(workdir), str(directory)), dict(os.environ))
    assert result.returncode == 1
    expected = "workdir is missing" if layout == "missing" else "overlaps the task workdir"
    assert expected in result.stderr
    assert not directory.exists() and not (task / "sessions").exists()
