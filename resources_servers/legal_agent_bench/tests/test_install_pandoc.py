# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


INSTALLER = Path(__file__).resolve().parents[1] / "install-pandoc.sh"


def _run_installer(
    tmp_path: Path, *, architecture: str = "x86_64", exit_code: int = 0, corrupt: bool = False
) -> subprocess.CompletedProcess[str]:
    archive = tmp_path / "release.tar.gz"
    binary = f'#!/bin/sh\nprintf "pandoc 3.11\\nFixture build\\n"\nexit {exit_code}\n'.encode()
    with tarfile.open(archive, "w:gz") as package:
        member = tarfile.TarInfo("pandoc-3.11/bin/pandoc")
        member.mode = 0o755
        member.size = len(binary)
        package.addfile(member, io.BytesIO(binary))

    # Substitute only release hashes, so the real checksum command verifies a local fixture.
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    script, count = re.subn(r"sha256=[0-9a-f]{64}", f"sha256={digest}", INSTALLER.read_text())
    assert count == 2
    installer = tmp_path / "installer.sh"
    installer.write_text(script)
    if corrupt:
        archive.write_bytes(archive.read_bytes() + b"changed after checksum")

    commands = tmp_path / "commands"
    commands.mkdir()
    scripts = {
        "uname": f"printf '%s\\n' {shlex.quote(architecture)}\n",
        "curl": 'test "$1" = -fsSL && test "$2" = -o && test "$4" = "$EXPECTED_URL" || exit 91\n'
        'cp "$RELEASE_ARCHIVE" "$3"\n',
        "tar": 'touch "$EXTRACTION_MARKER"\n' + f'exec {shlex.quote(shutil.which("tar"))} "$@"\n',
    }
    for name, body in scripts.items():
        command = commands / name
        command.write_text("#!/bin/sh\nset -eu\n" + body)
        command.chmod(0o755)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    arch = "amd64" if architecture == "x86_64" else "arm64"
    return subprocess.run(
        ["sh", str(installer), str(tmp_path / "prefix")],
        env={
            **os.environ,
            "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
            "TMPDIR": str(scratch),
            "RELEASE_ARCHIVE": str(archive),
            "EXTRACTION_MARKER": str(tmp_path / "extracted"),
            "EXPECTED_URL": f"https://github.com/jgm/pandoc/releases/download/3.11/pandoc-3.11-linux-{arch}.tar.gz",
        },
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )


@pytest.mark.parametrize("architecture", ["x86_64", "aarch64"])
def test_installer_verifies_extracts_and_runs_release(tmp_path: Path, architecture: str) -> None:
    result = _run_installer(tmp_path, architecture=architecture)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "pandoc 3.11\n"
    assert (tmp_path / "extracted").exists()
    assert os.access(tmp_path / "prefix/bin/pandoc", os.X_OK)
    assert not list((tmp_path / "scratch").iterdir())


def test_checksum_mismatch_stops_before_extraction_or_installation(tmp_path: Path) -> None:
    result = _run_installer(tmp_path, corrupt=True)
    assert result.returncode != 0
    assert not (tmp_path / "extracted").exists()
    assert not (tmp_path / "prefix").exists()
    assert not list((tmp_path / "scratch").iterdir())


def test_installed_binary_failure_is_not_hidden_by_output_formatting(tmp_path: Path) -> None:
    result = _run_installer(tmp_path, exit_code=42)
    assert result.returncode == 42, result.stderr
    assert (tmp_path / "prefix/bin/pandoc").is_file()
    assert not list((tmp_path / "scratch").iterdir())
