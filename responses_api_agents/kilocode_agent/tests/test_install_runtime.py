# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the real installer against stubbed system tools and a fake Node archive; no network or packages."""

import hashlib
import io
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


INSTALLER = Path(__file__).parents[1] / "install_kilo_runtime.sh"
ARCHIVE = "node-v22.19.0-linux-x64.tar.gz"

# Stands in for Node: reports its version, runs "npm install" by creating Kilo's launcher, and runs that launcher.
FAKE_NODE = """#!/bin/bash
case "$1" in
  --version) echo v22.19.0 ;;
  */npm-cli.js)
    echo "$*" >> "$TEST_ROOT/npm.log"
    while [ "$#" -gt 0 ]; do [ "$1" = --prefix ] && prefix=$2; shift; done
    mkdir -p "$prefix/node_modules/@kilocode/cli/bin"
    : > "$prefix/node_modules/@kilocode/cli/bin/kilo" ;;
  */@kilocode/cli/bin/kilo) echo "${TEST_KILO_VERSION:-7.4.15}" ;;
  *) exit 2 ;;
esac
"""


def _node_archive() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, text, mode in [
            ("bin/node", FAKE_NODE, 0o755),
            ("lib/node_modules/npm/bin/npm-cli.js", "", 0o644),
        ]:
            data = text.encode()
            info = tarfile.TarInfo(f"node-v22.19.0-linux-x64/{name}")
            info.size, info.mode = len(data), mode
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """An isolated PATH: real core tools, stubbed platform probes, package manager and downloads."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("bash", "mkdir", "tar", "gzip", "sha256sum", "awk", "grep"):
        executable = shutil.which(name)
        if executable is None:
            pytest.skip(f"Installer test requires {name}")
        (bindir / name).symlink_to(executable)
    (bindir / "python3").symlink_to(sys.executable)
    archive = _node_archive()
    (tmp_path / ARCHIVE).write_bytes(archive)
    (tmp_path / "SHASUMS256.txt").write_text(f"{hashlib.sha256(archive).hexdigest()}  {ARCHIVE}\n")
    scripts = {
        "uname": '#!/bin/bash\n[ "$1" = -m ] && echo x86_64 || echo Linux\n',
        "getconf": "#!/bin/bash\necho glibc 2.35\n",
        "id": '#!/bin/bash\necho "${TEST_UID:-0}"\n',
        # The first install provides the fake curl below, as apt-get would provide the real one.
        "apt-get": """#!/bin/bash
echo "$*" >> "$TEST_ROOT/packages.log"
if [ "$1" = install ]; then
  [ "${TEST_INSTALL_FAIL:-0}" = 0 ] || exit 100
  cp "$TEST_ROOT/curl" "$TEST_ROOT/bin/curl"
fi
""",
        "cp": f'#!/bin/bash\nexec {shutil.which("cp")} "$@"\n',
    }
    for name, script in scripts.items():
        (bindir / name).write_text(script)
        (bindir / name).chmod(0o755)
    curl = tmp_path / "curl"
    curl.write_text(
        """#!/bin/bash
while [ "$#" -gt 0 ]; do
  case "$1" in -o) output=$2; shift ;; https://*) url=$1 ;; esac
  shift
done
echo "$url" >> "$TEST_ROOT/download.log"
[ "${TEST_DOWNLOAD_FAIL:-0}" = 0 ] || exit 19
exec "$TEST_ROOT/bin/cp" "$TEST_ROOT/${url##*/}" "$output"
"""
    )
    curl.chmod(0o755)
    return tmp_path, os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)}


def run_installer(root: Path, env: dict[str, str], version: str = "7.4.15") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(root / "bin/bash"), str(INSTALLER), str(root / "runtime"), version],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )


def test_installs_verified_node_and_pinned_kilo(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = run_installer(root, env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "7.4.15"
    assert (root / "download.log").read_text().splitlines() == [
        f"https://nodejs.org/dist/v22.19.0/{ARCHIVE}",
        "https://nodejs.org/dist/v22.19.0/SHASUMS256.txt",
    ]
    npm = (root / "npm.log").read_text()
    assert f"--prefix {root}/runtime/kilo" in npm
    assert npm.rstrip().endswith("@kilocode/cli@7.4.15")
    assert (root / "runtime/kilo/node_modules/@kilocode/cli/bin/kilo").exists()


def test_version_mismatch_fails(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = run_installer(root, env | {"TEST_KILO_VERSION": "7.4.16"})
    assert result.returncode == 1
    assert "Kilo version mismatch: 7.4.16" in result.stderr


def test_checksum_mismatch_stops_before_installing_kilo(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / "SHASUMS256.txt").write_text(f"{'0' * 64}  {ARCHIVE}\n")
    result = run_installer(root, env)
    assert result.returncode != 0
    assert "Kilo installer failed" in result.stderr
    assert not (root / "npm.log").exists()


def test_inexact_version_is_rejected_before_any_work(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = run_installer(root, env, version="latest")
    assert result.returncode == 1
    assert "exact Kilo version" in result.stderr
    assert not (root / "packages.log").exists()


@pytest.mark.parametrize("missing", ["root", "apt-get"])
def test_missing_tools_without_bootstrap_support_explain_remedy(
    sandbox: tuple[Path, dict[str, str]], missing: str
) -> None:
    root, env = sandbox
    if missing == "root":
        env["TEST_UID"] = "1000"
    else:
        (root / "bin/apt-get").unlink()
    result = run_installer(root, env)
    assert result.returncode == 1
    assert "preinstall them" in result.stderr
    assert not (root / "download.log").exists()


def test_package_failure_reports_failing_command(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = run_installer(root, env | {"TEST_INSTALL_FAIL": "1"})
    assert result.returncode == 100
    assert "Kilo installer failed (exit 100)" in result.stderr
    assert not (root / "download.log").exists()
