# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-sandbox installer, run with bash against a local stand-in for the npm registry."""

import hashlib
import io
import os
import platform
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


INSTALLER = Path(__file__).resolve().parents[1] / "install_cline_runtime.sh"
VERSION = "3.0.55"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() not in ("x86_64", "aarch64"),
    reason="The installer targets Linux x86_64/aarch64 sandboxes",
)


def _package(tmp_path: Path, *, reports: str = VERSION) -> tuple[Path, str]:
    """An npm tarball whose package/bin/cline is a script reporting ``reports`` as its version."""
    binary = f'#!/bin/sh\n[ "$1" = --version ] && echo {reports}\n'.encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("package/bin/cline")
        info.size, info.mode = len(binary), 0o755
        archive.addfile(info, io.BytesIO(binary))
        assets = tarfile.TarInfo("package/cline-hub/webview/index.html")
        archive.addfile(assets, io.BytesIO(b""))
    path = tmp_path / "cli.tgz"
    path.write_bytes(buffer.getvalue())
    return path, hashlib.sha512(buffer.getvalue()).hexdigest()


def _install(tmp_path: Path, runtime: Path, digest: str, *, package: Path | None, getconf_fails: bool = False):
    """Run the installer with curl (and optionally getconf) replaced; record each curl call."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    curl = bin_dir / "curl"
    source = f'cp {package} "$out"' if package else "exit 22"
    curl.write_text(
        "#!/bin/bash\n"
        f'echo "$@" >> {tmp_path}/curl.log\n'
        "while [ $# -gt 0 ]; do case $1 in -o) out=$2; shift 2 ;; *) shift ;; esac; done\n"
        f"{source}\n"
    )
    curl.chmod(0o755)
    getconf = bin_dir / "getconf"
    if getconf_fails:
        getconf.write_text("#!/bin/sh\nexit 1\n")
        getconf.chmod(0o755)
    elif getconf.exists():
        getconf.unlink()
    arch_digests = [digest, digest]
    return subprocess.run(
        ["bash", str(INSTALLER), str(runtime), VERSION, *arch_digests],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        timeout=60,
    )


def test_cold_install_publishes_only_the_verified_binary_and_reuses_it(tmp_path):
    package, digest = _package(tmp_path)
    runtime = tmp_path / "runtime-3.0.55"
    installed = _install(tmp_path, runtime, digest, package=package)
    assert installed.returncode == 0, installed.stderr
    assert installed.stdout.strip().endswith(VERSION)
    assert sorted(path.name for path in runtime.iterdir()) == ["cline", "ready"]
    assert "registry.npmjs.org/@cline/cli-linux-" in (tmp_path / "curl.log").read_text()

    # A ready runtime is reused without downloading again.
    reused = _install(tmp_path, runtime, digest, package=None)
    assert reused.returncode == 0, reused.stderr
    assert len((tmp_path / "curl.log").read_text().splitlines()) == 1


def test_checksum_mismatch_publishes_nothing(tmp_path):
    package, _ = _package(tmp_path)
    runtime = tmp_path / "runtime-3.0.55"
    installed = _install(tmp_path, runtime, "0" * 128, package=package)
    assert installed.returncode != 0
    assert "FAILED" in installed.stdout + installed.stderr
    assert not runtime.exists()
    assert not list(tmp_path.glob("runtime-3.0.55.*"))


def test_wrong_version_and_partial_runtime_are_replaced_or_refused(tmp_path):
    runtime = tmp_path / "runtime-3.0.55"
    runtime.mkdir()
    (runtime / "cline").write_text("partial")
    package, digest = _package(tmp_path, reports="3.0.54")
    refused = _install(tmp_path, runtime, digest, package=package)
    assert refused.returncode != 0
    assert "Cline version mismatch: 3.0.54" in refused.stderr

    package, digest = _package(tmp_path)
    installed = _install(tmp_path, runtime, digest, package=package)
    assert installed.returncode == 0, installed.stderr
    assert (runtime / "ready").exists()


def test_non_glibc_sandbox_is_refused_before_download(tmp_path):
    package, digest = _package(tmp_path)
    installed = _install(tmp_path, tmp_path / "runtime-3.0.55", digest, package=package, getconf_fails=True)
    assert installed.returncode != 0
    assert "ships only glibc Linux binaries" in installed.stderr
    assert not (tmp_path / "curl.log").exists()


def test_missing_checksum_for_the_architecture_is_refused(tmp_path):
    package, _ = _package(tmp_path)
    installed = _install(tmp_path, tmp_path / "runtime-3.0.55", "", package=package)
    assert installed.returncode != 0
    assert "No pinned SHA-512" in installed.stderr
