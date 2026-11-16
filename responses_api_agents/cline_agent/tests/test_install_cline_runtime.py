# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-sandbox installer, run with bash against a local stand-in for the npm registry."""

import fcntl
import hashlib
import io
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
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


def _installer(
    tmp_path: Path, runtime: Path, digest: str, *, package: Path | None, getconf_fails: bool = False
) -> tuple[list[str], dict[str, str]]:
    """The installer command and environment, with curl (and optionally getconf) replaced and logged."""
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
    command = ["bash", str(INSTALLER), str(runtime), VERSION, *arch_digests]
    return command, {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}


def _install(tmp_path: Path, runtime: Path, digest: str, *, package: Path | None, getconf_fails: bool = False):
    """Run the installer to completion."""
    command, env = _installer(tmp_path, runtime, digest, package=package, getconf_fails=getconf_fails)
    return subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)


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


@pytest.mark.skipif(shutil.which("flock") is None, reason="needs flock")
def test_a_runtime_another_installer_published_is_not_replaced(tmp_path):
    runtime = tmp_path / "runtime-3.0.55"
    runtime.mkdir()
    (runtime / "cline").write_text("partial")
    package, digest = _package(tmp_path)
    command, env = _installer(tmp_path, runtime, digest, package=package)
    with open(f"{runtime}.lock", "a") as lock:
        # Another installer holds the lock while it replaces the stale runtime.
        fcntl.flock(lock, fcntl.LOCK_EX)
        installer = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        deadline = time.monotonic() + 30
        while not list(tmp_path.glob(f"{runtime.name}.*/ready")):
            assert installer.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        time.sleep(1)
        assert (runtime / "cline").read_text() == "partial", "the installer replaced the runtime without the lock"
        shutil.rmtree(runtime)
        runtime.mkdir()
        (runtime / "cline").write_text(f'#!/bin/sh\n[ "$1" = --version ] && echo {VERSION}\n')
        (runtime / "cline").chmod(0o755)
        (runtime / "ready").touch()
        (runtime / "published-by-the-other-installer").touch()
        fcntl.flock(lock, fcntl.LOCK_UN)
    stdout, stderr = installer.communicate(timeout=60)
    assert installer.returncode == 0, stderr
    assert stdout.strip().endswith(VERSION)
    # The published runtime is kept, not replaced by this installer's copy.
    assert (runtime / "published-by-the-other-installer").exists()
    assert not [path for path in tmp_path.glob(f"{runtime.name}.*") if path.is_dir()]


def test_without_flock_cold_install_and_reuse_work_but_stale_runtime_is_not_removed(tmp_path):
    package, digest = _package(tmp_path)
    runtime = tmp_path / "runtime-3.0.55"
    command, env = _installer(tmp_path, runtime, digest, package=package)
    bin_dir = tmp_path / "bin"
    # Use the real shell and tools, excluding only flock from this sandbox's PATH.
    for name in (
        "bash",
        "sh",
        "uname",
        "getconf",
        "tar",
        "gzip",
        "sha512sum",
        "mkdir",
        "dirname",
        "mktemp",
        "rm",
        "mv",
        "chmod",
        "touch",
        "cp",
    ):
        (bin_dir / name).symlink_to(shutil.which(name))
    env["PATH"] = str(bin_dir)
    first = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
    assert first.returncode == 0, first.stderr
    reused = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
    assert reused.returncode == 0, reused.stderr
    assert len((tmp_path / "curl.log").read_text().splitlines()) == 1

    (runtime / "ready").unlink()
    partial = (runtime / "cline").read_bytes()
    refused = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
    assert refused.returncode != 0
    assert "Cannot replace stale Cline runtime safely without flock" in refused.stderr
    assert (runtime / "cline").read_bytes() == partial
    assert not (runtime / "ready").exists()
    assert not [path for path in tmp_path.glob(f"{runtime.name}.*") if path.is_dir()]


def test_lock_wait_is_bounded_and_does_not_remove_stale_runtime(tmp_path):
    runtime = tmp_path / "runtime-3.0.55"
    runtime.mkdir()
    (runtime / "cline").write_text("partial")
    package, digest = _package(tmp_path)
    command, env = _installer(tmp_path, runtime, digest, package=package)
    flock = tmp_path / "bin" / "flock"
    flock.write_text('#!/bin/sh\n[ "$1" = -w ] && [ "$2" = 60 ] && [ "$3" = 9 ] || exit 99\nexit 1\n')
    flock.chmod(0o755)
    refused = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
    assert refused.returncode != 0
    assert "Timed out waiting for Cline runtime lock" in refused.stderr
    assert (runtime / "cline").read_text() == "partial"


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


@pytest.mark.parametrize("valid_digest,reports", [(True, VERSION), (False, VERSION), (True, "3.0.54")])
def test_uploaded_binary_uses_locked_publisher_without_download_tools(tmp_path, valid_digest, reports):
    source = tmp_path / "cline.upload"
    source.write_text(f'#!/bin/sh\n[ "$1" = --version ] && echo {reports}\n')
    digest = hashlib.sha512(source.read_bytes()).hexdigest() if valid_digest else "0" * 128
    runtime = tmp_path / "runtime-3.0.55"
    command, env = _installer(tmp_path, runtime, "0" * 128, package=None)
    command.extend([str(source), digest])
    bin_dir = tmp_path / "bin"
    for name in (
        "bash",
        "sh",
        "uname",
        "getconf",
        "sha512sum",
        "mkdir",
        "dirname",
        "mktemp",
        "rm",
        "mv",
        "chmod",
        "touch",
        "cp",
    ):
        (bin_dir / name).symlink_to(shutil.which(name))
    # No curl, tar, gzip or package manager: offline publication must not need them.
    (bin_dir / "curl").unlink()
    env["PATH"] = str(bin_dir)
    installed = subprocess.run(command, capture_output=True, text=True, env=env, timeout=60)
    if valid_digest and reports == VERSION:
        assert installed.returncode == 0, installed.stderr
        assert sorted(p.name for p in runtime.iterdir()) == ["cline", "ready"]
    else:
        assert installed.returncode != 0
        assert not runtime.exists()
    assert not (tmp_path / "curl.log").exists()
