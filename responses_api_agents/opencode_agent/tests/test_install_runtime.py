# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


INSTALLER = Path(__file__).parents[1] / "install_opencode_runtime.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="OpenCode sandbox installer requires Linux")


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Run the real installer with an isolated PATH and no access to the host package manager."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("bash", "uname", "getconf", "mkdir", "cp", "tar", "gzip", "chmod"):
        executable = shutil.which(name)
        if executable is None:
            pytest.skip(f"Installer test requires {name}")
        (bindir / name).symlink_to(executable)
    (bindir / "python3").symlink_to(sys.executable)
    scripts = {
        "rg": '#!/bin/bash\necho "ripgrep test fixture"\n',
        "id": '#!/bin/bash\necho "${TEST_UID:-0}"\n',
        "apt-get": """#!/bin/bash
echo "$*" >> "$TEST_ROOT/packages.log"
if [ "$1" = update ]; then
  [ "${TEST_UPDATE_FAIL:-0}" = 0 ] || exit 100
fi
if [ "$1" = install ]; then
  [ "${TEST_INSTALL_FAIL:-0}" = 0 ] || exit 100
  cp "$TEST_ROOT/curl" "$TEST_ROOT/bin/curl"
  if [ "${@: -1}" = ripgrep ]; then
    cp "$TEST_ROOT/ripgrep" "$TEST_ROOT/bin/rg"
  fi
fi
""",
    }
    for name, script in scripts.items():
        (bindir / name).write_text(script)
        (bindir / name).chmod(0o755)
    shutil.copy(bindir / "rg", tmp_path / "ripgrep")
    # Stop at the first download: neither apt nor external network requests are real in these tests.
    curl = tmp_path / "curl"
    curl.write_text('#!/bin/bash\necho "$*" > "$TEST_ROOT/download.log"\nexit 19\n')
    curl.chmod(0o755)
    # Isolate certificate locations without touching the machine's trust store.
    (tmp_path / "debian-ca.crt").write_text("test certificate bundle")
    (tmp_path / "install.sh").write_text(
        INSTALLER.read_text()
        .replace("/etc/ssl/certs/ca-certificates.crt", str(tmp_path / "debian-ca.crt"))
        .replace("/etc/pki/tls/certs/ca-bundle.crt", str(tmp_path / "rhel-ca.crt"))
    )
    return tmp_path, os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)}


def run_installer(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(root / "bin/bash"), str(root / "install.sh"), str(root / "runtime"), "1.17.11"],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )


def test_missing_curl_is_installed_before_downloading_opencode(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (root / "packages.log").read_text().splitlines() == [
        "update",
        "install -y --no-install-recommends curl",
    ]
    assert (
        "https://github.com/anomalyco/opencode/releases/download/v1.17.11/opencode-linux-"
        in (root / "download.log").read_text()
    )
    assert not (root / "runtime/ready").exists()


def test_existing_curl_does_not_install_packages(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    shutil.copy(root / "curl", root / "bin/curl")
    result = run_installer(root, env | {"TEST_UID": "1000"})
    assert result.returncode == 19, result.stderr
    assert (root / "download.log").exists()
    assert not (root / "packages.log").exists()


@pytest.mark.parametrize("missing", ["root", "apt-get"])
def test_missing_curl_without_bootstrap_support_explains_remedy(
    sandbox: tuple[Path, dict[str, str]], missing: str
) -> None:
    root, env = sandbox
    if missing == "root":
        env["TEST_UID"] = "1000"
    else:
        (root / "bin/apt-get").unlink()
    result = run_installer(root, env)
    assert result.returncode == 1
    assert "preinstall them in the task image" in result.stderr
    assert not (root / "packages.log").exists()
    assert not (root / "download.log").exists()


def test_package_failure_stops_before_download(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = run_installer(root, env | {"TEST_INSTALL_FAIL": "1"})
    assert result.returncode == 100
    assert not (root / "download.log").exists()
    assert not (root / "runtime/ready").exists()


def test_cached_runtime_needs_no_package_installation(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / "runtime").mkdir()
    (root / "runtime/opencode").write_text("#!/bin/bash\necho 1.17.11\n")
    (root / "runtime/opencode").chmod(0o755)
    result = run_installer(root, env)
    assert result.returncode == 0, result.stderr
    assert not (root / "packages.log").exists()
    assert not (root / "download.log").exists()


def test_wrong_cached_version_is_not_reused(sandbox):
    root, env = sandbox
    (root / "runtime").mkdir()
    binary = root / "runtime/opencode"
    binary.write_text("#!/bin/bash\necho 1.17.10\n")
    binary.chmod(0o755)
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (root / "download.log").exists()


def test_rhel_certificate_bundle_does_not_require_root_or_package_manager(sandbox):
    root, env = sandbox
    (root / "debian-ca.crt").unlink()
    (root / "rhel-ca.crt").write_text("test RHEL bundle")
    (root / "bin/apt-get").unlink()
    (root / "bin/rg").unlink()
    shutil.copy(root / "curl", root / "bin/curl")
    result = run_installer(root, env | {"TEST_UID": "1000"})
    assert result.returncode == 19, result.stderr
    assert (root / "download.log").exists()
    assert not (root / "packages.log").exists()
    assert "OpenCode can download it on demand" in result.stderr


@pytest.mark.parametrize("tool", ["tar", "gzip"])
def test_missing_archive_tool_is_installed(sandbox, tool):
    root, env = sandbox
    (root / "bin" / tool).unlink()
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert f"install -y --no-install-recommends curl {tool}" in (root / "packages.log").read_text()


@pytest.mark.parametrize("cached", [False, True])
def test_ripgrep_is_provisioned_even_with_a_cached_opencode(sandbox, cached):
    root, env = sandbox
    (root / "bin/rg").unlink()
    if cached:
        (root / "runtime").mkdir()
        binary = root / "runtime/opencode"
        binary.write_text("#!/bin/bash\necho 1.17.11\n")
        binary.chmod(0o755)
    result = run_installer(root, env)
    assert result.returncode == (0 if cached else 19), result.stderr
    expected = [] if cached else ["update", "install -y --no-install-recommends curl"]
    assert (root / "packages.log").read_text().splitlines() == expected + [
        "update",
        "install -y --no-install-recommends ripgrep",
    ]
    assert (root / "bin/rg").exists()
    assert "OpenCode can download it on demand" not in result.stderr


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("unavailable", ["nonroot", "no_manager", "apt_update", "apt_install", "apk"])
def test_optional_ripgrep_failure_does_not_block_runtime(sandbox, cached, unavailable):
    root, env = sandbox
    (root / "bin/rg").unlink()
    shutil.copy(root / "curl", root / "bin/curl")
    if cached:
        (root / "runtime").mkdir()
        binary = root / "runtime/opencode"
        binary.write_text("#!/bin/bash\necho 1.17.11\n")
        binary.chmod(0o755)
    if unavailable == "nonroot":
        env["TEST_UID"] = "1000"
    elif unavailable == "no_manager":
        (root / "bin/apt-get").unlink()
        (root / "debian-ca.crt").unlink()
        (root / "rhel-ca.crt").write_text("test RHEL bundle")
    elif unavailable == "apt_update":
        env["TEST_UPDATE_FAIL"] = "1"
    elif unavailable == "apt_install":
        env["TEST_INSTALL_FAIL"] = "1"
    else:
        (root / "bin/apk").write_text("#!/bin/bash\nexit 100\n")
        (root / "bin/apk").chmod(0o755)
    result = run_installer(root, env)
    assert result.returncode == (0 if cached else 19), result.stderr
    assert "OpenCode can download it on demand" in result.stderr
    assert (root / "download.log").exists() is not cached
    if unavailable in {"nonroot", "no_manager"}:
        assert not (root / "packages.log").exists()


@pytest.mark.skipif(shutil.which("python3.8") is None, reason="Python 3.8 is not installed")
def test_python38_can_bootstrap_runtime(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / "bin/python3").unlink()
    (root / "bin/python3").symlink_to(shutil.which("python3.8"))
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (root / "download.log").exists()


def test_musl_bootstraps_with_apk_and_downloads_matching_binary(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / "bin/getconf").unlink()
    for name, script in {
        "getconf": "#!/bin/bash\nexit 1\n",
        "ldd": "#!/bin/bash\necho 'musl libc (x86_64)' >&2\nexit 1\n",
        "apk": '#!/bin/bash\necho "$*" >> "$TEST_ROOT/packages.log"\ncp "$TEST_ROOT/curl" "$TEST_ROOT/bin/curl"\n',
    }.items():
        (root / "bin" / name).write_text(script)
        (root / "bin" / name).chmod(0o755)
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (root / "packages.log").read_text().splitlines() == ["add --no-cache curl"]
    arch = {"x86_64": "x64-baseline", "aarch64": "arm64"}[os.uname().machine]
    assert f"opencode-linux-{arch}-musl.tar.gz" in (root / "download.log").read_text()


@pytest.mark.parametrize("valid_digest", [True, False])
def test_preinstalled_binary_needs_no_download_and_checks_digest(sandbox, valid_digest):
    root, env = sandbox
    binary = root / "preinstalled-opencode"
    binary.write_text("#!/bin/bash\necho 1.17.11\n")
    binary.chmod(0o755)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest() if valid_digest else "0" * 64
    result = subprocess.run(
        [str(root / "bin/bash"), str(root / "install.sh"), str(root / "runtime"), "1.17.11", str(binary), digest],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert (result.returncode == 0) is valid_digest, result.stderr
    assert not (root / "download.log").exists()
    assert not (root / "packages.log").exists()
    if valid_digest:
        assert (root / "runtime/opencode").read_bytes() == binary.read_bytes()
    else:
        assert "digest mismatch" in result.stderr
        assert not (root / "runtime/opencode").exists()


def seed_binary(
    root: Path, env: dict[str, str], binary: Path, *, version: str = "1.17.11", digest: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(root / "bin/bash"),
            str(root / "install.sh"),
            str(root / "runtime"),
            version,
            str(binary),
            digest if digest is not None else hashlib.sha256(binary.read_bytes()).hexdigest(),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("failure", ["digest", "version"])
def test_invalid_seed_preserves_existing_executable(sandbox, failure):
    root, env = sandbox
    binary = root / "preinstalled-opencode"
    binary.write_text("#!/bin/bash\necho 1.17.11\n")
    assert seed_binary(root, env, binary).returncode == 0
    target = root / "runtime/opencode"
    original, inode = target.read_bytes(), target.stat().st_ino
    binary.write_text("#!/bin/bash\necho 1.17.10\n")
    result = seed_binary(root, env, binary, digest="0" * 64 if failure == "digest" else None)
    assert result.returncode != 0 and f"{failure} mismatch" in result.stderr
    assert target.read_bytes() == original and target.stat().st_ino == inode
    assert not list(target.parent.glob(".opencode-*"))
    assert not (root / "download.log").exists() and not (root / "packages.log").exists()


@pytest.mark.skipif(shutil.which("cc") is None, reason="Native executable regression requires a C compiler")
def test_reseeding_running_executable_publishes_a_separate_inode(sandbox):
    root, env = sandbox
    source = root / "opencode.c"
    source.write_text(
        "#include <stdio.h>\n#include <string.h>\n#include <unistd.h>\n"
        "int main(int argc, char **argv) {\n"
        '  if (argc == 2 && strcmp(argv[1], "--version") == 0) { puts("1.17.11"); return 0; }\n'
        '  puts("ready"); fflush(stdout);\n'
        "  for (;;) pause();\n}\n"
    )
    binary = root / "preinstalled-opencode"
    subprocess.run([shutil.which("cc"), str(source), "-o", str(binary)], check=True, timeout=30)
    first = seed_binary(root, env, binary)
    assert first.returncode == 0, first.stderr
    target = root / "runtime/opencode"
    inode = target.stat().st_ino
    with subprocess.Popen([str(target)], stdout=subprocess.PIPE, text=True) as process:
        try:
            assert process.stdout.readline() == "ready\n"
            assert Path(f"/proc/{process.pid}/exe").stat().st_ino == inode
            second = seed_binary(root, env, binary)
            assert second.returncode == 0, second.stderr
            assert process.poll() is None
            assert Path(f"/proc/{process.pid}/exe").stat().st_ino == inode
            assert target.stat().st_ino != inode and target.read_bytes() == binary.read_bytes()
        finally:
            process.terminate()
            process.wait(timeout=10)
    assert not list(target.parent.glob(".opencode-*"))
    assert not (root / "download.log").exists() and not (root / "packages.log").exists()
