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
    for name in ("bash", "uname", "getconf", "mkdir", "cp", "tar", "gzip", "chmod", "mktemp", "mv", "rm"):
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


def run_installer(root: Path, env: dict[str, str], *pinned: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(root / "bin/bash"), str(root / "install.sh"), str(root / "runtime"), "1.17.11", *pinned],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )


def pinned_binary(root: Path, script: str = "#!/bin/bash\necho 1.17.11\n") -> tuple[Path, str]:
    """A staged binary as a mount would provide it: not executable, with its trusted digest."""
    binary = root / "mounted/opencode-linux-x64"
    binary.parent.mkdir(exist_ok=True)
    binary.write_text(script)
    binary.chmod(0o444)
    return binary, hashlib.sha256(binary.read_bytes()).hexdigest()


def leftovers(root: Path) -> list[str]:
    return [path.name for path in (root / "runtime").glob(".opencode.*")]


def test_pinned_binary_installs_offline_without_packages_or_download(sandbox):
    root, env = sandbox
    # Neither the download tools nor a package manager or CA bundle are needed for a pin.
    for name in ("apt-get", "tar", "gzip"):
        (root / "bin" / name).unlink()
    (root / "debian-ca.crt").unlink()
    binary, digest = pinned_binary(root)
    result = run_installer(root, env | {"TEST_UID": "1000"}, str(binary), digest)
    assert result.returncode == 0, result.stderr
    installed = root / "runtime/opencode"
    assert installed.read_bytes() == binary.read_bytes()
    assert os.access(installed, os.X_OK)
    assert not (root / "download.log").exists()
    assert not (root / "packages.log").exists()
    assert leftovers(root) == []


def test_pinned_sha_mismatch_exits_3_without_installing(sandbox):
    root, env = sandbox
    binary, _ = pinned_binary(root)
    result = run_installer(root, env, str(binary), "0" * 64)
    assert result.returncode == 3, result.stderr
    assert f"Pinned OpenCode executable SHA-256 mismatch: expected {'0' * 64}" in result.stderr
    assert not (root / "runtime/opencode").exists()
    assert not (root / "download.log").exists()
    assert leftovers(root) == []


@pytest.mark.parametrize(
    ("script", "found"),
    [
        ("#!/bin/bash\necho 1.17.10\n", "found 1.17.10"),
        ("#!/bin/bash\necho broken >&2\nexit 7\n", "no version (exit 7)"),
    ],
)
def test_pinned_binary_with_another_version_exits_3_without_installing(sandbox, script, found):
    root, env = sandbox
    binary, digest = pinned_binary(root, script)
    result = run_installer(root, env, str(binary), digest)
    assert result.returncode == 3, result.stderr
    assert "Pinned OpenCode executable version mismatch: requested 1.17.11" in result.stderr
    assert found in result.stderr
    assert not (root / "runtime/opencode").exists()
    assert leftovers(root) == []


@pytest.mark.parametrize(
    ("script", "code", "message"),
    [
        ("#!/bin/bash\nkill -KILL $$\n", 1, "--version did not complete (exit 137)"),
        # Bash reports a missing interpreter as 127, or as 126 ("bad interpreter") before 5.2.
        ("#!/nonexistent/interpreter\n", 1, "--version did not complete (exit 12"),
        ("\x7fELF\x00\x00\x00 not a program", 3, "version mismatch: requested 1.17.11, found no version (exit 126)"),
    ],
    ids=["signal", "missing interpreter", "exec format"],
)
def test_only_a_version_check_that_finishes_can_reject_the_pin(sandbox, script, code, message):
    root, env = sandbox
    binary, digest = pinned_binary(root, script)
    result = run_installer(root, env, str(binary), digest)
    assert result.returncode == code, result.stderr
    assert message in result.stderr
    assert not (root / "runtime/opencode").exists()
    assert leftovers(root) == []


def test_missing_pinned_source_is_not_a_rejected_pin(sandbox):
    root, env = sandbox
    result = run_installer(root, env, str(root / "mounted/absent"), "0" * 64)
    assert result.returncode == 1
    assert "Pinned OpenCode executable not found in the sandbox" in result.stderr
    assert not (root / "download.log").exists()


@pytest.mark.parametrize("matching", [True, False])
def test_cached_pinned_runtime_is_hashed_before_it_runs(sandbox, matching):
    root, env = sandbox
    binary, digest = pinned_binary(root)
    (root / "runtime").mkdir()
    cached = root / "runtime/opencode"
    # A different cached build would leave a marker if it were executed before its hash check.
    cached.write_text(binary.read_text() if matching else '#!/bin/bash\ntouch "$TEST_ROOT/ran"\necho 1.17.11\n')
    cached.chmod(0o755)
    if matching:
        binary.unlink()
    result = run_installer(root, env, str(binary), digest)
    assert result.returncode == 0, result.stderr
    assert not (root / "ran").exists()
    assert hashlib.sha256(cached.read_bytes()).hexdigest() == digest
    assert not (root / "download.log").exists()


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


@pytest.mark.parametrize("pinned", [False, True])
def test_pinned_ripgrep_replaces_the_best_effort_package_install(sandbox, pinned):
    root, env = sandbox
    (root / "bin/rg").unlink()
    (root / "runtime").mkdir()
    cached = root / "runtime/opencode"
    cached.write_text("#!/bin/bash\necho 1.17.11\n")
    cached.chmod(0o755)
    result = run_installer(root, env | ({"NG_OPENCODE_SKIP_RIPGREP": "1"} if pinned else {}))
    assert result.returncode == 0, result.stderr
    assert (root / "packages.log").exists() is not pinned
    assert ("ripgrep" in result.stderr or (root / "bin/rg").exists()) is not pinned
