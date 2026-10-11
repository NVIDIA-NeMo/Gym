# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


INSTALLER = Path(__file__).parents[1] / "install_pi_runtime.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Pi requires Linux")


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Run the real installer with an isolated PATH and no access to the host package manager."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("bash", "uname", "getconf", "mkdir", "cp", "ln"):
        executable = shutil.which(name)
        if executable is None:
            pytest.skip(f"Installer test requires {name}")
        (bindir / name).symlink_to(executable)
    (bindir / "python3").symlink_to(sys.executable)
    scripts = {
        "id": '#!/bin/bash\necho "${TEST_UID:-0}"\n',
        "apt-get": """#!/bin/bash
echo "$*" >> "$TEST_ROOT/packages.log"
if [ "$1" = install ]; then
  [ "${TEST_INSTALL_FAIL:-0}" = 0 ] || exit 100
  case " $* " in *" python3 "*) ln -s "$TEST_PYTHON" "$TEST_ROOT/bin/python3" ;; esac
  case " $* " in *" curl "*) cp "$TEST_ROOT/curl" "$TEST_ROOT/bin/curl" ;; esac
fi
""",
    }
    for name, script in scripts.items():
        (bindir / name).write_text(script)
        (bindir / name).chmod(0o755)
    # Stop at the first download: neither apt nor external network requests are real in these tests.
    curl = tmp_path / "curl"
    curl.write_text('#!/bin/bash\necho "$*" > "$TEST_ROOT/download.log"\nexit 19\n')
    curl.chmod(0o755)
    return tmp_path, os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path), "TEST_PYTHON": sys.executable}


def run_installer(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(root / "bin/bash"), str(INSTALLER), str(root / "runtime"), "0.80.2"],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )


@pytest.fixture
def musl_sandbox(sandbox: tuple[Path, dict[str, str]]) -> tuple[Path, dict[str, str]]:
    root, env = sandbox
    (root / "bin/getconf").unlink()
    (root / "bin/uname").unlink()
    for name, script in {
        "uname": '#!/bin/bash\nif [ "$1" = -m ]; then echo x86_64; else echo Linux; fi\n',
        "getconf": "#!/bin/bash\nexit 1\n",
        "ldd": "#!/bin/bash\necho 'musl libc (x86_64)' >&2\nexit 1\n",
        "apk": '#!/bin/bash\necho "$*" >> "$TEST_ROOT/packages.log"\ncp "$TEST_ROOT/curl" "$TEST_ROOT/bin/curl"\n',
    }.items():
        (root / "bin" / name).write_text(script)
        (root / "bin" / name).chmod(0o755)
    return root, env


def test_missing_curl_is_installed_before_downloading_node(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (root / "packages.log").read_text().splitlines() == [
        "update",
        "install -y --no-install-recommends curl ca-certificates",
    ]
    assert "https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-" in (root / "download.log").read_text()
    assert not (root / "runtime/ready").exists()


def test_missing_python_is_bootstrapped_before_runtime_checks(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / "bin/python3").unlink()
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (root / "packages.log").read_text().splitlines() == [
        "update",
        "install -y --no-install-recommends python3",
        "update",
        "install -y --no-install-recommends curl ca-certificates",
    ]
    assert (root / "download.log").exists()


def test_missing_python_without_root_explains_remedy(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / "bin/python3").unlink()
    result = run_installer(root, env | {"TEST_UID": "1000"})
    assert result.returncode == 1
    assert "Pi requires python3: preinstall these packages" in result.stderr
    assert not (root / "download.log").exists()


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
    assert "Pi requires curl ca-certificates: preinstall these packages" in result.stderr
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
    (root / "runtime/ready").touch()
    result = run_installer(root, env)
    assert result.returncode == 0, result.stderr
    assert not (root / "packages.log").exists()
    assert not (root / "download.log").exists()


@pytest.mark.skipif(shutil.which("python3.8") is None, reason="Python 3.8 is not installed")
def test_python38_can_bootstrap_runtime(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    (root / "bin/python3").unlink()
    (root / "bin/python3").symlink_to(shutil.which("python3.8"))
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (root / "download.log").exists()


def test_unknown_libc_explains_failure_before_install(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    getconf = root / "bin/getconf"
    getconf.unlink()
    getconf.write_text('#!/bin/bash\necho "getconf: GNU_LIBC_VERSION: unknown variable" >&2\nexit 1\n')
    getconf.chmod(0o755)
    result = run_installer(root, env)
    assert result.returncode == 1
    assert "could not identify the sandbox libc" in result.stderr
    assert not (root / "packages.log").exists()
    assert not (root / "download.log").exists()


def test_musl_installs_curl_with_apk_and_selects_musl_node(musl_sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = musl_sandbox
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (root / "packages.log").read_text().splitlines() == ["add --no-cache curl ca-certificates"]
    assert (
        "https://unofficial-builds.nodejs.org/download/release/v22.19.0/node-v22.19.0-linux-x64-musl.tar.gz"
        in (root / "download.log").read_text()
    )
    assert not (root / "runtime/ready").exists()


def test_musl_existing_dependencies_need_no_root(musl_sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = musl_sandbox
    shutil.copy(root / "curl", root / "bin/curl")
    result = run_installer(root, env | {"TEST_UID": "1000"})
    assert result.returncode == 19, result.stderr
    assert "linux-x64-musl.tar.gz" in (root / "download.log").read_text()
    assert not (root / "packages.log").exists()


def test_musl_arm64_explains_unavailable_build(musl_sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = musl_sandbox
    (root / "bin/uname").unlink()
    (root / "bin/uname").write_text('#!/bin/bash\nif [ "$1" = -m ]; then echo aarch64; else echo Linux; fi\n')
    (root / "bin/uname").chmod(0o755)
    result = run_installer(root, env)
    assert result.returncode == 1
    assert "Node 22.19.0 musl build supports x86_64 only" in result.stderr
    assert not (root / "packages.log").exists()
    assert not (root / "download.log").exists()
