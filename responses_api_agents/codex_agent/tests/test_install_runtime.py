# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


INSTALLER = Path(__file__).parents[1] / "install_codex_runtime.sh"
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Native Codex requires Linux/glibc")


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Run the real installer with an isolated PATH and no access to the host package manager."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in (
        "bash",
        "uname",
        "getconf",
        "mkdir",
        "cp",
        "tar",
        "xz",
        "sha256sum",
        "awk",
        "touch",
        "flock",
        "sleep",
    ):
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
  cp "$TEST_ROOT/curl" "$TEST_ROOT/bin/curl"
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
    return tmp_path, os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)}


def run_installer(root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(root / "bin/bash"), str(INSTALLER), str(root / "runtime"), "0.144.4"],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )


def test_missing_curl_is_installed_before_downloading_node(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (root / "packages.log").read_text().splitlines() == [
        "update",
        "install -y --no-install-recommends curl ca-certificates tar xz-utils coreutils gawk",
    ]
    assert "https://nodejs.org/dist/v22.19.0/node-v22.19.0-linux-" in (root / "download.log").read_text()
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
    assert "preinstall them" in result.stderr
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
    (root / "runtime/node/bin").mkdir(parents=True)
    node = root / "runtime/node/bin/node"
    node.write_text('#!/bin/bash\necho "codex-cli 0.144.4"\n')
    node.chmod(0o755)
    shutil.copy(root / "curl", root / "bin/curl")
    result = run_installer(root, env)
    assert result.returncode == 0, result.stderr
    assert not (root / "packages.log").exists()
    assert not (root / "download.log").exists()


def test_installer_diagnostics_preserve_command_exit_and_stderr(sandbox):
    root, env = sandbox
    result = run_installer(root, env | {"TEST_INSTALL_FAIL": "1"})
    assert result.returncode == 100
    assert "Codex installer failed (exit 100)" in result.stderr
    assert "apt-get install" in result.stderr


def test_cached_runtime_version_mismatch_fails(sandbox):
    root, env = sandbox
    (root / "runtime/node/bin").mkdir(parents=True)
    (root / "runtime/ready").touch()
    node = root / "runtime/node/bin/node"
    node.write_text('#!/bin/bash\necho "codex-cli 0.0.1"\n')
    node.chmod(0o755)
    result = run_installer(root, env)
    assert result.returncode != 0
    assert "Codex version mismatch: codex-cli 0.0.1" in result.stderr
    assert not (root / "packages.log").exists()


def test_concurrent_installers_populate_runtime_once(sandbox):
    root, env = sandbox
    # Model the runtime materialized by tar/npm, while exercising the real shell + flock.
    (root / "runtime/node/bin").mkdir(parents=True)
    node = root / "runtime/node/bin/node"
    node.write_text("""#!/bin/bash
if [[ "$1" == *npm-cli.js ]]; then
  echo install >> "$TEST_ROOT/npm.log"
else
  echo 'codex-cli 0.144.4'
fi
""")
    node.chmod(0o755)
    for command, script in {
        "curl": """#!/bin/bash
if [[ "$*" == *SHASUMS256* ]]; then
  echo 'stub node-v22.19.0-linux-x64.tar.xz' > SHASUMS256.txt
else
  echo download >> "$TEST_ROOT/downloads.log"
fi
""",
        "tar": """#!/bin/bash
touch "$TEST_ROOT/entered"
while [ ! -f "$TEST_ROOT/release" ]; do sleep 0.01; done
""",
        "sha256sum": "#!/bin/bash\nwhile read -r line; do :; done\n",
    }.items():
        path = root / "bin" / command
        path.unlink(missing_ok=True)
        path.write_text(script)
        path.chmod(0o755)
    argv = [str(root / "bin/bash"), str(INSTALLER), str(root / "runtime"), "0.144.4"]
    processes = [subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)]
    try:
        deadline = time.monotonic() + 3
        while not (root / "entered").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert (root / "entered").exists()
        processes.append(subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        time.sleep(0.1)
        assert processes[1].poll() is None
        assert (root / "downloads.log").read_text().splitlines() == ["download"]
        (root / "release").touch()
        for process in processes:
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, (stdout, stderr)
        assert (root / "npm.log").read_text().splitlines() == ["install"]
        assert (root / "downloads.log").read_text().splitlines() == ["download"]
        assert (root / "runtime/ready").exists()
    finally:
        (root / "release").touch()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
