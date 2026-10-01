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
pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Native Codex requires Linux")


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
        "gzip",
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
        "install -y --no-install-recommends curl ca-certificates tar gzip coreutils gawk",
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
  echo 'stub node-v22.19.0-linux-x64.tar.gz' > SHASUMS256.txt
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


def _replace_command(root: Path, name: str, script: str) -> None:
    path = root / "bin" / name
    path.unlink(missing_ok=True)
    path.write_text("#!/bin/bash\n" + script)
    path.chmod(0o755)


def _use_musl_platform(root: Path, *, architecture: str = "x86_64") -> None:
    """Simulate libc detection without using host package managers or downloads."""
    _replace_command(root, "getconf", "exit 1\n")
    _replace_command(root, "ldd", "echo 'musl libc' >&2\nexit 1\n")
    _replace_command(root, "uname", f'case "$1" in -s) echo Linux;; -m) echo {architecture};; esac\n')
    # The task image has its existing C++ library, but our private Node may need
    # a newer version. Other Python calls still exercise the real interpreter.
    _replace_command(
        root,
        "python3",
        'if [[ "$*" == *ctypes.CDLL* ]]; then exit 0; fi\n' + f'exec "{sys.executable}" "$@"\n',
    )
    _replace_command(
        root,
        "apk",
        'echo "$*" >> "$TEST_ROOT/packages.log"\ncp "$TEST_ROOT/curl" "$TEST_ROOT/bin/curl"\n',
    )


@pytest.mark.parametrize("bootstrap", [False, True])
def test_musl_uses_pinned_x64_archive_and_apk(sandbox: tuple[Path, dict[str, str]], bootstrap: bool) -> None:
    root, env = sandbox
    _use_musl_platform(root)
    if not bootstrap:
        shutil.copy(root / "curl", root / "bin/curl")
    result = run_installer(root, env)
    assert result.returncode == 19, result.stderr
    assert (
        "https://unofficial-builds.nodejs.org/download/release/v22.19.0/node-v22.19.0-linux-x64-musl.tar.gz"
        in (root / "download.log").read_text()
    )
    if bootstrap:
        assert (root / "packages.log").read_text().splitlines() == [
            "add --no-cache curl ca-certificates tar gzip coreutils gawk"
        ]
    else:
        assert not (root / "packages.log").exists()
    assert not (root / "runtime/ready").exists()


def test_musl_arm64_is_rejected_before_installation(sandbox: tuple[Path, dict[str, str]]) -> None:
    root, env = sandbox
    _use_musl_platform(root, architecture="aarch64")
    result = run_installer(root, env)
    assert result.returncode == 1, result.stderr
    assert "musl build supports x86_64 only" in result.stderr
    assert not (root / "packages.log").exists()
    assert not (root / "download.log").exists()


@pytest.mark.parametrize("checksum_ok", [True, False])
def test_old_musl_cpp_repair_is_private_and_checksum_gated(
    sandbox: tuple[Path, dict[str, str]], checksum_ok: bool
) -> None:
    root, env = sandbox
    _use_musl_platform(root)
    commands = {
        "curl": """for last; do :; done
echo "$*" >> "$TEST_ROOT/download.log"
if [[ "$*" == *SHASUMS256* ]]; then
  echo 'stub  node-v22.19.0-linux-x64-musl.tar.gz' > "$last"
else
  echo fixture > "$last"
fi
""",
        "sha256sum": """while IFS= read -r line; do
  printf '%s\\n' "$line" >> "$TEST_ROOT/checksums.log"
  if [[ "$line" == *libstdc++.apk ]]; then exit "$TEST_APK_CHECKSUM_STATUS"; fi
done
""",
        "tar": 'echo "$*" >> "$TEST_ROOT/extract.log"\n',
        "patchelf": 'echo "$*" >> "$TEST_ROOT/patchelf.log"\n: > "$TEST_ROOT/patched"\n',
    }
    for name, script in commands.items():
        _replace_command(root, name, script)
    node = root / "runtime/node/bin/node"
    node.parent.mkdir(parents=True)
    node.write_text("""#!/bin/bash
case "$1" in
  --version) [ -f "$TEST_ROOT/patched" ] || exit 1; echo v22.19.0;;
  */npm-cli.js) echo install >> "$TEST_ROOT/npm.log";;
  *) echo 'codex-cli 0.144.4';;
esac
""")
    node.chmod(0o755)
    result = run_installer(root, env | {"TEST_APK_CHECKSUM_STATUS": "0" if checksum_ok else "1"})
    assert (
        "3cf66a7164240ef590106496d3c75f486bac46cba9cf2198c0c3b318c53ad027  libstdc++.apk"
        in (root / "checksums.log").read_text()
    )
    if not checksum_ok:
        assert result.returncode != 0
        assert not (root / "patchelf.log").exists()
        assert not (root / "npm.log").exists()
        assert not (root / "runtime/ready").exists()
        assert "libstdc++.apk" not in (root / "extract.log").read_text()
        return
    assert result.returncode == 0, result.stderr
    assert (root / "patchelf.log").read_text().splitlines() == [f"--set-rpath $ORIGIN/../../libstdcpp/usr/lib {node}"]
    assert "-xzf libstdc++.apk -C libstdcpp usr/lib" in (root / "extract.log").read_text()
    assert (root / "npm.log").read_text().splitlines() == ["install"]
    assert (root / "runtime/ready").exists()
    assert not (root / "packages.log").exists()
