# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from responses_api_agents.copilot_agent.app import _sandbox_prepare_command


INSTALLER = Path(__file__).parents[1] / "install_copilot_runtime.sh"
PACKAGE = "copilot-linux-x64"
URL = "https://registry.test/@github/copilot-linux-x64/-/copilot-linux-x64-1.0.103.tgz"
# Stands in for the standalone binary: the first run unpacks into $HOME/.cache like Copilot 1.0.103.
FAKE_BINARY = """#!/bin/sh
if [ -n "${COPILOT_CLI_DIST_DIR:-}" ]; then
  [ -d "$COPILOT_CLI_DIST_DIR" ] || exit 9
else
  mkdir -p "$HOME/.cache/copilot/pkg/linux-x64/1.0.103/builtin"
fi
echo "GitHub Copilot CLI 1.0.103."
"""
# Records each download and serves the package the test staged.
FAKE_CURL = """#!/bin/bash
while (($#)); do case $1 in -o) out=$2; shift 2 ;; *) url=$1; shift ;; esac; done
echo "$url" >> "$TEST_ROOT/downloads.log"
cat "$TEST_ROOT/served.tgz" > "$out"
"""


def package(binary: str = FAKE_BINARY) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, text in (("package/package.json", "{}"), ("package/copilot", binary)):
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o755
            tar.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


@pytest.fixture
def sandbox(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """Run the real installer with an isolated PATH: no python3, no package manager, and a fake curl."""
    if not sys.platform.startswith("linux"):
        pytest.skip("Installer publication requires Linux coreutils (mv -T)")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("bash", "env", "mkdir", "dirname", "rm", "mv", "chmod", "touch", "cat", "tar", "sha512sum"):
        found = shutil.which(name)
        if found is None:
            pytest.skip(f"Installer test requires {name}")
        (bindir / name).symlink_to(found)
    # GNU tar runs gzip; flock serializes concurrent installers where the image has it.
    for name in ("gzip", "flock"):
        if found := shutil.which(name):
            (bindir / name).symlink_to(found)
    if not (bindir / "flock").exists():
        executable(bindir / "flock", "#!/bin/sh\nexit 0\n")
    executable(bindir / "id", '#!/bin/bash\necho "${TEST_UID:-0}"\n')
    executable(bindir / "curl", FAKE_CURL)
    (tmp_path / "served.tgz").write_bytes(package())
    return tmp_path, os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)}


def install(root: Path, env: dict[str, str], *, sha512: str | None = None) -> subprocess.CompletedProcess[str]:
    digest = sha512 or hashlib.sha512((root / "served.tgz").read_bytes()).hexdigest()
    return subprocess.run(
        [str(root / "bin/bash"), str(INSTALLER), str(root / "runtime"), "1.0.103", PACKAGE, URL, digest],
        env=env,
        cwd="/",
        capture_output=True,
        text=True,
        errors="replace",
        timeout=30,
    )


def downloads(root: Path) -> list[str]:
    log = root / "downloads.log"
    return log.read_text().splitlines() if log.exists() else []


def test_cold_install_verifies_and_unpacks_into_a_private_home(sandbox):
    root, env = sandbox
    assert shutil.which("python3", path=env["PATH"]) is None
    installed = install(root, env)
    assert installed.returncode == 0, installed.stderr
    runtime = root / "runtime"
    record = json.loads(installed.stdout.strip().splitlines()[-1])
    assert record == {
        "binary": f"{runtime}/copilot",
        "dist_dir": f"{runtime}/home/.cache/copilot/pkg/linux-x64/1.0.103",
        "package": PACKAGE,
        "version": "1.0.103",
        "sha512": hashlib.sha512((root / "served.tgz").read_bytes()).hexdigest(),
    }
    assert json.loads((runtime / "runtime.json").read_text()) == record
    assert (runtime / "ready").is_file() and os.access(record["binary"], os.X_OK)
    # Only the binary is kept from the package.
    assert sorted(path.name for path in runtime.iterdir()) == ["copilot", "home", "ready", "runtime.json"]
    assert not list(root.glob("runtime.partial-*"))
    assert downloads(root) == [URL]


def test_ready_runtime_is_reused_without_a_download(sandbox):
    root, env = sandbox
    first = install(root, env)
    second = install(root, env)
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip().splitlines()[-1] == first.stdout.strip().splitlines()[-1]
    assert downloads(root) == [URL]


def test_checksum_mismatch_leaves_no_runtime(sandbox):
    root, env = sandbox
    installed = install(root, env, sha512=hashlib.sha512(b"another package").hexdigest())
    assert installed.returncode != 0
    assert "does not match the registry's SHA-512" in installed.stderr
    assert not (root / "runtime").exists()
    assert not list(root.glob("runtime.partial-*"))


@pytest.mark.parametrize(
    "binary",
    [FAKE_BINARY.replace("1.0.103.", "1.0.99."), FAKE_BINARY.replace('echo "GitHub', 'exit 3\necho "GitHub')],
    ids=["wrong-version", "exit-nonzero"],
)
def test_failed_health_check_publishes_no_ready_runtime(sandbox, binary):
    root, env = sandbox
    (root / "served.tgz").write_bytes(package(binary))
    installed = install(root, env)
    assert installed.returncode != 0
    assert "failed its health check" in installed.stderr
    assert not (root / "runtime/ready").exists()
    assert not list(root.glob("runtime.partial-*"))


@pytest.mark.parametrize("damage", ["no-ready-marker", "truncated-binary"])
def test_interrupted_installation_is_replaced(sandbox, damage):
    root, env = sandbox
    assert install(root, env).returncode == 0
    # Killed after publishing, before the ready marker; or a binary cut short. Another installer was staging.
    if damage == "no-ready-marker":
        (root / "runtime/ready").unlink()
    else:
        (root / "runtime/copilot").write_text("truncated")
    (root / "runtime.partial-1").mkdir()
    installed = install(root, env)
    assert installed.returncode == 0, installed.stderr
    assert (root / "runtime/ready").is_file()
    assert "GitHub Copilot CLI 1.0.103." in (root / "runtime/copilot").read_text()
    assert not list(root.glob("runtime.partial-*"))
    assert downloads(root) == [URL, URL]


@pytest.mark.parametrize("uid", [0, 1000])
def test_missing_downloader_is_installed_only_as_root(sandbox, uid):
    root, env = sandbox
    (root / "bin/curl").rename(root / "curl")
    executable(
        root / "bin/apt-get",
        '#!/bin/bash\necho "$*" >> "$TEST_ROOT/packages.log"\n'
        'if [ "$1" = install ]; then cat "$TEST_ROOT/curl" > "$TEST_ROOT/bin/curl"; chmod 755 "$TEST_ROOT/bin/curl"; fi\n',
    )
    installed = install(root, env | {"TEST_UID": str(uid)})
    if uid:
        assert installed.returncode == 1
        assert "Copilot requires curl ca-certificates: preinstall these packages" in installed.stderr
        assert not (root / "packages.log").exists() and not downloads(root)
        return
    assert installed.returncode == 0, installed.stderr
    assert (root / "packages.log").read_text().splitlines() == [
        "update",
        "install -y --no-install-recommends curl ca-certificates",
    ]
    assert json.loads(installed.stdout.strip().splitlines()[-1])["version"] == "1.0.103"


@pytest.mark.parametrize("package_manager", ["apk", "apt-get"])
@pytest.mark.parametrize("failure", [None, "nonroot", "unavailable"])
def test_bootstrap_installs_bash_only_when_missing(tmp_path, package_manager, failure):
    """The supervisor, the harness runner and the installer need bash; the bootstrap itself is POSIX sh."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for name in ("mkdir", "ln"):
        (bindir / name).symlink_to(shutil.which(name))
    executable(bindir / "id", f"#!/bin/sh\necho {1000 if failure == 'nonroot' else 0}\n")
    if failure != "unavailable":
        executable(
            bindir / package_manager,
            '#!/bin/sh\nprintf "%s\\n" "$*" >> "$TEST_ROOT/packages.log"\n'
            f'case " $* " in *" bash "*) ln -s {shlex.quote(shutil.which("bash"))} "$TEST_ROOT/bin/bash" ;; esac\n',
        )
    directory = tmp_path / "session with spaces"
    prepared = subprocess.run(
        ["/bin/sh", "-c", _sandbox_prepare_command(str(directory))],
        env=os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        errors="replace",
        timeout=10,
    )
    if failure:
        assert prepared.returncode == 1
        assert "requires bash" in prepared.stderr and not directory.exists()
        return
    assert prepared.returncode == 0, prepared.stderr
    assert directory.is_dir() and (bindir / "bash").exists()
    assert (tmp_path / "packages.log").read_text().splitlines() == (
        ["add --no-cache bash"] if package_manager == "apk" else ["update", "install -y --no-install-recommends bash"]
    )
    # An image that has bash installs nothing.
    (tmp_path / "packages.log").unlink()
    again = subprocess.run(
        ["/bin/sh", "-c", _sandbox_prepare_command(str(directory))],
        env=os.environ | {"PATH": str(bindir), "TEST_ROOT": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert again.returncode == 0, again.stderr
    assert not (tmp_path / "packages.log").exists()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="publication uses GNU mv -T")
def test_no_lock_keeps_cold_install_and_reuse_but_never_removes_a_stale_runtime(sandbox):
    root, env = sandbox
    (root / "bin/flock").unlink()
    assert install(root, env).returncode == 0
    assert install(root, env).returncode == 0
    assert len(downloads(root)) == 1
    (root / "runtime/ready").unlink()
    (root / "runtime/keep").write_text("existing runtime")

    refused = install(root, env)

    assert refused.returncode != 0
    assert "Cannot replace stale Copilot runtime safely without flock" in refused.stderr
    assert (root / "runtime/keep").read_text() == "existing runtime"
    assert not list(root.glob("runtime.partial-*"))


def test_lock_wait_is_bounded_and_preserves_runtime(sandbox):
    root, env = sandbox
    (root / "bin/flock").unlink()
    executable(root / "bin/flock", '#!/bin/sh\n[ "$1 $2 $3" = "-w 60 9" ] || exit 2\nexit 1\n')
    (root / "runtime").mkdir()
    (root / "runtime/keep").write_text("existing runtime")

    refused = install(root, env)

    assert refused.returncode != 0
    assert "Timed out waiting for Copilot runtime lock" in refused.stderr
    assert (root / "runtime/keep").read_text() == "existing runtime"
    assert not downloads(root)
