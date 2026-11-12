# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import io
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


INSTALLER = Path(__file__).parents[1] / "install_droid_runtime.sh"
# The installer runs with this PATH only, so the skip guard looks for its tools there too.
SYSTEM_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"
pytestmark = pytest.mark.skipif(
    any(shutil.which(tool, path=SYSTEM_PATH) is None for tool in ("bash", "sha256sum", "timeout", "tar", "gzip")),
    reason="the installer needs bash, coreutils sha256sum and timeout, tar and gzip",
)


def package(path: Path, *, reported: str = "1.1.0") -> str:
    """Write an npm-style package whose droid writes state under its home, as the real one does."""
    script = (
        f'#!/bin/sh\nmkdir -p "${{FACTORY_HOME_OVERRIDE:-$HOME}}/.factory"\n[ "$1" = --version ] && echo {reported}\n'
    )
    data = script.encode()
    # A fixed gzip mtime keeps the bytes, and so the SHA-256, the same when a test writes the package again.
    with gzip.GzipFile(path, "wb", mtime=0) as compressed, tarfile.open(fileobj=compressed, mode="w") as archive:
        info = tarfile.TarInfo("package/bin/droid")
        info.size, info.mode = len(data), 0o755
        archive.addfile(info, io.BytesIO(data))
        readme = tarfile.TarInfo("package/README.md")
        archive.addfile(readme, io.BytesIO(b""))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(tmp_path: Path, sha256: str, *args: str) -> subprocess.CompletedProcess:
    home = tmp_path / "task-home"
    home.mkdir(exist_ok=True)
    return subprocess.run(
        [
            "bash",
            str(INSTALLER),
            "--version",
            "1.1.0",
            "--runtime",
            str(tmp_path / "runtime"),
            "--sha256",
            sha256,
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": SYSTEM_PATH, "HOME": str(home)},
    )


def test_cold_install_then_reuse_without_the_package(tmp_path):
    source = tmp_path / "droid-package.tgz"
    sha256 = package(source)
    assert install(tmp_path, sha256, "--check-only").returncode == 3
    installed = install(tmp_path, sha256, "--source", str(source))
    assert installed.returncode == 0, installed.stderr
    assert not source.exists()
    assert sorted(path.name for path in (tmp_path / "runtime").iterdir()) == ["droid", "receipt"]
    assert install(tmp_path, sha256, "--check-only").returncode == 0
    # The health check gives Droid a throwaway home, never the task's.
    assert not list((tmp_path / "task-home").iterdir())


def test_package_with_other_bytes_is_refused(tmp_path):
    source = tmp_path / "droid-package.tgz"
    package(source)
    failed = install(tmp_path, "0" * 64, "--source", str(source))
    assert failed.returncode == 1 and "SHA-256 mismatch" in failed.stderr
    assert not (tmp_path / "runtime").exists()


def test_wrong_version_is_not_installed(tmp_path):
    source = tmp_path / "droid-package.tgz"
    sha256 = package(source, reported="1.0.0")
    failed = install(tmp_path, sha256, "--source", str(source))
    assert failed.returncode != 0 and "version mismatch" in failed.stderr
    assert not (tmp_path / "runtime").exists()
    assert not list(tmp_path.glob("runtime.staging-*"))


def test_changed_binary_is_reinstalled(tmp_path):
    source = tmp_path / "droid-package.tgz"
    sha256 = package(source)
    assert install(tmp_path, sha256, "--source", str(source)).returncode == 0
    binary = tmp_path / "runtime" / "droid"
    binary.write_text("#!/bin/sh\necho 1.1.0\n")
    assert install(tmp_path, sha256, "--check-only").returncode == 3
    package(source)
    assert install(tmp_path, sha256, "--source", str(source)).returncode == 0
    assert "FACTORY_HOME_OVERRIDE" in binary.read_text()


def test_absent_runtime_without_package_fails(tmp_path):
    failed = install(tmp_path, "a" * 64)
    assert failed.returncode == 1 and "provide a transferred package" in failed.stderr


def test_concurrent_installers_share_one_runtime(tmp_path):
    sources = [tmp_path / f"droid-package-{index}.tgz" for index in range(3)]
    sha256 = package(sources[0])
    for source in sources[1:]:
        shutil.copyfile(sources[0], source)
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(lambda source: install(tmp_path, sha256, "--source", str(source)), sources))
    assert [result.returncode for result in results] == [0, 0, 0], [result.stderr for result in results]
    assert install(tmp_path, sha256, "--check-only").returncode == 0
