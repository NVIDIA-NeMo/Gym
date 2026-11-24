# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import io
import os
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
    assert sorted(path.name for path in (tmp_path / "runtime").iterdir()) == ["droid", "fallback-bin", "receipt"]
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


def fallback_ps(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    """Run the installed fallback ps with no other ps on PATH, as in an image without procps."""
    source = tmp_path / "droid-package.tgz"
    if not (tmp_path / "runtime").exists():
        assert install(tmp_path, package(source), "--source", str(source)).returncode == 0
    return subprocess.run(
        [str(tmp_path / "runtime" / "fallback-bin" / "ps"), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PATH": SYSTEM_PATH},
    )


@pytest.mark.skipif(not Path("/proc/self/stat").is_file(), reason="the fallback ps reads /proc")
def test_fallback_ps_answers_the_parent_lookup_droid_makes(tmp_path):
    # Droid 1.1.0 and 1.12.1 run exactly `ps -o ppid= -p PID`, once per Execute call. A process whose name holds
    # spaces and parentheses must not shift the fields.
    child = tmp_path / "a ) (b"
    child.write_text("#!/bin/sh\nsleep 30\n")
    child.chmod(0o755)
    process = subprocess.Popen([str(child)])
    try:
        for pid, parent in ((process.pid, os.getpid()), (os.getpid(), os.getppid())):
            answer = fallback_ps(tmp_path, "-o", "ppid=", "-p", str(pid))
            assert (answer.returncode, answer.stdout.strip(), answer.stderr) == (0, str(parent), "")
    finally:
        process.kill()
        process.wait()


@pytest.mark.parametrize(
    "args",
    [("-o", "ppid=", "-p", "4194305"), ("-o", "ppid=", "-p", "../self"), ("-o", "ppid=", "-p", "")],
    ids=["no such process", "not a pid", "empty pid"],
)
def test_fallback_ps_fails_like_ps_for_a_process_it_cannot_find(tmp_path, args):
    answer = fallback_ps(tmp_path, *args)
    assert (answer.returncode, answer.stdout, answer.stderr) == (1, "", "")


@pytest.mark.parametrize("args", [(), ("aux",), ("-o", "pid=", "-p", "1"), ("-o", "ppid=", "-p", "1", "2")])
def test_fallback_ps_says_what_it_is_for_any_other_call(tmp_path, args):
    answer = fallback_ps(tmp_path, *args)
    assert answer.returncode == 1 and answer.stdout == ""
    assert "this image has no procps" in answer.stderr


@pytest.mark.parametrize("damage", ["missing", "changed", "not executable"])
def test_runtime_without_this_fallback_ps_is_reinstalled(tmp_path, damage):
    # A runtime left by an adapter that had no fallback ps, or another one, is replaced like a changed binary.
    source = tmp_path / "droid-package.tgz"
    sha256 = package(source)
    assert install(tmp_path, sha256, "--source", str(source)).returncode == 0
    fallback = tmp_path / "runtime" / "fallback-bin" / "ps"
    expected = fallback.read_text()
    if damage == "missing":
        shutil.rmtree(fallback.parent)
    elif damage == "changed":
        fallback.write_text("#!/bin/sh\nexit 0\n")
    else:
        fallback.chmod(0o644)
    assert install(tmp_path, sha256, "--check-only").returncode == 3
    package(source)
    assert install(tmp_path, sha256, "--source", str(source)).returncode == 0
    assert fallback.read_text() == expected and os.access(fallback, os.X_OK)


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
