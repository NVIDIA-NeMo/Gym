# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The in-sandbox installer: checksum before extraction, version check, reuse, and no partial runtime."""

import hashlib
import io
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


INSTALLER = Path(__file__).parents[1] / "install_goose_runtime.sh"


def archive(tmp_path: Path, version: str) -> Path:
    script = f"#!/bin/sh\necho ' {version}'\n".encode()
    path = tmp_path / "goose-x86_64-unknown-linux-musl.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo("goose")
        info.size, info.mode = len(script), 0o755
        tar.addfile(info, io.BytesIO(script))
    return path


def install(runtime: Path, source: Path, sha256: str, version: str = "1.54.0") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(INSTALLER), str(runtime), version, "x86_64", sha256, str(source)], capture_output=True, text=True
    )


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="needs coreutils sha256sum")
def test_installs_a_verified_archive_over_a_planted_binary(tmp_path):
    source = archive(tmp_path, "1.54.0")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    # A binary already at the runtime path is never trusted, even one that reports the pinned version.
    (runtime / "goose").write_text("#!/bin/sh\necho ' 1.54.0'\necho planted\n")
    assert install(runtime, source, hashlib.sha256(source.read_bytes()).hexdigest()).returncode == 0
    assert subprocess.run([runtime / "goose", "--version"], capture_output=True, text=True).stdout.strip() == "1.54.0"
    assert [path.name for path in runtime.iterdir()] == ["goose"]


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="needs coreutils sha256sum")
@pytest.mark.parametrize("problem", ["checksum", "version"])
def test_rejects_a_bad_archive_without_publishing_it(tmp_path, problem):
    source = archive(tmp_path, "1.53.0" if problem == "version" else "1.54.0")
    sha256 = "0" * 64 if problem == "checksum" else hashlib.sha256(source.read_bytes()).hexdigest()
    runtime = tmp_path / "runtime"
    result = install(runtime, source, sha256)
    assert result.returncode != 0
    if problem == "version":
        assert "expected Goose 1.54.0, got '1.53.0'" in result.stderr
    assert list(runtime.iterdir()) == []
