# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import tarfile
from pathlib import Path

import pytest

from responses_api_agents.opencode_sandboxed_agent import install_opencode as installer


@pytest.mark.parametrize(
    "machine,musl,flags,expected",
    [
        ("x86_64", False, "avx avx2 sse", "linux-x64"),
        ("x86_64", True, "avx2", "linux-x64-musl"),
        ("x86_64", False, "avx", "linux-x64-baseline"),
        ("x86_64", True, "", "linux-x64-baseline-musl"),
        ("aarch64", False, "", "linux-arm64"),
        ("aarch64", True, "", "linux-arm64-musl"),
    ],
)
def test_target_matches_cpu_and_libc(machine, musl, flags, expected):
    assert installer.linux_target(machine, musl, flags) == expected


def test_unsupported_architecture():
    with pytest.raises(ValueError, match="Unsupported"):
        installer.linux_target("unknown", False, "")


@pytest.mark.parametrize("version", ["latest", "1.2.3;echo bad", "", "v1.2.3"])
def test_requires_exact_version(version):
    with pytest.raises(ValueError, match="exact"):
        installer.install(version)


@pytest.mark.parametrize("archive_kind", ["binary", "wrong-version", "symlink", "traversal"])
def test_install_checks_archive_and_version_before_replacing(tmp_path, monkeypatch, archive_kind):
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    destination = tmp_path / ".opencode/bin/opencode"
    destination.parent.mkdir(parents=True)
    destination.write_text("existing installation")
    archive = io.BytesIO()
    version = "1.17.11" if archive_kind != "wrong-version" else "1.0.0"
    binary = f"#!/bin/sh\necho {version}\n".encode()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        member = tarfile.TarInfo("../opencode" if archive_kind == "traversal" else "opencode")
        if archive_kind == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/bin/sh"
        else:
            member.size = len(binary)
        tar.addfile(member, io.BytesIO(binary))
    downloads = []

    def download(url, timeout):
        downloads.append((url, timeout))
        return io.BytesIO(archive.getvalue())

    monkeypatch.setattr(installer, "urlopen", download)
    if archive_kind == "binary":
        installer.install("1.17.11")
        assert destination.read_bytes() == binary
        assert destination.stat().st_mode & 0o111
    else:
        with pytest.raises(ValueError):
            installer.install("1.17.11")
        assert destination.read_text() == "existing installation"
    assert len(downloads) == 1
    assert downloads[0][0].startswith("https://github.com/anomalyco/opencode/releases/download/v1.17.11/")
    assert downloads[0][1] == 120
    assert list(destination.parent.iterdir()) == [destination]
