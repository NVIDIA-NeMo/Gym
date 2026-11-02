# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify release selection and fail-closed installation without network access."""

import hashlib
import io
import shutil
import subprocess
import tarfile
from unittest.mock import Mock

import pytest

from responses_api_agents.pool_agent import install_pool


@pytest.fixture
def release(tmp_path, monkeypatch):
    archive = tmp_path / "release.tar.gz"

    def make(*, version="1.0.16", symlink=False):
        with tarfile.open(archive, "w:gz") as bundle:
            for target in install_pool.RELEASE_SHA256["v1.0.16"]:
                binary = f"#!/bin/sh\necho {version}\n".encode()
                member = tarfile.TarInfo(f"pool-{target}")
                if symlink:
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/bin/sh"
                else:
                    member.size = len(binary)
                bundle.addfile(member, None if symlink else io.BytesIO(binary))
        monkeypatch.setattr(
            install_pool,
            "RELEASE_SHA256",
            {
                "v1.0.16": dict.fromkeys(
                    install_pool.RELEASE_SHA256["v1.0.16"], hashlib.sha256(archive.read_bytes()).hexdigest()
                )
            },
        )

    make()
    run = subprocess.run

    def download(command, **kwargs):
        if command[0] == "curl":
            shutil.copyfile(archive, command[-1])
            return subprocess.CompletedProcess(command, 0)
        return run(command, **kwargs)

    calls = Mock(side_effect=download)
    monkeypatch.setattr(install_pool.subprocess, "run", calls)
    return archive, make, calls


@pytest.mark.parametrize("system", ["Linux", "Darwin"])
@pytest.mark.parametrize(
    "machine,arch", [("x86_64", "amd64"), ("amd64", "amd64"), ("aarch64", "arm64"), ("arm64", "arm64")]
)
def test_verified_release_installs_only_after_hash_and_version_checks(
    tmp_path, monkeypatch, release, system, machine, arch
):
    monkeypatch.setattr(install_pool.platform, "system", lambda: system)
    monkeypatch.setattr(install_pool.platform, "machine", lambda: machine)
    _, _, calls = release
    root = tmp_path / "session with spaces"
    install_pool.install(root, version="v1.0.16", accept_eula=True)
    command = calls.call_args_list[0].args[0]
    assert command[command.index("--retry") + 1] == "3"
    assert command[-3] == f"https://downloads.poolside.ai/pool/v1.0.16/pool-{system.lower()}-{arch}.tar.gz"
    assert (root / "bin/pool").stat().st_mode & 0o111
    assert list(root.iterdir()) == [root / "bin"]


@pytest.mark.parametrize("failure", ["checksum", "version", "symlink", "download"])
def test_bad_release_never_installs_and_removes_temporary_download(tmp_path, release, failure):
    archive, make, calls = release
    if failure == "checksum":
        archive.write_bytes(b"untrusted content")
    elif failure == "version":
        make(version="1.0.17")
    elif failure == "symlink":
        make(symlink=True)
    else:
        calls.side_effect = subprocess.CalledProcessError(22, "curl")
    root = tmp_path / "session"
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        install_pool.install(root, version="v1.0.16", accept_eula=True)
    assert list(root.iterdir()) == []
    if failure != "version":
        assert calls.call_count == 1  # No unverified binary is executed.


@pytest.mark.parametrize("failure", ["eula", "version", "os", "arch"])
def test_rejects_unaccepted_terms_and_unpinned_targets_before_io(tmp_path, monkeypatch, failure):
    calls = Mock(side_effect=AssertionError("must not download or execute"))
    monkeypatch.setattr(install_pool.subprocess, "run", calls)
    monkeypatch.setattr(install_pool.platform, "system", lambda: "Windows" if failure == "os" else "Linux")
    monkeypatch.setattr(install_pool.platform, "machine", lambda: "riscv64" if failure == "arch" else "x86_64")
    root = tmp_path / "session"
    with pytest.raises(ValueError, match="eula|No pinned Pool archive"):
        install_pool.install(
            root, version="v1.0.17" if failure == "version" else "v1.0.16", accept_eula=failure != "eula"
        )
    calls.assert_not_called()
    assert not root.exists()
