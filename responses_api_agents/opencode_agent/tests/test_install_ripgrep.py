# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
import tarfile

import pytest

from responses_api_agents.opencode_agent import install_ripgrep


def archive_bytes(binary_version="15.1.0", *, symlink=False):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as package:
        member = tarfile.TarInfo("ripgrep-15.1.0-x86_64-unknown-linux-musl/rg")
        binary = f"#!/bin/sh\necho 'ripgrep {binary_version} (rev fixture)'\n".encode()
        if symlink:
            member.type = tarfile.SYMTYPE
            member.linkname = "/bin/sh"
        else:
            member.size = len(binary)
        package.addfile(member, io.BytesIO(binary))
        # Unrelated archive paths must never be extracted, even with a verified archive.
        extra = tarfile.TarInfo("../../unexpected")
        extra.size = 1
        package.addfile(extra, io.BytesIO(b"x"))
    return stream.getvalue()


def prepare(monkeypatch, data):
    calls = []

    def download(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(data)

    monkeypatch.setattr(install_ripgrep.shutil, "which", lambda _: None)
    monkeypatch.setattr(install_ripgrep.urllib.request, "urlopen", download)
    return calls


def test_installs_verified_release_once_in_native_cache(tmp_path, monkeypatch):
    data = archive_bytes()
    calls = prepare(monkeypatch, data)
    destination = tmp_path / "cache/opencode/bin/rg"
    digest = hashlib.sha256(data).hexdigest()
    info = install_ripgrep.install("https://artifacts.example/rg.tar.gz", digest, "15.1.0", destination)
    assert info["source"] == "prefetched"
    assert info["archive_sha256"] == digest
    assert info["version"] == "ripgrep 15.1.0 (rev fixture)"
    assert install_ripgrep.version_of(destination) == "ripgrep 15.1.0 (rev fixture)"
    assert sorted(path.name for path in destination.parent.iterdir()) == ["rg"]
    assert not (tmp_path / "cache/unexpected").exists()
    assert install_ripgrep.install("unused", digest, "15.1.0", destination)["source"] == "cache"
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["digest", "version", "symlink"])
def test_invalid_release_never_installs_executable(tmp_path, monkeypatch, failure):
    data = archive_bytes("14.0.0" if failure == "version" else "15.1.0", symlink=failure == "symlink")
    prepare(monkeypatch, data)
    destination = tmp_path / "cache/opencode/bin/rg"
    digest = "0" * 64 if failure == "digest" else hashlib.sha256(data).hexdigest()
    with pytest.raises(RuntimeError):
        install_ripgrep.install("https://artifacts.example/rg.tar.gz", digest, "15.1.0", destination)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_preserves_native_system_binary_precedence(tmp_path, monkeypatch):
    system = tmp_path / "system-rg"
    system.write_text("#!/bin/sh\necho 'ripgrep 14.1.0'\n")
    system.chmod(0o755)
    monkeypatch.setattr(install_ripgrep.shutil, "which", lambda _: str(system))
    destination = tmp_path / "cache/opencode/bin/rg"
    info = install_ripgrep.install("invalid-url", "0" * 64, "15.1.0", destination)
    assert info == {"path": str(system), "version": "ripgrep 14.1.0", "source": "system"}
    assert not destination.exists()
