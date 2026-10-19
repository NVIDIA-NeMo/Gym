# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from responses_api_agents.claude_code_agent import install_claude_runtime as installer


@pytest.fixture
def binary(tmp_path):
    source = tmp_path / "source"
    source.write_text('#!/bin/sh\nprintf "2.1.108 (Claude Code)\\n"\n')
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


def test_offline_install_verifies_hash_and_reuses_healthy_cache(tmp_path, binary):
    source, checksum = binary
    runtime = tmp_path / "runtime"
    assert installer.install("2.1.108", runtime, source=str(source), expected=checksum) == "2.1.108 (Claude Code)"
    executable = runtime / "claude"
    original = executable.stat().st_ino
    source.unlink()
    assert installer.install("2.1.108", runtime, source=str(source), expected=checksum) == "2.1.108 (Claude Code)"
    assert executable.stat().st_ino == original
    assert executable.stat().st_mode & 0o777 == 0o700


def test_corrupted_cache_is_replaced_and_bad_source_cannot_replace_it(tmp_path, binary):
    source, checksum = binary
    runtime = tmp_path / "runtime"
    installer.install("2.1.108", runtime, source=str(source), expected=checksum)
    (runtime / "claude").write_bytes(b"corrupt")
    installer.install("2.1.108", runtime, source=str(source), expected=checksum)
    assert (runtime / "claude").read_bytes() == source.read_bytes()
    source.write_bytes(b"bad replacement")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        installer.install("2.1.108", runtime, source=str(source), expected="0" * 64)
    assert hashlib.sha256((runtime / "claude").read_bytes()).hexdigest() == checksum
    assert not list(runtime.glob(".claude-*"))


def test_concurrent_installers_share_one_verified_runtime(tmp_path, binary, monkeypatch):
    source, checksum = binary
    copy = MagicMock(wraps=installer.shutil.copyfile)
    monkeypatch.setattr(installer.shutil, "copyfile", copy)

    def install():
        return installer.install("2.1.108", tmp_path / "runtime", source=str(source), expected=checksum)

    with ThreadPoolExecutor(max_workers=4) as pool:
        versions = list(pool.map(lambda _: install(), range(4)))
    assert versions == ["2.1.108 (Claude Code)"] * 4
    assert copy.call_count == 1


def test_preinstalled_binary_needs_trusted_checksum_and_matching_version(tmp_path, binary):
    source, checksum = binary
    with pytest.raises(ValueError, match="trusted SHA-256"):
        installer.install("2.1.108", tmp_path / "runtime", source=str(source))
    with pytest.raises(ValueError, match="version mismatch"):
        installer.install("2.1.109", tmp_path / "runtime", source=str(source), expected=checksum)
    assert not (tmp_path / "runtime" / "claude").exists()


def test_download_uses_manifest_checksum_and_cached_receipt(tmp_path, binary, monkeypatch):
    source, checksum = binary
    content = source.read_bytes()
    monkeypatch.setattr(installer.platform, "system", lambda: "Linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(installer.platform, "libc_ver", lambda: ("glibc", "2.35"))

    def fetch(url, timeout):
        assert url.startswith("https://downloads.claude.ai/claude-code-releases/2.1.108/")
        if url.endswith("manifest.json"):
            return io.BytesIO(json.dumps({"platforms": {"linux-x64": {"checksum": checksum}}}).encode())
        assert url.endswith("linux-x64/claude")
        return io.BytesIO(content)

    download = MagicMock(side_effect=fetch)
    monkeypatch.setattr(installer.urllib.request, "urlopen", download)
    installer.install("2.1.108", tmp_path / "runtime")
    assert download.call_count == 2
    installer.install("2.1.108", tmp_path / "runtime")
    assert download.call_count == 2


def test_installer_and_runner_accept_python38_syntax():
    for filename in ("install_claude_runtime.py", "sandbox_runner.py"):
        path = Path(installer.__file__).with_name(filename)
        ast.parse(path.read_text(), feature_version=(3, 8))
    assert "file_digest" not in Path(installer.__file__).read_text()
