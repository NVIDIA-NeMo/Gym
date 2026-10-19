# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import hashlib
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


def test_installer_and_runner_accept_python38_syntax():
    for filename in ("install_claude_runtime.py", "sandbox_runner.py"):
        path = Path(installer.__file__).with_name(filename)
        ast.parse(path.read_text(), feature_version=(3, 8))
    tree = ast.parse(Path(installer.__file__).read_text())
    assert not any(isinstance(n, ast.Attribute) and n.attr == "file_digest" for n in ast.walk(tree))


def test_installer_verifies_decompressed_native_bytes(tmp_path, binary):
    import gzip

    source, checksum = binary
    compressed = tmp_path / "claude.gz"
    compressed.write_bytes(gzip.compress(source.read_bytes()))
    runtime = tmp_path / "runtime"
    installer.install("2.1.108", runtime, source=str(compressed), expected=checksum, compressed=True)
    assert (runtime / "claude").read_bytes() == source.read_bytes()
    assert installer.check_runtime("2.1.108", runtime, expected=checksum) == "2.1.108 (Claude Code)"
