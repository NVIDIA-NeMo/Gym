# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gzip
import hashlib
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from responses_api_agents.claude_code_agent import runtime as host_runtime


INSTALLER = Path(host_runtime.__file__).with_name("install_claude_runtime.sh")
pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("sha256sum") is None, reason="bash and sha256sum are required"
)


@pytest.fixture
def binary(tmp_path):
    source = tmp_path / "source"
    source.write_text('#!/bin/sh\necho run >> "$0.calls"\nprintf "2.1.108 (Claude Code)\\n"\n')
    return source, hashlib.sha256(source.read_bytes()).hexdigest()


def install(runtime: Path, *args: str, version: str = "2.1.108") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(INSTALLER), "--version", version, "--runtime", str(runtime), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_offline_install_verifies_hash_and_reuses_healthy_cache(tmp_path, binary):
    source, checksum = binary
    runtime = tmp_path / "runtime"
    first = install(runtime, "--sha256", checksum, "--source", str(source))
    assert (first.returncode, first.stdout) == (0, "2.1.108 (Claude Code)\n"), first.stderr
    executable = runtime / "claude"
    original = executable.stat().st_ino
    source.unlink()
    second = install(runtime, "--sha256", checksum.upper(), "--source", str(source))
    assert (second.returncode, second.stdout) == (0, "2.1.108 (Claude Code)\n"), second.stderr
    assert executable.stat().st_ino == original
    assert executable.stat().st_mode & 0o777 == 0o700


def test_corrupted_cache_is_replaced_and_bad_source_cannot_replace_it(tmp_path, binary):
    source, checksum = binary
    runtime = tmp_path / "runtime"
    assert install(runtime, "--sha256", checksum, "--source", str(source)).returncode == 0
    (runtime / "claude").write_bytes(b"corrupt")
    assert install(runtime, "--sha256", checksum, "--source", str(source)).returncode == 0
    assert (runtime / "claude").read_bytes() == source.read_bytes()
    source.write_bytes(b"bad replacement")
    failed = install(runtime, "--sha256", "0" * 64, "--source", str(source))
    assert failed.returncode != 0 and "SHA-256 mismatch" in failed.stderr
    assert hashlib.sha256((runtime / "claude").read_bytes()).hexdigest() == checksum
    assert not list(runtime.glob(".claude-??????"))  # temporary copies; the stub also logs .calls files


def test_concurrent_installers_share_one_verified_runtime(tmp_path, binary):
    source, checksum = binary
    runtime = tmp_path / "runtime"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: install(runtime, "--sha256", checksum, "--source", str(source)), range(4)))
    assert [(result.returncode, result.stdout) for result in results] == [(0, "2.1.108 (Claude Code)\n")] * 4
    # The stub logs each run under its own path: one temporary copy was made; the others reused the install.
    assert len(list(runtime.glob(".claude-*.calls"))) == 1
    assert (runtime / "claude.calls").read_text() == "run\n" * 3


def test_preinstalled_binary_needs_trusted_checksum_and_matching_version(tmp_path, binary):
    source, checksum = binary
    missing = install(tmp_path / "runtime", "--source", str(source))
    assert missing.returncode != 0 and "trusted SHA-256" in missing.stderr
    mismatch = install(tmp_path / "runtime", "--sha256", checksum, "--source", str(source), version="2.1.109")
    assert mismatch.returncode != 0 and "version mismatch" in mismatch.stderr
    assert not (tmp_path / "runtime" / "claude").exists()


def test_installer_verifies_decompressed_native_bytes_and_check_only_reports_absence(tmp_path, binary):
    source, checksum = binary
    runtime = tmp_path / "runtime"
    assert install(runtime, "--sha256", checksum, "--check-only").returncode == 3
    compressed = tmp_path / "claude.gz"
    compressed.write_bytes(gzip.compress(source.read_bytes()))
    assert install(runtime, "--sha256", checksum, "--source", str(compressed), "--gzip").returncode == 0
    assert (runtime / "claude").read_bytes() == source.read_bytes()
    checked = install(runtime, "--sha256", checksum, "--check-only")
    assert (checked.returncode, checked.stdout) == (0, "2.1.108 (Claude Code)\n")
