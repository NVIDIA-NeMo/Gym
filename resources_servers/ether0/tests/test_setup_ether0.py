# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check revision enforcement using local Git repositories without network access."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from resources_servers.ether0 import setup_ether0


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=path, encoding="utf-8", errors="replace").strip()


@pytest.fixture
def upstream(tmp_path: Path) -> tuple[Path, str, str]:
    path = tmp_path / "upstream"
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Ether0 setup test")
    _git(path, "config", "user.email", "test@example.invalid")
    (path / "src").mkdir()
    source = path / "src" / "version.py"
    source.write_text('VERSION = "first"\n')
    _git(path, "add", ".")
    _git(path, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "First revision")
    first = _git(path, "rev-parse", "HEAD")
    source.write_text('VERSION = "second"\n')
    _git(path, "-c", "commit.gpgsign=false", "commit", "--quiet", "-am", "Second revision")
    return path, first, _git(path, "rev-parse", "HEAD")


@pytest.fixture
def cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, upstream: tuple[Path, str, str]) -> Path:
    source, pinned, _ = upstream
    path = tmp_path / ".ether0"
    monkeypatch.setattr(setup_ether0, "ETHER0_PATH", path)
    monkeypatch.setattr(setup_ether0, "ETHER0_SRC_PATH", path / "src")
    monkeypatch.setattr(setup_ether0, "_REPO_URL", str(source))
    monkeypatch.setattr(setup_ether0, "_REPO_REVISION", pinned)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    return path


def test_matching_cache_is_reused_offline(
    cache: Path, upstream: tuple[Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source, pinned, _ = upstream
    shutil.copytree(source, cache)
    _git(cache, "checkout", "--quiet", "--detach", pinned)
    monkeypatch.setattr(setup_ether0, "_REPO_URL", str(cache.parent / "unavailable-upstream"))

    setup_ether0.ensure_ether0()
    setup_ether0.ensure_ether0()

    assert _git(cache, "rev-parse", "HEAD") == pinned
    assert sys.path.count(str(cache / "src")) == 1


def test_mismatched_cache_is_rejected_without_modifying_it(cache: Path, upstream: tuple[Path, str, str]) -> None:
    source, pinned, actual = upstream
    shutil.copytree(source, cache)
    tracked = cache / "src" / "version.py"
    tracked.write_text('VERSION = "user edit"\n')
    untracked = cache / "notes.txt"
    untracked.write_text("Keep these notes.\n")
    status = _git(cache, "status", "--porcelain")

    with pytest.raises(RuntimeError) as error:
        setup_ether0.ensure_ether0()

    assert str(cache) in str(error.value)
    assert pinned in str(error.value)
    assert actual in str(error.value)
    assert "Move" in str(error.value)
    assert str(cache / "src") not in sys.path
    assert _git(cache, "rev-parse", "HEAD") == actual
    assert _git(cache, "status", "--porcelain") == status
    assert tracked.read_text() == 'VERSION = "user edit"\n'
    assert untracked.read_text() == "Keep these notes.\n"


def test_fresh_checkout_uses_the_pinned_commit(cache: Path, upstream: tuple[Path, str, str]) -> None:
    _, pinned, _ = upstream

    setup_ether0.ensure_ether0()

    assert _git(cache, "rev-parse", "HEAD") == pinned
    assert (cache / "src" / "version.py").read_text() == 'VERSION = "first"\n'
    assert str(cache / "src") in sys.path
    assert not list(cache.parent.glob(".ether0-*"))


def test_failed_fetch_leaves_no_checkout(cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup_ether0, "_REPO_URL", str(cache.parent / "unavailable-upstream"))

    with pytest.raises(subprocess.CalledProcessError):
        setup_ether0.ensure_ether0()

    assert not cache.exists()
    assert not list(cache.parent.glob(".ether0-*"))
    assert str(cache / "src") not in sys.path
