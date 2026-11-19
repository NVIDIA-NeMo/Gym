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
"""The installer on a small fake upstream tree (its own manifest; the venv build replaced), and the worker's helpers."""

import hashlib
import json
import sys
import tarfile
from pathlib import Path

import pytest

from responses_api_agents.arc_agi_3_official_agent import setup_official_harness as setup


FILES = {
    "LICENSE": "MIT License\n",
    "benchmarking/__init__.py": "",
    "benchmarking/agent.py": "VALUE = 1\n",
}


def fake_python(dest: Path, body: str) -> None:
    python = dest / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(f"#!/bin/sh\n{body}\n")
    python.chmod(0o755)


def fake_tree(root: Path) -> Path:
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


@pytest.fixture
def fake_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    manifest = {
        "provenance": {"commit": "c0ffee", "repository": "https://example.invalid/repo"},
        "files": {rel: hashlib.sha256(text.encode()).hexdigest() for rel, text in FILES.items()},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(setup, "MANIFEST_PATH", path)
    builds = []

    def build_venv(dest: Path) -> None:  # a "venv" whose python runs this interpreter
        builds.append(dest)
        fake_python(dest, f'exec "{sys.executable}" "$@"')

    monkeypatch.setattr(setup, "_build_venv", build_venv)
    manifest["builds"] = builds
    return manifest


def test_install_from_a_directory_and_reuse(tmp_path: Path, fake_manifest: dict) -> None:
    source = fake_tree(tmp_path / "source")
    (source / ".git").mkdir()  # a checkout's .git is not part of the tree
    (source / ".git" / "HEAD").write_text("ref")
    dest = tmp_path / "install"
    assert setup.ensure_official_harness(str(source), dest) == dest
    assert sorted(setup._tree_files(dest / "src")) == sorted(FILES)  # .git and caches are not part of it
    assert setup.ensure_official_harness(None, dest) == dest  # verified, not rebuilt
    assert fake_manifest["builds"] == [dest]
    assert not dest.with_name("install.partial").exists()


def test_install_from_a_tarball_with_one_top_directory(tmp_path: Path, fake_manifest: dict) -> None:
    source = fake_tree(tmp_path / "repo-c0ffee")
    archive = tmp_path / "repo.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="repo-c0ffee")
    dest = tmp_path / "install"
    setup.ensure_official_harness(str(archive), dest)
    setup.verify_install(dest)


def test_a_changed_tree_is_refused_and_nothing_is_installed(tmp_path: Path, fake_manifest: dict) -> None:
    source = fake_tree(tmp_path / "source")
    (source / "benchmarking/agent.py").write_text("VALUE = 2\n")
    dest = tmp_path / "install"
    with pytest.raises(ValueError, match="sha256 differs: benchmarking/agent.py"):
        setup.ensure_official_harness(str(source), dest)
    assert not dest.exists() and fake_manifest["builds"] == []


def test_a_failed_venv_build_removes_the_install(
    tmp_path: Path, fake_manifest: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(dest: Path) -> None:
        raise RuntimeError("uv sync failed")

    monkeypatch.setattr(setup, "_build_venv", broken)
    dest = tmp_path / "install"
    with pytest.raises(RuntimeError, match="uv sync failed"):
        setup.ensure_official_harness(str(fake_tree(tmp_path / "source")), dest)
    assert not dest.exists()


def test_an_install_without_venv_or_with_a_broken_one_fails_verification(tmp_path: Path, fake_manifest: dict) -> None:
    dest = tmp_path / "install"
    fake_tree(dest / "src")
    with pytest.raises(FileNotFoundError, match="no venv"):
        setup.verify_install(dest)
    fake_python(dest, "echo 'ModuleNotFoundError: openai' >&2; exit 1")
    with pytest.raises(RuntimeError, match="cannot import the harness: ModuleNotFoundError: openai"):
        setup.verify_install(dest)
    (dest / "src/benchmarking/agent.py").write_text("VALUE = 2\n")
    with pytest.raises(ValueError, match="sha256 differs"):
        setup.verify_install(dest)


def test_unknown_sources_are_refused(tmp_path: Path, fake_manifest: dict) -> None:
    with pytest.raises(FileNotFoundError, match="not 'git'"):
        setup.ensure_official_harness(str(tmp_path / "missing"), tmp_path / "install")
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no benchmarking"):
        setup.ensure_official_harness(str(tmp_path / "empty"), tmp_path / "install")


def test_build_venv_needs_uv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    with pytest.raises(FileNotFoundError, match="uv"):
        setup._build_venv(tmp_path)
