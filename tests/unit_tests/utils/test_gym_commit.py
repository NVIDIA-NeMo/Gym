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

import json
import logging
import os
import subprocess
from importlib.metadata import PackageNotFoundError
from pathlib import Path

from pytest import MonkeyPatch

from nemo_gym.utils.gym_commit import installed_gym_commit


class _Distribution:
    def __init__(self, direct_url: dict | None) -> None:
        self.direct_url = direct_url

    def read_text(self, filename: str) -> str | None:
        assert filename == "direct_url.json"
        return None if self.direct_url is None else json.dumps(self.direct_url)


def _install(monkeypatch: MonkeyPatch, direct_url: dict | None) -> None:
    """What the distribution metadata says about the install; None for an index wheel."""
    monkeypatch.setattr("nemo_gym.utils.gym_commit.distribution", lambda name: _Distribution(direct_url))


def _not_installed(monkeypatch: MonkeyPatch) -> None:
    def missing(name: str):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("nemo_gym.utils.gym_commit.distribution", missing)


def _running_from(monkeypatch: MonkeyPatch, package_dir: Path) -> None:
    """Where the running nemo_gym package lives on disk."""
    monkeypatch.setattr("nemo_gym.utils.gym_commit._PACKAGE_DIR", package_dir)


_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}


def _checkout(repo: Path) -> tuple[Path, str]:
    """A one-commit checkout tracking `nemo_gym/__init__.py`; returns (package dir, HEAD)."""
    package_dir = repo / "nemo_gym"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "nemo_gym/__init__.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "seed"], check=True, env={**os.environ, **_GIT_IDENTITY}
    )
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return package_dir, head


def test_installed_gym_commit_reads_a_git_install(tmp_path: Path, monkeypatch: MonkeyPatch):
    """`pip install git+...@<sha>` records the sha in PEP 610 direct_url.json; the source is not consulted."""
    _install(
        monkeypatch,
        {
            "url": "https://github.com/NVIDIA-NeMo/gym.git",
            "vcs_info": {"vcs": "git", "commit_id": "c" * 40, "requested_revision": "main"},
        },
    )
    _running_from(monkeypatch, tmp_path / "site-packages" / "nemo_gym")
    assert installed_gym_commit() == "c" * 40


def test_installed_gym_commit_reads_a_clean_checkouts_head(tmp_path: Path, monkeypatch: MonkeyPatch):
    """An editable install records no commit; the running package's checkout is asked instead."""
    package_dir, head = _checkout(tmp_path)
    _install(monkeypatch, {"url": tmp_path.as_uri(), "dir_info": {"editable": True}})
    _running_from(monkeypatch, package_dir)
    assert installed_gym_commit() == head


def test_installed_gym_commit_marks_a_dirty_checkout(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Uncommitted edits ran too, so the HEAD alone must not read as the Gym that ran."""
    package_dir, head = _checkout(tmp_path)
    (package_dir / "__init__.py").write_text("edited")
    _install(monkeypatch, {"url": tmp_path.as_uri(), "dir_info": {"editable": True}})
    _running_from(monkeypatch, package_dir)
    assert installed_gym_commit() == f"{head}-dirty"


def test_installed_gym_commit_needs_no_distribution_metadata(tmp_path: Path, monkeypatch: MonkeyPatch):
    """A source tree on sys.path without any install (no dist-info) still names its checkout."""
    package_dir, head = _checkout(tmp_path)
    _not_installed(monkeypatch)
    _running_from(monkeypatch, package_dir)
    assert installed_gym_commit() == head


def test_installed_gym_commit_is_none_for_an_index_wheel(tmp_path: Path, monkeypatch: MonkeyPatch, caplog):
    """A wheel from an index records no commit and lives in site-packages, tracked by no checkout."""
    site = tmp_path / "site-packages" / "nemo_gym"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("")
    _install(monkeypatch, None)
    _running_from(monkeypatch, site)
    with caplog.at_level(logging.WARNING, logger="nemo_gym.utils.gym_commit"):
        assert installed_gym_commit() is None
    assert "not tracked in a git checkout" in caplog.text


def test_installed_gym_commit_ignores_a_checkout_the_venv_merely_sits_in(tmp_path: Path, monkeypatch: MonkeyPatch):
    """A `.venv` inside a Gym clone must not report the clone's HEAD for a wheel installed into it."""
    _checkout(tmp_path)
    site = tmp_path / ".venv" / "site-packages" / "nemo_gym"
    site.mkdir(parents=True)
    (site / "__init__.py").write_text("")
    _install(monkeypatch, None)
    _running_from(monkeypatch, site)
    assert installed_gym_commit() is None


def test_installed_gym_commit_is_none_without_git(tmp_path: Path, monkeypatch: MonkeyPatch, caplog):
    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr("nemo_gym.utils.gym_commit.subprocess.run", no_git)
    _install(monkeypatch, None)
    _running_from(monkeypatch, tmp_path)
    with caplog.at_level(logging.WARNING, logger="nemo_gym.utils.gym_commit"):
        assert installed_gym_commit() is None
    assert "git is not available" in caplog.text
