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
"""Runs the real ``anti_cheat_setup.sh`` with bash against a throwaway git repo built here.

The scrub must leave nothing reachable that comes after the base commit (later commits, their
tags, other branches, remotes). It must keep the tags on the base's ancestors: those reveal nothing
newer, and versioneer / setuptools-scm / ``git describe`` read them to compute the project's
version. With them gone the version becomes ``0+untagged...`` and, for example, pandas 2.1's own
pytest config turns the resulting pyarrow warning into an import error, so the agent cannot run
any test in its sandbox (issue #5745).

The script rewrites refs in its ``WORKING_DIRECTORY`` and runs ``git config --global``, so every
run here gets a minimal environment built from scratch: ``WORKING_DIRECTORY`` is the repo this test
created, ``HOME`` and ``GIT_CONFIG_GLOBAL`` point into ``tmp_path``, and no inherited ``GIT_*``
variable (``GIT_DIR`` from a git hook, say) can redirect git to another repository.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from resources_servers.swebench.anti_cheat import ANTI_CHEAT_SCRIPT_FPATH


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash on PATH"
)


def _hermetic_env(tmp_path: Path) -> dict[str, str]:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "GIT_CONFIG_GLOBAL": str(home / ".gitconfig"),
        "GIT_CONFIG_NOSYSTEM": "1",
        # Never let git walk up out of the test repo into an enclosing checkout.
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
        "LC_ALL": "C",
    }


class _Repo:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path.resolve()
        self.path = self.tmp_path / "repo"
        self.path.mkdir()
        self.env = _hermetic_env(self.tmp_path)
        self.git("init", "-q")
        self.git("symbolic-ref", "HEAD", "refs/heads/main")
        self._n = 0

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=self.path, env=self.env, capture_output=True, text=True, check=check)

    def out(self, *args: str) -> str:
        return self.git(*args).stdout.strip()

    def commit(self, message: str) -> str:
        self._n += 1
        (self.path / "file.txt").write_text(f"{self._n}\n")
        self.git("add", "file.txt")
        self.git("commit", "-q", "-m", message)
        return self.out("rev-parse", "HEAD")

    def refs(self) -> list[str]:
        return self.out("for-each-ref", "--format=%(refname)").splitlines()

    def has_object(self, sha: str) -> bool:
        return self.git("cat-file", "-e", sha, check=False).returncode == 0

    def run_scrub(self) -> dict[str, str]:
        """Run the script the way apply_anti_cheat_setup does (`bash anti_cheat_setup.sh` from the
        workdir, WORKING_DIRECTORY set) and return the fields of its GIT_CLEANUP line."""
        assert self.path.is_relative_to(self.tmp_path) and (self.path / ".git").is_dir()
        assert Path(self.out("rev-parse", "--show-toplevel")).resolve() == self.path
        # The script expands WORKING_DIRECTORY unquoted: whitespace or a glob character could send it elsewhere.
        if re.search(r"[\s*?\[\]]", str(self.path)):
            pytest.skip(f"temporary path {self.path} is unsafe for the script's unquoted WORKING_DIRECTORY")
        result = subprocess.run(
            ["bash", str(ANTI_CHEAT_SCRIPT_FPATH)],
            cwd=self.path,
            env=self.env | {"WORKING_DIRECTORY": str(self.path)},
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, result.stderr
        lines = [line for line in result.stdout.splitlines() if line.startswith("GIT_CLEANUP ")]
        assert len(lines) == 1, result.stdout
        assert "skipped" not in lines[0], lines[0]
        return dict(re.findall(r"(\w+)=(\S*)", lines[0]))


def _build_history(repo: _Repo) -> dict[str, str]:
    """c1 (v0.9, lightweight) -- c2 (v1.0, annotated) -- c3 = base, HEAD on main
    with c4 (v2.0, annotated; origin/main) after the base, and c5 (v1.1-side, lightweight; branch
    `feature`) forked off c2 -- neither c4 nor c5 is an ancestor of the base."""
    shas = {"c1": repo.commit("c1")}
    repo.git("tag", "v0.9")
    shas["c2"] = repo.commit("c2")
    repo.git("tag", "-a", "v1.0", "-m", "release 1.0")
    shas["c3"] = repo.commit("c3 (base)")
    repo.git("checkout", "-q", "-b", "next")
    shas["c4"] = repo.commit("c4 (the fix)")
    repo.git("tag", "-a", "v2.0", "-m", "release 2.0")
    shas["v2.0"] = repo.out("rev-parse", "refs/tags/v2.0")
    repo.git("remote", "add", "origin", "https://example.invalid/repo.git")
    repo.git("update-ref", "refs/remotes/origin/main", shas["c4"])
    repo.git("checkout", "-q", "-b", "feature", shas["c2"])
    shas["c5"] = repo.commit("c5 (side branch)")
    repo.git("tag", "v1.1-side")
    repo.git("checkout", "-q", "main")
    repo.git("branch", "-q", "-D", "next")
    shas["v1.0"] = repo.out("rev-parse", "refs/tags/v1.0")
    assert repo.out("rev-parse", "HEAD") == shas["c3"]
    return shas


@pytest.fixture(params=["loose_refs", "packed_refs"])
def scrubbed(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[_Repo, dict[str, str], dict[str, str]]:
    repo = _Repo(tmp_path)
    shas = _build_history(repo)
    if request.param == "packed_refs":
        repo.git("pack-refs", "--all")
        loose_tags = repo.path / ".git" / "refs" / "tags"
        assert not loose_tags.exists() or not any(loose_tags.iterdir())
    return repo, shas, repo.run_scrub()


class TestAntiCheatSetupScript:
    def test_keeps_tags_on_ancestors_of_the_base(self, scrubbed) -> None:
        repo, shas, _ = scrubbed
        assert {"refs/tags/v0.9", "refs/tags/v1.0"} <= set(repo.refs())
        assert repo.out("rev-parse", "refs/tags/v0.9") == shas["c1"]
        assert repo.out("rev-parse", "refs/tags/v1.0") == shas["v1.0"]  # the same annotated tag object
        assert repo.out("cat-file", "-t", "refs/tags/v1.0") == "tag"
        assert repo.out("describe", "--tags", "--abbrev=0") == "v1.0"
        assert re.fullmatch(r"v1\.0-1-g[0-9a-f]+", repo.out("describe", "--tags", "--long"))

    def test_removes_everything_after_the_base(self, scrubbed) -> None:
        repo, shas, _ = scrubbed
        assert set(repo.refs()) - {"refs/tags/v0.9", "refs/tags/v1.0"} == {"refs/heads/_nel_work"}
        assert repo.out("symbolic-ref", "HEAD") == "refs/heads/_nel_work"
        assert repo.out("rev-parse", "HEAD") == shas["c3"]
        assert repo.out("remote") == ""
        assert repo.out("rev-list", "--all", "--not", shas["c3"], "--count") == "0"
        for name in ("c4", "v2.0", "c5"):
            assert not repo.has_object(shas[name]), name
        assert (repo.path / "file.txt").read_text() == "3\n"

    def test_log_line_reports_kept_tags_and_no_non_ancestor_left(self, scrubbed) -> None:
        _, _, fields = scrubbed
        assert fields["non_ancestor_left"] == "0"
        assert fields["tags_kept"] == "2"
        assert fields["gc_rc"] == "0"


class TestAntiCheatSetupScriptEdgeCases:
    def test_a_repo_without_tags(self, tmp_path: Path) -> None:
        repo = _Repo(tmp_path)
        repo.commit("c1")
        base = repo.commit("c2 (base)")
        fields = repo.run_scrub()
        assert repo.refs() == ["refs/heads/_nel_work"]
        assert repo.out("rev-parse", "HEAD") == base
        assert fields["tags_kept"] == "0"
        assert fields["non_ancestor_left"] == "0"

    def test_tag_names_with_slashes(self, tmp_path: Path) -> None:
        repo = _Repo(tmp_path)
        c1 = repo.commit("c1")
        repo.git("tag", "-a", "release/1.0", "-m", "release 1.0")
        repo.commit("c2 (base)")
        repo.git("checkout", "-q", "-b", "next")
        c3 = repo.commit("c3 (the fix)")
        repo.git("tag", "release/2.0")
        repo.git("checkout", "-q", "main")
        repo.git("branch", "-q", "-D", "next")
        fields = repo.run_scrub()
        assert repo.refs() == ["refs/heads/_nel_work", "refs/tags/release/1.0"]
        assert repo.out("rev-parse", "refs/tags/release/1.0^{commit}") == c1
        assert not repo.has_object(c3)
        assert fields["tags_kept"] == "1"

    def test_replace_refs_do_not_make_later_commits_look_like_ancestors(self, tmp_path: Path) -> None:
        """A replace ref can graft a later commit into the base's history. The scrub deletes replace refs,
        so it must pick the tags to keep from the real history, or a kept tag would hold the later commit."""
        repo = _Repo(tmp_path)
        c1 = repo.commit("c1")
        repo.git("tag", "v1")
        repo.git("checkout", "-q", "-b", "side")
        fix = repo.commit("the fix")
        repo.git("tag", "-a", "v9-fix", "-m", "the fix")
        repo.git("checkout", "-q", "main")
        base = repo.commit("c2 (base)")
        repo.git("branch", "-q", "-D", "side")
        repo.git("replace", "--graft", base, c1, fix)
        fields = repo.run_scrub()
        assert repo.refs() == ["refs/heads/_nel_work", "refs/tags/v1"]
        assert not repo.has_object(fix)
        assert fields["tags_kept"] == "1"
        assert fields["non_ancestor_left"] == "0"

    def test_tags_on_trees_and_blobs_are_dropped(self, tmp_path: Path) -> None:
        """`--merged` only matches refs that peel to a commit, so tags on other objects go with the
        rest; the scrub never keeps a ref it cannot place relative to the base."""
        repo = _Repo(tmp_path)
        repo.commit("c1")
        base = repo.commit("c2 (base)")
        (tmp_path / "blob.txt").write_text("in no commit\n")
        blob = repo.out("hash-object", "-w", str(tmp_path / "blob.txt"))
        repo.git("tag", "a-blob", blob)
        repo.git("tag", "-a", "a-tree", "-m", "a tree", f"{base}^{{tree}}")
        fields = repo.run_scrub()
        assert repo.refs() == ["refs/heads/_nel_work"]
        assert fields["tags_kept"] == "0"
        assert fields["non_ancestor_left"] == "0"
